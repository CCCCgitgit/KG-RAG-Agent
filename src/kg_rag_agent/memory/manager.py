# -*- coding: utf-8 -*-
"""MemoryManager：Memory 模块对外统一门面。"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Mapping, Optional

from .models import (
    MemoryCandidate,
    MemoryContext,
    MemoryQuery,
    MemoryStatus,
    MemoryWriteResult,
)
from .policies import MemoryPolicy
from .retriever import MemoryRetriever, estimate_tokens, trim_text_to_token_budget
from .stores import InMemoryMemoryStore, JSONMemoryStore, MemoryStore
from .summarizer import ConversationSummarizer
from .writer import MemoryWriter


class MemoryManager:
    """管理短期消息、会话摘要和长期 Memory。"""

    def __init__(
        self,
        *,
        policy: MemoryPolicy,
        store: Optional[MemoryStore] = None,
        retriever: Optional[MemoryRetriever] = None,
        writer: Optional[MemoryWriter] = None,
        summarizer: Optional[ConversationSummarizer] = None,
    ) -> None:
        self.policy = policy
        self.store = store or InMemoryMemoryStore()
        self.retriever = retriever or MemoryRetriever(self.store)
        self.writer = writer or MemoryWriter(self.store, self.policy)
        self.summarizer = summarizer or ConversationSummarizer()
        self._recent: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.policy.max_messages)
        )
        self._summaries: dict[str, str] = {}
        self._lock = RLock()
        self._closed = False

    @classmethod
    def from_config(
        cls,
        config: Optional[Mapping[str, Any]],
        *,
        project_root: str | Path | None = None,
        llm_client: Any = None,
        store: Optional[MemoryStore] = None,
    ) -> "MemoryManager":
        raw = dict(config or {})
        policy = MemoryPolicy.from_config(raw)
        selected_store = store
        if selected_store is None:
            store_type = str(raw.get("store_type") or "in_memory").strip().lower()
            if store_type in {"json", "persistent", "file"}:
                root = Path(project_root or Path.cwd()).expanduser().resolve()
                configured_path = raw.get("store_path") or "outputs/memory/memories.json"
                path = Path(str(configured_path)).expanduser()
                if not path.is_absolute():
                    path = root / path
                selected_store = JSONMemoryStore(path)
            elif store_type == "in_memory":
                selected_store = InMemoryMemoryStore()
            else:
                raise ValueError(f"Unsupported memory store_type: {store_type}")
        return cls(
            policy=policy,
            store=selected_store,
            summarizer=ConversationSummarizer(llm_client=llm_client),
        )

    @property
    def enabled(self) -> bool:
        return self.policy.enabled

    def add_messages(
        self,
        *,
        session_id: str,
        messages: Iterable[Mapping[str, Any]],
        user_id: str = "",
        project_id: str = "",
    ) -> None:
        self._ensure_open()
        key = _session_key(
            session_id,
            user_id=user_id,
            project_id=project_id,
        )
        normalized = _normalize_messages(messages)
        if not normalized:
            return
        with self._lock:
            self._recent[key].extend(normalized)

    def get_recent_messages(
        self,
        session_id: str,
        *,
        user_id: str = "",
        project_id: str = "",
    ) -> list[dict[str, Any]]:
        self._ensure_open()
        key = _session_key(
            session_id,
            user_id=user_id,
            project_id=project_id,
        )
        with self._lock:
            return [dict(item) for item in self._recent.get(key, ())]

    def summarize_session(
        self,
        *,
        session_id: str,
        user_id: str = "",
        project_id: str = "",
        use_llm: bool = False,
        clear_summarized_messages: bool = False,
    ) -> str:
        self._ensure_open()
        key = _session_key(
            session_id,
            user_id=user_id,
            project_id=project_id,
        )
        with self._lock:
            messages = [dict(item) for item in self._recent.get(key, ())]
            previous = self._summaries.get(key, "")
        summary = self.summarizer.summarize(
            messages,
            token_budget=self.policy.max_summary_tokens,
            existing_summary=previous,
            use_llm=use_llm,
        )
        with self._lock:
            self._summaries[key] = summary
            if clear_summarized_messages:
                self._recent.pop(key, None)
        return summary

    def get_summary(
        self,
        session_id: str,
        *,
        user_id: str = "",
        project_id: str = "",
    ) -> str:
        self._ensure_open()
        key = _session_key(
            session_id,
            user_id=user_id,
            project_id=project_id,
        )
        with self._lock:
            return self._summaries.get(key, "")

    def load_context(
        self,
        *,
        query: str,
        user_id: str = "",
        project_id: str = "",
        session_id: str = "",
        namespace: Optional[str] = None,
        memory_types: tuple[Any, ...] = (),
    ) -> MemoryContext:
        self._ensure_open()
        if not self.policy.enabled:
            return MemoryContext()
        resolved_namespace = namespace or self.policy.namespace(project_id)
        self.policy.validate_read_scope(
            namespace=resolved_namespace,
            user_id=user_id,
        )
        recent = (
            self.get_recent_messages(
                session_id,
                user_id=user_id,
                project_id=project_id,
            )
            if session_id
            else []
        )
        summary = (
            self.get_summary(
                session_id,
                user_id=user_id,
                project_id=project_id,
            )
            if session_id
            else ""
        )
        memories = []
        if self.policy.long_term_enabled:
            memories = self.retriever.retrieve(
                MemoryQuery(
                    namespace=resolved_namespace,
                    user_id=user_id,
                    project_id=project_id,
                    session_id=session_id,
                    query=query,
                    top_k=self.policy.max_retrieved_items,
                    min_score=self.policy.min_relevance_score,
                    memory_types=tuple(memory_types),
                )
            )
        text = self._compose_context_text(
            recent_messages=recent,
            summary=summary,
            memories=memories,
        )
        text = trim_text_to_token_budget(text, self.policy.max_context_tokens)
        return MemoryContext(
            recent_messages=recent,
            summary=summary,
            memories=memories,
            text=text,
            estimated_tokens=estimate_tokens(text),
        )

    def write_candidates(
        self,
        candidates: Iterable[MemoryCandidate | Mapping[str, Any]],
        *,
        user_id: str,
        project_id: str = "",
        session_id: str = "",
        namespace: Optional[str] = None,
    ) -> MemoryWriteResult:
        self._ensure_open()
        resolved_namespace = namespace or self.policy.namespace(project_id)
        return self.writer.write(
            candidates,
            namespace=resolved_namespace,
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
        )

    def forget(self, memory_id: str, *, hard: bool = False) -> bool:
        self._ensure_open()
        return self.store.delete(memory_id, hard=hard)

    def supersede(self, memory_id: str) -> bool:
        self._ensure_open()
        return self.store.set_status(memory_id, MemoryStatus.SUPERSEDED) is not None

    def clear_session(
        self,
        session_id: str,
        *,
        user_id: str = "",
        project_id: str = "",
    ) -> None:
        self._ensure_open()
        key = _session_key(
            session_id,
            user_id=user_id,
            project_id=project_id,
        )
        with self._lock:
            self._recent.pop(key, None)
            self._summaries.pop(key, None)

    def summary(self) -> dict[str, Any]:
        self._ensure_open()
        with self._lock:
            sessions = len(set(self._recent).union(self._summaries))
        return {
            "enabled": self.policy.enabled,
            "write_enabled": self.policy.write_enabled,
            "long_term_enabled": self.policy.long_term_enabled,
            "store": type(self.store).__name__,
            "session_count": sessions,
            "namespace_prefix": self.policy.namespace_prefix,
        }

    def close(self) -> None:
        if self._closed:
            return
        close = getattr(self.store, "close", None)
        if callable(close):
            close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MemoryManager has already been closed.")

    def _compose_context_text(
        self,
        *,
        recent_messages: list[dict[str, Any]],
        summary: str,
        memories: list[Any],
    ) -> str:
        sections: list[str] = []
        if summary.strip():
            sections.append("[会话摘要]\n" + summary.strip())
        if memories:
            lines = ["[相关长期记忆]"]
            for item in memories:
                lines.append(
                    f"- ({item.record.memory_type.value}, {item.score:.3f}) "
                    f"{item.record.content}"
                )
            sections.append("\n".join(lines))
        if recent_messages:
            lines = ["[最近对话]"]
            for message in recent_messages:
                role = str(message.get("role") or "unknown")
                content = str(message.get("content") or "").strip()
                if content:
                    lines.append(f"- {role}: {content}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)


def _session_key(
    session_id: str,
    *,
    user_id: str = "",
    project_id: str = "",
) -> str:
    """构造短期 Memory 的租户隔离键。

    相同 ``session_id`` 在不同用户或项目中必须映射到不同缓冲区。
    长度前缀避免分隔符出现在标识符中时产生碰撞。
    """

    session = str(session_id or "").strip()
    if not session:
        raise ValueError("session_id must not be empty")

    user = str(user_id or "").strip() or "__anonymous__"
    project = str(project_id or "").strip() or "__default__"
    return "|".join(
        (
            f"{len(user)}:{user}",
            f"{len(project)}:{project}",
            f"{len(session)}:{session}",
        )
    )


def _normalize_messages(messages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, Mapping):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        result.append(
            {
                "role": str(item.get("role") or "unknown").strip() or "unknown",
                "content": content,
                **({"timestamp": item["timestamp"]} if "timestamp" in item else {}),
            }
        )
    return result


__all__ = ["MemoryManager"]
