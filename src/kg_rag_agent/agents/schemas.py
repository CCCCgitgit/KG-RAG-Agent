# -*- coding: utf-8 -*-
"""Agent 门面层使用的标准请求与响应结构。

本模块只定义 Agent 边界上的可序列化数据，不依赖 LangGraph、RuntimeContext、
MemoryManager 或任何外部连接对象。
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, TypedDict


class RequestOptions(TypedDict, total=False):
    """允许在单次请求中覆盖的轻量参数。"""

    retrieval_top_k: int
    path_max_depth: int
    temperature: float
    max_tokens: int
    language: str
    include_citations: bool
    allowed_tools: List[str]


class MemoryStatus(TypedDict, total=False):
    """对外可返回的 Memory 执行摘要。

    该结构只包含计数和状态，不暴露长期 Memory 正文、会话摘要正文或底层
    Store 信息。
    """

    loaded: bool
    written: bool
    recent_message_count: int
    retrieved_memory_count: int
    estimated_tokens: int
    summary_used: bool
    written_count: int
    skipped_count: int


@dataclass(slots=True)
class AgentRequest:
    """标准化的单次 Agent 请求。"""

    query: str
    request_id: str
    session_id: str
    user_id: str = ""
    project_id: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_options: RequestOptions = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.query = str(self.query or "").strip()
        self.request_id = _normalize_identifier(self.request_id)
        self.session_id = _normalize_identifier(self.session_id)
        self.user_id = _normalize_identifier(self.user_id)
        self.project_id = _normalize_identifier(self.project_id)
        self.messages = _copy_dict_list(self.messages)
        self.chat_history = _copy_dict_list(self.chat_history)
        self.metadata = _copy_mapping(self.metadata)
        self.request_options = copy.deepcopy(dict(self.request_options or {}))

        if not self.query:
            raise ValueError("query must not be empty.")
        if not self.request_id:
            raise ValueError("request_id must not be empty.")
        if not self.session_id:
            raise ValueError("session_id must not be empty.")

    def state_metadata(self) -> Dict[str, Any]:
        """生成写入 AgentState.metadata 的轻量副本。"""

        metadata = copy.deepcopy(self.metadata)
        if self.request_options:
            metadata["request_options"] = copy.deepcopy(self.request_options)
        else:
            metadata.pop("request_options", None)
        return metadata


@dataclass(slots=True)
class AgentResult:
    """Agent 的稳定返回结构。"""

    answer: str
    request_id: str
    route: str = ""
    answerability: str = ""
    semantic_score: float = 0.0
    citations: List[Dict[str, Any]] = field(default_factory=list)
    traces: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    has_error: bool = False
    error_message: str = ""

    # 请求标识默认不进入旧版 API 响应；调用方可显式要求输出。
    session_id: str = ""
    user_id: str = ""
    project_id: str = ""

    # 只保存受控摘要，不保存 Memory 正文。
    memory_status: MemoryStatus = field(default_factory=dict)
    raw_state: Optional[Dict[str, Any]] = None

    @classmethod
    def from_state(
        cls,
        final_state: Mapping[str, Any],
        *,
        request_id: str,
        include_raw_state: bool = False,
        fallback_answer: str = "目前没有生成有效回答。你可以换一种问法，或者补充更多上下文后再试。",
    ) -> "AgentResult":
        """从 LangGraph 最终状态构造标准结果。"""

        state = dict(final_state or {})
        answer = str(state.get("final_answer", "") or "").strip()
        if not answer:
            answer = fallback_answer

        return cls(
            answer=answer,
            request_id=str(state.get("request_id", request_id) or request_id),
            route=str(state.get("route", "") or ""),
            answerability=str(state.get("answerability", "") or ""),
            semantic_score=_safe_float(state.get("semantic_score", 0.0)),
            citations=_copy_dict_list(state.get("citations")),
            traces=_copy_dict_list(state.get("traces")),
            warnings=[str(item) for item in (state.get("warnings") or [])],
            has_error=bool(state.get("has_error", False)),
            error_message=str(state.get("error_message", "") or ""),
            session_id=_normalize_identifier(state.get("session_id")),
            user_id=_normalize_identifier(state.get("user_id")),
            project_id=_normalize_identifier(state.get("project_id")),
            memory_status=_build_memory_status(state),
            raw_state=copy.deepcopy(state) if include_raw_state else None,
        )

    @classmethod
    def error(
        cls,
        *,
        request_id: str,
        message: str,
        answer: str = "抱歉，我刚刚处理这个问题时遇到了一点问题。你可以换一种问法，或者补充更多上下文后再试。",
        session_id: str = "",
        user_id: str = "",
        project_id: str = "",
    ) -> "AgentResult":
        """构造标准错误结果。"""

        return cls(
            answer=answer,
            request_id=_normalize_identifier(request_id),
            route="error",
            answerability="unanswerable",
            semantic_score=0.0,
            has_error=True,
            error_message=str(message or "unknown error"),
            session_id=_normalize_identifier(session_id),
            user_id=_normalize_identifier(user_id),
            project_id=_normalize_identifier(project_id),
        )

    def to_dict(
        self,
        *,
        include_raw_state: bool = False,
        include_identifiers: bool = False,
        include_memory_status: bool = False,
    ) -> Dict[str, Any]:
        """转换为可 JSON 化字典。

        ``include_identifiers`` 与 ``include_memory_status`` 默认关闭，以保持现有
        FastAPI ``ChatResponse`` 的字段集合兼容。后续 API Schema 完成 Memory
        对接后可显式开启。
        """

        data = asdict(self)
        if not include_raw_state:
            data.pop("raw_state", None)
        if not include_identifiers:
            data.pop("session_id", None)
            data.pop("user_id", None)
            data.pop("project_id", None)
        if not include_memory_status:
            data.pop("memory_status", None)
        return data


def _build_memory_status(state: Mapping[str, Any]) -> MemoryStatus:
    context = state.get("memory_context")
    context_map = dict(context) if isinstance(context, Mapping) else {}

    write_result = state.get("memory_write_result")
    write_map = dict(write_result) if isinstance(write_result, Mapping) else {}

    recent_messages = context_map.get("recent_messages")
    memories = context_map.get("memories")
    skipped = write_map.get("skipped")

    return MemoryStatus(
        loaded=bool(state.get("memory_loaded", False)),
        written=bool(state.get("memory_written", False)),
        recent_message_count=_safe_len(recent_messages),
        retrieved_memory_count=_safe_len(memories),
        estimated_tokens=_safe_non_negative_int(
            context_map.get("estimated_tokens", 0)
        ),
        summary_used=bool(str(context_map.get("summary", "") or "").strip()),
        written_count=_safe_non_negative_int(
            write_map.get("written_count", 0)
        ),
        skipped_count=_safe_len(skipped),
    )


def _normalize_identifier(value: Any) -> str:
    return str(value or "").strip()


def _copy_mapping(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return copy.deepcopy(dict(value))


def _copy_dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [copy.deepcopy(dict(item)) for item in value if isinstance(item, Mapping)]


def _safe_len(value: Any) -> int:
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return 0


def _safe_non_negative_int(value: Any, default: int = 0) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "AgentRequest",
    "AgentResult",
    "MemoryStatus",
    "RequestOptions",
]
