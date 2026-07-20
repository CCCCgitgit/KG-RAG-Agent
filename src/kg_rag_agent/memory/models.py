# -*- coding: utf-8 -*-
"""Memory 领域数据结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Optional
from uuid import uuid4


def utc_now() -> str:
    """返回 UTC ISO-8601 时间。"""

    return datetime.now(timezone.utc).isoformat()


def normalize_scope(value: object) -> str:
    return str(value or "").strip()


class MemoryType(str, Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    CONSTRAINT = "constraint"
    PROJECT = "project"
    TASK = "task"
    SUMMARY = "summary"
    NOTE = "note"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"
    EXPIRED = "expired"


def _as_memory_type(value: MemoryType | str) -> MemoryType:
    return value if isinstance(value, MemoryType) else MemoryType(str(value))


def _as_memory_status(value: MemoryStatus | str) -> MemoryStatus:
    return value if isinstance(value, MemoryStatus) else MemoryStatus(str(value))


@dataclass(slots=True)
class MemoryRecord:
    """一条可持久化的长期 Memory。"""

    memory_id: str
    user_id: str
    namespace: str
    memory_type: MemoryType
    content: str
    source: str
    created_at: str
    updated_at: str
    confidence: float = 1.0
    status: MemoryStatus = MemoryStatus.ACTIVE
    project_id: str = ""
    session_id: str = ""
    expires_at: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.memory_id = normalize_scope(self.memory_id) or f"mem_{uuid4().hex}"
        self.user_id = normalize_scope(self.user_id)
        self.project_id = normalize_scope(self.project_id)
        self.session_id = normalize_scope(self.session_id)
        self.namespace = normalize_scope(self.namespace)
        self.content = str(self.content or "").strip()
        self.source = str(self.source or "").strip()
        self.confidence = min(max(float(self.confidence), 0.0), 1.0)
        if not self.namespace:
            raise ValueError("namespace must not be empty")
        if not self.content:
            raise ValueError("memory content must not be empty")
        self.memory_type = _as_memory_type(self.memory_type)
        self.status = _as_memory_status(self.status)
        self.metadata = dict(self.metadata or {})
        self.created_at = str(self.created_at or utc_now())
        self.updated_at = str(self.updated_at or self.created_at)
        if self.expires_at is not None:
            self.expires_at = str(self.expires_at)

    @classmethod
    def create(
        cls,
        *,
        user_id: str,
        namespace: str,
        memory_type: MemoryType | str,
        content: str,
        source: str,
        project_id: str = "",
        session_id: str = "",
        confidence: float = 1.0,
        expires_at: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        memory_id: Optional[str] = None,
    ) -> "MemoryRecord":
        now = utc_now()
        return cls(
            memory_id=memory_id or f"mem_{uuid4().hex}",
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
            namespace=namespace,
            memory_type=_as_memory_type(memory_type),
            content=content,
            source=source,
            created_at=now,
            updated_at=now,
            confidence=confidence,
            status=MemoryStatus.ACTIVE,
            expires_at=expires_at,
            metadata=dict(metadata or {}),
        )

    @property
    def is_active(self) -> bool:
        if self.status != MemoryStatus.ACTIVE:
            return False
        if not self.expires_at:
            return True
        try:
            expiry = datetime.fromisoformat(self.expires_at)
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            return expiry > datetime.now(timezone.utc)
        except ValueError:
            return False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["memory_type"] = self.memory_type.value
        result["status"] = self.status.value
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryRecord":
        return cls(
            memory_id=str(data.get("memory_id") or ""),
            user_id=str(data.get("user_id") or ""),
            project_id=str(data.get("project_id") or ""),
            session_id=str(data.get("session_id") or ""),
            namespace=str(data.get("namespace") or ""),
            memory_type=_as_memory_type(data.get("memory_type") or MemoryType.NOTE.value),
            content=str(data.get("content") or ""),
            source=str(data.get("source") or ""),
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
            confidence=float(data.get("confidence", 1.0)),
            status=_as_memory_status(data.get("status") or MemoryStatus.ACTIVE.value),
            expires_at=data.get("expires_at"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    """待写入 Memory 的候选。"""

    content: str
    memory_type: MemoryType = MemoryType.NOTE
    source: str = "agent"
    confidence: float = 1.0
    expires_at: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def normalized(self) -> "MemoryCandidate":
        content = str(self.content or "").strip()
        if not content:
            raise ValueError("memory candidate content must not be empty")
        memory_type = (
            self.memory_type
            if isinstance(self.memory_type, MemoryType)
            else MemoryType(str(self.memory_type))
        )
        return MemoryCandidate(
            content=content,
            memory_type=memory_type,
            source=str(self.source or "agent").strip() or "agent",
            confidence=min(max(float(self.confidence), 0.0), 1.0),
            expires_at=str(self.expires_at) if self.expires_at else None,
            metadata=dict(self.metadata or {}),
        )


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    """隔离后的长期 Memory 查询。"""

    namespace: str
    user_id: str = ""
    project_id: str = ""
    session_id: str = ""
    query: str = ""
    top_k: int = 8
    min_score: float = 0.0
    memory_types: tuple[MemoryType, ...] = ()
    include_session_memories: bool = True

    def normalized(self) -> "MemoryQuery":
        namespace = normalize_scope(self.namespace)
        if not namespace:
            raise ValueError("namespace must not be empty")
        types = tuple(
            _as_memory_type(value)
            for value in self.memory_types
        )
        return MemoryQuery(
            namespace=namespace,
            user_id=normalize_scope(self.user_id),
            project_id=normalize_scope(self.project_id),
            session_id=normalize_scope(self.session_id),
            query=str(self.query or "").strip(),
            top_k=max(1, min(int(self.top_k), 100)),
            min_score=min(max(float(self.min_score), 0.0), 1.0),
            memory_types=types,
            include_session_memories=bool(self.include_session_memories),
        )


@dataclass(frozen=True, slots=True)
class ScoredMemory:
    record: MemoryRecord
    score: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "score": float(self.score),
            "reason": self.reason,
        }


@dataclass(slots=True)
class MemoryContext:
    """注入 AgentState 的受控 Memory 上下文。"""

    recent_messages: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    memories: list[ScoredMemory] = field(default_factory=list)
    text: str = ""
    estimated_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "recent_messages": [dict(item) for item in self.recent_messages],
            "summary": self.summary,
            "memories": [item.to_dict() for item in self.memories],
            "text": self.text,
            "estimated_tokens": self.estimated_tokens,
        }


@dataclass(slots=True)
class MemoryWriteResult:
    created: list[MemoryRecord] = field(default_factory=list)
    updated: list[MemoryRecord] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    @property
    def written_count(self) -> int:
        return len(self.created) + len(self.updated)

    def to_dict(self) -> dict[str, Any]:
        return {
            "created": [item.to_dict() for item in self.created],
            "updated": [item.to_dict() for item in self.updated],
            "skipped": [dict(item) for item in self.skipped],
            "written_count": self.written_count,
        }


def normalize_candidates(
    values: Iterable[MemoryCandidate | Mapping[str, Any]],
) -> list[MemoryCandidate]:
    result: list[MemoryCandidate] = []
    for value in values:
        if isinstance(value, MemoryCandidate):
            result.append(value.normalized())
            continue
        if not isinstance(value, Mapping):
            raise TypeError("memory candidates must be mappings or MemoryCandidate")
        result.append(
            MemoryCandidate(
                content=str(value.get("content") or ""),
                memory_type=MemoryType(str(value.get("memory_type") or MemoryType.NOTE.value)),
                source=str(value.get("source") or "agent"),
                confidence=float(value.get("confidence", 1.0)),
                expires_at=value.get("expires_at"),
                metadata=dict(value.get("metadata") or {}),
            ).normalized()
        )
    return result


__all__ = [
    "MemoryType",
    "MemoryStatus",
    "MemoryRecord",
    "MemoryCandidate",
    "MemoryQuery",
    "ScoredMemory",
    "MemoryContext",
    "MemoryWriteResult",
    "normalize_candidates",
    "utc_now",
]
