# -*- coding: utf-8 -*-
"""Memory 读取、写入和隔离策略。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from .models import MemoryCandidate, MemoryType

_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(api[_ -]?key|access[_ -]?token|secret|password|passwd)\b\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bsk-[a-z0-9_-]{12,}\b"),
    re.compile(r"(?i)\b(bearer)\s+[a-z0-9._~-]{12,}\b"),
)


@dataclass(frozen=True, slots=True)
class MemoryPolicy:
    enabled: bool = False
    write_enabled: bool = False
    long_term_enabled: bool = False
    namespace_prefix: str = "kg_rag_agent"
    max_messages: int = 20
    max_summary_tokens: int = 1200
    max_retrieved_items: int = 8
    max_context_tokens: int = 4000
    min_relevance_score: float = 0.15
    max_write_candidates: int = 8
    allowed_memory_types: frozenset[MemoryType] = field(
        default_factory=lambda: frozenset(MemoryType)
    )
    require_user_id_for_long_term: bool = True
    reject_sensitive_content: bool = True

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "MemoryPolicy":
        raw = dict(config or {})
        types = raw.get("allowed_memory_types")
        allowed = frozenset(MemoryType)
        if isinstance(types, Sequence) and not isinstance(types, (str, bytes)):
            allowed = frozenset(MemoryType(str(item)) for item in types)
        return cls(
            enabled=bool(raw.get("enabled", False)),
            write_enabled=bool(raw.get("write_enabled", False)),
            long_term_enabled=bool(raw.get("long_term_enabled", False)),
            namespace_prefix=str(raw.get("namespace_prefix") or "kg_rag_agent").strip(),
            max_messages=max(1, min(int(raw.get("max_messages", 20)), 200)),
            max_summary_tokens=max(64, min(int(raw.get("max_summary_tokens", 1200)), 32000)),
            max_retrieved_items=max(1, min(int(raw.get("max_retrieved_items", 8)), 100)),
            max_context_tokens=max(128, min(int(raw.get("max_context_tokens", 4000)), 64000)),
            min_relevance_score=min(max(float(raw.get("min_relevance_score", 0.15)), 0.0), 1.0),
            max_write_candidates=max(1, min(int(raw.get("max_write_candidates", 8)), 100)),
            allowed_memory_types=allowed,
            require_user_id_for_long_term=bool(raw.get("require_user_id_for_long_term", True)),
            reject_sensitive_content=bool(raw.get("reject_sensitive_content", True)),
        )

    def namespace(self, project_id: str = "") -> str:
        prefix = self.namespace_prefix.strip() or "kg_rag_agent"
        project = str(project_id or "").strip()
        return f"{prefix}:{project}" if project else prefix

    def validate_read_scope(
        self,
        *,
        namespace: str,
        user_id: str = "",
    ) -> None:
        if not self.enabled:
            raise PermissionError("Memory is disabled by policy.")
        if not str(namespace or "").strip():
            raise ValueError("memory namespace must not be empty")
        if self.require_user_id_for_long_term and self.long_term_enabled and not str(user_id or "").strip():
            raise PermissionError("user_id is required for long-term memory access.")

    def validate_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        user_id: str,
    ) -> Optional[str]:
        if not self.enabled:
            return "memory_disabled"
        if not self.write_enabled:
            return "memory_write_disabled"
        if not self.long_term_enabled:
            return "long_term_memory_disabled"
        if self.require_user_id_for_long_term and not str(user_id or "").strip():
            return "missing_user_id"
        if candidate.memory_type not in self.allowed_memory_types:
            return "memory_type_not_allowed"
        if self.reject_sensitive_content and contains_sensitive_content(candidate.content):
            return "sensitive_content"
        if candidate.confidence <= 0:
            return "invalid_confidence"
        return None


def contains_sensitive_content(content: str) -> bool:
    text = str(content or "")
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


__all__ = ["MemoryPolicy", "contains_sensitive_content"]
