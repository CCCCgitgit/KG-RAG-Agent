# -*- coding: utf-8 -*-
"""Memory 候选过滤、去重、更新与写入。"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Optional

from .models import (
    MemoryCandidate,
    MemoryQuery,
    MemoryRecord,
    MemoryWriteResult,
    normalize_candidates,
    utc_now,
)
from .policies import MemoryPolicy
from .stores import MemoryStore

_SPACE_RE = re.compile(r"\s+")


class MemoryWriter:
    def __init__(self, store: MemoryStore, policy: MemoryPolicy) -> None:
        self.store = store
        self.policy = policy

    def write(
        self,
        candidates: Iterable[MemoryCandidate | Mapping[str, Any]],
        *,
        namespace: str,
        user_id: str,
        project_id: str = "",
        session_id: str = "",
    ) -> MemoryWriteResult:
        result = MemoryWriteResult()
        normalized = normalize_candidates(candidates)
        if len(normalized) > self.policy.max_write_candidates:
            overflow = normalized[self.policy.max_write_candidates :]
            normalized = normalized[: self.policy.max_write_candidates]
            result.skipped.extend(
                {"content": item.content, "reason": "write_candidate_budget"}
                for item in overflow
            )

        existing = self.store.list(
            MemoryQuery(
                namespace=namespace,
                user_id=user_id,
                project_id=project_id,
                session_id=session_id,
                query="",
                top_k=100,
                min_score=0.0,
            )
        )
        index = {
            (item.memory_type.value, _normalize_content(item.content)): item
            for item in existing
        }

        for candidate in normalized:
            reason = self.policy.validate_candidate(candidate, user_id=user_id)
            if reason:
                result.skipped.append({"content": candidate.content, "reason": reason})
                continue
            key = (candidate.memory_type.value, _normalize_content(candidate.content))
            current = index.get(key)
            if current is not None:
                current.confidence = max(current.confidence, candidate.confidence)
                current.updated_at = utc_now()
                current.source = candidate.source or current.source
                current.metadata.update(dict(candidate.metadata or {}))
                if candidate.expires_at:
                    current.expires_at = candidate.expires_at
                stored = self.store.upsert(current)
                result.updated.append(stored)
                index[key] = stored
                continue

            record = MemoryRecord.create(
                user_id=user_id,
                project_id=project_id,
                session_id=session_id,
                namespace=namespace,
                memory_type=candidate.memory_type,
                content=candidate.content,
                source=candidate.source,
                confidence=candidate.confidence,
                expires_at=candidate.expires_at,
                metadata=candidate.metadata,
            )
            stored = self.store.upsert(record)
            result.created.append(stored)
            index[key] = stored

        return result


def _normalize_content(content: str) -> str:
    return _SPACE_RE.sub(" ", str(content or "").strip().casefold())


__all__ = ["MemoryWriter"]
