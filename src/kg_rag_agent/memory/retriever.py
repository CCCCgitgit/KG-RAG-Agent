# -*- coding: utf-8 -*-
"""长期 Memory 检索与上下文预算控制。"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

from .models import MemoryQuery, MemoryRecord, ScoredMemory
from .stores import MemoryStore

_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class MemoryRetriever:
    def __init__(
        self,
        store: MemoryStore,
        *,
        scorer: Optional[Callable[[str, MemoryRecord], float]] = None,
    ) -> None:
        self.store = store
        self.scorer = scorer

    def retrieve(self, query: MemoryQuery) -> list[ScoredMemory]:
        query = query.normalized()
        records = self.store.list(query)
        scored: list[ScoredMemory] = []
        for record in records:
            score, reason = self._score(query.query, record)
            if score < query.min_score:
                continue
            scored.append(ScoredMemory(record=record, score=score, reason=reason))
        scored.sort(
            key=lambda item: (item.score, item.record.updated_at),
            reverse=True,
        )
        return scored[: query.top_k]

    def _score(self, text: str, record: MemoryRecord) -> tuple[float, str]:
        if self.scorer is not None:
            score = min(max(float(self.scorer(text, record)), 0.0), 1.0)
            return score, "custom_scorer"

        query = str(text or "").strip().lower()
        content = record.content.lower()
        if not query:
            lexical = 0.35
            reason = "recent_active_memory"
        elif query in content or content in query:
            lexical = 1.0
            reason = "substring_match"
        else:
            query_tokens = set(_TOKEN_RE.findall(query))
            content_tokens = set(_TOKEN_RE.findall(content))
            if not query_tokens or not content_tokens:
                lexical = 0.0
            else:
                intersection = len(query_tokens.intersection(content_tokens))
                union = len(query_tokens.union(content_tokens))
                lexical = intersection / max(union, 1)
            reason = "token_overlap"

        recency = _recency_score(record.updated_at)
        score = 0.72 * lexical + 0.18 * record.confidence + 0.10 * recency
        return min(max(score, 0.0), 1.0), reason


def _recency_score(timestamp: str) -> float:
    try:
        value = datetime.fromisoformat(timestamp)
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        age_days = max((datetime.now(timezone.utc) - value).total_seconds() / 86400.0, 0.0)
        return math.exp(-age_days / 180.0)
    except (TypeError, ValueError):
        return 0.0


def estimate_tokens(text: str) -> int:
    """无 tokenizer 依赖的保守估算。"""

    value = str(text or "")
    if not value:
        return 0
    ascii_chars = sum(1 for char in value if ord(char) < 128)
    non_ascii_chars = len(value) - ascii_chars
    return max(1, int(ascii_chars / 4.0 + non_ascii_chars / 1.5))


def trim_text_to_token_budget(text: str, budget: int) -> str:
    budget = max(int(budget), 0)
    value = str(text or "")
    if estimate_tokens(value) <= budget:
        return value
    if budget <= 0:
        return ""
    low, high = 0, len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(value[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    return value[:low].rstrip()


__all__ = ["MemoryRetriever", "estimate_tokens", "trim_text_to_token_budget"]
