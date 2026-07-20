# -*- coding: utf-8 -*-
"""Evaluation 层的唯一数据协议。"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


_STRING_LIST_ALIASES = {
    "keywords": ("expected_keywords", "keywords", "reference_keywords"),
    "mentions": ("expected_mentions", "mentions"),
    "entities": ("expected_entities", "entities"),
    "evidence_ids": ("expected_evidence_ids", "evidence_ids"),
    "tags": ("tags",),
}


@dataclass(slots=True, init=False)
class EvaluationCase:
    """单条评估样例。缺少参考字段时，相应指标不参与汇总。"""

    case_id: str
    query: str
    reference_answer: str = ""
    expected_route: str = ""
    expected_answerability: str = ""
    keywords: List[str] = field(default_factory=list)
    expected_mentions: List[str] = field(default_factory=list)
    expected_entities: List[str] = field(default_factory=list)
    expected_evidence_ids: List[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "normal"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        case_id: str,
        query: str,
        reference_answer: str = "",
        expected_route: str = "",
        expected_answerability: str = "",
        keywords: Optional[List[str]] = None,
        expected_mentions: Optional[List[str]] = None,
        expected_entities: Optional[List[str]] = None,
        expected_evidence_ids: Optional[List[str]] = None,
        category: str = "general",
        difficulty: str = "normal",
        tags: Optional[List[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        *,
        expected_answer: Optional[str] = None,
        expected_keywords: Optional[List[str]] = None,
    ) -> None:
        self.case_id = str(case_id or "").strip()
        self.query = str(query or "").strip()
        self.reference_answer = str(
            reference_answer if expected_answer is None else expected_answer
        )
        self.expected_route = str(expected_route or "").strip()
        self.expected_answerability = str(expected_answerability or "").strip()
        self.keywords = _normalize_string_list(
            keywords if expected_keywords is None else expected_keywords
        )
        self.expected_mentions = _normalize_string_list(expected_mentions)
        self.expected_entities = _normalize_string_list(expected_entities)
        self.expected_evidence_ids = _normalize_string_list(expected_evidence_ids)
        self.category = str(category or "general").strip() or "general"
        self.difficulty = str(difficulty or "normal").strip() or "normal"
        self.tags = _normalize_string_list(tags)
        self.metadata = copy.deepcopy(dict(metadata or {}))
        if not self.case_id:
            raise ValueError("case_id must not be empty.")
        if not self.query:
            raise ValueError(f"query must not be empty for case '{self.case_id}'.")

    @property
    def expected_answer(self) -> str:
        """旧字段兼容。"""
        return self.reference_answer

    @property
    def expected_keywords(self) -> List[str]:
        """旧字段兼容。"""
        return list(self.keywords)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        index: int = 0,
    ) -> "EvaluationCase":
        query = _first_text(data, "query", "question", "input", "user_query", "text")
        case_id = _first_text(data, "case_id", "id") or f"case_{index:04d}"
        metadata = data.get("metadata", {})
        return cls(
            case_id=case_id,
            query=query,
            reference_answer=_first_text(
                data,
                "reference_answer",
                "expected_answer",
                "answer",
                "reference",
            ),
            expected_route=_first_text(data, "expected_route", "route"),
            expected_answerability=_first_text(
                data,
                "expected_answerability",
                "answerability",
            ),
            keywords=_first_list(data, *_STRING_LIST_ALIASES["keywords"]),
            expected_mentions=_first_list(data, *_STRING_LIST_ALIASES["mentions"]),
            expected_entities=_first_list(data, *_STRING_LIST_ALIASES["entities"]),
            expected_evidence_ids=_first_list(data, *_STRING_LIST_ALIASES["evidence_ids"]),
            category=_first_text(data, "category", "type") or "general",
            difficulty=_first_text(data, "difficulty") or "normal",
            tags=_first_list(data, *_STRING_LIST_ALIASES["tags"]),
            metadata=metadata if isinstance(metadata, Mapping) else {},
        )


@dataclass(slots=True)
class EvaluationRecord:
    """单条评估执行记录。"""

    case_id: str
    query: str
    prediction: str
    reference_answer: str = ""
    request_id: str = ""
    route: str = ""
    expected_route: str = ""
    answerability: str = ""
    expected_answerability: str = ""
    semantic_score: float = 0.0
    latency_ms: float = 0.0
    has_error: bool = False
    error_message: str = ""
    error_stage: str = ""
    num_mentions: int = 0
    num_grounded_entities: int = 0
    num_evidence: int = 0
    num_citations: int = 0
    num_warnings: int = 0
    citations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    state_summary: Dict[str, Any] = field(default_factory=dict)
    raw_state: Optional[Dict[str, Any]] = None
    category: str = "general"
    difficulty: str = "normal"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def answer(self) -> str:
        """旧字段兼容。"""
        return self.prediction

    def to_dict(self, *, include_raw_state: bool = False) -> Dict[str, Any]:
        result = asdict(self)
        result["answer"] = self.prediction
        result["expected_answer"] = self.reference_answer
        if not include_raw_state:
            result.pop("raw_state", None)
        return result


@dataclass(slots=True)
class EvaluationSummary:
    """一次评估运行的汇总。"""

    run_id: str
    total: int
    success: int
    failed: int
    started_at: str
    finished_at: str
    duration_ms: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    output_dir: str = ""

    @property
    def avg_latency_ms(self) -> float:
        latency = self.metrics.get("numeric_metrics", {}).get("latency_ms", {})
        try:
            return float(latency.get("avg", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["avg_latency_ms"] = self.avg_latency_ms
        return result


@dataclass(slots=True)
class EvaluationRunResult:
    """评估执行结果。"""

    summary: EvaluationSummary
    records: List[EvaluationRecord]
    manifest: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_raw_state: bool = False) -> Dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "manifest": copy.deepcopy(self.manifest),
            "records": [
                item.to_dict(include_raw_state=include_raw_state)
                for item in self.records
            ],
        }


def _first_text(data: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _first_list(data: Mapping[str, Any], *keys: str) -> List[str]:
    for key in keys:
        if key in data:
            return _normalize_string_list(data.get(key))
    return []


def _normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        result: List[str] = []
        seen = set()
        for item in value:
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result
    return []


__all__ = [
    "EvaluationCase",
    "EvaluationRecord",
    "EvaluationSummary",
    "EvaluationRunResult",
]
