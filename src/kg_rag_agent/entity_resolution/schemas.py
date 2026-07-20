# -*- coding: utf-8 -*-
"""Entity-resolution domain schemas.

These structures are independent of LangGraph. Graph nodes may convert them to
partial AgentState updates, while the domain layer remains reusable by API,
CLI, tools, evaluation, and MCP adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, TypedDict


class Mention(TypedDict, total=False):
    text: str
    start: int
    end: int
    type: str
    confidence: float


class EntityCandidate(TypedDict, total=False):
    mention: str
    entity_id: str
    entity_name: str
    score: float
    source: str
    aliases: List[str]
    metadata: Dict[str, Any]


class GroundedEntity(TypedDict, total=False):
    mention: str
    entity_id: str
    entity_name: str
    node_key: str
    confidence: float
    in_graph: bool
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class MentionExtractionOptions:
    use_llm: bool = False
    min_confidence: float = 0.0
    max_mentions: int = 16

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> "MentionExtractionOptions":
        data = dict(value or {})
        return cls(
            use_llm=bool(data.get("use_llm", False)),
            min_confidence=_clip_float(data.get("min_confidence", 0.0), 0.0, 1.0, 0.0),
            max_mentions=max(_safe_int(data.get("max_mentions", 16), 16), 1),
        )


@dataclass(frozen=True)
class LinkingOptions:
    top_k: int = 5

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> "LinkingOptions":
        data = dict(value or {})
        return cls(top_k=max(_safe_int(data.get("top_k", 5), 5), 1))


@dataclass(frozen=True)
class GroundingOptions:
    min_score: float = 0.0
    max_entities: int = 16
    one_entity_per_mention: bool = True
    require_in_graph: bool = True
    allow_linear_scan: bool = True

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> "GroundingOptions":
        data = dict(value or {})
        return cls(
            min_score=_clip_float(data.get("min_score", 0.0), 0.0, 1.0, 0.0),
            max_entities=max(_safe_int(data.get("max_entities", 16), 16), 1),
            one_entity_per_mention=bool(data.get("one_entity_per_mention", True)),
            require_in_graph=bool(data.get("require_in_graph", True)),
            allow_linear_scan=bool(data.get("allow_linear_scan", True)),
        )


@dataclass
class MentionExtractionResult:
    mentions: List[Mention] = field(default_factory=list)
    raw_output: str = ""
    extractor_type: str = "rule"
    warnings: List[str] = field(default_factory=list)


@dataclass
class LinkingResult:
    candidates_by_mention: Dict[str, List[EntityCandidate]] = field(default_factory=dict)
    raw_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    unlinked_mentions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class GroundingResult:
    grounded_entities: List[GroundedEntity] = field(default_factory=list)
    ungrounded_mentions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class EntityResolutionResult:
    query: str
    mentions: List[Mention] = field(default_factory=list)
    candidates_by_mention: Dict[str, List[EntityCandidate]] = field(default_factory=dict)
    grounded_entities: List[GroundedEntity] = field(default_factory=list)
    unlinked_mentions: List[str] = field(default_factory=list)
    ungrounded_mentions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clip_float(value: Any, minimum: float, maximum: float, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, result))
