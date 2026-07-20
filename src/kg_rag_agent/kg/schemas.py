# -*- coding: utf-8 -*-
"""Stable KG query and retrieval schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, TypedDict


class Triple(TypedDict, total=False):
    head: str
    relation: str
    tail: str


class RelationItem(TypedDict, total=False):
    head: str
    relation: str
    tail: str
    direction: str
    score: float
    text: str
    metadata: Dict[str, Any]


class PathItem(TypedDict, total=False):
    path: List[str]
    triples: List[Triple]
    text: str
    score: float
    path_length: int
    metadata: Dict[str, Any]


class NeighborItem(TypedDict, total=False):
    source: str
    relation: str
    target: str
    direction: str
    score: float
    text: str
    metadata: Dict[str, Any]


class SubgraphNode(TypedDict, total=False):
    id: str
    label: str
    depth: int
    is_center: bool
    metadata: Dict[str, Any]


class SubgraphEdge(TypedDict, total=False):
    head: str
    relation: str
    tail: str
    score: float
    text: str
    metadata: Dict[str, Any]


class EvidenceItem(TypedDict, total=False):
    evidence_id: str
    evidence_type: str
    source_entity: str
    target_entity: str
    relation: str
    path: List[str]
    triples: List[Triple]
    text: str
    score: float
    metadata: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class KGRetrievalOptions:
    """Request-scoped KG retrieval parameters with explicit boundaries."""

    enable_relation: bool = True
    enable_path: bool = True
    enable_neighbor: bool = True
    enable_subgraph: bool = True

    relation_directed: bool = True
    relation_include_reverse: bool = True
    max_relations: int = 20

    path_directed: bool = True
    max_paths: int = 5
    max_path_length: int = 4

    neighbor_direction: str = "both"
    max_neighbors: int = 20

    subgraph_direction: str = "both"
    subgraph_max_depth: int = 2
    subgraph_max_nodes: int = 50
    subgraph_max_edges: int = 100

    min_evidence_score: float = 0.0
    max_evidence: int = 30

    @classmethod
    def from_mapping(
        cls,
        value: Optional[Mapping[str, Any]],
    ) -> "KGRetrievalOptions":
        data = dict(value or {})
        return cls(
            enable_relation=_as_bool(data.get("enable_relation"), True),
            enable_path=_as_bool(data.get("enable_path"), True),
            enable_neighbor=_as_bool(data.get("enable_neighbor"), True),
            enable_subgraph=_as_bool(data.get("enable_subgraph"), True),
            relation_directed=_as_bool(data.get("relation_directed"), True),
            relation_include_reverse=_as_bool(
                data.get("relation_include_reverse"), True
            ),
            max_relations=_bounded_int(data.get("max_relations"), 20, 1, 500),
            path_directed=_as_bool(data.get("path_directed"), True),
            max_paths=_bounded_int(data.get("max_paths"), 5, 1, 100),
            max_path_length=_bounded_int(
                data.get("max_path_length", data.get("path_max_depth")),
                4,
                2,
                12,
            ),
            neighbor_direction=_direction(
                data.get("neighbor_direction"), "both"
            ),
            max_neighbors=_bounded_int(data.get("max_neighbors"), 20, 1, 1000),
            subgraph_direction=_direction(
                data.get("subgraph_direction"), "both"
            ),
            subgraph_max_depth=_bounded_int(
                data.get("subgraph_max_depth"), 2, 0, 8
            ),
            subgraph_max_nodes=_bounded_int(
                data.get("subgraph_max_nodes"), 50, 1, 5000
            ),
            subgraph_max_edges=_bounded_int(
                data.get("subgraph_max_edges"), 100, 1, 10000
            ),
            min_evidence_score=_bounded_float(
                data.get("min_evidence_score"), 0.0, 0.0, 1.0
            ),
            max_evidence=_bounded_int(data.get("max_evidence"), 30, 1, 1000),
        )


@dataclass(slots=True)
class KGRetrievalResult:
    entities: List[str] = field(default_factory=list)
    relation_results: List[Dict[str, Any]] = field(default_factory=list)
    path_results: List[Dict[str, Any]] = field(default_factory=list)
    neighbor_results: List[Dict[str, Any]] = field(default_factory=list)
    subgraph_results: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return bool(self.evidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": list(self.entities),
            "found": self.found,
            "relation_results": list(self.relation_results),
            "path_results": list(self.path_results),
            "neighbor_results": list(self.neighbor_results),
            "subgraph_results": list(self.subgraph_results),
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    return max(minimum, min(maximum, result))


def _bounded_float(
    value: Any,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    return max(minimum, min(maximum, result))


def _direction(value: Any, default: str) -> str:
    normalized = str(value or default).strip().lower()
    return normalized if normalized in {"in", "out", "both"} else default
