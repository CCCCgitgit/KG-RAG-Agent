# -*- coding: utf-8 -*-
"""Unified KG retrieval façade.

This module orchestrates the existing relation, path, neighbor, subgraph, and
EvidenceBuilder implementations. It contains no LangGraph state logic and no
answer-generation logic.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from .evidence_builder import EvidenceBuilder
from .neighbor_search import neighbor_search
from .path_search import path_search
from .relation_search import relation_search
from .schemas import KGRetrievalOptions, KGRetrievalResult
from .subgraph_search import subgraph_search


class KGRetriever:
    """Reusable orchestration layer over the project's proven KG algorithms."""

    def __init__(
        self,
        *,
        graph: Any = None,
        graph_loader: Any = None,
        evidence_builder: Optional[EvidenceBuilder] = None,
        default_options: Optional[
            KGRetrievalOptions | Mapping[str, Any]
        ] = None,
    ) -> None:
        if graph is None and graph_loader is None:
            raise ValueError("graph or graph_loader must be provided")
        self._graph = graph
        self.graph_loader = graph_loader
        self.evidence_builder = evidence_builder
        self.default_options = self._resolve_options(default_options)

    @property
    def graph(self) -> Any:
        if self._graph is not None:
            return self._graph
        loader = self.graph_loader
        getter = getattr(loader, "get_graph", None)
        if not callable(getter):
            raise TypeError("graph_loader must provide get_graph()")
        self._graph = getter()
        if self._graph is None:
            raise RuntimeError("graph_loader returned None")
        return self._graph

    def clear_graph_reference(self) -> None:
        """Drop only the local reference; GraphLoader keeps its own cache."""
        if self.graph_loader is not None:
            self._graph = None

    def retrieve(
        self,
        entities: Sequence[Any],
        *,
        relation_pairs: Optional[Sequence[Sequence[Any]]] = None,
        options: Optional[KGRetrievalOptions | Mapping[str, Any]] = None,
    ) -> KGRetrievalResult:
        resolved = self._resolve_options(options)
        normalized_entities = _normalize_entities(entities)
        result = KGRetrievalResult(entities=normalized_entities)

        if not normalized_entities:
            result.warnings.append("No valid grounded entity was provided.")
            return result

        graph = self.graph
        pairs = _normalize_pairs(relation_pairs, normalized_entities)

        if resolved.enable_relation:
            for source, target in pairs:
                result.relation_results.append(
                    relation_search(
                        graph,
                        source=source,
                        target=target,
                        directed=resolved.relation_directed,
                        include_reverse=resolved.relation_include_reverse,
                        max_results=resolved.max_relations,
                    )
                )

        if resolved.enable_path:
            for source, target in pairs:
                result.path_results.append(
                    path_search(
                        graph,
                        source=source,
                        target=target,
                        max_paths=resolved.max_paths,
                        max_path_length=resolved.max_path_length,
                        directed=resolved.path_directed,
                        include_triples=True,
                    )
                )

        if resolved.enable_neighbor:
            for entity in normalized_entities:
                result.neighbor_results.append(
                    neighbor_search(
                        graph,
                        entity=entity,
                        max_neighbors=resolved.max_neighbors,
                        direction=resolved.neighbor_direction,
                        include_relation=True,
                    )
                )

        if resolved.enable_subgraph:
            result.subgraph_results.append(
                subgraph_search(
                    graph,
                    entities=normalized_entities,
                    max_depth=resolved.subgraph_max_depth,
                    max_nodes=resolved.subgraph_max_nodes,
                    max_edges=resolved.subgraph_max_edges,
                    direction=resolved.subgraph_direction,
                    include_center=True,
                )
            )

        builder = self.evidence_builder or EvidenceBuilder(
            max_evidence=resolved.max_evidence,
            min_score=resolved.min_evidence_score,
            deduplicate=True,
        )
        raw_evidence = []
        for item in result.relation_results:
            raw_evidence.extend(builder.from_relation_result(item))
        for item in result.path_results:
            raw_evidence.extend(builder.from_path_result(item))
        for item in result.neighbor_results:
            raw_evidence.extend(builder.from_neighbor_result(item))
        for item in result.subgraph_results:
            raw_evidence.extend(builder.from_subgraph_result(item))
        result.evidence = builder.postprocess(raw_evidence)
        result.metadata = {
            "entity_count": len(normalized_entities),
            "relation_pair_count": len(pairs),
            "relation_query_count": len(result.relation_results),
            "path_query_count": len(result.path_results),
            "neighbor_query_count": len(result.neighbor_results),
            "subgraph_query_count": len(result.subgraph_results),
            "evidence_count": len(result.evidence),
        }
        return result

    def search_relation(
        self,
        source: Any,
        target: Any,
        *,
        options: Optional[KGRetrievalOptions | Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        resolved = self._resolve_options(options)
        return relation_search(
            self.graph,
            source=_entity_key(source),
            target=_entity_key(target),
            directed=resolved.relation_directed,
            include_reverse=resolved.relation_include_reverse,
            max_results=resolved.max_relations,
        )

    def search_path(
        self,
        source: Any,
        target: Any,
        *,
        options: Optional[KGRetrievalOptions | Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        resolved = self._resolve_options(options)
        return path_search(
            self.graph,
            source=_entity_key(source),
            target=_entity_key(target),
            max_paths=resolved.max_paths,
            max_path_length=resolved.max_path_length,
            directed=resolved.path_directed,
            include_triples=True,
        )

    def search_neighbors(
        self,
        entity: Any,
        *,
        options: Optional[KGRetrievalOptions | Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        resolved = self._resolve_options(options)
        return neighbor_search(
            self.graph,
            entity=_entity_key(entity),
            max_neighbors=resolved.max_neighbors,
            direction=resolved.neighbor_direction,
            include_relation=True,
        )

    def search_subgraph(
        self,
        entities: Sequence[Any],
        *,
        options: Optional[KGRetrievalOptions | Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        resolved = self._resolve_options(options)
        return subgraph_search(
            self.graph,
            entities=_normalize_entities(entities),
            max_depth=resolved.subgraph_max_depth,
            max_nodes=resolved.subgraph_max_nodes,
            max_edges=resolved.subgraph_max_edges,
            direction=resolved.subgraph_direction,
            include_center=True,
        )

    def _resolve_options(
        self,
        value: Optional[KGRetrievalOptions | Mapping[str, Any]],
    ) -> KGRetrievalOptions:
        if isinstance(value, KGRetrievalOptions):
            return value
        if value is None and hasattr(self, "default_options"):
            return self.default_options
        return KGRetrievalOptions.from_mapping(value)


def _normalize_entities(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values or []:
        key = _entity_key(value)
        marker = key.casefold()
        if not key or marker in seen:
            continue
        seen.add(marker)
        result.append(key)
    return result


def _entity_key(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("node_key", "graph_node", "entity_id", "entity_name", "name"):
            candidate = str(value.get(key, "") or "").strip()
            if candidate:
                return candidate
        return ""
    return str(value or "").strip()


def _normalize_pairs(
    pairs: Optional[Sequence[Sequence[Any]]],
    entities: Sequence[str],
) -> List[Tuple[str, str]]:
    if pairs is None:
        return list(combinations(entities, 2))
    result: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for pair in pairs:
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)):
            continue
        if len(pair) < 2:
            continue
        source = _entity_key(pair[0])
        target = _entity_key(pair[1])
        if not source or not target or source == target:
            continue
        marker = (source.casefold(), target.casefold())
        if marker in seen:
            continue
        seen.add(marker)
        result.append((source, target))
    return result
