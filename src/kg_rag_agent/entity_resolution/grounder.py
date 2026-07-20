# -*- coding: utf-8 -*-
"""Candidate selection and graph grounding.

This is the domain implementation extracted from the original grounding node.
It accepts an injected graph or graph loader and never stores runtime objects in
request state.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .normalizer import normalize_node_key
from .schemas import EntityCandidate, GroundedEntity, GroundingOptions, GroundingResult


class EntityGrounder:
    def __init__(
        self,
        *,
        graph: Any = None,
        graph_loader: Any = None,
        default_options: Optional[GroundingOptions] = None,
    ) -> None:
        if graph is None and graph_loader is None:
            raise ValueError("EntityGrounder requires graph or graph_loader.")
        self._graph = graph
        self._graph_loader = graph_loader
        self.default_options = default_options or GroundingOptions()
        self._node_index: Optional[Dict[str, str]] = None
        self._indexed_graph_identity: Optional[int] = None

    @property
    def graph(self) -> Any:
        if self._graph is None:
            loader = self._graph_loader
            if hasattr(loader, "get_graph"):
                self._graph = loader.get_graph()
            elif hasattr(loader, "load"):
                self._graph = loader.load()
            elif callable(loader):
                self._graph = loader()
            else:
                raise TypeError("Unsupported graph_loader; expected get_graph(), load(), or callable.")
        return self._graph

    def invalidate_index(self) -> None:
        self._node_index = None
        self._indexed_graph_identity = None

    def ground(
        self,
        candidates_by_mention: Mapping[str, List[EntityCandidate]],
        *,
        options: Optional[GroundingOptions | Mapping[str, Any]] = None,
    ) -> GroundingResult:
        resolved = self._resolve_options(options)
        grounded: List[GroundedEntity] = []
        ungrounded: List[str] = []
        warnings: List[str] = []

        for raw_mention, raw_candidates in dict(candidates_by_mention or {}).items():
            mention = str(raw_mention or "").strip()
            if not mention:
                continue
            candidates = list(raw_candidates or [])
            if not candidates:
                ungrounded.append(mention)
                warnings.append(f"No candidates available for mention: {mention}")
                continue

            selected = self._ground_one(mention, candidates, resolved)
            if not selected:
                ungrounded.append(mention)
                warnings.append(f"No grounded entity found for mention: {mention}")
                continue
            grounded.extend(selected)
            if len(grounded) >= resolved.max_entities:
                grounded = grounded[: resolved.max_entities]
                warnings.append(f"Grounded entities truncated to max_entities={resolved.max_entities}.")
                break

        return GroundingResult(
            grounded_entities=grounded,
            ungrounded_mentions=ungrounded,
            warnings=warnings,
        )

    def _resolve_options(
        self,
        options: Optional[GroundingOptions | Mapping[str, Any]],
    ) -> GroundingOptions:
        if isinstance(options, GroundingOptions):
            return options
        if options is not None:
            return GroundingOptions.from_mapping(options)
        return self.default_options

    def _ground_one(
        self,
        mention: str,
        candidates: List[EntityCandidate],
        options: GroundingOptions,
    ) -> List[GroundedEntity]:
        result: List[GroundedEntity] = []
        for candidate in sorted(candidates, key=_candidate_sort_key, reverse=True):
            score = _safe_float(candidate.get("score", 0.0), 0.0)
            if score < options.min_score:
                continue
            entity_name = str(candidate.get("entity_name", "") or "").strip()
            entity_id = str(candidate.get("entity_id", "") or "").strip()
            if not entity_name and not entity_id:
                continue

            node_key, in_graph, match_method = self._resolve_node_key(
                entity_name=entity_name,
                entity_id=entity_id,
                candidate=candidate,
                allow_linear_scan=options.allow_linear_scan,
            )
            if options.require_in_graph and not in_graph:
                continue

            result.append(
                GroundedEntity(
                    mention=mention,
                    entity_id=entity_id,
                    entity_name=entity_name,
                    node_key=node_key,
                    confidence=score,
                    in_graph=in_graph,
                    metadata={
                        "source": candidate.get("source", ""),
                        "match_method": match_method,
                        "candidate_metadata": candidate.get("metadata", {}) or {},
                        "aliases": candidate.get("aliases", []) or [],
                    },
                )
            )
            if options.one_entity_per_mention:
                break
        return result

    def _resolve_node_key(
        self,
        *,
        entity_name: str,
        entity_id: str,
        candidate: EntityCandidate,
        allow_linear_scan: bool,
    ) -> Tuple[str, bool, str]:
        graph = self.graph
        metadata = candidate.get("metadata", {}) or {}
        direct_keys = [
            metadata.get("node_key"),
            metadata.get("graph_node"),
            metadata.get("graph_key"),
            entity_id,
            entity_name,
        ]
        for raw_key in direct_keys:
            key = str(raw_key or "").strip()
            if key and _graph_has_node(graph, key):
                return key, True, "direct_node_key"

        if allow_linear_scan:
            node_index = self._get_node_index(graph)
            for value, method in (
                (entity_id, "normalized_entity_id"),
                (entity_name, "normalized_entity_name"),
            ):
                normalized = normalize_node_key(value)
                if normalized and normalized in node_index:
                    return node_index[normalized], True, method
            for alias in candidate.get("aliases", []) or []:
                normalized = normalize_node_key(alias)
                if normalized and normalized in node_index:
                    return node_index[normalized], True, "normalized_alias"

        return entity_id or entity_name, False, "not_found"

    def _get_node_index(self, graph: Any) -> Dict[str, str]:
        identity = id(graph)
        if self._node_index is not None and self._indexed_graph_identity == identity:
            return self._node_index
        self._node_index = _build_node_index(graph)
        self._indexed_graph_identity = identity
        return self._node_index


def _candidate_sort_key(candidate: EntityCandidate) -> Tuple[int, float, int]:
    metadata = candidate.get("metadata", {}) or {}
    selected = 1 if metadata.get("selected", False) else 0
    score = _safe_float(candidate.get("score", 0.0), 0.0)
    rank = _safe_int(metadata.get("rank", candidate.get("rank", 999999)), 999999)
    return selected, score, -rank


def _build_node_index(graph: Any) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for node_key in _iter_graph_nodes(graph):
        node_key_text = str(node_key)
        normalized = normalize_node_key(node_key_text)
        if normalized:
            index.setdefault(normalized, node_key_text)
        attrs = _get_node_attrs(graph, node_key)
        for attr_key in ("id", "entity_id", "name", "entity_name", "label", "title", "text"):
            value = attrs.get(attr_key)
            normalized = normalize_node_key(value)
            if normalized:
                index.setdefault(normalized, node_key_text)
        aliases = attrs.get("aliases") or attrs.get("alias") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        if isinstance(aliases, (list, tuple, set)):
            for alias in aliases:
                normalized = normalize_node_key(alias)
                if normalized:
                    index.setdefault(normalized, node_key_text)
    return index


def _graph_has_node(graph: Any, node_key: str) -> bool:
    if graph is None:
        return False
    if hasattr(graph, "has_node"):
        try:
            return bool(graph.has_node(node_key))
        except Exception:
            pass
    try:
        return node_key in graph
    except Exception:
        return False


def _iter_graph_nodes(graph: Any) -> Iterable[Any]:
    if graph is None:
        return []
    if hasattr(graph, "nodes"):
        nodes = graph.nodes
        try:
            return list(nodes())
        except TypeError:
            try:
                return list(nodes)
            except Exception:
                return []
    if isinstance(graph, dict):
        return list(graph.keys())
    if isinstance(graph, (list, tuple, set)):
        return list(graph)
    return []


def _get_node_attrs(graph: Any, node_key: Any) -> Dict[str, Any]:
    if graph is None:
        return {}
    if hasattr(graph, "nodes"):
        try:
            attrs = graph.nodes[node_key]
            return attrs if isinstance(attrs, dict) else {}
        except Exception:
            pass
    if isinstance(graph, dict):
        value = graph.get(node_key)
        return value if isinstance(value, dict) else {}
    return {}


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
