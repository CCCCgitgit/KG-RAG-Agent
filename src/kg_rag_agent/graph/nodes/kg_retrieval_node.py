# -*- coding: utf-8 -*-
"""Knowledge-graph retrieval LangGraph node.

The node is a thin orchestration adapter between ``AgentState`` and the domain
implementation in :mod:`kg_rag_agent.kg`.  It does not implement graph search
algorithms, semantic scoring, reasoning, or answer generation.

Runtime boundary:
    * graph objects, loaders, retrievers, and builders live in RuntimeContext;
    * AgentState stores only serializable request data and retrieval results;
    * ``state["config"]`` remains a migration-only fallback for direct legacy
      node calls without RuntimeContext.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from kg_rag_agent.kg import (
    EvidenceBuilder,
    KGRetrievalOptions,
    KGRetrievalResult,
    KGRetriever,
    GraphLoader,
    normalize_evidence_list,
)

from ..state import AgentState, EvidenceItem, make_error, utc_now

if TYPE_CHECKING:
    from kg_rag_agent.runtime import RuntimeContext


DEFAULT_MAX_EVIDENCE = 30
DEFAULT_MAX_RELATIONS = 20
DEFAULT_MAX_PATHS = 5
DEFAULT_MAX_PATH_LENGTH = 4
DEFAULT_MAX_NEIGHBORS = 10
DEFAULT_SUBGRAPH_MAX_DEPTH = 2
DEFAULT_SUBGRAPH_MAX_NODES = 50
DEFAULT_SUBGRAPH_MAX_EDGES = 100

MAX_GROUNDED_ENTITIES = 256
MAX_EVIDENCE_LIMIT = 1000
MAX_RELATION_LIMIT = 500
MAX_PATH_LIMIT = 100
MAX_PATH_LENGTH_LIMIT = 12
MAX_NEIGHBOR_LIMIT = 1000

# Only used when this node is called without RuntimeContext during migration.
_LEGACY_RETRIEVER_CACHE: Dict[str, KGRetriever] = {}


def create_kg_retrieval_node(
    runtime: Optional["RuntimeContext"] = None,
):
    """Create a KG retrieval node bound to one RuntimeContext."""

    _ensure_runtime_open(runtime)
    bound_retriever: Optional[Any] = None
    if runtime is not None:
        config = _get_kg_retrieval_config(state={}, runtime=runtime)
        options = _build_options(config=config, request_options={})
        bound_retriever = _resolve_retriever(
            runtime=runtime,
            retrieval_config=config,
            options=options,
            legacy_config={},
        )

    def _node(state: AgentState) -> AgentState:
        return kg_retrieval_node(
            state,
            runtime=runtime,
            retriever=bound_retriever,
        )

    _node.__name__ = "kg_retrieval_node"
    _node.__qualname__ = "kg_retrieval_node"
    _node.__doc__ = kg_retrieval_node.__doc__
    return _node


def kg_retrieval_node(
    state: AgentState,
    *,
    runtime: Optional["RuntimeContext"] = None,
    retriever: Optional[Any] = None,
) -> AgentState:
    """Retrieve graph evidence for the grounded entities in ``state``."""

    try:
        _ensure_runtime_open(runtime)
        grounded_entities, normalization_warnings = _normalize_grounded_entities(
            state.get("grounded_entities")
        )

        if not grounded_entities:
            return _build_no_entities_update(
                state=state,
                warnings=normalization_warnings,
            )

        retrieval_config = _get_kg_retrieval_config(
            state=state,
            runtime=runtime,
        )
        request_options = _mapping_copy(state.get("request_options"))
        options = _build_options(
            config=retrieval_config,
            request_options=request_options,
        )

        resolved_retriever = retriever or _resolve_retriever(
            runtime=runtime,
            retrieval_config=retrieval_config,
            options=options,
            legacy_config=state.get("config"),
        )
        raw_result = _invoke_retriever(
            retriever=resolved_retriever,
            grounded_entities=grounded_entities,
            options=options,
        )

        builder = _resolve_evidence_builder(
            runtime=runtime,
            retriever=resolved_retriever,
            options=options,
        )
        normalized_result = _normalize_retrieval_result(
            raw_result=raw_result,
            builder=builder,
            options=options,
        )

        warnings = _deduplicate_texts(
            list(normalization_warnings)
            + list(normalized_result["warnings"])
        )
        evidence = normalized_result["evidence"]
        raw_evidence = normalized_result["raw_evidence"]
        evidence_text = normalized_result["evidence_text"]

        if not evidence:
            warnings.append("No usable evidence found after KG retrieval.")

        return _build_success_update(
            state=state,
            grounded_entities=grounded_entities,
            raw_evidence=raw_evidence,
            evidence=evidence,
            evidence_text=evidence_text,
            warnings=warnings,
            options=options,
            result_metadata=normalized_result["metadata"],
            raw_result=raw_result,
            runtime=runtime,
            retriever=resolved_retriever,
        )

    except Exception as exc:
        _log_failure(runtime, exc)
        return make_error(
            stage="kg_retrieval",
            message=str(exc),
            detail={
                "grounded_entities": _safe_grounded_entity_summary(
                    state.get("grounded_entities")
                ),
                "error_type": type(exc).__name__,
            },
        )


# ---------------------------------------------------------------------------
# Runtime and domain-object resolution
# ---------------------------------------------------------------------------


def _ensure_runtime_open(runtime: Optional["RuntimeContext"]) -> None:
    if runtime is None:
        return
    method = getattr(runtime, "ensure_open", None)
    if callable(method):
        method()


def _runtime_get(runtime: Any, name: str) -> Any:
    if runtime is None:
        return None

    getter = getattr(runtime, "get", None)
    if callable(getter):
        try:
            value = getter(name, None)
        except TypeError:
            try:
                value = getter(name)
            except Exception:
                value = None
        except Exception:
            value = None
        if value is not None:
            return value

    value = getattr(runtime, name, None)
    if value is not None:
        return value

    extras = getattr(runtime, "extras", None)
    if isinstance(extras, Mapping):
        return extras.get(name)
    return None


def _register_runtime_extra(runtime: Any, name: str, value: Any) -> None:
    if runtime is None or value is None:
        return

    register = getattr(runtime, "register", None)
    if callable(register):
        try:
            register(name, value, overwrite=False)
            return
        except KeyError:
            return
        except Exception:
            pass

    extras = getattr(runtime, "extras", None)
    if isinstance(extras, dict):
        extras.setdefault(name, value)


def _resolve_retriever(
    *,
    runtime: Optional["RuntimeContext"],
    retrieval_config: Mapping[str, Any],
    options: KGRetrievalOptions,
    legacy_config: Any,
) -> Any:
    if runtime is not None:
        direct = _runtime_get(runtime, "kg_retriever")
        if _is_retriever(direct):
            return direct

        for dependency_name in ("kg_service", "knowledge_graph_service"):
            dependency = _runtime_get(runtime, dependency_name)
            for attribute_name in ("retriever", "kg_retriever"):
                nested = getattr(dependency, attribute_name, None)
                if _is_retriever(nested):
                    return nested

        graph_loader = (
            _runtime_get(runtime, "graph_loader")
            or _runtime_get(runtime, "graph_store")
        )
        if graph_loader is not None:
            created = KGRetriever(
                graph_loader=graph_loader,
                evidence_builder=_resolve_evidence_builder(
                    runtime=runtime,
                    retriever=None,
                    options=options,
                ),
                default_options=options,
            )
            _register_runtime_extra(runtime, "kg_retriever", created)
            return created

        get_graph = getattr(runtime, "get_graph", None)
        if callable(get_graph):
            created = KGRetriever(
                graph=get_graph(),
                evidence_builder=_resolve_evidence_builder(
                    runtime=runtime,
                    retriever=None,
                    options=options,
                ),
                default_options=options,
            )
            _register_runtime_extra(runtime, "kg_retriever", created)
            return created

    graph_path = _resolve_graph_path(
        runtime=runtime,
        retrieval_config=retrieval_config,
        legacy_config=legacy_config,
    )
    cache_key = str(Path(graph_path).expanduser().resolve())
    cached = _LEGACY_RETRIEVER_CACHE.get(cache_key)
    if _is_retriever(cached):
        return cached

    loader = GraphLoader(
        graph_path=graph_path,
        use_cache=_as_bool(retrieval_config.get("use_cache", True), True),
        validate=_as_bool(retrieval_config.get("validate_graph", True), True),
    )
    created = KGRetriever(
        graph_loader=loader,
        evidence_builder=EvidenceBuilder(
            max_evidence=options.max_evidence,
            min_score=options.min_evidence_score,
            deduplicate=True,
        ),
        default_options=options,
    )
    _LEGACY_RETRIEVER_CACHE[cache_key] = created
    return created


def _resolve_evidence_builder(
    *,
    runtime: Optional["RuntimeContext"],
    retriever: Any,
    options: KGRetrievalOptions,
) -> EvidenceBuilder:
    existing = getattr(retriever, "evidence_builder", None)
    if _is_evidence_builder(existing):
        return existing

    runtime_builder = _runtime_get(runtime, "evidence_builder")
    if _is_evidence_builder(runtime_builder):
        return runtime_builder

    created = EvidenceBuilder(
        max_evidence=options.max_evidence,
        min_score=options.min_evidence_score,
        deduplicate=True,
    )
    _register_runtime_extra(runtime, "evidence_builder", created)
    return created


def _is_retriever(value: Any) -> bool:
    return value is not None and callable(getattr(value, "retrieve", None))


def _is_evidence_builder(value: Any) -> bool:
    return (
        value is not None
        and callable(getattr(value, "postprocess", None))
        and callable(getattr(value, "build_evidence_text", None))
    )


# ---------------------------------------------------------------------------
# Invocation and result normalization
# ---------------------------------------------------------------------------


def _invoke_retriever(
    *,
    retriever: Any,
    grounded_entities: List[Dict[str, Any]],
    options: KGRetrievalOptions,
) -> Any:
    method = getattr(retriever, "retrieve", None)
    if not callable(method):
        raise TypeError("KG retriever must expose a retrieve() method.")

    try:
        return method(grounded_entities, options=options)
    except TypeError as first_error:
        try:
            return method(grounded_entities, options=asdict(options))
        except TypeError:
            try:
                return method(grounded_entities)
            except TypeError:
                raise first_error


def _normalize_retrieval_result(
    *,
    raw_result: Any,
    builder: EvidenceBuilder,
    options: KGRetrievalOptions,
) -> Dict[str, Any]:
    result_mapping = _to_mapping(raw_result)

    evidence = normalize_evidence_list(result_mapping.get("evidence", []))
    raw_evidence = normalize_evidence_list(
        result_mapping.get("raw_evidence", [])
    )

    if not raw_evidence:
        raw_evidence = _build_raw_evidence_from_query_results(
            result_mapping=result_mapping,
            builder=builder,
        )

    if not evidence:
        evidence = builder.postprocess(raw_evidence)
    else:
        # Apply the request-scoped limits even when a custom retriever returns
        # already-normalized evidence with broader defaults.
        bounded_builder = EvidenceBuilder(
            max_evidence=options.max_evidence,
            min_score=options.min_evidence_score,
            deduplicate=True,
        )
        evidence = bounded_builder.postprocess(evidence)

    evidence_text = _normalize_text(result_mapping.get("evidence_text"))
    if not evidence_text:
        evidence_text = builder.build_evidence_text(
            evidence,
            max_items=options.max_evidence,
            include_score=True,
        )

    warnings = _normalize_warnings(result_mapping.get("warnings"))
    metadata = _mapping_copy(result_mapping.get("metadata"))
    metadata.update(_query_result_counts(result_mapping))

    return {
        "raw_evidence": raw_evidence,
        "evidence": evidence,
        "evidence_text": evidence_text,
        "warnings": warnings,
        "metadata": metadata,
    }


def _build_raw_evidence_from_query_results(
    *,
    result_mapping: Mapping[str, Any],
    builder: EvidenceBuilder,
) -> List[EvidenceItem]:
    raw: List[EvidenceItem] = []
    converters = (
        ("relation_results", "from_relation_result"),
        ("path_results", "from_path_result"),
        ("neighbor_results", "from_neighbor_result"),
        ("subgraph_results", "from_subgraph_result"),
    )

    for field_name, method_name in converters:
        method = getattr(builder, method_name, None)
        if not callable(method):
            continue
        for item in _as_sequence(result_mapping.get(field_name)):
            try:
                converted = method(item)
            except Exception:
                continue
            raw.extend(normalize_evidence_list(converted))

    return normalize_evidence_list(raw)


def _to_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return copy.deepcopy(dict(converted))

    if is_dataclass(value):
        converted = asdict(value)
        if isinstance(converted, Mapping):
            return converted

    fields = (
        "entities",
        "relation_results",
        "path_results",
        "neighbor_results",
        "subgraph_results",
        "raw_evidence",
        "evidence",
        "evidence_text",
        "warnings",
        "metadata",
    )
    result: Dict[str, Any] = {}
    for field_name in fields:
        if hasattr(value, field_name):
            result[field_name] = copy.deepcopy(getattr(value, field_name))
    return result


# ---------------------------------------------------------------------------
# Configuration and parameter boundaries
# ---------------------------------------------------------------------------


def _get_kg_retrieval_config(
    *,
    state: Mapping[str, Any],
    runtime: Optional["RuntimeContext"],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}

    settings = getattr(runtime, "settings", None)
    if settings is not None:
        kg_section = getattr(settings, "kg", None)
        if isinstance(kg_section, Mapping):
            merged = _deep_merge_dicts(merged, dict(kg_section))

        section_method = getattr(settings, "section", None)
        if callable(section_method):
            try:
                node_section = section_method("kg_retrieval")
            except Exception:
                node_section = {}
            if isinstance(node_section, Mapping):
                merged = _deep_merge_dicts(merged, dict(node_section))

    legacy = state.get("config") if isinstance(state, Mapping) else None
    if isinstance(legacy, Mapping):
        kg_section = legacy.get("kg")
        if isinstance(kg_section, Mapping):
            merged = _deep_merge_dicts(merged, dict(kg_section))

        top_level = legacy.get("kg_retrieval")
        if isinstance(top_level, Mapping):
            merged = _deep_merge_dicts(merged, dict(top_level))

        graph_section = legacy.get("graph")
        if isinstance(graph_section, Mapping):
            nested = graph_section.get("kg_retrieval")
            if isinstance(nested, Mapping):
                merged = _deep_merge_dicts(merged, dict(nested))

    return _flatten_nested_config(merged)


def _flatten_nested_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(config))
    subgraph = result.get("subgraph")
    if isinstance(subgraph, Mapping):
        result.setdefault("subgraph_max_depth", subgraph.get("max_depth"))
        result.setdefault("subgraph_max_nodes", subgraph.get("max_nodes"))
        result.setdefault("subgraph_max_edges", subgraph.get("max_edges"))
        result.setdefault("subgraph_direction", subgraph.get("direction"))

    evidence = result.get("evidence")
    if isinstance(evidence, Mapping):
        result.setdefault("min_score", evidence.get("min_score"))
        result.setdefault("min_evidence_score", evidence.get("min_score"))
        result.setdefault("max_evidence", evidence.get("max_evidence"))

    return result


def _build_options(
    *,
    config: Mapping[str, Any],
    request_options: Mapping[str, Any],
) -> KGRetrievalOptions:
    data: Dict[str, Any] = {
        "enable_relation": _first_present(
            config, "enable_relation", "enable_relation_search", default=True
        ),
        "enable_path": _first_present(
            config, "enable_path", "enable_path_search", default=True
        ),
        "enable_neighbor": _first_present(
            config, "enable_neighbor", "enable_neighbor_search", default=True
        ),
        "enable_subgraph": _first_present(
            config, "enable_subgraph", "enable_subgraph_search", default=False
        ),
        "relation_directed": config.get("relation_directed", True),
        "relation_include_reverse": config.get(
            "relation_include_reverse", True
        ),
        "max_relations": _first_present(
            config, "max_relations", default=DEFAULT_MAX_RELATIONS
        ),
        "path_directed": config.get("path_directed", True),
        "max_paths": _first_present(
            config, "max_paths", "max_paths_per_pair", default=DEFAULT_MAX_PATHS
        ),
        "max_path_length": _first_present(
            config,
            "max_path_length",
            "path_max_depth",
            default=DEFAULT_MAX_PATH_LENGTH,
        ),
        "neighbor_direction": config.get("neighbor_direction", "both"),
        "max_neighbors": _first_present(
            config,
            "max_neighbors",
            "max_neighbors_per_entity",
            default=DEFAULT_MAX_NEIGHBORS,
        ),
        "subgraph_direction": config.get("subgraph_direction", "both"),
        "subgraph_max_depth": _first_present(
            config, "subgraph_max_depth", default=DEFAULT_SUBGRAPH_MAX_DEPTH
        ),
        "subgraph_max_nodes": _first_present(
            config, "subgraph_max_nodes", default=DEFAULT_SUBGRAPH_MAX_NODES
        ),
        "subgraph_max_edges": _first_present(
            config, "subgraph_max_edges", default=DEFAULT_SUBGRAPH_MAX_EDGES
        ),
        "min_evidence_score": _first_present(
            config,
            "min_evidence_score",
            "min_score",
            default=0.0,
        ),
        "max_evidence": _first_present(
            config, "max_evidence", default=DEFAULT_MAX_EVIDENCE
        ),
    }

    if "retrieval_top_k" in request_options:
        data["max_evidence"] = request_options["retrieval_top_k"]
    if "path_max_depth" in request_options:
        data["max_path_length"] = request_options["path_max_depth"]

    # Explicitly bound before handing values to the domain schema so aliases
    # cannot bypass the same system limits.
    data["max_evidence"] = _bounded_int(
        data["max_evidence"], DEFAULT_MAX_EVIDENCE, 1, MAX_EVIDENCE_LIMIT
    )
    data["max_relations"] = _bounded_int(
        data["max_relations"], DEFAULT_MAX_RELATIONS, 1, MAX_RELATION_LIMIT
    )
    data["max_paths"] = _bounded_int(
        data["max_paths"], DEFAULT_MAX_PATHS, 1, MAX_PATH_LIMIT
    )
    data["max_path_length"] = _bounded_int(
        data["max_path_length"],
        DEFAULT_MAX_PATH_LENGTH,
        2,
        MAX_PATH_LENGTH_LIMIT,
    )
    data["max_neighbors"] = _bounded_int(
        data["max_neighbors"], DEFAULT_MAX_NEIGHBORS, 1, MAX_NEIGHBOR_LIMIT
    )

    return KGRetrievalOptions.from_mapping(data)


def _resolve_graph_path(
    *,
    runtime: Optional["RuntimeContext"],
    retrieval_config: Mapping[str, Any],
    legacy_config: Any,
) -> str:
    graph_path = _normalize_text(retrieval_config.get("graph_path"))

    if not graph_path and isinstance(legacy_config, Mapping):
        kg_section = legacy_config.get("kg")
        graph_section = legacy_config.get("graph")
        if isinstance(kg_section, Mapping):
            graph_path = _normalize_text(kg_section.get("graph_path"))
        if not graph_path and isinstance(graph_section, Mapping):
            graph_path = _normalize_text(graph_section.get("graph_path"))

    graph_path = graph_path or "data/demo/kg/graph.pkl"
    settings = getattr(runtime, "settings", None)
    resolve_path = getattr(settings, "resolve_path", None)
    if callable(resolve_path):
        return str(resolve_path(graph_path))

    candidate = Path(graph_path).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())

    project_root = _find_project_root(Path.cwd())
    return str((project_root / candidate).resolve())


# ---------------------------------------------------------------------------
# State updates and diagnostics
# ---------------------------------------------------------------------------


def _build_no_entities_update(
    *,
    state: Mapping[str, Any],
    warnings: List[str],
) -> AgentState:
    normalized_warnings = _deduplicate_texts(
        list(warnings) + ["No grounded entities found for retrieval."]
    )
    metadata = _mapping_copy(state.get("metadata"))
    metadata["kg_retrieval"] = {
        "status": "skipped",
        "reason": "no_grounded_entities",
        "num_grounded_entities": 0,
        "num_raw_evidence": 0,
        "num_evidence": 0,
    }
    return AgentState(
        raw_evidence=[],
        evidence=[],
        evidence_text="",
        metadata=metadata,
        has_error=False,
        error_stage="unknown",
        error_message="",
        error_detail={},
        traces=[
            {
                "stage": "kg_retrieval",
                "message": "No grounded entities found. KG retrieval skipped.",
                "timestamp": utc_now(),
                "payload": {"num_grounded_entities": 0},
            }
        ],
        warnings=normalized_warnings,
    )


def _build_success_update(
    *,
    state: Mapping[str, Any],
    grounded_entities: List[Dict[str, Any]],
    raw_evidence: List[EvidenceItem],
    evidence: List[EvidenceItem],
    evidence_text: str,
    warnings: List[str],
    options: KGRetrievalOptions,
    result_metadata: Mapping[str, Any],
    raw_result: Any,
    runtime: Optional["RuntimeContext"],
    retriever: Any,
) -> AgentState:
    metadata = _mapping_copy(state.get("metadata"))
    metadata["kg_retrieval"] = {
        "status": "completed",
        "num_grounded_entities": len(grounded_entities),
        "num_raw_evidence": len(raw_evidence),
        "num_evidence": len(evidence),
        "options": asdict(options),
        "result": _lightweight_result_metadata(result_metadata, raw_result),
        "runtime_managed": runtime is not None,
        "retriever_type": type(retriever).__name__,
    }

    return AgentState(
        raw_evidence=copy.deepcopy(raw_evidence),
        evidence=copy.deepcopy(evidence),
        evidence_text=evidence_text,
        metadata=metadata,
        has_error=False,
        error_stage="unknown",
        error_message="",
        error_detail={},
        traces=[
            {
                "stage": "kg_retrieval",
                "message": "KG retrieval completed.",
                "timestamp": utc_now(),
                "payload": {
                    "num_grounded_entities": len(grounded_entities),
                    "num_raw_evidence": len(raw_evidence),
                    "num_evidence": len(evidence),
                    "evidence_types": _evidence_type_counts(evidence),
                },
            }
        ],
        warnings=_deduplicate_texts(warnings),
    )


def _lightweight_result_metadata(
    metadata: Mapping[str, Any],
    raw_result: Any,
) -> Dict[str, Any]:
    result = _mapping_copy(metadata)
    result.update(_query_result_counts(_to_mapping(raw_result)))
    return _sanitize_metadata(result)


def _query_result_counts(result_mapping: Mapping[str, Any]) -> Dict[str, int]:
    return {
        "relation_query_count": len(_as_sequence(result_mapping.get("relation_results"))),
        "path_query_count": len(_as_sequence(result_mapping.get("path_results"))),
        "neighbor_query_count": len(_as_sequence(result_mapping.get("neighbor_results"))),
        "subgraph_query_count": len(_as_sequence(result_mapping.get("subgraph_results"))),
    }


def _evidence_type_counts(evidence: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for item in evidence:
        evidence_type = _normalize_text(item.get("evidence_type")) or "unknown"
        result[evidence_type] = result.get(evidence_type, 0) + 1
    return result


def _log_failure(runtime: Optional["RuntimeContext"], exc: Exception) -> None:
    logger = getattr(runtime, "logger", None)
    if logger is not None and callable(getattr(logger, "exception", None)):
        logger.exception("KG retrieval node failed: %s", exc)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _normalize_grounded_entities(value: Any) -> tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    result: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for index, item in enumerate(_as_sequence(value)[:MAX_GROUNDED_ENTITIES]):
        mapping = _to_mapping(item)
        node_key = _first_non_empty(
            mapping,
            "node_key",
            "graph_node",
            "entity_id",
            "entity_name",
            "name",
        )
        if not node_key:
            warnings.append(f"Ignored grounded entity at index {index}: missing node key.")
            continue

        marker = node_key.casefold()
        if marker in seen:
            continue
        seen.add(marker)

        normalized = {
            "mention": _normalize_text(mapping.get("mention")),
            "entity_id": _normalize_text(mapping.get("entity_id")) or node_key,
            "entity_name": _normalize_text(mapping.get("entity_name")) or node_key,
            "node_key": node_key,
            "confidence": _bounded_float(
                mapping.get("confidence", mapping.get("score", 0.0)),
                0.0,
                0.0,
                1.0,
            ),
            "in_graph": _as_bool(mapping.get("in_graph", True), True),
            "metadata": _mapping_copy(mapping.get("metadata")),
        }
        result.append(normalized)

    if len(_as_sequence(value)) > MAX_GROUNDED_ENTITIES:
        warnings.append(
            f"Grounded entities were truncated to {MAX_GROUNDED_ENTITIES}."
        )
    return result, _deduplicate_texts(warnings)


def _normalize_warnings(value: Any) -> List[str]:
    if isinstance(value, str):
        return [_normalize_text(value)] if _normalize_text(value) else []
    return _deduplicate_texts(
        [_normalize_text(item) for item in _as_sequence(value)]
    )


def _safe_grounded_entity_summary(value: Any) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for item in _as_sequence(value)[:20]:
        mapping = _to_mapping(item)
        result.append(
            {
                "mention": _normalize_text(mapping.get("mention")),
                "node_key": _first_non_empty(
                    mapping,
                    "node_key",
                    "graph_node",
                    "entity_id",
                    "entity_name",
                ),
            }
        )
    return result


def _sanitize_metadata(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return _normalize_text(value)[:500]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 100:
                result["_truncated"] = True
                break
            result[str(key)] = _sanitize_metadata(item, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _sanitize_metadata(item, depth=depth + 1)
            for item in list(value)[:100]
        ]
    return _normalize_text(value)[:500]


def _mapping_copy(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    return {}


def _as_sequence(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _first_present(
    mapping: Mapping[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _first_non_empty(mapping: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = _normalize_text(mapping.get(key))
        if value:
            return value
    return ""


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _deduplicate_texts(values: Sequence[Any]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = _normalize_text(value)
        marker = text.casefold()
        if not text or marker in seen:
            continue
        seen.add(marker)
        result.append(text)
    return result


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


def _deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _find_project_root(start: Path) -> Path:
    current = start.expanduser().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / "configs").is_dir():
            return candidate
    return current


__all__ = [
    "kg_retrieval_node",
    "create_kg_retrieval_node",
]
