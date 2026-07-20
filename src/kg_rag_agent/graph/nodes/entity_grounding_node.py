# -*- coding: utf-8 -*-
"""Entity Grounding LangGraph 节点。

本节点只负责：
    1. 读取 ``AgentState["entity_candidates"]``；
    2. 调用 ``entity_resolution.EntityGrounder``；
    3. 校验候选实体是否能够落到真实知识图谱节点；
    4. 写回 ``grounded_entities``、``ungrounded_mentions``、Metadata 和 Trace。

职责边界：
    * 不执行 Mention 抽取或实体链接；
    * 不执行关系、路径、邻居或子图检索；
    * 不生成 Evidence、Reasoning 或最终答案；
    * Graph、GraphLoader、索引和 RuntimeContext 不得写入 AgentState。

迁移兼容：
    * 正式路径优先复用 ``RuntimeContext`` 中的 GraphLoader 或 EntityGrounder；
    * 无 Runtime 的旧入口仍可从 ``state["config"]`` 创建 GraphLoader；
    * 同时提供 ``entity_grounding_node`` 和 ``create_entity_grounding_node``。
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from kg_rag_agent.entity_resolution import EntityGrounder, GroundingOptions
from kg_rag_agent.kg import GraphLoader

from ..state import (
    AgentState,
    EntityCandidate,
    GroundedEntity,
    make_error,
    utc_now,
)

if TYPE_CHECKING:
    from kg_rag_agent.runtime import RuntimeContext


DEFAULT_MIN_SCORE = 0.0
DEFAULT_MAX_ENTITIES = 16
DEFAULT_MAX_MENTIONS = 128
MAX_ENTITIES_LIMIT = 256
MAX_MENTIONS_LIMIT = 512

# 仅供无 Runtime 的旧式直接调用使用。正式 Graph 由 Runtime 管理生命周期。
_LEGACY_GROUNDER_CACHE: Dict[str, EntityGrounder] = {}


def create_entity_grounding_node(
    runtime: Optional["RuntimeContext"] = None,
):
    """创建已绑定 RuntimeContext 的 Entity Grounding Node。"""

    _ensure_runtime_open(runtime)

    # Runtime 存在时可提前绑定领域对象；无 Runtime 时必须等到收到 State 后，
    # 才能读取迁移期 config 中的 graph_path。
    bound_grounder: Optional[Any] = None
    if runtime is not None:
        bound_grounder = _resolve_grounder(
            runtime=runtime,
            grounding_config=_get_entity_grounding_config(
                state={},
                runtime=runtime,
            ),
            legacy_config={},
        )

    def _node(state: AgentState) -> AgentState:
        return entity_grounding_node(
            state,
            runtime=runtime,
            grounder=bound_grounder,
        )

    _node.__name__ = "entity_grounding_node"
    _node.__qualname__ = "entity_grounding_node"
    _node.__doc__ = entity_grounding_node.__doc__
    return _node


def entity_grounding_node(
    state: AgentState,
    *,
    runtime: Optional["RuntimeContext"] = None,
    grounder: Optional[Any] = None,
) -> AgentState:
    """将实体候选映射为可供 KG 检索使用的真实图谱实体。"""

    try:
        _ensure_runtime_open(runtime)

        grounding_config = _get_entity_grounding_config(
            state=state,
            runtime=runtime,
        )

        max_mentions = _bounded_int(
            grounding_config.get("max_mentions", DEFAULT_MAX_MENTIONS),
            default=DEFAULT_MAX_MENTIONS,
            minimum=1,
            maximum=MAX_MENTIONS_LIMIT,
        )
        normalized_candidates, normalization_warnings = _normalize_candidates_by_mention(
            state.get("entity_candidates"),
            max_mentions=max_mentions,
        )

        if not normalized_candidates:
            return _build_no_candidates_update(
                state=state,
                warnings=normalization_warnings,
            )

        options = GroundingOptions(
            min_score=_bounded_float(
                grounding_config.get("min_score", DEFAULT_MIN_SCORE),
                default=DEFAULT_MIN_SCORE,
                minimum=0.0,
                maximum=1.0,
            ),
            max_entities=_bounded_int(
                grounding_config.get("max_entities", DEFAULT_MAX_ENTITIES),
                default=DEFAULT_MAX_ENTITIES,
                minimum=1,
                maximum=MAX_ENTITIES_LIMIT,
            ),
            one_entity_per_mention=_as_bool(
                grounding_config.get("one_entity_per_mention", True),
                default=True,
            ),
            require_in_graph=_as_bool(
                grounding_config.get("require_in_graph", True),
                default=True,
            ),
            allow_linear_scan=_as_bool(
                grounding_config.get("allow_linear_scan", True),
                default=True,
            ),
        )

        resolved_grounder = grounder or _resolve_grounder(
            runtime=runtime,
            grounding_config=grounding_config,
            legacy_config=state.get("config"),
        )

        raw_result = _invoke_grounder(
            grounder=resolved_grounder,
            candidates_by_mention=normalized_candidates,
            options=options,
        )
        grounded_entities, result_ungrounded, result_warnings = _normalize_grounding_result(
            raw_result=raw_result,
            candidates_by_mention=normalized_candidates,
            require_in_graph=options.require_in_graph,
            max_entities=options.max_entities,
        )

        # 领域实现可以只返回 grounded_entities；Node 统一补齐未落地 Mention。
        grounded_mentions = {
            _normalize_text(item.get("mention"))
            for item in grounded_entities
            if _normalize_text(item.get("mention"))
        }
        ungrounded_mentions = _deduplicate_texts(
            list(result_ungrounded)
            + [
                mention
                for mention in normalized_candidates
                if mention not in grounded_mentions
            ]
        )

        warnings = _deduplicate_texts(
            list(normalization_warnings) + list(result_warnings)
        )
        for mention in ungrounded_mentions:
            message = f"No grounded entity found for mention: {mention}"
            if message not in warnings:
                warnings.append(message)

        return _build_success_update(
            state=state,
            candidates_by_mention=normalized_candidates,
            grounded_entities=grounded_entities,
            ungrounded_mentions=ungrounded_mentions,
            options=options,
            max_mentions=max_mentions,
            warnings=warnings,
            runtime=runtime,
            grounder=resolved_grounder,
        )

    except Exception as exc:
        _log_failure(runtime, exc)
        return make_error(
            stage="entity_grounding",
            message=str(exc),
            detail={
                "candidate_mentions": list(
                    (state.get("entity_candidates") or {}).keys()
                )
                if isinstance(state.get("entity_candidates"), Mapping)
                else [],
                "error_type": type(exc).__name__,
            },
        )


# ---------------------------------------------------------------------------
# Runtime and domain-object resolution
# ---------------------------------------------------------------------------


def _ensure_runtime_open(runtime: Optional["RuntimeContext"]) -> None:
    if runtime is None:
        return
    ensure_open = getattr(runtime, "ensure_open", None)
    if callable(ensure_open):
        ensure_open()


def _runtime_get(runtime: Any, name: str) -> Any:
    if runtime is None:
        return None

    get_method = getattr(runtime, "get", None)
    if callable(get_method):
        try:
            value = get_method(name, None)
        except TypeError:
            try:
                value = get_method(name)
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


def _resolve_grounder(
    *,
    runtime: Optional["RuntimeContext"],
    grounding_config: Mapping[str, Any],
    legacy_config: Any,
) -> Any:
    """优先复用 Runtime 依赖，最后才创建迁移期 EntityGrounder。"""

    if runtime is not None:
        direct = _runtime_get(runtime, "entity_grounder")
        if _is_grounder(direct):
            return direct

        for dependency_name in (
            "entity_resolution_pipeline",
            "entity_resolver",
        ):
            dependency = _runtime_get(runtime, dependency_name)
            for attribute_name in ("grounder", "entity_grounder"):
                nested = getattr(dependency, attribute_name, None)
                if _is_grounder(nested):
                    return nested

        graph_loader = (
            _runtime_get(runtime, "graph_loader")
            or _runtime_get(runtime, "graph_store")
        )
        if graph_loader is not None:
            created = EntityGrounder(
                graph_loader=graph_loader,
                default_options=GroundingOptions.from_mapping(grounding_config),
            )
            _register_runtime_extra(runtime, "entity_grounder", created)
            return created

        get_graph = getattr(runtime, "get_graph", None)
        if callable(get_graph):
            created = EntityGrounder(
                graph=get_graph(),
                default_options=GroundingOptions.from_mapping(grounding_config),
            )
            _register_runtime_extra(runtime, "entity_grounder", created)
            return created

    config = _runtime_settings_dict(runtime)
    if isinstance(legacy_config, Mapping):
        config = _deep_merge_dicts(
            copy.deepcopy(dict(legacy_config)),
            config,
        )

    graph_path = _resolve_graph_path(
        config=config,
        grounding_config=grounding_config,
    )
    cache_key = str(Path(graph_path).expanduser())
    cached = _LEGACY_GROUNDER_CACHE.get(cache_key)
    if _is_grounder(cached):
        return cached

    loader = GraphLoader(
        graph_path=graph_path,
        use_cache=_as_bool(
            grounding_config.get("use_graph_cache", True),
            default=True,
        ),
        validate=_as_bool(
            grounding_config.get("validate_graph", True),
            default=True,
        ),
    )
    created = EntityGrounder(
        graph_loader=loader,
        default_options=GroundingOptions.from_mapping(grounding_config),
    )
    _LEGACY_GROUNDER_CACHE[cache_key] = created
    return created


def _register_runtime_extra(
    runtime: Optional["RuntimeContext"],
    name: str,
    value: Any,
) -> None:
    if runtime is None:
        return

    register = getattr(runtime, "register", None)
    if callable(register):
        try:
            register(name, value)
            return
        except KeyError:
            return
        except Exception:
            pass

    extras = getattr(runtime, "extras", None)
    if isinstance(extras, dict):
        extras.setdefault(name, value)


def _is_grounder(value: Any) -> bool:
    return value is not None and any(
        callable(getattr(value, method_name, None))
        for method_name in ("ground", "ground_entities", "resolve")
    )


def _runtime_settings_dict(
    runtime: Optional["RuntimeContext"],
) -> Dict[str, Any]:
    if runtime is None:
        return {}

    settings = getattr(runtime, "settings", None)
    if settings is None:
        return {}

    to_dict = getattr(settings, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return copy.deepcopy(dict(value))

    if isinstance(settings, Mapping):
        return copy.deepcopy(dict(settings))

    return {}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _get_entity_grounding_config(
    *,
    state: Mapping[str, Any],
    runtime: Optional["RuntimeContext"],
) -> Dict[str, Any]:
    """读取 Grounding 配置；Runtime 配置覆盖迁移期 State 配置。"""

    merged = _get_grounding_config(
        state.get("config") if isinstance(state, Mapping) else {}
    )

    if runtime is None:
        return merged

    settings = getattr(runtime, "settings", None)
    if settings is None:
        return merged

    runtime_config: Dict[str, Any] = {}
    section_method = getattr(settings, "section", None)
    if callable(section_method):
        value = section_method("entity_grounding")
        if isinstance(value, Mapping):
            runtime_config = copy.deepcopy(dict(value))
    elif isinstance(settings, Mapping):
        runtime_config = _get_grounding_config(dict(settings))
    else:
        get_method = getattr(settings, "get", None)
        if callable(get_method):
            direct = get_method("entity_grounding", {})
            nested = get_method("graph.entity_grounding", {})
            if isinstance(direct, Mapping):
                runtime_config.update(copy.deepcopy(dict(direct)))
            if isinstance(nested, Mapping):
                runtime_config.update(copy.deepcopy(dict(nested)))

    merged.update(runtime_config)
    return merged


def _get_grounding_config(config: Any) -> Dict[str, Any]:
    """兼容 ``entity_grounding`` 与 ``graph.entity_grounding`` 两种位置。"""

    if not isinstance(config, Mapping):
        return {}

    result: Dict[str, Any] = {}
    direct = config.get("entity_grounding")
    graph = config.get("graph")

    if isinstance(direct, Mapping):
        result.update(copy.deepcopy(dict(direct)))
    if isinstance(graph, Mapping):
        nested = graph.get("entity_grounding")
        if isinstance(nested, Mapping):
            result.update(copy.deepcopy(dict(nested)))

    return result


def _resolve_graph_path(
    *,
    config: Mapping[str, Any],
    grounding_config: Mapping[str, Any],
) -> str:
    kg_config = config.get("kg") if isinstance(config, Mapping) else {}
    graph_config = config.get("graph") if isinstance(config, Mapping) else {}

    candidates = [
        grounding_config.get("graph_path"),
        kg_config.get("graph_path") if isinstance(kg_config, Mapping) else None,
        graph_config.get("graph_path") if isinstance(graph_config, Mapping) else None,
        config.get("graph_path") if isinstance(config, Mapping) else None,
        "data/demo/kg/graph.pkl",
    ]
    for candidate in candidates:
        path = _normalize_text(candidate)
        if path:
            return path
    return "data/demo/kg/graph.pkl"


# ---------------------------------------------------------------------------
# Domain invocation and normalization
# ---------------------------------------------------------------------------


def _invoke_grounder(
    *,
    grounder: Any,
    candidates_by_mention: Mapping[str, List[EntityCandidate]],
    options: GroundingOptions,
) -> Any:
    if not _is_grounder(grounder):
        raise TypeError(
            "Entity grounder must provide ground(), ground_entities(), or resolve()."
        )

    method = None
    for method_name in ("ground", "ground_entities", "resolve"):
        candidate = getattr(grounder, method_name, None)
        if callable(candidate):
            method = candidate
            break

    if method is None:
        raise TypeError("No callable grounding method found.")

    try:
        return method(candidates_by_mention, options=options)
    except TypeError as first_error:
        # 兼容只接受 Mapping options 或不接受 options 的旧适配器。
        try:
            return method(candidates_by_mention, options=asdict(options))
        except TypeError:
            try:
                return method(candidates_by_mention)
            except TypeError:
                raise first_error


def _normalize_grounding_result(
    *,
    raw_result: Any,
    candidates_by_mention: Mapping[str, List[EntityCandidate]],
    require_in_graph: bool,
    max_entities: int,
) -> tuple[List[GroundedEntity], List[str], List[str]]:
    mapping = _result_to_mapping(raw_result)

    if isinstance(raw_result, (list, tuple)):
        raw_grounded: Any = list(raw_result)
        raw_ungrounded: Any = []
        raw_warnings: Any = []
    else:
        raw_grounded = (
            mapping.get("grounded_entities")
            or mapping.get("entities")
            or mapping.get("results")
            or []
        )
        raw_ungrounded = (
            mapping.get("ungrounded_mentions")
            or mapping.get("unresolved_mentions")
            or []
        )
        raw_warnings = mapping.get("warnings") or []

    if isinstance(raw_grounded, Mapping):
        raw_grounded = list(raw_grounded.values())
    if not isinstance(raw_grounded, (list, tuple)):
        raw_grounded = []

    grounded: List[GroundedEntity] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for item in raw_grounded:
        normalized = _normalize_grounded_entity(
            item=item,
            candidates_by_mention=candidates_by_mention,
        )
        if not normalized:
            continue
        if require_in_graph and not bool(normalized.get("in_graph", False)):
            continue

        identity = (
            _normalize_text(normalized.get("mention")),
            _normalize_text(normalized.get("node_key")),
            _normalize_text(normalized.get("entity_id")),
        )
        if identity in seen_keys:
            continue
        seen_keys.add(identity)
        grounded.append(normalized)
        if len(grounded) >= max_entities:
            break

    ungrounded = _normalize_text_list(raw_ungrounded)
    warnings = _normalize_text_list(raw_warnings)
    return grounded, ungrounded, warnings


def _normalize_grounded_entity(
    *,
    item: Any,
    candidates_by_mention: Mapping[str, List[EntityCandidate]],
) -> GroundedEntity:
    if is_dataclass(item):
        item = asdict(item)
    if not isinstance(item, Mapping):
        return GroundedEntity()

    raw = copy.deepcopy(dict(item))
    mention = _first_non_empty_text(
        raw,
        ("mention", "mention_text", "query_mention"),
    )
    entity_id = _first_non_empty_text(
        raw,
        ("entity_id", "id", "node_id", "qid"),
    )
    entity_name = _first_non_empty_text(
        raw,
        ("entity_name", "name", "entity", "label", "title"),
    )
    node_key = _first_non_empty_text(
        raw,
        ("node_key", "graph_node", "graph_key", "node"),
    )

    # 某些旧适配器只返回 node_key；补齐可读实体名。
    if not entity_name:
        entity_name = node_key or entity_id
    if not node_key and bool(raw.get("in_graph", False)):
        node_key = entity_id or entity_name

    if not mention:
        mention = _infer_mention(
            entity_id=entity_id,
            entity_name=entity_name,
            node_key=node_key,
            candidates_by_mention=candidates_by_mention,
        )

    if not mention or not (entity_name or entity_id or node_key):
        return GroundedEntity()

    confidence = _bounded_float(
        raw.get("confidence", raw.get("score", 0.0)),
        default=0.0,
        minimum=0.0,
        maximum=1.0,
    )
    in_graph = _as_bool(
        raw.get("in_graph", bool(node_key)),
        default=bool(node_key),
    )

    raw_metadata = raw.get("metadata")
    metadata = (
        copy.deepcopy(dict(raw_metadata))
        if isinstance(raw_metadata, Mapping)
        else {}
    )
    reserved = {
        "mention", "mention_text", "query_mention",
        "entity_id", "id", "node_id", "qid",
        "entity_name", "name", "entity", "label", "title",
        "node_key", "graph_node", "graph_key", "node",
        "confidence", "score", "in_graph", "metadata",
    }
    metadata.update(
        {
            key: copy.deepcopy(value)
            for key, value in raw.items()
            if key not in reserved
        }
    )

    return GroundedEntity(
        mention=mention,
        entity_id=entity_id,
        entity_name=entity_name,
        node_key=node_key,
        confidence=confidence,
        in_graph=in_graph,
        metadata=metadata,
    )


def _result_to_mapping(result: Any) -> Dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, Mapping):
        return copy.deepcopy(dict(result))
    if is_dataclass(result):
        value = asdict(result)
        return value if isinstance(value, dict) else {}
    result_dict = getattr(result, "__dict__", None)
    if isinstance(result_dict, Mapping):
        return copy.deepcopy(dict(result_dict))
    return {}


# ---------------------------------------------------------------------------
# Candidate normalization
# ---------------------------------------------------------------------------


def _normalize_candidates_by_mention(
    value: Any,
    *,
    max_mentions: int,
) -> tuple[Dict[str, List[EntityCandidate]], List[str]]:
    if value is None:
        return {}, []
    if not isinstance(value, Mapping):
        raise TypeError("entity_candidates must be a mapping by mention.")

    result: Dict[str, List[EntityCandidate]] = {}
    warnings: List[str] = []

    for raw_mention, raw_candidates in value.items():
        mention = _normalize_text(raw_mention)
        if not mention:
            warnings.append("Ignored entity candidate group with empty mention.")
            continue
        if mention in result:
            warnings.append(f"Duplicate candidate group ignored: {mention}")
            continue
        if len(result) >= max_mentions:
            warnings.append(
                f"Entity candidate groups truncated to max_mentions={max_mentions}."
            )
            break

        if raw_candidates is None:
            result[mention] = []
            continue
        if isinstance(raw_candidates, Mapping):
            candidate_items: Sequence[Any] = [raw_candidates]
        elif isinstance(raw_candidates, Sequence) and not isinstance(
            raw_candidates, (str, bytes, bytearray)
        ):
            candidate_items = raw_candidates
        else:
            warnings.append(
                f"Invalid candidate list ignored for mention: {mention}"
            )
            result[mention] = []
            continue

        candidates: List[EntityCandidate] = []
        seen: set[tuple[str, str]] = set()
        for raw_candidate in candidate_items:
            candidate = _normalize_candidate(
                mention=mention,
                item=raw_candidate,
            )
            if not candidate:
                continue
            identity = (
                _normalize_text(candidate.get("entity_id")),
                _normalize_text(candidate.get("entity_name")),
            )
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(candidate)

        candidates.sort(
            key=lambda item: (
                bool((item.get("metadata") or {}).get("selected", False)),
                float(item.get("score", 0.0)),
                _normalize_text(item.get("entity_name")),
            ),
            reverse=True,
        )
        result[mention] = candidates

    return result, warnings


def _normalize_candidate(
    *,
    mention: str,
    item: Any,
) -> EntityCandidate:
    if is_dataclass(item):
        item = asdict(item)
    if isinstance(item, str):
        entity_name = _normalize_text(item)
        if not entity_name:
            return EntityCandidate()
        return EntityCandidate(
            mention=mention,
            entity_id="",
            entity_name=entity_name,
            score=1.0,
            source="string",
            aliases=[],
            metadata={},
        )
    if not isinstance(item, Mapping):
        return EntityCandidate()

    raw = copy.deepcopy(dict(item))
    entity_id = _first_non_empty_text(
        raw,
        ("entity_id", "id", "node_id", "qid", "key"),
    )
    entity_name = _first_non_empty_text(
        raw,
        ("entity_name", "name", "entity", "label", "title", "document"),
    )
    if not entity_name:
        entity_name = entity_id
    if not entity_name and not entity_id:
        return EntityCandidate()

    score = _bounded_float(
        raw.get("score", raw.get("confidence", raw.get("similarity", 0.0))),
        default=0.0,
        minimum=0.0,
        maximum=1.0,
    )
    source = _normalize_text(raw.get("source")) or "unknown"

    raw_aliases = raw.get("aliases") or []
    if isinstance(raw_aliases, str):
        raw_aliases = [raw_aliases]
    aliases = (
        _deduplicate_texts(raw_aliases)
        if isinstance(raw_aliases, (list, tuple, set))
        else []
    )

    raw_metadata = raw.get("metadata")
    metadata = (
        copy.deepcopy(dict(raw_metadata))
        if isinstance(raw_metadata, Mapping)
        else {}
    )
    reserved = {
        "mention", "entity_id", "id", "node_id", "qid", "key",
        "entity_name", "name", "entity", "label", "title", "document",
        "score", "confidence", "similarity", "source", "aliases", "metadata",
    }
    metadata.update(
        {
            key: copy.deepcopy(value)
            for key, value in raw.items()
            if key not in reserved
        }
    )

    return EntityCandidate(
        mention=mention,
        entity_id=entity_id,
        entity_name=entity_name,
        score=score,
        source=source,
        aliases=aliases,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# State updates
# ---------------------------------------------------------------------------


def _build_no_candidates_update(
    *,
    state: Mapping[str, Any],
    warnings: List[str],
) -> AgentState:
    metadata = _copy_metadata(state)
    metadata["entity_grounding"] = {
        "status": "skipped",
        "reason": "no_entity_candidates",
        "num_candidate_mentions": 0,
        "num_grounded_entities": 0,
        "num_ungrounded_mentions": 0,
    }

    output_warnings = list(warnings)
    message = "No entity candidates found for grounding."
    if message not in output_warnings:
        output_warnings.append(message)

    return AgentState(
        grounded_entities=[],
        ungrounded_mentions=[],
        has_error=False,
        error_stage="unknown",
        error_message="",
        error_detail={},
        metadata=metadata,
        traces=[
            {
                "stage": "entity_grounding",
                "message": "No entity candidates found. Grounding skipped.",
                "timestamp": utc_now(),
                "payload": {},
            }
        ],
        warnings=output_warnings,
    )


def _build_success_update(
    *,
    state: Mapping[str, Any],
    candidates_by_mention: Mapping[str, List[EntityCandidate]],
    grounded_entities: List[GroundedEntity],
    ungrounded_mentions: List[str],
    options: GroundingOptions,
    max_mentions: int,
    warnings: List[str],
    runtime: Optional["RuntimeContext"],
    grounder: Any,
) -> AgentState:
    metadata = _copy_metadata(state)
    metadata["entity_grounding"] = {
        "status": "completed",
        "num_candidate_mentions": len(candidates_by_mention),
        "num_candidates": sum(len(items) for items in candidates_by_mention.values()),
        "num_grounded_entities": len(grounded_entities),
        "num_ungrounded_mentions": len(ungrounded_mentions),
        "ungrounded_mentions": list(ungrounded_mentions),
        "min_score": options.min_score,
        "max_entities": options.max_entities,
        "max_mentions": max_mentions,
        "one_entity_per_mention": options.one_entity_per_mention,
        "require_in_graph": options.require_in_graph,
        "allow_linear_scan": options.allow_linear_scan,
        "grounder_type": type(grounder).__name__,
        "runtime_managed": runtime is not None,
    }

    return AgentState(
        grounded_entities=grounded_entities,
        ungrounded_mentions=ungrounded_mentions,
        has_error=False,
        error_stage="unknown",
        error_message="",
        error_detail={},
        metadata=metadata,
        traces=[
            {
                "stage": "entity_grounding",
                "message": "Entity grounding completed.",
                "timestamp": utc_now(),
                "payload": {
                    "num_candidate_mentions": len(candidates_by_mention),
                    "num_grounded_entities": len(grounded_entities),
                    "num_ungrounded_mentions": len(ungrounded_mentions),
                },
            }
        ],
        warnings=warnings,
    )


def _copy_metadata(state: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = state.get("metadata") if isinstance(state, Mapping) else None
    return (
        copy.deepcopy(dict(metadata))
        if isinstance(metadata, Mapping)
        else {}
    )


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _infer_mention(
    *,
    entity_id: str,
    entity_name: str,
    node_key: str,
    candidates_by_mention: Mapping[str, List[EntityCandidate]],
) -> str:
    targets = {
        _normalize_text(entity_id),
        _normalize_text(entity_name),
        _normalize_text(node_key),
    }
    targets.discard("")

    for mention, candidates in candidates_by_mention.items():
        for candidate in candidates:
            values = {
                _normalize_text(candidate.get("entity_id")),
                _normalize_text(candidate.get("entity_name")),
                _normalize_text((candidate.get("metadata") or {}).get("node_key")),
            }
            if targets.intersection(values):
                return mention
    return ""


def _first_non_empty_text(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
) -> str:
    for key in keys:
        value = _normalize_text(mapping.get(key))
        if value:
            return value
    return ""


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return []
    return _deduplicate_texts(value)


def _deduplicate_texts(values: Sequence[Any]) -> List[str]:
    result: List[str] = []
    for value in values:
        text = _normalize_text(value)
        if text and text not in result:
            result.append(text)
    return result


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = _normalize_text(value).lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _bounded_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        return default
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, result))


def _bounded_float(
    value: Any,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, result))


def _deep_merge_dicts(
    base: Dict[str, Any],
    override: Mapping[str, Any],
) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _log_failure(
    runtime: Optional["RuntimeContext"],
    exc: Exception,
) -> None:
    logger = getattr(runtime, "logger", None) if runtime is not None else None
    if logger is None:
        return
    try:
        logger.exception("Entity grounding failed: %s", exc)
    except Exception:
        pass


__all__ = [
    "entity_grounding_node",
    "create_entity_grounding_node",
]
