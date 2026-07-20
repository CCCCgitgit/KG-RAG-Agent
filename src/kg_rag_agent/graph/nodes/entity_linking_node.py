# -*- coding: utf-8 -*-
"""Entity Linking LangGraph 节点。

本节点只负责：
    1. 读取 ``AgentState[\"mentions\"]``；
    2. 调用 ``entity_resolution.EntityLinker`` 领域能力；
    3. 将不同底层返回格式统一为 ``EntityCandidate``；
    4. 把候选、轻量 Metadata、Trace 与 Warning 写回 AgentState。

职责边界：
    * 不执行实体 Grounding，不验证实体是否真实存在于知识图谱；
    * 不执行关系、路径、邻居或子图查询；
    * 不创建第二套向量检索或 Embedding 实现；
    * RuntimeContext、Client、Store 和 Manager 不得写入 AgentState。

迁移兼容：
    * 正式路径优先复用 ``RuntimeContext.entity_linker``；
    * ``state[\"config\"]`` 和 ``_ENTITY_LINKER_CACHE`` 仅保留给旧入口与旧测试；
    * 同时提供 ``entity_linking_node`` 和 ``create_entity_linking_node``。
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from kg_rag_agent.entity_resolution import EntityLinker

from ..state import AgentState, EntityCandidate, make_error, utc_now

if TYPE_CHECKING:
    from kg_rag_agent.runtime import RuntimeContext


DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.0
MAX_TOP_K = 100
MAX_MENTIONS = 128
_MAX_TRACE_CANDIDATES_PER_MENTION = 5

# 迁移期兼容缓存。正式 Graph 通过 RuntimeContext 复用 EntityLinker，不使用此缓存。
_ENTITY_LINKER_CACHE: Dict[str, Any] = {}


def create_entity_linking_node(
    runtime: Optional["RuntimeContext"] = None,
):
    """创建已绑定 RuntimeContext 的 Entity Linking Node。"""

    _ensure_runtime_open(runtime)
    linking_config = _get_entity_linking_config(state={}, runtime=runtime)
    linker = _resolve_entity_linker(
        runtime=runtime,
        linking_config=linking_config,
        legacy_config={},
    )

    def _node(state: AgentState) -> AgentState:
        return entity_linking_node(
            state,
            runtime=runtime,
            linker=linker,
        )

    _node.__name__ = "entity_linking_node"
    _node.__qualname__ = "entity_linking_node"
    _node.__doc__ = entity_linking_node.__doc__
    return _node


def entity_linking_node(
    state: AgentState,
    *,
    runtime: Optional["RuntimeContext"] = None,
    linker: Optional[Any] = None,
) -> AgentState:
    """为 State 中的 Mention 生成实体候选，并返回部分状态更新。"""

    try:
        _ensure_runtime_open(runtime)

        linking_config = _get_entity_linking_config(
            state=state,
            runtime=runtime,
        )
        mentions, mention_warnings = _normalize_mentions(
            state.get("mentions"),
            max_mentions=_bounded_int(
                linking_config.get("max_mentions", MAX_MENTIONS),
                default=MAX_MENTIONS,
                minimum=1,
                maximum=MAX_MENTIONS,
            ),
        )

        if not mentions:
            return _build_no_mentions_update(
                state=state,
                warnings=mention_warnings,
            )

        top_k = _bounded_int(
            linking_config.get("top_k", DEFAULT_TOP_K),
            default=DEFAULT_TOP_K,
            minimum=1,
            maximum=MAX_TOP_K,
        )
        min_score = _bounded_float(
            linking_config.get("min_score", DEFAULT_MIN_SCORE),
            default=DEFAULT_MIN_SCORE,
            minimum=0.0,
            maximum=1.0,
        )
        keep_empty_candidates = _as_bool(
            linking_config.get("keep_empty_candidates", True),
            default=True,
        )
        fail_fast = _as_bool(
            linking_config.get("fail_fast", False),
            default=False,
        )

        resolved_linker = linker or _resolve_entity_linker(
            runtime=runtime,
            linking_config=linking_config,
            legacy_config=state.get("config"),
        )

        entity_candidates: Dict[str, List[EntityCandidate]] = {}
        warnings = list(mention_warnings)
        failed_mentions: List[str] = []
        linking_methods: Dict[str, str] = {}

        for mention_text in mentions:
            try:
                raw_result = _link_single_mention(
                    linker=resolved_linker,
                    mention_text=mention_text,
                    top_k=top_k,
                )
                raw_mapping = _result_to_mapping(raw_result)
                linking_methods[mention_text] = _normalize_text(
                    raw_mapping.get("linking_method")
                ) or "unknown"

                candidates = _normalize_linking_result(
                    mention_text=mention_text,
                    raw_result=raw_result,
                    min_score=min_score,
                )[:top_k]

                if candidates:
                    entity_candidates[mention_text] = candidates
                    continue

                warning = f"No candidate found for mention: {mention_text}"
                if warning not in warnings:
                    warnings.append(warning)
                if keep_empty_candidates:
                    entity_candidates[mention_text] = []

            except Exception as exc:
                if fail_fast:
                    raise

                failed_mentions.append(mention_text)
                warning = (
                    f"Entity linking failed for mention {mention_text!r}: "
                    f"{type(exc).__name__}: {exc}"
                )
                if warning not in warnings:
                    warnings.append(warning)
                if keep_empty_candidates:
                    entity_candidates[mention_text] = []

        # 所有 Mention 都执行失败时，不伪装为正常无候选。
        if failed_mentions and len(failed_mentions) == len(mentions):
            raise RuntimeError(
                "Entity linking failed for all mentions: "
                + ", ".join(failed_mentions)
            )

        return _build_success_update(
            state=state,
            mentions=mentions,
            entity_candidates=entity_candidates,
            top_k=top_k,
            min_score=min_score,
            keep_empty_candidates=keep_empty_candidates,
            failed_mentions=failed_mentions,
            linking_methods=linking_methods,
            warnings=warnings,
        )

    except Exception as exc:
        _log_failure(runtime, exc)
        return make_error(
            stage="entity_linking",
            message=str(exc),
            detail={
                "mention_texts": [
                    _get_mention_text(item)
                    for item in (state.get("mentions") or [])
                    if _get_mention_text(item)
                ],
                "error_type": type(exc).__name__,
            },
        )


# ---------------------------------------------------------------------------
# Runtime and linker resolution
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


def _resolve_entity_linker(
    *,
    runtime: Optional["RuntimeContext"],
    linking_config: Mapping[str, Any],
    legacy_config: Any,
) -> Any:
    """优先复用 Runtime 中的领域对象，最后才创建迁移期实例。"""

    if runtime is not None:
        direct = _runtime_get(runtime, "entity_linker")
        if _is_linker(direct):
            return direct

        for dependency_name in (
            "entity_resolution_pipeline",
            "entity_resolver",
        ):
            dependency = _runtime_get(runtime, dependency_name)
            nested = getattr(dependency, "entity_linker", None)
            if _is_linker(nested):
                return nested

    cache_key = _normalize_text(linking_config.get("cache_key")) or "default"
    cached = _ENTITY_LINKER_CACHE.get(cache_key)
    if _is_linker(cached):
        return cached

    full_config = _runtime_settings_dict(runtime)
    if isinstance(legacy_config, Mapping):
        full_config = _deep_merge_dicts(
            copy.deepcopy(dict(legacy_config)),
            full_config,
        )

    kwargs = _build_linker_init_kwargs(
        linking_config=dict(linking_config),
        config=full_config,
    )
    created = EntityLinker(**kwargs)

    # 若 Runtime 已管理实体向量库，显式复用，避免新建第二套 Store。
    entity_store = _runtime_get(runtime, "entity_vector_store")
    if entity_store is not None and hasattr(created, "vector_store"):
        created.vector_store = entity_store

    # 无 Runtime 的旧式直接调用才使用模块缓存。
    if runtime is None:
        _ENTITY_LINKER_CACHE[cache_key] = created

    return created


def _is_linker(value: Any) -> bool:
    return any(
        callable(getattr(value, method_name, None))
        for method_name in ("link", "search", "retrieve")
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


def _get_entity_linking_config(
    *,
    state: Mapping[str, Any],
    runtime: Optional["RuntimeContext"],
) -> Dict[str, Any]:
    """读取 Entity Linking 配置；Runtime 配置覆盖迁移期 State 配置。"""

    merged = _get_linking_config(
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
        value = section_method("entity_linking")
        if isinstance(value, Mapping):
            runtime_config = copy.deepcopy(dict(value))
    elif isinstance(settings, Mapping):
        runtime_config = _get_linking_config(dict(settings))
    else:
        get_method = getattr(settings, "get", None)
        if callable(get_method):
            direct = get_method("entity_linking", {})
            nested = get_method("graph.entity_linking", {})
            if isinstance(direct, Mapping):
                runtime_config.update(copy.deepcopy(dict(direct)))
            if isinstance(nested, Mapping):
                runtime_config.update(copy.deepcopy(dict(nested)))

    merged.update(runtime_config)
    return merged


def _get_linking_config(config: Any) -> Dict[str, Any]:
    """兼容 ``entity_linking`` 与 ``graph.entity_linking`` 两种配置位置。"""

    if not isinstance(config, Mapping):
        return {}

    result: Dict[str, Any] = {}
    direct = config.get("entity_linking")
    graph = config.get("graph")

    if isinstance(direct, Mapping):
        result.update(copy.deepcopy(dict(direct)))
    if isinstance(graph, Mapping):
        nested = graph.get("entity_linking")
        if isinstance(nested, Mapping):
            result.update(copy.deepcopy(dict(nested)))

    return result


def _build_linker_init_kwargs(
    *,
    linking_config: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """只提取 EntityLinker 明确支持的初始化参数。"""

    retrieval_config = (
        config.get("retrieval", {}) if isinstance(config, Mapping) else {}
    ) or {}
    kg_config = (
        config.get("kg", {}) if isinstance(config, Mapping) else {}
    ) or {}

    kwargs: Dict[str, Any] = {}

    chroma_dir = (
        linking_config.get("chroma_dir")
        or linking_config.get("entity_chroma_dir")
        or retrieval_config.get("entity_chroma_dir")
    )
    if chroma_dir:
        kwargs["chroma_dir"] = str(chroma_dir)

    collection_name = (
        linking_config.get("collection_name")
        or linking_config.get("entity_collection_name")
        or retrieval_config.get("entity_collection_name")
    )
    if collection_name:
        kwargs["collection_name"] = str(collection_name)

    model_name = (
        linking_config.get("model_name")
        or linking_config.get("embedding_model")
        or retrieval_config.get("embedding_model")
    )
    if model_name:
        kwargs["model_name"] = str(model_name)

    alias_path = linking_config.get("alias_path") or kg_config.get("alias_path")
    if alias_path:
        kwargs["alias_path"] = str(alias_path)

    if "auto_select_threshold" in linking_config:
        kwargs["auto_select_threshold"] = _bounded_float(
            linking_config.get("auto_select_threshold"),
            default=0.72,
            minimum=0.0,
            maximum=1.0,
        )

    if "margin_threshold" in linking_config:
        kwargs["margin_threshold"] = _bounded_float(
            linking_config.get("margin_threshold"),
            default=0.05,
            minimum=0.0,
            maximum=1.0,
        )

    if "local_files_only" in linking_config:
        kwargs["local_files_only"] = _as_bool(
            linking_config.get("local_files_only"),
            default=True,
        )
    elif "local_files_only" in retrieval_config:
        kwargs["local_files_only"] = _as_bool(
            retrieval_config.get("local_files_only"),
            default=True,
        )

    if "lazy_load" in linking_config:
        kwargs["lazy_load"] = _as_bool(
            linking_config.get("lazy_load"),
            default=True,
        )

    return kwargs


# ---------------------------------------------------------------------------
# Linker invocation and result normalization
# ---------------------------------------------------------------------------


def _link_single_mention(
    *,
    linker: Any,
    mention_text: str,
    top_k: int,
) -> Any:
    """兼容 link、search 和 retrieve 三种领域接口。"""

    for method_name in ("link", "search", "retrieve"):
        method = getattr(linker, method_name, None)
        if not callable(method):
            continue
        try:
            return method(mention_text, top_k=top_k)
        except TypeError:
            return method(mention_text)

    raise AttributeError("Entity linker must provide link/search/retrieve method.")


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


def _normalize_linking_result(
    *,
    mention_text: str,
    raw_result: Any,
    min_score: float,
) -> List[EntityCandidate]:
    """将底层 Linker 的不同返回格式统一为候选列表。"""

    if raw_result is None:
        return []

    if is_dataclass(raw_result):
        raw_result = asdict(raw_result)

    if isinstance(raw_result, str):
        entity_name = _normalize_text(raw_result)
        if not entity_name:
            return []
        return [
            EntityCandidate(
                mention=mention_text,
                entity_id="",
                entity_name=entity_name,
                score=1.0,
                source="string",
                aliases=[],
                metadata={
                    "selected": True,
                    "need_confirmation": False,
                    "linking_method": "string",
                },
            )
        ]

    selected_entity = ""
    selected_entity_id = ""
    need_confirmation = False
    linking_method = "unknown"
    raw_message = ""
    score_margin = 0.0
    normalized_mention = ""

    if isinstance(raw_result, Mapping):
        result = dict(raw_result)

        # LinkingResult / pipeline 风格返回。
        candidates_by_mention = result.get("candidates_by_mention")
        if isinstance(candidates_by_mention, Mapping):
            raw_candidates = (
                candidates_by_mention.get(mention_text)
                or candidates_by_mention.get(
                    _normalize_text(result.get("mention") or mention_text)
                )
                or []
            )
            raw_results = result.get("raw_results")
            if isinstance(raw_results, Mapping):
                nested_raw = raw_results.get(mention_text)
                if isinstance(nested_raw, Mapping):
                    result = _deep_merge_dicts(dict(result), dict(nested_raw))
        else:
            raw_candidates = result.get("candidates", []) or []

        selected_entity = _normalize_text(result.get("selected_entity"))
        selected_entity_id = _normalize_text(result.get("selected_entity_id"))
        need_confirmation = _as_bool(
            result.get("need_confirmation", False),
            default=False,
        )
        linking_method = _normalize_text(result.get("linking_method")) or "unknown"
        raw_message = _normalize_text(result.get("message"))
        score_margin = _bounded_float(
            result.get("score_margin", 0.0),
            default=0.0,
            minimum=-1.0,
            maximum=1.0,
        )
        normalized_mention = _normalize_text(result.get("normalized_mention"))

    elif isinstance(raw_result, (list, tuple)):
        raw_candidates = list(raw_result)
    else:
        return []

    if is_dataclass(raw_candidates):
        raw_candidates = [asdict(raw_candidates)]
    if isinstance(raw_candidates, Mapping):
        raw_candidates = [dict(raw_candidates)]
    if not isinstance(raw_candidates, (list, tuple)):
        return []

    candidates: List[EntityCandidate] = []
    for item in raw_candidates:
        candidate = _normalize_candidate_item(
            mention_text=mention_text,
            item=item,
            selected_entity=selected_entity,
            selected_entity_id=selected_entity_id,
            need_confirmation=need_confirmation,
            linking_method=linking_method,
            raw_message=raw_message,
            score_margin=score_margin,
            normalized_mention=normalized_mention,
        )
        if not candidate:
            continue
        if float(candidate.get("score", 0.0)) < min_score:
            continue
        candidates.append(candidate)

    # 某些 Adapter 只返回 selected_entity；将其补成一个标准候选。
    if not candidates and selected_entity:
        selected_score = _extract_score(
            raw_result if isinstance(raw_result, Mapping) else {},
            default=1.0,
        )
        if selected_score >= min_score:
            candidates.append(
                EntityCandidate(
                    mention=mention_text,
                    entity_id=selected_entity_id,
                    entity_name=selected_entity,
                    score=selected_score,
                    source=linking_method,
                    aliases=[],
                    metadata={
                        "selected": True,
                        "need_confirmation": need_confirmation,
                        "linking_method": linking_method,
                        "raw_message": raw_message,
                        "score_margin": score_margin,
                        "normalized_mention": normalized_mention,
                    },
                )
            )

    candidates = _deduplicate_candidates(candidates)
    candidates.sort(
        key=lambda item: (
            bool((item.get("metadata") or {}).get("selected", False)),
            float(item.get("score", 0.0)),
            str(item.get("entity_name", "")),
        ),
        reverse=True,
    )
    return candidates


def _normalize_candidate_item(
    *,
    mention_text: str,
    item: Any,
    selected_entity: str,
    selected_entity_id: str,
    need_confirmation: bool,
    linking_method: str,
    raw_message: str,
    score_margin: float,
    normalized_mention: str,
) -> EntityCandidate:
    if is_dataclass(item):
        item = asdict(item)

    if isinstance(item, str):
        entity_name = _normalize_text(item)
        if not entity_name:
            return EntityCandidate()
        return EntityCandidate(
            mention=mention_text,
            entity_id="",
            entity_name=entity_name,
            score=1.0,
            source="string",
            aliases=[],
            metadata={
                "selected": _is_selected(
                    entity_name=entity_name,
                    entity_id="",
                    selected_entity=selected_entity or entity_name,
                    selected_entity_id=selected_entity_id,
                ),
                "need_confirmation": need_confirmation,
                "linking_method": linking_method,
                "raw_message": raw_message,
                "score_margin": score_margin,
                "normalized_mention": normalized_mention,
            },
        )

    if not isinstance(item, Mapping):
        return EntityCandidate()

    raw = dict(item)
    entity_id = _first_non_empty_str(
        raw,
        keys=["entity_id", "id", "node_id", "qid", "key"],
    )
    entity_name = _first_non_empty_str(
        raw,
        keys=["entity_name", "name", "entity", "title", "label", "document"],
    )
    if not entity_name and entity_id:
        entity_name = entity_id
    if not entity_name:
        return EntityCandidate()

    score = _extract_score(raw)
    source = _normalize_text(raw.get("source")) or linking_method or "unknown"

    aliases = raw.get("aliases", [])
    if isinstance(aliases, str):
        aliases = [aliases]
    if not isinstance(aliases, (list, tuple, set)):
        aliases = []
    normalized_aliases: List[str] = []
    for alias in aliases:
        alias_text = _normalize_text(alias)
        if alias_text and alias_text not in normalized_aliases:
            normalized_aliases.append(alias_text)

    reserved_keys = {
        "entity_id", "id", "node_id", "qid", "key",
        "entity_name", "name", "entity", "title", "label", "document",
        "score", "similarity", "confidence", "distance",
        "source", "aliases", "metadata",
    }
    raw_metadata = raw.get("metadata")
    metadata = (
        copy.deepcopy(dict(raw_metadata))
        if isinstance(raw_metadata, Mapping)
        else {}
    )
    metadata.update(
        {
            key: copy.deepcopy(value)
            for key, value in raw.items()
            if key not in reserved_keys
        }
    )
    metadata.update(
        {
            "selected": _is_selected(
                entity_name=entity_name,
                entity_id=entity_id,
                selected_entity=selected_entity,
                selected_entity_id=selected_entity_id,
            ),
            "need_confirmation": need_confirmation,
            "linking_method": linking_method,
            "raw_message": raw_message,
            "score_margin": score_margin,
            "normalized_mention": normalized_mention,
        }
    )

    return EntityCandidate(
        mention=mention_text,
        entity_id=entity_id,
        entity_name=entity_name,
        score=score,
        source=source,
        aliases=normalized_aliases,
        metadata=metadata,
    )


def _deduplicate_candidates(
    candidates: List[EntityCandidate],
) -> List[EntityCandidate]:
    best: Dict[str, EntityCandidate] = {}

    for candidate in candidates:
        entity_id = _normalize_text(candidate.get("entity_id"))
        entity_name = _normalize_text(candidate.get("entity_name"))
        key = (entity_id or entity_name).casefold()
        if not key:
            continue

        current = best.get(key)
        if current is None:
            best[key] = candidate
            continue

        current_selected = bool((current.get("metadata") or {}).get("selected"))
        candidate_selected = bool((candidate.get("metadata") or {}).get("selected"))
        if candidate_selected and not current_selected:
            best[key] = candidate
        elif candidate_selected == current_selected and float(
            candidate.get("score", 0.0)
        ) > float(current.get("score", 0.0)):
            best[key] = candidate

    return list(best.values())


# ---------------------------------------------------------------------------
# State update construction
# ---------------------------------------------------------------------------


def _build_no_mentions_update(
    *,
    state: AgentState,
    warnings: List[str],
) -> AgentState:
    metadata = _copy_metadata(state.get("metadata"))
    metadata["entity_linking"] = {
        "num_mentions": 0,
        "num_mentions_with_candidates": 0,
        "num_candidates": 0,
        "candidate_summary": {},
    }

    final_warnings = list(warnings)
    message = "No mentions found for entity linking."
    if message not in final_warnings:
        final_warnings.append(message)

    return AgentState(
        entity_candidates={},
        has_error=False,
        error_stage="unknown",
        error_message="",
        error_detail={},
        metadata=metadata,
        traces=[
            {
                "stage": "entity_linking",
                "message": "No mentions found. Skip entity linking.",
                "timestamp": utc_now(),
                "payload": {},
            }
        ],
        warnings=final_warnings,
    )


def _build_success_update(
    *,
    state: AgentState,
    mentions: List[str],
    entity_candidates: Dict[str, List[EntityCandidate]],
    top_k: int,
    min_score: float,
    keep_empty_candidates: bool,
    failed_mentions: List[str],
    linking_methods: Dict[str, str],
    warnings: List[str],
) -> AgentState:
    metadata = _copy_metadata(state.get("metadata"))
    candidate_summary = _build_candidate_summary(entity_candidates)
    num_candidates = sum(len(items) for items in entity_candidates.values())

    metadata["entity_linking"] = {
        "num_mentions": len(mentions),
        "num_mentions_with_candidates": _count_mentions_with_candidates(
            entity_candidates
        ),
        "num_candidates": num_candidates,
        "top_k": top_k,
        "min_score": min_score,
        "keep_empty_candidates": keep_empty_candidates,
        "failed_mentions": list(failed_mentions),
        "linking_methods": copy.deepcopy(linking_methods),
        "candidate_summary": candidate_summary,
    }

    return AgentState(
        entity_candidates=entity_candidates,
        has_error=False,
        error_stage="unknown",
        error_message="",
        error_detail={},
        metadata=metadata,
        traces=[
            {
                "stage": "entity_linking",
                "message": "Entity linking completed.",
                "timestamp": utc_now(),
                "payload": {
                    "num_mentions": len(mentions),
                    "num_mentions_with_candidates": _count_mentions_with_candidates(
                        entity_candidates
                    ),
                    "num_candidates": num_candidates,
                    "failed_mentions": list(failed_mentions),
                    "candidate_summary": candidate_summary,
                },
            }
        ],
        warnings=_deduplicate_strings(warnings),
    )


# ---------------------------------------------------------------------------
# Compatibility helpers and generic utilities
# ---------------------------------------------------------------------------


def _normalize_mentions(
    value: Any,
    *,
    max_mentions: int,
) -> tuple[List[str], List[str]]:
    if value is None:
        return [], []
    if not isinstance(value, (list, tuple)):
        raise TypeError("AgentState['mentions'] must be a list or tuple.")

    mentions: List[str] = []
    warnings: List[str] = []
    seen: set[str] = set()

    for item in value:
        mention_text = _get_mention_text(item)
        if not mention_text:
            warning = "Empty mention skipped in entity linking."
            if warning not in warnings:
                warnings.append(warning)
            continue

        key = mention_text.casefold()
        if key in seen:
            continue
        seen.add(key)
        mentions.append(mention_text)

        if len(mentions) >= max_mentions:
            break

    return mentions, warnings


def _get_mention_text(mention: Any) -> str:
    if isinstance(mention, str):
        return _normalize_text(mention)
    if isinstance(mention, Mapping):
        return _normalize_text(mention.get("text"))
    return ""


def _first_non_empty_str(item: Mapping[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = item.get(key)
        text = _normalize_text(value)
        if text:
            return text
    return ""


def _extract_score(item: Mapping[str, Any], default: float = 0.0) -> float:
    if "score" in item:
        return _clip_score(item.get("score"))
    if "similarity" in item:
        return _clip_score(item.get("similarity"))
    if "confidence" in item:
        return _clip_score(item.get("confidence"))
    if "distance" in item:
        return _clip_score(1.0 - _safe_float(item.get("distance"), default=1.0))
    return _clip_score(default)


def _clip_score(score: Any) -> float:
    return _bounded_float(
        score,
        default=0.0,
        minimum=0.0,
        maximum=1.0,
    )


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    return max(minimum, min(maximum, _safe_int(value, default)))


def _bounded_float(
    value: Any,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(minimum, min(maximum, _safe_float(value, default)))


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "y"}:
        return True
    if normalized in {"0", "false", "no", "off", "n"}:
        return False
    return default


def _is_selected(
    *,
    entity_name: str,
    entity_id: str,
    selected_entity: str,
    selected_entity_id: str,
) -> bool:
    normalized_entity_id = _normalize_text(entity_id)
    normalized_selected_id = _normalize_text(selected_entity_id)
    if normalized_selected_id:
        return bool(normalized_entity_id) and (
            normalized_entity_id == normalized_selected_id
        )

    normalized_name = _normalize_text(entity_name).casefold()
    normalized_selected_name = _normalize_text(selected_entity).casefold()
    return bool(normalized_name and normalized_selected_name) and (
        normalized_name == normalized_selected_name
    )


def _count_mentions_with_candidates(
    entity_candidates: Dict[str, List[EntityCandidate]],
) -> int:
    return sum(1 for candidates in entity_candidates.values() if candidates)


def _build_candidate_summary(
    entity_candidates: Dict[str, List[EntityCandidate]],
) -> Dict[str, List[Dict[str, Any]]]:
    summary: Dict[str, List[Dict[str, Any]]] = {}

    for mention_text, candidates in entity_candidates.items():
        items: List[Dict[str, Any]] = []
        for candidate in candidates[:_MAX_TRACE_CANDIDATES_PER_MENTION]:
            metadata = candidate.get("metadata") or {}
            items.append(
                {
                    "entity_id": candidate.get("entity_id", ""),
                    "entity_name": candidate.get("entity_name", ""),
                    "score": candidate.get("score", 0.0),
                    "source": candidate.get("source", ""),
                    "selected": bool(
                        metadata.get("selected", False)
                        if isinstance(metadata, Mapping)
                        else False
                    ),
                }
            )
        summary[mention_text] = items

    return summary


def _copy_metadata(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return copy.deepcopy(dict(value))


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _deduplicate_strings(values: List[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        text = _normalize_text(value)
        if text and text not in result:
            result.append(text)
    return result


def _deep_merge_dicts(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge_dicts(
                dict(result[key]),
                dict(value),
            )
        else:
            result[key] = copy.deepcopy(value)
    return result


def _log_failure(runtime: Optional["RuntimeContext"], exc: Exception) -> None:
    if runtime is None:
        return
    logger = getattr(runtime, "logger", None)
    log_method = getattr(logger, "exception", None)
    if callable(log_method):
        log_method("Entity linking node failed: %s", exc)


__all__ = [
    "entity_linking_node",
    "create_entity_linking_node",
]
