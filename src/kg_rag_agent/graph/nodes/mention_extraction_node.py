# -*- coding: utf-8 -*-
"""Mention Extraction LangGraph 节点。

本节点只负责：
    1. 从当前问题中抽取待解析 Mention；
    2. 调用 ``entity_resolution.MentionExtractor`` 领域能力；
    3. 将标准化 Mention 写回 AgentState；
    4. 记录轻量 Metadata、Trace 与 Warning。

职责边界：
    * 不访问知识图谱或向量库；
    * 不生成实体候选，不执行实体链接与 Grounding；
    * 不创建共享 LLM Client；
    * RuntimeContext、Client、Store 和 Manager 不得写入 AgentState。

迁移兼容：
    * 系统配置优先从 RuntimeContext.settings 读取；
    * ``state[\"config\"]`` 仅作为旧 Graph 的过渡兼容来源；
    * Node 同时提供 ``mention_extraction_node`` 与
      ``create_mention_extraction_node`` 两种入口。
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from kg_rag_agent.entity_resolution import MentionExtractor

from ..state import AgentState, Mention, make_error, utc_now

if TYPE_CHECKING:
    from kg_rag_agent.runtime import RuntimeContext


DEFAULT_EMPTY_QUERY_QUESTION = "请你补充一个具体问题，我再继续帮你识别其中的对象。"
_MAX_RAW_OUTPUT_CHARS = 2000


def create_mention_extraction_node(
    runtime: Optional["RuntimeContext"] = None,
):
    """创建已绑定 RuntimeContext 的 Mention Extraction Node。"""

    _ensure_runtime_open(runtime)
    extractor = _resolve_mention_extractor(runtime)

    def _node(state: AgentState) -> AgentState:
        return mention_extraction_node(
            state,
            runtime=runtime,
            extractor=extractor,
        )

    _node.__name__ = "mention_extraction_node"
    _node.__qualname__ = "mention_extraction_node"
    _node.__doc__ = mention_extraction_node.__doc__
    return _node


def mention_extraction_node(
    state: AgentState,
    *,
    runtime: Optional["RuntimeContext"] = None,
    extractor: Optional[Any] = None,
) -> AgentState:
    """抽取用户问题中的 Mention，并返回部分 AgentState 更新。"""

    query = _normalize_text(
        state.get("normalized_query") or state.get("query") or ""
    )

    if not query:
        return _build_empty_query_update(state)

    try:
        _ensure_runtime_open(runtime)

        extraction_config = _get_mention_extraction_config(
            state=state,
            runtime=runtime,
        )
        resolved_extractor = extractor or _resolve_mention_extractor(runtime)
        result = _invoke_extractor(
            extractor=resolved_extractor,
            query=query,
            extraction_config=extraction_config,
        )

        mentions = _normalize_mentions(
            result.get("mentions", []),
            query=query,
            max_mentions=_bounded_int(
                extraction_config.get("max_mentions", 16),
                default=16,
                minimum=1,
                maximum=128,
            ),
            min_confidence=_bounded_float(
                extraction_config.get("min_confidence", 0.0),
                default=0.0,
                minimum=0.0,
                maximum=1.0,
            ),
        )

        warnings = _normalize_warnings(result.get("warnings", []))
        if not mentions and not warnings:
            warnings.append(
                "No mention extracted. The graph will continue according to edge routing."
            )

        return _build_success_update(
            state=state,
            mentions=mentions,
            extractor_type=_normalize_text(result.get("extractor_type")) or "rule",
            raw_output=_normalize_text(result.get("raw_output")),
            use_llm_requested=_as_bool(extraction_config.get("use_llm", False)),
            warnings=warnings,
        )

    except Exception as exc:
        _log_failure(runtime, exc)
        return make_error(
            stage="mention_extraction",
            message=str(exc),
            detail={
                "query_length": len(query),
                "error_type": type(exc).__name__,
            },
        )


# ---------------------------------------------------------------------------
# Runtime and domain-service resolution
# ---------------------------------------------------------------------------


def _ensure_runtime_open(runtime: Optional["RuntimeContext"]) -> None:
    if runtime is None:
        return
    ensure_open = getattr(runtime, "ensure_open", None)
    if callable(ensure_open):
        ensure_open()


def _resolve_mention_extractor(
    runtime: Optional["RuntimeContext"],
) -> Any:
    """优先复用 Runtime 中注册的领域对象，否则创建轻量适配实例。"""

    if runtime is not None:
        direct = _runtime_get(runtime, "mention_extractor")
        if direct is not None and callable(getattr(direct, "extract", None)):
            return direct

        pipeline = _runtime_get(runtime, "entity_resolution_pipeline")
        nested = getattr(pipeline, "mention_extractor", None)
        if nested is not None and callable(getattr(nested, "extract", None)):
            return nested

        resolver = _runtime_get(runtime, "entity_resolver")
        nested = getattr(resolver, "mention_extractor", None)
        if nested is not None and callable(getattr(nested, "extract", None)):
            return nested

        llm_client = _runtime_get(runtime, "llm_client")
        return MentionExtractor(llm_client=llm_client)

    return MentionExtractor()


def _runtime_get(runtime: Any, name: str) -> Any:
    get_method = getattr(runtime, "get", None)
    if callable(get_method):
        try:
            return get_method(name, None)
        except TypeError:
            try:
                return get_method(name)
            except Exception:
                pass
        except Exception:
            pass

    value = getattr(runtime, name, None)
    if value is not None:
        return value

    extras = getattr(runtime, "extras", None)
    if isinstance(extras, Mapping):
        return extras.get(name)

    return None


def _invoke_extractor(
    *,
    extractor: Any,
    query: str,
    extraction_config: Mapping[str, Any],
) -> Dict[str, Any]:
    extract_method = getattr(extractor, "extract", None)
    if not callable(extract_method):
        raise TypeError("Mention extractor must expose an extract() method.")

    config_wrapper = {
        "mention_extraction": copy.deepcopy(dict(extraction_config)),
        "graph": {
            "mention_extraction": copy.deepcopy(dict(extraction_config)),
        },
    }

    try:
        raw_result = extract_method(
            query,
            options=extraction_config,
            config=config_wrapper,
        )
    except TypeError:
        try:
            raw_result = extract_method(query, config=config_wrapper)
        except TypeError:
            raw_result = extract_method(query)

    return _result_to_mapping(raw_result)


def _result_to_mapping(result: Any) -> Dict[str, Any]:
    if result is None:
        return {
            "mentions": [],
            "raw_output": "",
            "extractor_type": "unknown",
            "warnings": [],
        }

    if isinstance(result, Mapping):
        return copy.deepcopy(dict(result))

    if is_dataclass(result):
        value = asdict(result)
        return value if isinstance(value, dict) else {"mentions": []}

    if isinstance(result, (list, tuple)):
        return {
            "mentions": list(result),
            "raw_output": "",
            "extractor_type": "custom",
            "warnings": [],
        }

    result_dict = getattr(result, "__dict__", None)
    if isinstance(result_dict, Mapping):
        return copy.deepcopy(dict(result_dict))

    raise TypeError(
        "Mention extractor result must be a mapping, dataclass, list, or tuple."
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _get_mention_extraction_config(
    *,
    state: AgentState,
    runtime: Optional["RuntimeContext"],
) -> Dict[str, Any]:
    """读取 Mention Extraction 配置；Runtime 配置覆盖迁移期 State 配置。"""

    merged = _legacy_mention_extraction_config(state.get("config"))

    if runtime is None:
        return merged

    settings = getattr(runtime, "settings", None)
    if settings is None:
        return merged

    runtime_config: Dict[str, Any] = {}
    section_method = getattr(settings, "section", None)
    if callable(section_method):
        value = section_method("mention_extraction")
        if isinstance(value, Mapping):
            runtime_config = copy.deepcopy(dict(value))
    elif isinstance(settings, Mapping):
        runtime_config = _legacy_mention_extraction_config(settings)
    else:
        get_method = getattr(settings, "get", None)
        if callable(get_method):
            direct = get_method("mention_extraction", {})
            nested = get_method("graph.mention_extraction", {})
            if isinstance(direct, Mapping):
                runtime_config.update(copy.deepcopy(dict(direct)))
            if isinstance(nested, Mapping):
                runtime_config.update(copy.deepcopy(dict(nested)))

    merged.update(runtime_config)
    return merged


def _legacy_mention_extraction_config(config: Any) -> Dict[str, Any]:
    if not isinstance(config, Mapping):
        return {}

    result: Dict[str, Any] = {}
    direct = config.get("mention_extraction")
    graph = config.get("graph")

    if isinstance(direct, Mapping):
        result.update(copy.deepcopy(dict(direct)))
    if isinstance(graph, Mapping):
        nested = graph.get("mention_extraction")
        if isinstance(nested, Mapping):
            result.update(copy.deepcopy(dict(nested)))

    return result


# ---------------------------------------------------------------------------
# State update construction
# ---------------------------------------------------------------------------


def _build_success_update(
    *,
    state: AgentState,
    mentions: List[Mention],
    extractor_type: str,
    raw_output: str,
    use_llm_requested: bool,
    warnings: List[str],
) -> AgentState:
    metadata = _copy_metadata(state.get("metadata"))
    raw_preview, raw_truncated = _truncate_text(raw_output, _MAX_RAW_OUTPUT_CHARS)

    extraction_metadata: Dict[str, Any] = {
        "extractor_type": extractor_type,
        "use_llm_requested": bool(use_llm_requested),
        "num_mentions": len(mentions),
        "mention_texts": [str(item.get("text", "")) for item in mentions],
        "raw_output_length": len(raw_output),
    }
    if raw_preview:
        # 保留旧 metadata 键名，但限制长度，避免把完整模型输出写入 State。
        extraction_metadata["raw_output"] = raw_preview
        extraction_metadata["raw_output_truncated"] = raw_truncated

    metadata["mention_extraction"] = extraction_metadata

    return AgentState(
        mentions=mentions,
        need_clarification=False,
        clarifying_question="",
        has_error=False,
        error_stage="unknown",
        error_message="",
        error_detail={},
        metadata=metadata,
        traces=[
            {
                "stage": "mention_extraction",
                "message": "Mention extraction completed.",
                "timestamp": utc_now(),
                "payload": {
                    "extractor_type": extractor_type,
                    "use_llm_requested": bool(use_llm_requested),
                    "num_mentions": len(mentions),
                    "mention_texts": [
                        str(item.get("text", "")) for item in mentions
                    ],
                },
            }
        ],
        warnings=warnings,
    )


def _build_empty_query_update(state: AgentState) -> AgentState:
    metadata = _copy_metadata(state.get("metadata"))
    metadata["mention_extraction"] = {
        "extractor_type": "empty_input",
        "use_llm_requested": False,
        "num_mentions": 0,
        "mention_texts": [],
        "raw_output_length": 0,
    }

    return AgentState(
        mentions=[],
        need_clarification=True,
        clarifying_question=DEFAULT_EMPTY_QUERY_QUESTION,
        has_error=False,
        error_stage="unknown",
        error_message="",
        error_detail={},
        metadata=metadata,
        traces=[
            {
                "stage": "mention_extraction",
                "message": "Empty query detected; clarification requested.",
                "timestamp": utc_now(),
                "payload": {},
            }
        ],
        warnings=["Empty query detected in mention_extraction_node."],
    )


# ---------------------------------------------------------------------------
# Mention normalization
# ---------------------------------------------------------------------------


def _normalize_mentions(
    value: Any,
    *,
    query: str,
    max_mentions: int,
    min_confidence: float,
) -> List[Mention]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise TypeError("Mention extraction result 'mentions' must be a list.")

    normalized: List[Mention] = []
    seen: set[tuple[str, int, int]] = set()

    for item in value:
        if isinstance(item, str):
            raw: Mapping[str, Any] = {"text": item}
        elif isinstance(item, Mapping):
            raw = item
        else:
            continue

        text = _normalize_text(raw.get("text"))
        if not text:
            continue

        confidence = _bounded_float(
            raw.get("confidence", 1.0),
            default=1.0,
            minimum=0.0,
            maximum=1.0,
        )
        if confidence < min_confidence:
            continue

        start = _safe_int(raw.get("start"), default=-1)
        end = _safe_int(raw.get("end"), default=-1)
        if start < 0 or end <= start or query[start:end] != text:
            start = query.find(text)
            end = start + len(text) if start >= 0 else -1

        if start < 0 or end <= start:
            continue

        key = (text.casefold(), start, end)
        if key in seen:
            continue
        seen.add(key)

        normalized.append(
            Mention(
                text=text,
                start=start,
                end=end,
                type=_normalize_text(raw.get("type")) or "unknown",
                confidence=confidence,
            )
        )

        if len(normalized) >= max_mentions:
            break

    normalized.sort(
        key=lambda item: (
            int(item.get("start", 0)),
            -len(str(item.get("text", ""))),
        )
    )
    return normalized


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _copy_metadata(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return copy.deepcopy(dict(value))


def _normalize_warnings(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return []

    result: List[str] = []
    for item in value:
        text = _normalize_text(item)
        if text and text not in result:
            result.append(text)
    return result


def _truncate_text(value: str, limit: int) -> tuple[str, bool]:
    text = str(value or "")
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value or "").strip().lower()
    return normalized in {"1", "true", "yes", "on", "y"}


def _safe_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bounded_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    return max(minimum, min(maximum, _safe_int(value, default=default)))


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
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _log_failure(runtime: Optional["RuntimeContext"], exc: Exception) -> None:
    if runtime is None:
        return
    logger = getattr(runtime, "logger", None)
    if logger is None:
        return
    log_method = getattr(logger, "exception", None)
    if callable(log_method):
        log_method("Mention extraction node failed: %s", exc)


__all__ = [
    "mention_extraction_node",
    "create_mention_extraction_node",
]
