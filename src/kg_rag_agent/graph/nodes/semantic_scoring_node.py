# -*- coding: utf-8 -*-
"""Semantic Scoring LangGraph 节点。

本节点只负责对 KG/Retrieval 阶段产生的 Evidence 进行相关性评分、筛选和
可回答性判断。核心评分算法位于 ``answering/evidence_selector.py``；Node 仅
完成 State 读取、运行时依赖获取、参数标准化和部分状态更新。

边界：
    * 不重新查询知识图谱或向量库；
    * 不执行实体链接或 Grounding；
    * 不生成最终答案；
    * 不在 AgentState 中保存 LLM、Selector 或 RuntimeContext；
    * 系统配置和共享 LLMClient 由 RuntimeContext 提供。
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from kg_rag_agent.answering import (
    EvidenceSelection,
    EvidenceSelectionOptions,
    EvidenceSelector,
)

from ..state import (
    AgentState,
    AnswerabilityType,
    EvidenceItem,
    SemanticScoringResult,
    make_error,
    utc_now,
)

if TYPE_CHECKING:
    from kg_rag_agent.runtime import RuntimeContext


DEFAULT_OPTIONS = EvidenceSelectionOptions(
    use_llm=False,
    max_selected_evidence=8,
    min_evidence_score=0.15,
    answerable_threshold=0.55,
    uncertain_threshold=0.25,
)


def create_semantic_scoring_node(
    runtime: Optional["RuntimeContext"] = None,
    *,
    selector: Optional[Any] = None,
):
    """创建已绑定 RuntimeContext 的 Semantic Scoring Node。"""

    _ensure_runtime_open(runtime)

    def _node(state: AgentState) -> AgentState:
        return semantic_scoring_node(
            state,
            runtime=runtime,
            selector=selector,
        )

    _node.__name__ = "semantic_scoring_node"
    _node.__qualname__ = "semantic_scoring_node"
    _node.__doc__ = semantic_scoring_node.__doc__
    return _node


def semantic_scoring_node(
    state: AgentState,
    *,
    runtime: Optional["RuntimeContext"] = None,
    selector: Optional[Any] = None,
) -> AgentState:
    """筛选 Evidence 并写入语义分数与可回答性结果。"""

    try:
        _ensure_runtime_open(runtime)

        query = _normalize_text(
            state.get("normalized_query") or state.get("query") or ""
        )
        evidence, normalization_warnings = _normalize_evidence(
            state.get("evidence") or state.get("raw_evidence") or []
        )
        scoring_config = _get_scoring_config(state=state, runtime=runtime)
        options = _build_options(scoring_config)

        resolved_selector, selector_warnings = _resolve_selector(
            runtime=runtime,
            selector=selector,
            use_llm=options.use_llm,
        )
        selection = _invoke_selector(
            selector=resolved_selector,
            query=query,
            evidence=evidence,
            options=options,
        )
        normalized = _normalize_selection(
            selection=selection,
            input_evidence=evidence,
            options=options,
        )

        warnings = _deduplicate_texts(
            normalization_warnings
            + selector_warnings
            + normalized["warnings"]
        )
        if not evidence:
            warnings.append("No evidence available for semantic scoring.")

        result: SemanticScoringResult = normalized["result"]
        selected_evidence: List[EvidenceItem] = normalized["evidence"]
        evidence_text = normalized["evidence_text"]
        scoring_type = normalized["scoring_type"]
        answerability = _normalize_answerability(
            result.get("answerability", "uncertain")
        )
        semantic_score = _bounded_float(
            result.get("score", 0.0),
            default=0.0,
            minimum=0.0,
            maximum=1.0,
        )
        reason = _normalize_text(result.get("reason", ""))
        if not reason:
            reason = _default_reason(
                answerability=answerability,
                selected_count=len(selected_evidence),
            )

        result = SemanticScoringResult(
            score=semantic_score,
            answerability=answerability,
            reason=reason,
            selected_evidence_ids=_normalize_id_list(
                result.get("selected_evidence_ids")
            ),
            rejected_evidence_ids=_normalize_id_list(
                result.get("rejected_evidence_ids")
            ),
        )

        metadata = _update_metadata(
            state,
            {
                "num_input_evidence": len(evidence),
                "num_selected_evidence": len(selected_evidence),
                "scoring_type": scoring_type,
                "use_llm_requested": options.use_llm,
                "max_selected_evidence": options.max_selected_evidence,
                "min_evidence_score": options.min_evidence_score,
                "answerable_threshold": options.answerable_threshold,
                "uncertain_threshold": options.uncertain_threshold,
            },
        )

        return AgentState(
            semantic_scoring=result,
            semantic_score=semantic_score,
            answerability=answerability,
            scoring_reason=reason,
            evidence=copy.deepcopy(selected_evidence),
            evidence_text=evidence_text,
            has_error=False,
            error_stage="unknown",
            error_message="",
            error_detail={},
            metadata=metadata,
            traces=[
                {
                    "stage": "semantic_scoring",
                    "message": "Semantic scoring completed.",
                    "timestamp": utc_now(),
                    "payload": {
                        "semantic_score": semantic_score,
                        "answerability": answerability,
                        "scoring_type": scoring_type,
                        "num_input_evidence": len(evidence),
                        "num_selected_evidence": len(selected_evidence),
                        "selected_evidence_ids": result.get(
                            "selected_evidence_ids", []
                        ),
                        "rejected_evidence_ids": result.get(
                            "rejected_evidence_ids", []
                        ),
                    },
                }
            ],
            warnings=warnings,
        )

    except Exception as exc:
        _log_failure(runtime, exc)
        return make_error(
            stage="semantic_scoring",
            message=str(exc),
            detail={
                "query_length": len(
                    str(
                        state.get("normalized_query")
                        or state.get("query")
                        or ""
                    )
                ),
                "num_evidence": _safe_sequence_length(
                    state.get("evidence") or state.get("raw_evidence")
                ),
                "error_type": type(exc).__name__,
            },
        )


# ---------------------------------------------------------------------------
# Runtime and selector resolution
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


def _resolve_selector(
    *,
    runtime: Optional["RuntimeContext"],
    selector: Optional[Any],
    use_llm: bool,
) -> tuple[Any, List[str]]:
    warnings: List[str] = []

    if selector is not None:
        _validate_selector(selector)
        return selector, warnings

    runtime_selector = _runtime_get(runtime, "evidence_selector")
    if runtime_selector is not None:
        _validate_selector(runtime_selector)
        return runtime_selector, warnings

    llm_client = _runtime_get(runtime, "llm_client") if use_llm else None
    if use_llm and llm_client is None:
        warnings.append(
            "LLM semantic scoring was requested but no shared LLM client was "
            "available; rule scoring was used."
        )

    return EvidenceSelector(llm_client=llm_client), warnings


def _validate_selector(selector: Any) -> None:
    if not callable(getattr(selector, "select", None)):
        raise TypeError("evidence selector must provide a callable select method")


def _invoke_selector(
    *,
    selector: Any,
    query: str,
    evidence: List[EvidenceItem],
    options: EvidenceSelectionOptions,
) -> Any:
    select = getattr(selector, "select")

    try:
        return select(query=query, evidence=evidence, options=options)
    except TypeError as first_error:
        # 迁移期兼容较早的 selector 接口。只有在关键字签名不兼容时才回退，
        # 领域逻辑内部抛出的其他异常仍由外层标准错误处理捕获。
        try:
            return select(query, evidence, options)
        except TypeError:
            raise first_error


# ---------------------------------------------------------------------------
# Configuration and options
# ---------------------------------------------------------------------------


def _get_scoring_config(
    *,
    state: AgentState,
    runtime: Optional["RuntimeContext"],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}

    legacy = state.get("config")
    if isinstance(legacy, Mapping):
        direct = legacy.get("semantic_scoring")
        if isinstance(direct, Mapping):
            merged.update(copy.deepcopy(dict(direct)))

        graph = legacy.get("graph")
        if isinstance(graph, Mapping):
            nested = graph.get("semantic_scoring")
            if isinstance(nested, Mapping):
                merged.update(copy.deepcopy(dict(nested)))

    settings = getattr(runtime, "settings", None) if runtime is not None else None
    section = getattr(settings, "section", None)
    if callable(section):
        value = section("semantic_scoring")
        if isinstance(value, Mapping):
            merged.update(copy.deepcopy(dict(value)))
    elif settings is not None:
        get = getattr(settings, "get", None)
        if callable(get):
            direct = get("semantic_scoring", {})
            nested = get("graph.semantic_scoring", {})
            if isinstance(direct, Mapping):
                merged.update(copy.deepcopy(dict(direct)))
            if isinstance(nested, Mapping):
                merged.update(copy.deepcopy(dict(nested)))

    return merged


def _build_options(config: Mapping[str, Any]) -> EvidenceSelectionOptions:
    return EvidenceSelectionOptions(
        use_llm=_coerce_bool(
            config.get("use_llm", DEFAULT_OPTIONS.use_llm),
            default=DEFAULT_OPTIONS.use_llm,
        ),
        max_selected_evidence=_bounded_int(
            config.get(
                "max_selected_evidence",
                DEFAULT_OPTIONS.max_selected_evidence,
            ),
            default=DEFAULT_OPTIONS.max_selected_evidence,
            minimum=1,
            maximum=100,
        ),
        min_evidence_score=_bounded_float(
            config.get(
                "min_evidence_score",
                DEFAULT_OPTIONS.min_evidence_score,
            ),
            default=DEFAULT_OPTIONS.min_evidence_score,
            minimum=0.0,
            maximum=1.0,
        ),
        answerable_threshold=_bounded_float(
            config.get(
                "answerable_threshold",
                DEFAULT_OPTIONS.answerable_threshold,
            ),
            default=DEFAULT_OPTIONS.answerable_threshold,
            minimum=0.0,
            maximum=1.0,
        ),
        uncertain_threshold=_bounded_float(
            config.get(
                "uncertain_threshold",
                DEFAULT_OPTIONS.uncertain_threshold,
            ),
            default=DEFAULT_OPTIONS.uncertain_threshold,
            minimum=0.0,
            maximum=1.0,
        ),
    ).normalized()


# ---------------------------------------------------------------------------
# Input/output normalization
# ---------------------------------------------------------------------------


def _normalize_evidence(value: Any) -> tuple[List[EvidenceItem], List[str]]:
    warnings: List[str] = []
    if value is None:
        return [], warnings
    if not _is_non_string_sequence(value):
        raise TypeError("evidence must be a sequence of mappings")

    result: List[EvidenceItem] = []
    seen_ids: set[str] = set()

    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            warnings.append(f"Ignored invalid evidence item at index {index}.")
            continue

        item = copy.deepcopy(dict(raw))
        evidence_id = _normalize_text(item.get("evidence_id"))
        if not evidence_id:
            evidence_id = f"ev_{index + 1}"
            item["evidence_id"] = evidence_id

        if evidence_id in seen_ids:
            base = evidence_id
            suffix = 2
            while f"{base}_{suffix}" in seen_ids:
                suffix += 1
            evidence_id = f"{base}_{suffix}"
            item["evidence_id"] = evidence_id
            warnings.append(
                "Duplicate evidence_id was normalized: "
                f"{base} -> {evidence_id}."
            )

        seen_ids.add(evidence_id)
        item["text"] = _normalize_text(item.get("text"))
        item["score"] = _bounded_float(
            item.get("score", 0.0),
            default=0.0,
            minimum=0.0,
            maximum=1.0,
        )
        metadata = item.get("metadata")
        item["metadata"] = (
            copy.deepcopy(dict(metadata))
            if isinstance(metadata, Mapping)
            else {}
        )
        result.append(EvidenceItem(**item))

    return result, warnings


def _normalize_selection(
    *,
    selection: Any,
    input_evidence: List[EvidenceItem],
    options: EvidenceSelectionOptions,
) -> Dict[str, Any]:
    warnings: List[str] = []

    if isinstance(selection, EvidenceSelection):
        raw_result = selection.result
        raw_evidence = selection.evidence
        evidence_text = selection.evidence_text
        scoring_type = selection.scoring_type
    elif isinstance(selection, Mapping):
        raw_result = selection.get("result", selection.get("semantic_scoring", {}))
        raw_evidence = selection.get("evidence", [])
        evidence_text = selection.get("evidence_text", "")
        scoring_type = selection.get("scoring_type", "custom")
    else:
        raw_result = getattr(selection, "result", {})
        raw_evidence = getattr(selection, "evidence", [])
        evidence_text = getattr(selection, "evidence_text", "")
        scoring_type = getattr(selection, "scoring_type", "custom")

    result_mapping = dict(raw_result) if isinstance(raw_result, Mapping) else {}
    selected, selected_warnings = _normalize_evidence(raw_evidence)
    warnings.extend(selected_warnings)

    valid_by_id = {
        str(item.get("evidence_id", "")): item
        for item in input_evidence
        if item.get("evidence_id")
    }
    normalized_selected: List[EvidenceItem] = []
    seen: set[str] = set()
    for item in selected:
        evidence_id = _normalize_text(item.get("evidence_id"))
        if not evidence_id or evidence_id in seen:
            continue
        if evidence_id not in valid_by_id:
            warnings.append(
                f"Ignored selector output with unknown evidence_id: {evidence_id}."
            )
            continue
        normalized_selected.append(item)
        seen.add(evidence_id)
        if len(normalized_selected) >= options.max_selected_evidence:
            break

    # 自定义 Selector 若只返回 selected_evidence_ids，则由输入 Evidence 补全。
    selected_ids = _normalize_id_list(
        result_mapping.get("selected_evidence_ids")
    )
    if not normalized_selected and selected_ids:
        for evidence_id in selected_ids:
            if evidence_id in valid_by_id and evidence_id not in seen:
                normalized_selected.append(copy.deepcopy(valid_by_id[evidence_id]))
                seen.add(evidence_id)
                if len(normalized_selected) >= options.max_selected_evidence:
                    break

    if input_evidence and not normalized_selected:
        # EvidenceSelector 正常情况下会负责规则回退。此处只保护自定义实现，避免
        # 非法空选择使后续 Node 丢失全部可用材料。
        fallback_selector = EvidenceSelector(llm_client=None)
        fallback = fallback_selector.select(
            query="",
            evidence=input_evidence,
            options=EvidenceSelectionOptions(
                use_llm=False,
                max_selected_evidence=options.max_selected_evidence,
                min_evidence_score=options.min_evidence_score,
                answerable_threshold=options.answerable_threshold,
                uncertain_threshold=options.uncertain_threshold,
            ),
        )
        normalized_selected = copy.deepcopy(fallback.evidence)
        result_mapping = dict(fallback.result)
        evidence_text = fallback.evidence_text
        scoring_type = "rule_fallback"
        warnings.append(
            "Selector returned no valid evidence; rule scoring fallback was used."
        )

    selected_ids = [
        str(item.get("evidence_id", ""))
        for item in normalized_selected
        if item.get("evidence_id")
    ]
    rejected_ids = _normalize_id_list(
        result_mapping.get("rejected_evidence_ids")
    )
    if not rejected_ids:
        selected_set = set(selected_ids)
        rejected_ids = [
            str(item.get("evidence_id", ""))
            for item in input_evidence
            if item.get("evidence_id")
            and str(item.get("evidence_id", "")) not in selected_set
        ]

    answerability = _normalize_answerability(
        result_mapping.get("answerability", "uncertain")
    )
    score = _bounded_float(
        result_mapping.get("score", 0.0),
        default=0.0,
        minimum=0.0,
        maximum=1.0,
    )
    reason = _normalize_text(result_mapping.get("reason"))

    if not input_evidence:
        answerability = "unanswerable"
        score = 0.0
        reason = reason or "当前可用信息不足，无法可靠回答。"
        normalized_selected = []
        selected_ids = []
        rejected_ids = []
        evidence_text = ""
        scoring_type = "none"
    elif not reason:
        reason = _default_reason(
            answerability=answerability,
            selected_count=len(normalized_selected),
        )

    normalized_text = _normalize_text(evidence_text)
    if normalized_selected and not normalized_text:
        normalized_text = _build_evidence_text(normalized_selected)

    result = SemanticScoringResult(
        score=score,
        answerability=answerability,
        reason=reason,
        selected_evidence_ids=selected_ids,
        rejected_evidence_ids=rejected_ids,
    )

    return {
        "result": result,
        "evidence": normalized_selected,
        "evidence_text": normalized_text,
        "scoring_type": _normalize_text(scoring_type) or "custom",
        "warnings": warnings,
    }


def _build_evidence_text(evidence: List[EvidenceItem]) -> str:
    lines: List[str] = []
    for index, item in enumerate(evidence, start=1):
        evidence_id = _normalize_text(item.get("evidence_id")) or f"ev_{index}"
        text = _normalize_text(item.get("text"))
        if not text:
            source = _normalize_text(item.get("source_entity"))
            relation = _normalize_text(item.get("relation"))
            target = _normalize_text(item.get("target_entity"))
            text = " ".join(part for part in (source, relation, target) if part)
        if text:
            lines.append(f"[{evidence_id}] {text}")
    return "\n".join(lines)


def _normalize_answerability(value: Any) -> AnswerabilityType:
    normalized = _normalize_text(value).lower()
    aliases = {
        "answerable": "answerable",
        "yes": "answerable",
        "sufficient": "answerable",
        "uncertain": "uncertain",
        "partial": "uncertain",
        "partially_answerable": "uncertain",
        "unanswerable": "unanswerable",
        "no": "unanswerable",
        "insufficient": "unanswerable",
    }
    return aliases.get(normalized, "uncertain")  # type: ignore[return-value]


def _normalize_id_list(value: Any) -> List[str]:
    if not _is_non_string_sequence(value):
        return []
    result: List[str] = []
    for item in value:
        normalized = _normalize_text(item)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


# ---------------------------------------------------------------------------
# State metadata, diagnostics and generic validation
# ---------------------------------------------------------------------------


def _update_metadata(
    state: AgentState,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_metadata = state.get("metadata")
    metadata = (
        copy.deepcopy(dict(raw_metadata))
        if isinstance(raw_metadata, Mapping)
        else {}
    )
    existing = metadata.get("semantic_scoring")
    section = (
        copy.deepcopy(dict(existing))
        if isinstance(existing, Mapping)
        else {}
    )
    section.update(copy.deepcopy(dict(payload)))
    metadata["semantic_scoring"] = section
    return metadata


def _default_reason(
    *,
    answerability: AnswerabilityType,
    selected_count: int,
) -> str:
    if answerability == "answerable":
        return f"已筛选出 {selected_count} 条与问题较匹配的信息，可以支撑回答。"
    if answerability == "unanswerable":
        return "当前可用信息与问题的匹配度不足，无法可靠回答。"
    return f"已找到 {selected_count} 条相关信息，但支撑程度仍有限。"


def _log_failure(runtime: Optional["RuntimeContext"], exc: Exception) -> None:
    logger = getattr(runtime, "logger", None) if runtime is not None else None
    if logger is None:
        return
    exception = getattr(logger, "exception", None)
    if callable(exception):
        exception("Semantic scoring node failed: %s", exc)


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "disabled"}:
        return False
    return default


def _bounded_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError):
        normalized = int(default)
    return max(minimum, min(normalized, maximum))


def _bounded_float(
    value: Any,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError):
        normalized = float(default)
    if normalized != normalized or normalized in {float("inf"), float("-inf")}:
        normalized = float(default)
    return max(minimum, min(normalized, maximum))


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _deduplicate_texts(values: Sequence[Any]) -> List[str]:
    result: List[str] = []
    for value in values:
        text = _normalize_text(value)
        if text and text not in result:
            result.append(text)
    return result


def _is_non_string_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _safe_sequence_length(value: Any) -> int:
    return len(value) if _is_non_string_sequence(value) else 0


__all__ = [
    "create_semantic_scoring_node",
    "semantic_scoring_node",
]
