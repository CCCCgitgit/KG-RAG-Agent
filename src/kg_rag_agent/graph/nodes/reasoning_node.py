# -*- coding: utf-8 -*-
"""Reasoning LangGraph 节点。

本节点只负责读取已经筛选完成的 Evidence，并调用
``answering.reasoner.AnswerReasoner`` 形成结构化中间判断。最终答案仍由
``generation_node.py`` 生成。

边界：
    * 不重新执行实体解析、Grounding、KG 查询或语义评分；
    * 不直接生成 ``final_answer`` 或 Citation；
    * 不在 AgentState 中保存 LLM、Reasoner 或 RuntimeContext；
    * 系统配置和共享 LLMClient 由 RuntimeContext 提供；
    * Node 只返回自己负责的部分状态更新。
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from kg_rag_agent.answering import (
    AnswerReasoner,
    ReasoningOptions,
    ReasoningOutput,
)
from kg_rag_agent.answering.reasoner import (
    _build_reasoning_text,
    _build_rule_conclusion,
    _clip_score,
    _estimate_reasoning_confidence,
    _evidence_to_reasoning_step,
    _evidence_type_priority,
    _fallback_conclusion_by_answerability,
    _normalize_answerability,
    _normalize_reasoning_result,
    _preview_text,
    _reason_with_rules,
    _safe_float,
    _safe_int,
    _sanitize_text,
    _select_evidence_for_reasoning,
)

from ..state import (
    AgentState,
    AnswerabilityType,
    EvidenceItem,
    ReasoningResult,
    make_error,
    utc_now,
)

if TYPE_CHECKING:
    from kg_rag_agent.runtime import RuntimeContext


DEFAULT_OPTIONS = ReasoningOptions(
    use_llm=False,
    max_reasoning_steps=5,
)


def create_reasoning_node(
    runtime: Optional["RuntimeContext"] = None,
    *,
    reasoner: Optional[Any] = None,
):
    """创建已绑定 RuntimeContext 的 Reasoning Node。"""

    _ensure_runtime_open(runtime)

    def _node(state: AgentState) -> AgentState:
        return reasoning_node(
            state,
            runtime=runtime,
            reasoner=reasoner,
        )

    _node.__name__ = "reasoning_node"
    _node.__qualname__ = "reasoning_node"
    _node.__doc__ = reasoning_node.__doc__
    return _node


def reasoning_node(
    state: AgentState,
    *,
    runtime: Optional["RuntimeContext"] = None,
    reasoner: Optional[Any] = None,
) -> AgentState:
    """基于已筛选 Evidence 生成结构化中间判断。"""

    try:
        _ensure_runtime_open(runtime)

        query = _normalize_text(
            state.get("normalized_query") or state.get("query") or ""
        )
        evidence, normalization_warnings = _normalize_evidence(
            state.get("evidence") or []
        )
        evidence_text = _normalize_text(state.get("evidence_text") or "")
        answerability = _normalize_answerability(
            state.get("answerability", "uncertain")
        )
        semantic_score = _clip_score(
            _safe_float(state.get("semantic_score", 0.0), default=0.0)
        )

        reasoning_config = _get_reasoning_config(
            state=state,
            runtime=runtime,
        )
        options = _build_options(reasoning_config)

        resolved_reasoner, resolution_warnings = _resolve_reasoner(
            runtime=runtime,
            reasoner=reasoner,
            use_llm=options.use_llm,
        )

        if not evidence:
            reasoning_type = "empty"
            normalized_result = ReasoningResult(
                reasoning_chain=["当前可用信息不足，无法形成可靠判断。"],
                conclusion="当前信息不足，不能可靠回答该问题。",
                used_evidence_ids=[],
                confidence=0.0,
                metadata={
                    "reasoning_type": reasoning_type,
                    "answerability": "unanswerable",
                    "semantic_score": semantic_score,
                    "num_evidence": 0,
                },
            )
            reasoning_text = _build_reasoning_text(normalized_result)
            output_warnings = ["No evidence available for reasoning."]
        else:
            raw_output = _invoke_reasoner(
                reasoner=resolved_reasoner,
                query=query,
                evidence=evidence,
                evidence_text=evidence_text,
                answerability=answerability,
                semantic_score=semantic_score,
                options=options,
            )
            result, reasoning_text, reasoning_type, output_warnings = (
                _normalize_reasoning_output(
                    output=raw_output,
                    evidence=evidence,
                    answerability=answerability,
                    semantic_score=semantic_score,
                    max_reasoning_steps=options.max_reasoning_steps,
                )
            )
            normalized_result = result

        warnings = _deduplicate_texts(
            normalization_warnings
            + resolution_warnings
            + output_warnings
        )

        used_evidence_ids = list(
            normalized_result.get("used_evidence_ids", []) or []
        )
        confidence = _clip_score(
            _safe_float(
                normalized_result.get("confidence", 0.0),
                default=0.0,
            )
        )

        metadata = _update_metadata(
            state,
            {
                "reasoning_type": reasoning_type,
                "num_input_evidence": len(evidence),
                "num_used_evidence": len(used_evidence_ids),
                "used_evidence_ids": used_evidence_ids,
                "confidence": confidence,
                "answerability": (
                    "unanswerable" if not evidence else answerability
                ),
                "semantic_score": semantic_score,
                "use_llm_requested": options.use_llm,
                "max_reasoning_steps": options.max_reasoning_steps,
            },
        )

        return AgentState(
            reasoning=copy.deepcopy(normalized_result),
            reasoning_text=reasoning_text,
            has_error=False,
            error_stage="unknown",
            error_message="",
            error_detail={},
            metadata=metadata,
            traces=[
                {
                    "stage": "reasoning",
                    "message": "Reasoning completed.",
                    "timestamp": utc_now(),
                    "payload": {
                        "reasoning_type": reasoning_type,
                        "num_input_evidence": len(evidence),
                        "num_used_evidence": len(used_evidence_ids),
                        "used_evidence_ids": used_evidence_ids,
                        "confidence": confidence,
                        "answerability": (
                            "unanswerable" if not evidence else answerability
                        ),
                        "conclusion": _preview_text(
                            str(normalized_result.get("conclusion", "") or ""),
                            max_len=240,
                        ),
                    },
                }
            ],
            warnings=warnings,
        )

    except Exception as exc:
        _log_failure(runtime, exc)
        return make_error(
            stage="reasoning",
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
                    state.get("evidence")
                ),
                "error_type": type(exc).__name__,
            },
        )


# ---------------------------------------------------------------------------
# Runtime and reasoner resolution
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


def _resolve_reasoner(
    *,
    runtime: Optional["RuntimeContext"],
    reasoner: Optional[Any],
    use_llm: bool,
) -> tuple[Any, List[str]]:
    warnings: List[str] = []

    if reasoner is not None:
        _validate_reasoner(reasoner)
        return reasoner, warnings

    for dependency_name in (
        "answer_reasoner",
        "answering_reasoner",
        "reasoner",
    ):
        runtime_reasoner = _runtime_get(runtime, dependency_name)
        if runtime_reasoner is not None:
            _validate_reasoner(runtime_reasoner)
            return runtime_reasoner, warnings

    llm_client = _runtime_get(runtime, "llm_client") if use_llm else None
    if use_llm and llm_client is None:
        warnings.append(
            "LLM reasoning was requested but no shared LLM client was "
            "available; rule reasoning was used."
        )

    return AnswerReasoner(llm_client=llm_client), warnings


def _validate_reasoner(reasoner: Any) -> None:
    if not callable(getattr(reasoner, "reason", None)):
        raise TypeError("reasoner must provide a callable reason method")


def _invoke_reasoner(
    *,
    reasoner: Any,
    query: str,
    evidence: List[EvidenceItem],
    evidence_text: str,
    answerability: AnswerabilityType,
    semantic_score: float,
    options: ReasoningOptions,
) -> Any:
    reason = getattr(reasoner, "reason")

    try:
        return reason(
            query=query,
            evidence=evidence,
            evidence_text=evidence_text,
            answerability=answerability,
            semantic_score=semantic_score,
            options=options,
        )
    except TypeError as first_error:
        # 迁移期兼容较早的 Reasoner 接口。只有关键字签名不兼容时回退。
        try:
            return reason(
                query,
                evidence,
                evidence_text,
                answerability,
                semantic_score,
                options,
            )
        except TypeError:
            raise first_error


# ---------------------------------------------------------------------------
# Configuration and options
# ---------------------------------------------------------------------------


def _get_reasoning_config(
    config: Optional[Mapping[str, Any]] = None,
    *,
    state: Optional[AgentState] = None,
    runtime: Optional["RuntimeContext"] = None,
) -> Dict[str, Any]:
    """读取 reasoning 配置。

    兼容旧调用 ``_get_reasoning_config(config)``。正式 Node 调用传入
    ``state`` 和 ``runtime``；Runtime 配置优先于迁移期 State.config。
    """

    merged: Dict[str, Any] = {}

    legacy_sources: List[Mapping[str, Any]] = []
    if isinstance(config, Mapping):
        legacy_sources.append(config)
    if isinstance(state, Mapping):
        legacy = state.get("config")
        if isinstance(legacy, Mapping):
            legacy_sources.append(legacy)

    for source in legacy_sources:
        direct = source.get("reasoning")
        if isinstance(direct, Mapping):
            merged.update(copy.deepcopy(dict(direct)))

        graph = source.get("graph")
        if isinstance(graph, Mapping):
            nested = graph.get("reasoning")
            if isinstance(nested, Mapping):
                merged.update(copy.deepcopy(dict(nested)))

    settings = getattr(runtime, "settings", None) if runtime is not None else None
    section = getattr(settings, "section", None)
    if callable(section):
        value = section("reasoning")
        if isinstance(value, Mapping):
            merged.update(copy.deepcopy(dict(value)))
    elif settings is not None:
        get = getattr(settings, "get", None)
        if callable(get):
            direct = get("reasoning", {})
            nested = get("graph.reasoning", {})
            if isinstance(direct, Mapping):
                merged.update(copy.deepcopy(dict(direct)))
            if isinstance(nested, Mapping):
                merged.update(copy.deepcopy(dict(nested)))

    return merged


def _build_options(config: Mapping[str, Any]) -> ReasoningOptions:
    return ReasoningOptions(
        use_llm=_coerce_bool(
            config.get("use_llm", DEFAULT_OPTIONS.use_llm),
            default=DEFAULT_OPTIONS.use_llm,
        ),
        max_reasoning_steps=_bounded_int(
            config.get(
                "max_reasoning_steps",
                DEFAULT_OPTIONS.max_reasoning_steps,
            ),
            default=DEFAULT_OPTIONS.max_reasoning_steps,
            minimum=1,
            maximum=20,
        ),
    ).normalized()


# ---------------------------------------------------------------------------
# Input/output normalization
# ---------------------------------------------------------------------------


def _normalize_evidence(value: Any) -> tuple[List[EvidenceItem], List[str]]:
    warnings: List[str] = []
    if value is None:
        return [], warnings
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise TypeError("evidence must be a sequence of mappings")

    result: List[EvidenceItem] = []
    seen_ids: set[str] = set()

    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            warnings.append(
                f"Ignored non-mapping evidence at index {index}."
            )
            continue

        normalized = copy.deepcopy(dict(item))
        evidence_id = _normalize_text(normalized.get("evidence_id"))
        if not evidence_id:
            evidence_id = f"ev_reasoning_{index + 1}"
            warnings.append(
                f"Generated missing evidence_id for evidence at index {index}."
            )

        base_id = evidence_id
        suffix = 2
        while evidence_id in seen_ids:
            evidence_id = f"{base_id}_{suffix}"
            suffix += 1
        if evidence_id != base_id:
            warnings.append(
                f"Renamed duplicate evidence_id '{base_id}' to '{evidence_id}'."
            )
        seen_ids.add(evidence_id)

        normalized["evidence_id"] = evidence_id
        normalized["evidence_type"] = _normalize_text(
            normalized.get("evidence_type") or normalized.get("type") or "text"
        ).lower() or "text"
        normalized["source_entity"] = _normalize_text(
            normalized.get("source_entity") or normalized.get("source")
        )
        normalized["target_entity"] = _normalize_text(
            normalized.get("target_entity") or normalized.get("target")
        )
        normalized["relation"] = _normalize_text(normalized.get("relation"))
        normalized["text"] = _normalize_text(normalized.get("text"))
        normalized["score"] = _clip_score(
            _safe_float(normalized.get("score", 0.0), default=0.0)
        )

        path = normalized.get("path")
        if isinstance(path, Sequence) and not isinstance(
            path, (str, bytes, bytearray)
        ):
            normalized["path"] = [
                _normalize_text(part)
                for part in path
                if _normalize_text(part)
            ]
        elif path is not None:
            normalized["path"] = []

        triples = normalized.get("triples")
        if isinstance(triples, Sequence) and not isinstance(
            triples, (str, bytes, bytearray)
        ):
            normalized["triples"] = [
                copy.deepcopy(dict(triple))
                for triple in triples
                if isinstance(triple, Mapping)
            ]
        elif triples is not None:
            normalized["triples"] = []

        metadata = normalized.get("metadata")
        normalized["metadata"] = (
            copy.deepcopy(dict(metadata))
            if isinstance(metadata, Mapping)
            else {}
        )

        result.append(EvidenceItem(**normalized))

    return result, warnings


def _normalize_reasoning_output(
    *,
    output: Any,
    evidence: List[EvidenceItem],
    answerability: AnswerabilityType,
    semantic_score: float,
    max_reasoning_steps: int,
) -> tuple[ReasoningResult, str, str, List[str]]:
    warnings: List[str] = []
    reasoning_type = "rule"
    reasoning_text = ""
    raw_result: Any = output

    if isinstance(output, ReasoningOutput):
        raw_result = output.result
        reasoning_text = _normalize_text(output.reasoning_text)
        reasoning_type = _normalize_text(output.reasoning_type) or "rule"
    elif isinstance(output, Mapping):
        if isinstance(output.get("result"), Mapping):
            raw_result = output.get("result")
            reasoning_text = _normalize_text(output.get("reasoning_text"))
            reasoning_type = (
                _normalize_text(output.get("reasoning_type")) or "rule"
            )
    elif hasattr(output, "result"):
        raw_result = getattr(output, "result")
        reasoning_text = _normalize_text(
            getattr(output, "reasoning_text", "")
        )
        reasoning_type = (
            _normalize_text(getattr(output, "reasoning_type", "")) or "rule"
        )
    elif isinstance(output, tuple):
        if output:
            raw_result = output[0]
        if len(output) > 1:
            reasoning_type = _normalize_text(output[1]) or "rule"

    if not isinstance(raw_result, Mapping):
        raise TypeError("reasoner output must contain a mapping result")

    result = _normalize_reasoning_result(
        reasoning_result=ReasoningResult(**copy.deepcopy(dict(raw_result))),
        evidence=evidence,
        answerability=answerability,
        semantic_score=semantic_score,
        max_reasoning_steps=max_reasoning_steps,
        fallback_type=reasoning_type,
    )

    result_metadata = result.get("metadata")
    metadata = (
        copy.deepcopy(dict(result_metadata))
        if isinstance(result_metadata, Mapping)
        else {}
    )
    metadata["reasoning_type"] = reasoning_type
    metadata.setdefault("answerability", answerability)
    metadata.setdefault("semantic_score", semantic_score)
    metadata.setdefault("num_evidence", len(evidence))
    result["metadata"] = metadata

    if not reasoning_text:
        reasoning_text = _build_reasoning_text(result)

    if reasoning_type not in {"rule", "llm", "empty", "none", "custom"}:
        warnings.append(
            f"Unknown reasoning_type '{reasoning_type}' was preserved."
        )

    return result, reasoning_text, reasoning_type, warnings


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
    existing = metadata.get("reasoning")
    section = (
        copy.deepcopy(dict(existing))
        if isinstance(existing, Mapping)
        else {}
    )
    section.update(copy.deepcopy(dict(payload)))
    metadata["reasoning"] = section
    return metadata


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\x00", " ").split()).strip()


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
    parsed = _safe_int(value, default=default)
    return max(minimum, min(parsed, maximum))


def _deduplicate_texts(values: Sequence[Any]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = _normalize_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _safe_sequence_length(value: Any) -> int:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return len(value)
    return 0


def _log_failure(
    runtime: Optional["RuntimeContext"],
    exc: Exception,
) -> None:
    logger = getattr(runtime, "logger", None) if runtime is not None else None
    if logger is None:
        return
    exception = getattr(logger, "exception", None)
    if callable(exception):
        exception("Reasoning node failed: %s", exc)


__all__ = [
    "DEFAULT_OPTIONS",
    "create_reasoning_node",
    "reasoning_node",
    # 迁移期兼容 helper；核心实现仍位于 answering.reasoner。
    "_build_reasoning_text",
    "_build_rule_conclusion",
    "_clip_score",
    "_estimate_reasoning_confidence",
    "_evidence_to_reasoning_step",
    "_evidence_type_priority",
    "_fallback_conclusion_by_answerability",
    "_get_reasoning_config",
    "_normalize_answerability",
    "_normalize_reasoning_result",
    "_preview_text",
    "_reason_with_rules",
    "_safe_float",
    "_safe_int",
    "_sanitize_text",
    "_select_evidence_for_reasoning",
    "_update_metadata",
]
