# -*- coding: utf-8 -*-
"""Generation LangGraph 节点。

本节点是 KG-RAG 主链中唯一负责最终用户侧回答的步骤。它只读取已经形成的
Evidence、Semantic Scoring 和 Reasoning 结果，并调用 ``answering/`` 中的
``AnswerComposer`` 完成答案与 Citation 对齐。

边界：
    * 不重新查询 KG 或向量库；
    * 不重新执行 Mention、Entity Linking 或 Grounding；
    * 不重新进行语义评分和领域推理；
    * 不在 AgentState 中保存 LLM、PromptManager、Composer 或 RuntimeContext；
    * 系统配置和共享组件由 RuntimeContext 注入；
    * 请求级 temperature、max_tokens、include_citations 只从白名单参数读取。

迁移兼容：
    原项目 ``tests/test_generation.py`` 直接导入了若干回答辅助函数。它们的核心
    实现现已迁移至 ``answering/composer.py``，本文件继续导出同名符号，但不再
    维护第二套实现。
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from kg_rag_agent.answering import (
    AnswerComposer,
    AnswerResult,
    GenerationOptions,
)
from kg_rag_agent.answering.citation_builder import (
    build_citations as _answering_build_citations,
)
from kg_rag_agent.answering.composer import (
    _answer_by_relation_intent,
    _answer_relationship_question,
    _build_basis_sentence,
    _build_clarification_answer,
    _build_evidence_brief,
    _build_error_answer,
    _build_generation_prompt,
    _build_unanswerable_answer,
    _call_llm,
    _canonical_relation,
    _choose_best_relation_item_for_query,
    _clip_score,
    _describe_neighbor_items,
    _describe_path_items,
    _describe_relation_items,
    _detect_relation_intent,
    _display_entity,
    _entity_mentioned_in_query,
    _extract_answer,
    _extract_entities_from_evidence_text,
    _generate_with_rules,
    _get_generation_config,
    _get_system_prompt,
    _normalize_answerability,
    _normalize_text,
    _relation_label_zh,
    _remove_internal_debug_lines,
    _safe_float,
    _safe_int,
    _sanitize_user_answer,
    _token_overlap_score,
    _try_build_direct_fact_answer,
)

from ..state import (
    AgentState,
    Citation,
    EvidenceItem,
    ReasoningResult,
    get_request_options,
    make_error,
    utc_now,
)

if TYPE_CHECKING:
    from kg_rag_agent.runtime import RuntimeContext


DEFAULT_OPTIONS = GenerationOptions(
    use_llm=True,
    temperature=0.2,
    max_tokens=1200,
    include_citations=True,
)


def create_generation_node(
    runtime: Optional["RuntimeContext"] = None,
    *,
    composer: Optional[Any] = None,
):
    """创建已绑定 RuntimeContext 的 Generation Node。"""

    _ensure_runtime_open(runtime)

    def _node(state: AgentState) -> AgentState:
        return generation_node(
            state,
            runtime=runtime,
            composer=composer,
        )

    _node.__name__ = "generation_node"
    _node.__qualname__ = "generation_node"
    _node.__doc__ = generation_node.__doc__
    return _node


def generation_node(
    state: AgentState,
    *,
    runtime: Optional["RuntimeContext"] = None,
    composer: Optional[Any] = None,
) -> AgentState:
    """生成最终答案和 Citation，并返回本节点负责的部分 State 更新。"""

    try:
        _ensure_runtime_open(runtime)

        query = _clean_text(
            state.get("normalized_query") or state.get("query") or ""
        )
        evidence, evidence_warnings = _normalize_evidence(
            state.get("evidence") or []
        )
        reasoning, reasoning_warnings = _normalize_reasoning(
            state.get("reasoning") or {}
        )
        evidence_text = _clean_text(state.get("evidence_text") or "")
        reasoning_text = _clean_text(state.get("reasoning_text") or "")
        answerability = _normalize_answerability(
            state.get("answerability", "uncertain")
        )
        semantic_score = _clip_score(
            _safe_float(state.get("semantic_score", 0.0), default=0.0)
        )

        generation_config = _get_node_generation_config(
            state=state,
            runtime=runtime,
        )
        options = _build_options(
            state=state,
            generation_config=generation_config,
        )

        need_clarification = bool(state.get("need_clarification", False)) or (
            state.get("route") == "clarify"
        )
        has_existing_error = bool(state.get("has_error", False)) or (
            state.get("route") == "error"
        )
        requires_llm = (
            options.use_llm
            and not need_clarification
            and not has_existing_error
            and answerability != "unanswerable"
            and bool(evidence)
        )

        resolved_composer, composer_warnings = _resolve_composer(
            runtime=runtime,
            composer=composer,
            use_llm=requires_llm,
        )
        system_prompt, prompt_warnings = _resolve_system_prompt(
            runtime=runtime,
            state=state,
        )

        result = _invoke_composer(
            composer=resolved_composer,
            query=query,
            evidence=evidence,
            reasoning=reasoning,
            evidence_text=evidence_text,
            reasoning_text=reasoning_text,
            answerability=answerability,
            semantic_score=semantic_score,
            options=options,
            clarifying_question=_clean_text(
                state.get("clarifying_question") or ""
            ),
            need_clarification=need_clarification,
            has_error=has_existing_error,
            ungrounded_mentions=_normalize_text_list(
                state.get("ungrounded_mentions") or []
            ),
            scoring_reason=_clean_text(state.get("scoring_reason") or ""),
            system_prompt=system_prompt,
        )
        normalized_result = _normalize_answer_result(
            result=result,
            evidence=evidence,
            options=options,
            answerability=answerability,
            semantic_score=semantic_score,
        )

        final_answer = _sanitize_user_answer(normalized_result.answer)
        if not final_answer:
            fallback, fallback_type = _generate_with_rules(
                query=query,
                evidence=evidence,
                reasoning=reasoning,
                answerability=answerability,
                semantic_score=semantic_score,
            )
            final_answer = _sanitize_user_answer(fallback)
            normalized_result.generation_type = fallback_type

        if not final_answer:
            final_answer = _build_unanswerable_answer(
                query=query,
                state={
                    "ungrounded_mentions": state.get("ungrounded_mentions", []),
                    "scoring_reason": state.get("scoring_reason", ""),
                },
            )
            normalized_result.generation_type = "unanswerable"

        citations = _normalize_citations(normalized_result.citations)
        if not options.include_citations:
            citations = []

        warnings = _deduplicate_texts(
            evidence_warnings
            + reasoning_warnings
            + composer_warnings
            + prompt_warnings
        )

        generation_type = _clean_token(
            normalized_result.generation_type
        ) or "rule"
        metadata = _update_metadata(
            state=state,
            payload={
                "generation_type": generation_type,
                "model_called": generation_type == "llm",
                "answerability": normalized_result.answerability,
                "semantic_score": normalized_result.semantic_score,
                "num_evidence": len(evidence),
                "num_citations": len(citations),
                "answer_length": len(final_answer),
                "include_citations": options.include_citations,
                "temperature": options.temperature,
                "max_tokens": options.max_tokens,
            },
        )

        update = AgentState(
            final_answer=final_answer,
            citations=citations,
            has_error=has_existing_error,
            error_stage=(
                state.get("error_stage", "unknown")
                if has_existing_error
                else "unknown"
            ),
            error_message=(
                _clean_text(state.get("error_message") or "")
                if has_existing_error
                else ""
            ),
            error_detail=(
                copy.deepcopy(state.get("error_detail") or {})
                if has_existing_error
                else {}
            ),
            metadata=metadata,
            traces=[
                {
                    "stage": "generation",
                    "message": "Final answer generated.",
                    "timestamp": utc_now(),
                    "payload": {
                        "generation_type": generation_type,
                        "answerability": normalized_result.answerability,
                        "semantic_score": normalized_result.semantic_score,
                        "answer_length": len(final_answer),
                        "num_evidence": len(evidence),
                        "num_citations": len(citations),
                        "model_called": generation_type == "llm",
                    },
                }
            ],
            warnings=warnings,
        )
        return update

    except Exception as exc:
        _log_failure(runtime, exc)
        return make_error(
            stage="generation",
            message=str(exc),
            detail={
                "query_length": len(
                    str(
                        state.get("normalized_query")
                        or state.get("query")
                        or ""
                    )
                ),
                "num_evidence": _safe_sequence_length(state.get("evidence")),
                "error_type": type(exc).__name__,
            },
        )


# ---------------------------------------------------------------------------
# Runtime and composer resolution
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


def _resolve_composer(
    *,
    runtime: Optional["RuntimeContext"],
    composer: Optional[Any],
    use_llm: bool,
) -> tuple[Any, List[str]]:
    warnings: List[str] = []

    if composer is not None:
        _validate_composer(composer)
        return composer, warnings

    runtime_composer = _runtime_get(runtime, "answer_composer")
    if runtime_composer is not None:
        _validate_composer(runtime_composer)
        return runtime_composer, warnings

    llm_client = _runtime_get(runtime, "llm_client") if use_llm else None
    if use_llm and llm_client is None:
        warnings.append(
            "LLM generation was requested but no shared LLM client was "
            "available; rule generation was used."
        )

    return AnswerComposer(llm_client=llm_client), warnings


def _validate_composer(composer: Any) -> None:
    if not callable(getattr(composer, "compose", None)):
        raise TypeError("answer composer must provide a callable compose method")


def _invoke_composer(*, composer: Any, **kwargs: Any) -> Any:
    compose = getattr(composer, "compose")

    try:
        return compose(**kwargs)
    except TypeError as first_error:
        # 迁移期兼容较早的 AnswerComposer 位置参数接口。
        try:
            return compose(
                kwargs["query"],
                kwargs["evidence"],
                kwargs["reasoning"],
                kwargs["evidence_text"],
                kwargs["reasoning_text"],
                kwargs["answerability"],
                kwargs["semantic_score"],
                kwargs["options"],
            )
        except TypeError:
            raise first_error


# ---------------------------------------------------------------------------
# Configuration, request options and PromptManager
# ---------------------------------------------------------------------------


def _get_node_generation_config(
    *,
    state: AgentState,
    runtime: Optional["RuntimeContext"],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}

    legacy_config = state.get("config")
    if isinstance(legacy_config, Mapping):
        merged.update(_get_generation_config(dict(legacy_config)))

    settings = getattr(runtime, "settings", None) if runtime is not None else None
    if settings is not None:
        section = getattr(settings, "section", None)
        if callable(section):
            try:
                value = section("generation")
            except Exception:
                value = None
            if isinstance(value, Mapping):
                merged.update(dict(value))
        else:
            raw = getattr(settings, "raw", None)
            if isinstance(raw, Mapping):
                merged.update(_get_generation_config(dict(raw)))

    return merged


def _build_options(
    *,
    state: AgentState,
    generation_config: Mapping[str, Any],
) -> GenerationOptions:
    request_options = get_request_options(state)

    use_llm = _as_bool(
        generation_config.get("use_llm", DEFAULT_OPTIONS.use_llm),
        default=DEFAULT_OPTIONS.use_llm,
    )
    temperature = request_options.get(
        "temperature",
        generation_config.get("temperature", DEFAULT_OPTIONS.temperature),
    )
    max_tokens = request_options.get(
        "max_tokens",
        generation_config.get("max_tokens", DEFAULT_OPTIONS.max_tokens),
    )
    include_citations = request_options.get(
        "include_citations",
        generation_config.get(
            "include_citations",
            DEFAULT_OPTIONS.include_citations,
        ),
    )

    return GenerationOptions(
        use_llm=use_llm,
        temperature=_safe_float(
            temperature,
            default=DEFAULT_OPTIONS.temperature,
        ),
        max_tokens=_safe_int(
            max_tokens,
            default=DEFAULT_OPTIONS.max_tokens,
        ),
        include_citations=_as_bool(
            include_citations,
            default=DEFAULT_OPTIONS.include_citations,
        ),
    ).normalized()


def _resolve_system_prompt(
    *,
    runtime: Optional["RuntimeContext"],
    state: AgentState,
) -> tuple[str, List[str]]:
    warnings: List[str] = []
    prompt_manager = _runtime_get(runtime, "prompt_manager")

    if prompt_manager is not None:
        for name in (
            "generation.system",
            "generation_system_prompt",
            "system_prompt",
        ):
            text = _prompt_manager_get(prompt_manager, name)
            if text:
                return text, warnings
        warnings.append(
            "PromptManager did not contain a generation system prompt; "
            "the configured fallback prompt was used."
        )

    config: Dict[str, Any] = {}
    legacy_config = state.get("config")
    if isinstance(legacy_config, Mapping):
        config = dict(legacy_config)
    return _get_system_prompt(config), warnings


def _prompt_manager_get(prompt_manager: Any, name: str) -> str:
    for method_name in ("get", "get_prompt"):
        method = getattr(prompt_manager, method_name, None)
        if not callable(method):
            continue
        try:
            value = method(name, default="")
        except TypeError:
            try:
                value = method(name)
            except Exception:
                continue
        except Exception:
            continue
        normalized = _clean_text(value)
        if normalized:
            return normalized
    return ""


# ---------------------------------------------------------------------------
# Domain result normalization
# ---------------------------------------------------------------------------


def _normalize_evidence(value: Any) -> tuple[List[EvidenceItem], List[str]]:
    warnings: List[str] = []
    if value is None:
        return [], warnings
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return [], ["Invalid evidence container was ignored during generation."]

    result: List[EvidenceItem] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            warnings.append(f"Evidence item {index} was not a mapping and was ignored.")
            continue
        normalized = dict(item)
        normalized["evidence_id"] = (
            _clean_text(normalized.get("evidence_id")) or f"E{index + 1}"
        )
        normalized["text"] = _clean_text(normalized.get("text"))
        normalized["score"] = _clip_score(
            _safe_float(normalized.get("score", 0.0), default=0.0)
        )
        result.append(EvidenceItem(**normalized))
    return result, warnings


def _normalize_reasoning(value: Any) -> tuple[ReasoningResult, List[str]]:
    if value is None:
        return ReasoningResult(), []
    if not isinstance(value, Mapping):
        return ReasoningResult(), [
            "Invalid reasoning result was ignored during generation."
        ]

    normalized = dict(value)
    normalized["reasoning_chain"] = _normalize_text_list(
        normalized.get("reasoning_chain") or []
    )
    normalized["conclusion"] = _clean_text(normalized.get("conclusion"))
    normalized["used_evidence_ids"] = _normalize_text_list(
        normalized.get("used_evidence_ids") or []
    )
    normalized["confidence"] = _clip_score(
        _safe_float(normalized.get("confidence", 0.0), default=0.0)
    )
    metadata = normalized.get("metadata")
    normalized["metadata"] = dict(metadata) if isinstance(metadata, Mapping) else {}
    return ReasoningResult(**normalized), []


def _normalize_answer_result(
    *,
    result: Any,
    evidence: List[EvidenceItem],
    options: GenerationOptions,
    answerability: str,
    semantic_score: float,
) -> AnswerResult:
    if isinstance(result, AnswerResult):
        normalized = result
    elif isinstance(result, Mapping):
        normalized = AnswerResult(
            answer=_clean_text(
                result.get("answer") or result.get("final_answer") or ""
            ),
            citations=list(result.get("citations") or []),
            generation_type=_clean_token(
                result.get("generation_type") or "rule"
            ),
            answerability=_normalize_answerability(
                result.get("answerability", answerability)
            ),
            semantic_score=_clip_score(
                _safe_float(
                    result.get("semantic_score", semantic_score),
                    default=semantic_score,
                )
            ),
            metadata=(
                dict(result.get("metadata") or {})
                if isinstance(result.get("metadata"), Mapping)
                else {}
            ),
        )
    elif isinstance(result, str):
        normalized = AnswerResult(
            answer=result,
            citations=[],
            generation_type="custom",
            answerability=_normalize_answerability(answerability),
            semantic_score=_clip_score(semantic_score),
        )
    else:
        raise TypeError("AnswerComposer returned an unsupported result type.")

    normalized.answer = _clean_text(normalized.answer)
    normalized.generation_type = (
        _clean_token(normalized.generation_type) or "rule"
    )
    normalized.answerability = _normalize_answerability(
        normalized.answerability or answerability
    )
    normalized.semantic_score = _clip_score(
        _safe_float(normalized.semantic_score, default=semantic_score)
    )

    if options.include_citations and not normalized.citations:
        normalized.citations = _build_citations(evidence)
    return normalized


def _build_citations(evidence: List[EvidenceItem]) -> List[Citation]:
    """将 Evidence 转换为稳定 Citation；核心实现位于 answering/。"""

    return _normalize_citations(_answering_build_citations(evidence or []))


def _normalize_citations(value: Any) -> List[Citation]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return []

    citations: List[Citation] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            continue
        normalized = dict(item)
        normalized["citation_id"] = (
            _clean_text(normalized.get("citation_id")) or f"E{index + 1}"
        )
        normalized["evidence_id"] = (
            _clean_text(normalized.get("evidence_id"))
            or normalized["citation_id"]
        )
        normalized["text"] = _clean_text(normalized.get("text"))
        normalized["score"] = _clip_score(
            _safe_float(normalized.get("score", 0.0), default=0.0)
        )
        citations.append(Citation(**normalized))
    return citations


# ---------------------------------------------------------------------------
# Metadata, logging and common helpers
# ---------------------------------------------------------------------------


def _update_metadata(
    *,
    state: AgentState,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """更新 ``metadata['generation']``，不修改输入 State。"""

    current = state.get("metadata")
    metadata = copy.deepcopy(dict(current)) if isinstance(current, Mapping) else {}
    metadata["generation"] = copy.deepcopy(dict(payload or {}))
    return metadata


def _log_failure(runtime: Optional["RuntimeContext"], exc: Exception) -> None:
    logger = getattr(runtime, "logger", None) if runtime is not None else None
    if logger is None:
        return
    method = getattr(logger, "exception", None)
    if callable(method):
        try:
            method("Generation node failed: %s", exc)
        except Exception:
            pass


def _clean_text(value: Any) -> str:
    """清理首尾和重复空白，同时保留原始大小写与下划线。"""

    text = str(value or "").strip()
    return " ".join(text.split())


def _clean_token(value: Any) -> str:
    """清理内部枚举/类型标识，不改变下划线语义。"""

    return str(value or "").strip().lower().replace(" ", "_")


def _normalize_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, (bytes, bytearray)) or not isinstance(value, Sequence):
        return []

    result: List[str] = []
    seen: set[str] = set()
    for item in value:
        text = _clean_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _deduplicate_texts(items: Sequence[Any]) -> List[str]:
    return _normalize_text_list(list(items or []))


def _as_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _safe_sequence_length(value: Any) -> int:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return len(value)
    return 0


__all__ = [
    "create_generation_node",
    "generation_node",
    # 迁移期辅助函数导出。
    "_answer_by_relation_intent",
    "_answer_relationship_question",
    "_build_basis_sentence",
    "_build_citations",
    "_build_clarification_answer",
    "_build_error_answer",
    "_build_evidence_brief",
    "_build_generation_prompt",
    "_build_unanswerable_answer",
    "_call_llm",
    "_canonical_relation",
    "_choose_best_relation_item_for_query",
    "_clip_score",
    "_describe_neighbor_items",
    "_describe_path_items",
    "_describe_relation_items",
    "_detect_relation_intent",
    "_display_entity",
    "_entity_mentioned_in_query",
    "_extract_answer",
    "_extract_entities_from_evidence_text",
    "_generate_with_rules",
    "_get_generation_config",
    "_get_system_prompt",
    "_normalize_answerability",
    "_normalize_text",
    "_relation_label_zh",
    "_remove_internal_debug_lines",
    "_safe_float",
    "_safe_int",
    "_sanitize_user_answer",
    "_token_overlap_score",
    "_try_build_direct_fact_answer",
    "_update_metadata",
]
