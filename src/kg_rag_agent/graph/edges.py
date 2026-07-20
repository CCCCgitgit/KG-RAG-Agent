# -*- coding: utf-8 -*-
"""LangGraph 条件边与节点名称常量。

本模块只负责读取 :class:`AgentState` 并决定下一跳，不修改状态、不调用
LLM、不访问知识图谱，也不创建任何 Runtime 对象。

设计原则：
    - Node 负责完成原子业务步骤；
    - Edge 只负责流程控制；
    - 所有返回值必须与 ``builder.py`` 注册的节点名称一致；
    - 错误和澄清请求统一进入 ``generation``，由生成节点输出标准结果；
    - 对迁移期的新旧 State 写法保持有限兼容，但不放宽非法路由。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal, cast

from .state import AgentState, AnswerabilityType, RouteType


# =========================================================
# Node names
# =========================================================

QUERY_ROUTER_NODE: Final[str] = "query_router"
DIRECT_LLM_NODE: Final[str] = "direct_llm"
MENTION_EXTRACTION_NODE: Final[str] = "mention_extraction"
ENTITY_LINKING_NODE: Final[str] = "entity_linking"
ENTITY_GROUNDING_NODE: Final[str] = "entity_grounding"
KG_RETRIEVAL_NODE: Final[str] = "kg_retrieval"
SEMANTIC_SCORING_NODE: Final[str] = "semantic_scoring"
REASONING_NODE: Final[str] = "reasoning"
GENERATION_NODE: Final[str] = "generation"

ALL_GRAPH_NODES: Final[frozenset[str]] = frozenset(
    {
        QUERY_ROUTER_NODE,
        DIRECT_LLM_NODE,
        MENTION_EXTRACTION_NODE,
        ENTITY_LINKING_NODE,
        ENTITY_GROUNDING_NODE,
        KG_RETRIEVAL_NODE,
        SEMANTIC_SCORING_NODE,
        REASONING_NODE,
        GENERATION_NODE,
    }
)


# =========================================================
# Edge return types
# =========================================================

RouteAfterQueryRouter = Literal[
    "direct_llm",
    "mention_extraction",
    "generation",
]

RouteAfterMentionExtraction = Literal[
    "entity_linking",
    "direct_llm",
    "generation",
]

RouteAfterEntityLinking = Literal[
    "entity_grounding",
    "generation",
]

RouteAfterEntityGrounding = Literal[
    "kg_retrieval",
    "generation",
]

RouteAfterKGRetrieval = Literal[
    "semantic_scoring",
    "generation",
]

RouteAfterSemanticScoring = Literal[
    "reasoning",
    "generation",
]

RouteAfterReasoning = Literal["generation"]


# =========================================================
# Generic state predicates
# =========================================================


def _has_error(state: AgentState) -> bool:
    """Return whether the request is already in an error state."""

    if bool(state.get("has_error", False)):
        return True

    # 迁移期兼容：部分旧节点只写 error_message 或 route=error。
    route = _normalize_route(state.get("route"))
    error_message = str(state.get("error_message", "") or "").strip()
    return route == "error" or bool(error_message)


def _need_clarification(state: AgentState) -> bool:
    """Return whether the request should stop for clarification."""

    if bool(state.get("need_clarification", False)):
        return True

    route = _normalize_route(state.get("route"))
    clarifying_question = str(
        state.get("clarifying_question", "") or ""
    ).strip()
    return route == "clarify" or bool(clarifying_question)


def _has_mentions(state: AgentState) -> bool:
    """Return whether mention extraction produced at least one usable item."""

    mentions = state.get("mentions")
    if not _is_non_string_sequence(mentions):
        return False

    for mention in mentions:
        if isinstance(mention, Mapping):
            text = str(mention.get("text", "") or "").strip()
            if text:
                return True
        elif str(mention or "").strip():
            # 兼容过渡期的 ``List[str]``。
            return True

    return False


def _has_entity_candidates(state: AgentState) -> bool:
    """Return whether entity linking produced at least one usable candidate."""

    candidates_by_mention = state.get("entity_candidates")
    if not isinstance(candidates_by_mention, Mapping):
        return False

    for candidates in candidates_by_mention.values():
        if not _is_non_string_sequence(candidates):
            continue

        for candidate in candidates:
            if isinstance(candidate, Mapping):
                entity_key = str(
                    candidate.get("entity_id")
                    or candidate.get("entity_name")
                    or candidate.get("node_key")
                    or ""
                ).strip()
                if entity_key:
                    return True
            elif str(candidate or "").strip():
                return True

    return False


def _has_grounded_entities(state: AgentState) -> bool:
    """Return whether at least one entity can be used for KG retrieval.

    ``in_graph`` 缺省时按 True 处理，以兼容旧 Grounding 节点；但显式写为
    False 的实体不会被视为可检索实体。
    """

    grounded_entities = state.get("grounded_entities")
    if not _is_non_string_sequence(grounded_entities):
        return False

    for entity in grounded_entities:
        if not isinstance(entity, Mapping):
            continue

        node_key = str(
            entity.get("node_key")
            or entity.get("entity_id")
            or entity.get("entity_name")
            or ""
        ).strip()
        if not node_key:
            continue

        if _coerce_bool(entity.get("in_graph", True), default=True):
            return True

    return False


def _has_evidence(state: AgentState) -> bool:
    """Return whether KG retrieval produced evidence for semantic scoring."""

    evidence = state.get("evidence")
    if _is_non_string_sequence(evidence):
        for item in evidence:
            if isinstance(item, Mapping):
                evidence_id = str(item.get("evidence_id", "") or "").strip()
                text = str(item.get("text", "") or "").strip()
                triples = item.get("triples")
                path = item.get("path")
                if evidence_id or text or bool(triples) or bool(path):
                    return True
            elif str(item or "").strip():
                return True

    evidence_text = state.get("evidence_text")
    if isinstance(evidence_text, str) and evidence_text.strip():
        return True

    return False


def _is_answerable(state: AgentState) -> bool:
    """Return whether semantic scoring permits the reasoning stage.

    ``answerable`` 与 ``uncertain`` 均进入 reasoning；``unanswerable`` 直接进入
    generation。优先读取顶层标准字段，并兼容嵌套 ``semantic_scoring`` 结果。
    """

    top_level = _normalize_answerability(state.get("answerability"))

    nested_value: Any = None
    semantic_scoring = state.get("semantic_scoring")
    if isinstance(semantic_scoring, Mapping):
        nested_value = semantic_scoring.get("answerability")
    nested = _normalize_answerability(nested_value)

    # 初始 State 的默认值是 uncertain。若评分节点只更新嵌套结果，则优先采用
    # 明确的嵌套 answerable/unanswerable，避免默认值掩盖真实评分结论。
    if nested in {"answerable", "unanswerable"}:
        resolved = nested
    elif top_level is not None:
        resolved = top_level
    elif nested is not None:
        resolved = nested
    else:
        resolved = "uncertain"

    return resolved in {"answerable", "uncertain"}


# =========================================================
# Conditional routes
# =========================================================


def route_after_query_router(state: AgentState) -> RouteAfterQueryRouter:
    """Route after ``query_router``."""

    if _has_error(state) or _need_clarification(state):
        return GENERATION_NODE

    route = _normalize_route(state.get("route"))
    if route == "kg_rag":
        return MENTION_EXTRACTION_NODE
    if route == "direct_llm":
        return DIRECT_LLM_NODE

    # clarify、error、缺失值及任何非法值统一安全兜底。
    return GENERATION_NODE


def route_after_mention_extraction(
    state: AgentState,
) -> RouteAfterMentionExtraction:
    """Route after ``mention_extraction``."""

    if _has_error(state) or _need_clarification(state):
        return GENERATION_NODE

    if not _has_mentions(state):
        return DIRECT_LLM_NODE

    return ENTITY_LINKING_NODE


def route_after_entity_linking(
    state: AgentState,
) -> RouteAfterEntityLinking:
    """Route after ``entity_linking``."""

    if _has_error(state) or _need_clarification(state):
        return GENERATION_NODE

    if not _has_entity_candidates(state):
        return GENERATION_NODE

    return ENTITY_GROUNDING_NODE


def route_after_entity_grounding(
    state: AgentState,
) -> RouteAfterEntityGrounding:
    """Route after ``entity_grounding``."""

    if _has_error(state) or _need_clarification(state):
        return GENERATION_NODE

    if not _has_grounded_entities(state):
        return GENERATION_NODE

    return KG_RETRIEVAL_NODE


def route_after_kg_retrieval(
    state: AgentState,
) -> RouteAfterKGRetrieval:
    """Route after ``kg_retrieval``."""

    if _has_error(state) or _need_clarification(state):
        return GENERATION_NODE

    if not _has_evidence(state):
        return GENERATION_NODE

    return SEMANTIC_SCORING_NODE


def route_after_semantic_scoring(
    state: AgentState,
) -> RouteAfterSemanticScoring:
    """Route after ``semantic_scoring``."""

    if _has_error(state) or _need_clarification(state):
        return GENERATION_NODE

    if not _is_answerable(state):
        return GENERATION_NODE

    return REASONING_NODE


def route_after_reasoning(state: AgentState) -> RouteAfterReasoning:
    """Reasoning always converges to final answer generation."""

    del state  # 明确说明本路由不依赖具体字段。
    return GENERATION_NODE


# =========================================================
# Validation and normalization helpers
# =========================================================


def validate_edge_target(target: str) -> bool:
    """Return whether ``target`` is a registered graph node name."""

    return isinstance(target, str) and target in ALL_GRAPH_NODES


def validate_route_value(route: Any) -> bool:
    """Return whether a value is a supported query-router route."""

    return _normalize_route(route) is not None


def _normalize_route(value: Any) -> RouteType | None:
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower()
    if normalized in {"kg_rag", "direct_llm", "clarify", "error"}:
        return cast(RouteType, normalized)
    return None


def _normalize_answerability(value: Any) -> AnswerabilityType | None:
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower()
    if normalized in {"answerable", "uncertain", "unanswerable"}:
        return cast(AnswerabilityType, normalized)
    return None


def _is_non_string_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


__all__ = [
    "ALL_GRAPH_NODES",
    "QUERY_ROUTER_NODE",
    "DIRECT_LLM_NODE",
    "MENTION_EXTRACTION_NODE",
    "ENTITY_LINKING_NODE",
    "ENTITY_GROUNDING_NODE",
    "KG_RETRIEVAL_NODE",
    "SEMANTIC_SCORING_NODE",
    "REASONING_NODE",
    "GENERATION_NODE",
    "RouteAfterQueryRouter",
    "RouteAfterMentionExtraction",
    "RouteAfterEntityLinking",
    "RouteAfterEntityGrounding",
    "RouteAfterKGRetrieval",
    "RouteAfterSemanticScoring",
    "RouteAfterReasoning",
    "route_after_query_router",
    "route_after_mention_extraction",
    "route_after_entity_linking",
    "route_after_entity_grounding",
    "route_after_kg_retrieval",
    "route_after_semantic_scoring",
    "route_after_reasoning",
    "validate_edge_target",
    "validate_route_value",
]
