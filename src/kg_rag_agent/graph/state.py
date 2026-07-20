# -*- coding: utf-8 -*-
"""LangGraph 全局状态协议与状态辅助函数。

本模块只定义一次请求在 Graph 中流转的数据，不创建或保存任何运行时对象。
LLM Client、Embedding Model、Graph Store、Vector Store、PromptManager、
ToolRegistry、Memory Store 和 MCP Session 等共享依赖必须由 RuntimeContext 管理。

迁移说明：
    ``config`` 字段和 ``build_initial_state(..., config=...)`` 参数仅用于兼容尚未
    完成 Runtime 注入的旧 Node。新 Node 必须优先读取 ``request_options``，并从
    RuntimeContext 获取系统配置和共享组件。完成全部 Node 迁移后可删除该兼容字段。
"""

from __future__ import annotations

import copy
import operator
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Mapping, Optional, TypedDict

from typing_extensions import Annotated

RouteType = Literal["kg_rag", "direct_llm", "clarify", "error"]
AnswerabilityType = Literal["answerable", "uncertain", "unanswerable"]

ErrorStageType = Literal[
    "query_router",
    "direct_llm",
    "mention_extraction",
    "entity_linking",
    "entity_grounding",
    "kg_retrieval",
    "semantic_scoring",
    "reasoning",
    "generation",
    "tool_calling",
    "memory",
    "mcp",
    "runtime",
    "unknown",
]


class Mention(TypedDict, total=False):
    """用户问题中抽取出的实体 Mention。"""

    text: str
    start: int
    end: int
    type: str
    confidence: float


class EntityCandidate(TypedDict, total=False):
    """实体链接阶段产生的候选实体。"""

    mention: str
    entity_id: str
    entity_name: str
    score: float
    source: str
    aliases: List[str]
    metadata: Dict[str, Any]


class GroundedEntity(TypedDict, total=False):
    """已确认映射到知识图谱节点的实体。"""

    mention: str
    entity_id: str
    entity_name: str
    node_key: str
    confidence: float
    in_graph: bool
    metadata: Dict[str, Any]


class EvidenceItem(TypedDict, total=False):
    """KG 或检索模块返回的统一证据。"""

    evidence_id: str
    evidence_type: str
    source_entity: str
    target_entity: str
    relation: str
    path: List[str]
    triples: List[Dict[str, Any]]
    text: str
    score: float
    metadata: Dict[str, Any]


class SemanticScoringResult(TypedDict, total=False):
    """证据语义评分结果。"""

    score: float
    answerability: AnswerabilityType
    reason: str
    selected_evidence_ids: List[str]
    rejected_evidence_ids: List[str]


class ReasoningResult(TypedDict, total=False):
    """受证据约束的推理结果。"""

    reasoning_chain: List[str]
    conclusion: str
    used_evidence_ids: List[str]
    confidence: float
    metadata: Dict[str, Any]


class Citation(TypedDict, total=False):
    """最终回答中的证据引用。"""

    citation_id: str
    evidence_id: str
    type: str
    source_entity: str
    relation: str
    target_entity: str
    text: str
    score: float


class RequestOptions(TypedDict, total=False):
    """允许在单次请求中覆盖的轻量参数。"""

    retrieval_top_k: int
    path_max_depth: int
    temperature: float
    max_tokens: int
    language: str
    include_citations: bool
    allowed_tools: List[str]


class TraceEvent(TypedDict, total=False):
    """一次可序列化的运行 Trace。"""

    stage: str
    message: str
    timestamp: str
    payload: Dict[str, Any]


class ToolCallRecord(TypedDict, total=False):
    """AgentState 中保存的轻量工具调用记录。"""

    call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    status: Literal["pending", "running", "succeeded", "failed", "denied"]
    source: Literal["local", "mcp", "unknown"]
    timestamp: str
    metadata: Dict[str, Any]


class ToolResultRecord(TypedDict, total=False):
    """AgentState 中保存的轻量工具结果摘要。"""

    call_id: str
    tool_name: str
    success: bool
    content: Any
    error: str
    duration_ms: float
    timestamp: str
    metadata: Dict[str, Any]


class AgentState(TypedDict, total=False):
    """KG-RAG Agent 在 LangGraph 中流转的统一请求状态。

    Node 只返回自己负责字段的部分更新。``traces``、``warnings``、
    ``tool_calls`` 和 ``tool_results`` 使用加法 reducer，允许各 Node 追加记录。
    """

    # 请求上下文
    request_id: str
    session_id: str
    user_id: str
    project_id: str
    query: str
    normalized_query: str
    messages: List[Dict[str, Any]]
    chat_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    request_options: RequestOptions

    # 路由
    route: RouteType
    route_reason: str

    # 实体解析
    mentions: List[Mention]
    entity_candidates: Dict[str, List[EntityCandidate]]
    grounded_entities: List[GroundedEntity]
    ungrounded_mentions: List[str]

    # 证据与评分
    raw_evidence: List[EvidenceItem]
    evidence: List[EvidenceItem]
    evidence_text: str
    semantic_scoring: SemanticScoringResult
    semantic_score: float
    answerability: AnswerabilityType
    scoring_reason: str

    # 推理与答案
    reasoning: ReasoningResult
    reasoning_text: str
    final_answer: str
    citations: List[Citation]

    # Memory（只保存可序列化快照，不保存 Manager 或 Store）
    memory_loaded: bool
    memory_context: Dict[str, Any]
    memory_text: str
    memory_candidates: List[Dict[str, Any]]
    memory_written: bool
    memory_write_result: Dict[str, Any]

    # 澄清
    need_clarification: bool
    clarifying_question: str

    # 错误
    has_error: bool
    error_stage: ErrorStageType
    error_message: str
    error_detail: Dict[str, Any]

    # 运行观测
    traces: Annotated[List[TraceEvent], operator.add]
    warnings: Annotated[List[str], operator.add]
    tool_calls: Annotated[List[ToolCallRecord], operator.add]
    tool_results: Annotated[List[ToolResultRecord], operator.add]

    # 迁移期兼容字段。禁止向其中写入 Client、连接、Store 或 Registry。
    config: Dict[str, Any]


def build_initial_state(
    query: str,
    *,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    project_id: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    request_options: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> AgentState:
    """构造完整且可序列化的初始 AgentState。

    ``request_options`` 是正式的请求级配置入口。``config`` 仅用于旧 Node 迁移，
    会被深拷贝后放入兼容字段，且其中可识别的请求参数会同步提取到
    ``request_options``。
    """

    normalized_query = str(query or "").strip()
    safe_metadata = _copy_mapping(metadata)
    safe_config = _copy_mapping(config)
    normalized_options = normalize_request_options(
        request_options=request_options,
        metadata=safe_metadata,
        legacy_config=safe_config,
    )

    # request_options 使用独立字段；metadata 中只保留轻量镜像以兼容已有入口。
    if normalized_options:
        safe_metadata["request_options"] = copy.deepcopy(normalized_options)
    else:
        safe_metadata.pop("request_options", None)

    resolved_request_id = _normalize_identifier(request_id) or _make_id("req")
    resolved_session_id = _normalize_identifier(session_id) or _make_id("sess")
    resolved_user_id = _normalize_identifier(user_id)
    resolved_project_id = _normalize_identifier(project_id)

    initial_trace = TraceEvent(
        stage="init",
        message="Initial AgentState created.",
        timestamp=utc_now(),
        payload={
            "request_id": resolved_request_id,
            "session_id": resolved_session_id,
            "user_id": resolved_user_id,
            "project_id": resolved_project_id,
            "query": normalized_query,
        },
    )

    state = AgentState(
        request_id=resolved_request_id,
        session_id=resolved_session_id,
        user_id=resolved_user_id,
        project_id=resolved_project_id,
        query=normalized_query,
        normalized_query=normalized_query,
        messages=_copy_dict_list(messages),
        chat_history=_copy_dict_list(chat_history),
        metadata=safe_metadata,
        request_options=normalized_options,
        route="kg_rag",
        route_reason="",
        mentions=[],
        entity_candidates={},
        grounded_entities=[],
        ungrounded_mentions=[],
        raw_evidence=[],
        evidence=[],
        evidence_text="",
        semantic_scoring={},
        semantic_score=0.0,
        answerability="uncertain",
        scoring_reason="",
        reasoning={},
        reasoning_text="",
        final_answer="",
        citations=[],
        memory_loaded=False,
        memory_context={},
        memory_text="",
        memory_candidates=[],
        memory_written=False,
        memory_write_result={},
        need_clarification=False,
        clarifying_question="",
        has_error=False,
        error_stage="unknown",
        error_message="",
        error_detail={},
        traces=[initial_trace],
        warnings=[],
        tool_calls=[],
        tool_results=[],
    )

    if safe_config:
        state["config"] = safe_config

    return state


def normalize_request_options(
    request_options: Optional[Mapping[str, Any]] = None,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    legacy_config: Optional[Mapping[str, Any]] = None,
) -> RequestOptions:
    """合并并校验请求级参数。

    优先级从低到高为：旧配置映射、metadata.request_options、显式
    ``request_options``。未知参数会被拒绝，避免请求覆盖系统路径、凭据或权限。
    """

    raw: Dict[str, Any] = {}
    raw.update(_extract_legacy_request_options(legacy_config))

    metadata_options = None
    if isinstance(metadata, Mapping):
        metadata_options = metadata.get("request_options")
    if isinstance(metadata_options, Mapping):
        raw.update(dict(metadata_options))

    if request_options is not None:
        if not isinstance(request_options, Mapping):
            raise TypeError("request_options must be a mapping.")
        raw.update(dict(request_options))

    allowed = {
        "retrieval_top_k",
        "path_max_depth",
        "temperature",
        "max_tokens",
        "language",
        "include_citations",
        "allowed_tools",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            "Unsupported request options: " + ", ".join(sorted(unknown))
        )

    result: RequestOptions = {}

    if "retrieval_top_k" in raw:
        result["retrieval_top_k"] = _bounded_int(
            raw["retrieval_top_k"], "retrieval_top_k", 1, 100
        )
    if "path_max_depth" in raw:
        result["path_max_depth"] = _bounded_int(
            raw["path_max_depth"], "path_max_depth", 1, 6
        )
    if "temperature" in raw:
        result["temperature"] = _bounded_float(
            raw["temperature"], "temperature", 0.0, 2.0
        )
    if "max_tokens" in raw:
        result["max_tokens"] = _bounded_int(
            raw["max_tokens"], "max_tokens", 1, 8192
        )
    if "language" in raw:
        language = str(raw["language"] or "").strip()
        if not language or len(language) > 32:
            raise ValueError("language must contain 1 to 32 characters.")
        result["language"] = language
    if "include_citations" in raw:
        value = raw["include_citations"]
        if not isinstance(value, bool):
            raise TypeError("include_citations must be bool.")
        result["include_citations"] = value
    if "allowed_tools" in raw:
        result["allowed_tools"] = _normalize_tool_names(raw["allowed_tools"])

    return result


def get_request_options(state: Mapping[str, Any]) -> RequestOptions:
    """从 State 返回经过校验的请求级参数副本。"""

    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping.")
    return normalize_request_options(
        state.get("request_options"),
        metadata=state.get("metadata"),
        legacy_config=state.get("config"),
    )


def get_request_option(
    state: Mapping[str, Any],
    name: str,
    default: Any = None,
) -> Any:
    """读取单个请求级参数。"""

    options = get_request_options(state)
    return options.get(name, default)


def make_trace(
    stage: str,
    message: str,
    payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, List[TraceEvent]]:
    """构造可由 LangGraph reducer 追加的 Trace 更新。"""

    return {
        "traces": [
            TraceEvent(
                stage=str(stage or "unknown").strip() or "unknown",
                message=str(message or "").strip(),
                timestamp=utc_now(),
                payload=_copy_mapping(payload),
            )
        ]
    }


def make_warning(message: str) -> Dict[str, List[str]]:
    """构造可由 LangGraph reducer 追加的 Warning 更新。"""

    normalized = str(message or "").strip()
    return {"warnings": [normalized]} if normalized else {"warnings": []}


def make_error(
    stage: ErrorStageType,
    message: str,
    detail: Optional[Mapping[str, Any]] = None,
) -> AgentState:
    """构造标准错误部分更新。"""

    normalized_stage = _normalize_error_stage(stage)
    normalized_message = str(message or "Unknown error.").strip() or "Unknown error."
    safe_detail = _copy_mapping(detail)

    return AgentState(
        has_error=True,
        route="error",
        error_stage=normalized_stage,
        error_message=normalized_message,
        error_detail=safe_detail,
        traces=[
            TraceEvent(
                stage=normalized_stage,
                message=f"Error occurred: {normalized_message}",
                timestamp=utc_now(),
                payload=safe_detail,
            )
        ],
        warnings=[normalized_message],
    )


def make_tool_call(
    *,
    tool_name: str,
    arguments: Optional[Mapping[str, Any]] = None,
    call_id: Optional[str] = None,
    source: Literal["local", "mcp", "unknown"] = "local",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, List[ToolCallRecord]]:
    """构造待执行工具调用记录，不在 State 中保存执行器或连接对象。"""

    normalized_name = str(tool_name or "").strip()
    if not normalized_name:
        raise ValueError("tool_name must not be empty.")

    record = ToolCallRecord(
        call_id=_normalize_identifier(call_id) or _make_id("tool"),
        tool_name=normalized_name,
        arguments=_copy_mapping(arguments),
        status="pending",
        source=source,
        timestamp=utc_now(),
        metadata=_copy_mapping(metadata),
    )
    return {"tool_calls": [record]}


def make_tool_result(
    *,
    call_id: str,
    tool_name: str,
    success: bool,
    content: Any = None,
    error: str = "",
    duration_ms: float = 0.0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, List[ToolResultRecord]]:
    """构造轻量工具结果记录。"""

    normalized_call_id = _normalize_identifier(call_id)
    normalized_name = str(tool_name or "").strip()
    if not normalized_call_id:
        raise ValueError("call_id must not be empty.")
    if not normalized_name:
        raise ValueError("tool_name must not be empty.")

    record = ToolResultRecord(
        call_id=normalized_call_id,
        tool_name=normalized_name,
        success=bool(success),
        content=copy.deepcopy(content),
        error=str(error or "").strip(),
        duration_ms=max(_safe_float(duration_ms, 0.0), 0.0),
        timestamp=utc_now(),
        metadata=_copy_mapping(metadata),
    )
    return {"tool_results": [record]}


def utc_now() -> str:
    """返回 RFC 3339 UTC 时间字符串。"""

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _extract_legacy_request_options(
    config: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(config, Mapping):
        return {}

    data = dict(config)
    result: Dict[str, Any] = {}

    for key in (
        "retrieval_top_k",
        "path_max_depth",
        "temperature",
        "max_tokens",
        "language",
        "include_citations",
        "allowed_tools",
    ):
        if key in data:
            result[key] = data[key]

    retrieval = data.get("retrieval")
    if isinstance(retrieval, Mapping) and "top_k" in retrieval:
        result.setdefault("retrieval_top_k", retrieval["top_k"])

    graph = data.get("graph")
    if isinstance(graph, Mapping):
        entity_linking = graph.get("entity_linking")
        if isinstance(entity_linking, Mapping) and "top_k" in entity_linking:
            result.setdefault("retrieval_top_k", entity_linking["top_k"])

        kg_retrieval = graph.get("kg_retrieval")
        if isinstance(kg_retrieval, Mapping):
            depth = kg_retrieval.get(
                "path_max_depth", kg_retrieval.get("max_path_length")
            )
            if depth is not None:
                result.setdefault("path_max_depth", depth)

        generation = graph.get("generation")
        if isinstance(generation, Mapping):
            for key in ("temperature", "max_tokens"):
                if key in generation:
                    result.setdefault(key, generation[key])

    top_level_kg_retrieval = data.get("kg_retrieval")
    if isinstance(top_level_kg_retrieval, Mapping):
        depth = top_level_kg_retrieval.get(
            "path_max_depth", top_level_kg_retrieval.get("max_path_length")
        )
        if depth is not None:
            result.setdefault("path_max_depth", depth)

    for section_name in ("generation", "model"):
        section = data.get(section_name)
        if isinstance(section, Mapping):
            for key in ("temperature", "max_tokens"):
                if key in section:
                    result.setdefault(key, section[key])

    return result


def _normalize_tool_names(value: Any) -> List[str]:
    if not isinstance(value, (list, tuple, set)):
        raise TypeError("allowed_tools must be a list of strings.")

    result: List[str] = []
    for item in value:
        name = str(item or "").strip()
        if not name:
            raise ValueError("allowed_tools cannot contain empty names.")
        if name not in result:
            result.append(name)
    if len(result) > 64:
        raise ValueError("allowed_tools cannot contain more than 64 names.")
    return result


def _normalize_error_stage(value: Any) -> ErrorStageType:
    allowed = {
        "query_router",
        "direct_llm",
        "mention_extraction",
        "entity_linking",
        "entity_grounding",
        "kg_retrieval",
        "semantic_scoring",
        "reasoning",
        "generation",
        "tool_calling",
        "memory",
        "mcp",
        "runtime",
        "unknown",
    }
    normalized = str(value or "unknown").strip()
    return normalized if normalized in allowed else "unknown"  # type: ignore[return-value]


def _copy_mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping.")
    return copy.deepcopy(dict(value))


def _copy_dict_list(value: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("messages and chat_history must be lists.")

    result: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("messages and chat_history items must be mappings.")
        result.append(copy.deepcopy(dict(item)))
    return result


def _normalize_identifier(value: Any) -> str:
    return str(value or "").strip()


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be int.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be int.") from exc
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return number


def _bounded_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be float.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be float.") from exc
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return number


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "RouteType",
    "AnswerabilityType",
    "ErrorStageType",
    "RequestOptions",
    "Mention",
    "EntityCandidate",
    "GroundedEntity",
    "EvidenceItem",
    "SemanticScoringResult",
    "ReasoningResult",
    "Citation",
    "TraceEvent",
    "ToolCallRecord",
    "ToolResultRecord",
    "AgentState",
    "build_initial_state",
    "normalize_request_options",
    "get_request_options",
    "get_request_option",
    "make_trace",
    "make_warning",
    "make_error",
    "make_tool_call",
    "make_tool_result",
    "utc_now",
]
