# -*- coding: utf-8 -*-
"""KG-RAG Agent 的 LangGraph 编排层公共接口。

``graph`` 包只负责状态协议、路由规则、节点名称和 Graph 构建入口。
实体解析、知识图谱查询、检索、推理、答案生成与 Memory 的具体实现
分别位于对应领域模块和 ``graph.nodes`` 中。

导入 ``kg_rag_agent.graph`` 时不会立即加载 LangGraph 或全部 Node；
Graph Builder 与 Memory Node 均采用惰性导入。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Final

from .edges import (
    ALL_GRAPH_NODES as ROUTING_GRAPH_NODES,
    DIRECT_LLM_NODE,
    ENTITY_GROUNDING_NODE,
    ENTITY_LINKING_NODE,
    GENERATION_NODE,
    KG_RETRIEVAL_NODE,
    MENTION_EXTRACTION_NODE,
    QUERY_ROUTER_NODE,
    REASONING_NODE,
    SEMANTIC_SCORING_NODE,
    route_after_entity_grounding,
    route_after_entity_linking,
    route_after_kg_retrieval,
    route_after_mention_extraction,
    route_after_query_router,
    route_after_reasoning,
    route_after_semantic_scoring,
    validate_edge_target,
    validate_route_value,
)
from .state import (
    AgentState,
    AnswerabilityType,
    Citation,
    EntityCandidate,
    ErrorStageType,
    EvidenceItem,
    GroundedEntity,
    Mention,
    ReasoningResult,
    RequestOptions,
    RouteType,
    SemanticScoringResult,
    ToolCallRecord,
    ToolResultRecord,
    TraceEvent,
    build_initial_state,
    get_request_option,
    get_request_options,
    make_error,
    make_tool_call,
    make_tool_result,
    make_trace,
    make_warning,
    normalize_request_options,
    utc_now,
)

# Memory 节点是流程首尾节点，不参与 edges.py 中的条件路由。
MEMORY_LOAD_NODE: Final[str] = "memory_load"
MEMORY_WRITE_NODE: Final[str] = "memory_write"

CORE_GRAPH_NODE_NAMES: Final[tuple[str, ...]] = (
    QUERY_ROUTER_NODE,
    DIRECT_LLM_NODE,
    MENTION_EXTRACTION_NODE,
    ENTITY_LINKING_NODE,
    ENTITY_GROUNDING_NODE,
    KG_RETRIEVAL_NODE,
    SEMANTIC_SCORING_NODE,
    REASONING_NODE,
    GENERATION_NODE,
)

ALL_GRAPH_NODES: Final[frozenset[str]] = frozenset(
    (
        MEMORY_LOAD_NODE,
        *CORE_GRAPH_NODE_NAMES,
        MEMORY_WRITE_NODE,
    )
)

if TYPE_CHECKING:
    from .builder import (
        GRAPH_NODE_NAMES,
        NodeCallable,
        NodeRegistry,
        build_graph,
        build_graph_builder,
    )
    from .nodes import (
        create_memory_load_node,
        create_memory_write_node,
        memory_load_node,
        memory_write_node,
    )


_LAZY_BUILDER_EXPORTS: Final[frozenset[str]] = frozenset(
    {
        "GRAPH_NODE_NAMES",
        "NodeCallable",
        "NodeRegistry",
        "build_graph",
        "build_graph_builder",
    }
)

_LAZY_MEMORY_NODE_EXPORTS: Final[frozenset[str]] = frozenset(
    {
        "memory_load_node",
        "create_memory_load_node",
        "memory_write_node",
        "create_memory_write_node",
    }
)


def __getattr__(name: str) -> Any:
    """按需加载 Graph Builder 或 Memory Node 公共成员。"""

    if name in _LAZY_BUILDER_EXPORTS:
        module = import_module(".builder", __name__)
    elif name in _LAZY_MEMORY_NODE_EXPORTS:
        module = import_module(".nodes", __name__)
    else:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """让 IDE 和交互式环境发现惰性导出的公共成员。"""

    return sorted(
        set(globals())
        | set(_LAZY_BUILDER_EXPORTS)
        | set(_LAZY_MEMORY_NODE_EXPORTS)
    )


__all__ = [
    # State protocol
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
    # State helpers
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
    # Node names
    "ROUTING_GRAPH_NODES",
    "CORE_GRAPH_NODE_NAMES",
    "ALL_GRAPH_NODES",
    "MEMORY_LOAD_NODE",
    "QUERY_ROUTER_NODE",
    "DIRECT_LLM_NODE",
    "MENTION_EXTRACTION_NODE",
    "ENTITY_LINKING_NODE",
    "ENTITY_GROUNDING_NODE",
    "KG_RETRIEVAL_NODE",
    "SEMANTIC_SCORING_NODE",
    "REASONING_NODE",
    "GENERATION_NODE",
    "MEMORY_WRITE_NODE",
    # Edge routing
    "route_after_query_router",
    "route_after_mention_extraction",
    "route_after_entity_linking",
    "route_after_entity_grounding",
    "route_after_kg_retrieval",
    "route_after_semantic_scoring",
    "route_after_reasoning",
    "validate_edge_target",
    "validate_route_value",
    # Lazy graph builder exports
    "GRAPH_NODE_NAMES",
    "NodeCallable",
    "NodeRegistry",
    "build_graph",
    "build_graph_builder",
    # Lazy Memory Node exports
    "memory_load_node",
    "create_memory_load_node",
    "memory_write_node",
    "create_memory_write_node",
]
