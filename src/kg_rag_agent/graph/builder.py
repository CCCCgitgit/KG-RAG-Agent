# -*- coding: utf-8 -*-
"""KG-RAG Agent 的 LangGraph 构建与编译入口。

本模块只负责注册 Node、Edge 和编译 Graph，不实现实体解析、KG 检索、
Evidence 评分、推理或答案生成逻辑。

运行时边界：
    * RuntimeContext 由 Agent 或应用入口创建，并通过 ``build_graph`` 注入。
    * RuntimeContext 不得写入 AgentState。
    * 迁移期旧 Node 仍可保持 ``node(state)`` 形式；新 Node 可以提供
      ``create_<node_name>_node(runtime)`` 工厂，或声明 ``runtime`` 参数。

正式调用链：
    AgentService -> KGRAGAgent -> build_graph(runtime=...) -> CompiledGraph
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from .edges import (
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
)
from .state import AgentState

MEMORY_LOAD_NODE = "memory_load"
MEMORY_WRITE_NODE = "memory_write"

if TYPE_CHECKING:
    from kg_rag_agent.runtime import RuntimeContext


NodeCallable = Callable[[AgentState], Mapping[str, Any]]
NodeRegistry = Mapping[str, Callable[..., Any]]


@dataclass(frozen=True, slots=True)
class _NodeSpec:
    """默认 Node 的导入信息。"""

    name: str
    module: str
    function: str
    factory: str


_NODE_SPECS: tuple[_NodeSpec, ...] = (
    _NodeSpec(
        MEMORY_LOAD_NODE,
        ".nodes.memory_load_node",
        "memory_load_node",
        "create_memory_load_node",
    ),
    _NodeSpec(
        QUERY_ROUTER_NODE,
        ".nodes.query_router_node",
        "query_router_node",
        "create_query_router_node",
    ),
    _NodeSpec(
        DIRECT_LLM_NODE,
        ".nodes.direct_llm_node",
        "direct_llm_node",
        "create_direct_llm_node",
    ),
    _NodeSpec(
        MENTION_EXTRACTION_NODE,
        ".nodes.mention_extraction_node",
        "mention_extraction_node",
        "create_mention_extraction_node",
    ),
    _NodeSpec(
        ENTITY_LINKING_NODE,
        ".nodes.entity_linking_node",
        "entity_linking_node",
        "create_entity_linking_node",
    ),
    _NodeSpec(
        ENTITY_GROUNDING_NODE,
        ".nodes.entity_grounding_node",
        "entity_grounding_node",
        "create_entity_grounding_node",
    ),
    _NodeSpec(
        KG_RETRIEVAL_NODE,
        ".nodes.kg_retrieval_node",
        "kg_retrieval_node",
        "create_kg_retrieval_node",
    ),
    _NodeSpec(
        SEMANTIC_SCORING_NODE,
        ".nodes.semantic_scoring_node",
        "semantic_scoring_node",
        "create_semantic_scoring_node",
    ),
    _NodeSpec(
        REASONING_NODE,
        ".nodes.reasoning_node",
        "reasoning_node",
        "create_reasoning_node",
    ),
    _NodeSpec(
        GENERATION_NODE,
        ".nodes.generation_node",
        "generation_node",
        "create_generation_node",
    ),
    _NodeSpec(
        MEMORY_WRITE_NODE,
        ".nodes.memory_write_node",
        "memory_write_node",
        "create_memory_write_node",
    ),
)

GRAPH_NODE_NAMES: tuple[str, ...] = tuple(spec.name for spec in _NODE_SPECS)


# =========================================================
# Public graph construction API
# =========================================================


def build_graph(
    *,
    runtime: Optional["RuntimeContext"] = None,
    checkpointer: Optional[Any] = None,
    interrupt_before: Optional[Sequence[str]] = None,
    interrupt_after: Optional[Sequence[str]] = None,
    nodes: Optional[NodeRegistry] = None,
) -> Any:
    """构建并编译 KG-RAG Agent LangGraph。

    Args:
        runtime:
            应用级 RuntimeContext。新 Node 从中读取 LLM、图谱、向量库、
            PromptManager、ToolRegistry 等共享对象。该对象不会进入 AgentState。
        checkpointer:
            LangGraph Checkpointer。为空时不启用状态持久化。
        interrupt_before:
            需要在执行前中断的 Node 名称。
        interrupt_after:
            需要在执行后中断的 Node 名称。
        nodes:
            测试或定制场景使用的 Node 覆盖映射。允许只覆盖部分 Node；未覆盖
            Node 仍从 ``graph/nodes`` 加载。未知 Node 名称会被拒绝。

    Returns:
        编译后的 LangGraph 对象。
    """

    _ensure_runtime_open(runtime)
    graph_builder = build_graph_builder(runtime=runtime, nodes=nodes)

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    normalized_before = _normalize_interrupt_nodes(
        interrupt_before,
        parameter_name="interrupt_before",
    )
    normalized_after = _normalize_interrupt_nodes(
        interrupt_after,
        parameter_name="interrupt_after",
    )
    if normalized_before is not None:
        compile_kwargs["interrupt_before"] = normalized_before
    if normalized_after is not None:
        compile_kwargs["interrupt_after"] = normalized_after

    compiled_graph = graph_builder.compile(**compile_kwargs)
    _log_graph_ready(runtime, compiled=True)
    return compiled_graph


def build_graph_builder(
    *,
    runtime: Optional["RuntimeContext"] = None,
    nodes: Optional[NodeRegistry] = None,
) -> StateGraph:
    """返回尚未编译的 ``StateGraph``。

    该入口用于结构测试、Graph 可视化和自定义 Checkpointer。业务入口应优先
    使用 :func:`build_graph`。
    """

    _ensure_runtime_open(runtime)
    graph_builder = StateGraph(AgentState)
    resolved_nodes = _resolve_nodes(runtime=runtime, overrides=nodes)

    _add_nodes(graph_builder, resolved_nodes)
    _add_edges(graph_builder)

    _log_graph_ready(runtime, compiled=False)
    return graph_builder


# =========================================================
# Node registration and dependency binding
# =========================================================


def _add_nodes(
    graph_builder: StateGraph,
    nodes: Mapping[str, NodeCallable],
) -> None:
    """注册全部 Graph Node。"""

    missing = set(GRAPH_NODE_NAMES) - set(nodes)
    unknown = set(nodes) - set(GRAPH_NODE_NAMES)
    if missing:
        raise ValueError(
            "Missing graph nodes: " + ", ".join(sorted(missing))
        )
    if unknown:
        raise ValueError(
            "Unknown graph nodes: " + ", ".join(sorted(unknown))
        )

    for node_name in GRAPH_NODE_NAMES:
        node = nodes[node_name]
        if not callable(node):
            raise TypeError(f"Graph node is not callable: {node_name}")
        graph_builder.add_node(node_name, node)


def _resolve_nodes(
    *,
    runtime: Optional["RuntimeContext"],
    overrides: Optional[NodeRegistry],
) -> dict[str, NodeCallable]:
    """加载默认 Node，并应用显式覆盖。"""

    override_map = dict(overrides or {})
    unknown = set(override_map) - set(GRAPH_NODE_NAMES)
    if unknown:
        raise ValueError(
            "Unknown node overrides: " + ", ".join(sorted(unknown))
        )

    resolved: dict[str, NodeCallable] = {}
    for spec in _NODE_SPECS:
        if spec.name in override_map:
            resolved[spec.name] = _bind_runtime(
                override_map[spec.name],
                runtime=runtime,
                node_name=spec.name,
            )
            continue

        module = importlib.import_module(spec.module, package=__package__)
        factory = getattr(module, spec.factory, None)

        if callable(factory):
            node = _call_node_factory(
                factory,
                runtime=runtime,
                node_name=spec.name,
            )
        else:
            node = getattr(module, spec.function, None)
            if not callable(node):
                raise ImportError(
                    f"Node module {module.__name__!r} must expose callable "
                    f"{spec.function!r} or {spec.factory!r}."
                )
            node = _bind_runtime(
                node,
                runtime=runtime,
                node_name=spec.name,
            )

        resolved[spec.name] = node

    return resolved


def _call_node_factory(
    factory: Callable[..., Any],
    *,
    runtime: Optional["RuntimeContext"],
    node_name: str,
) -> NodeCallable:
    """调用 Node Factory，并验证其返回值。"""

    signature = _safe_signature(factory)
    parameters = signature.parameters if signature is not None else {}

    if _accepts_keyword(parameters, "runtime"):
        node = factory(runtime=runtime)
    elif _accepts_positional(parameters):
        node = factory(runtime)
    elif not parameters:
        node = factory()
    else:
        raise TypeError(
            f"Node factory for {node_name!r} does not support RuntimeContext "
            "injection. Expected factory(runtime) or factory(*, runtime=...)."
        )

    if not callable(node):
        raise TypeError(
            f"Node factory for {node_name!r} returned a non-callable object."
        )
    return _bind_runtime(node, runtime=runtime, node_name=node_name)


def _bind_runtime(
    node: Callable[..., Any],
    *,
    runtime: Optional["RuntimeContext"],
    node_name: str,
) -> NodeCallable:
    """兼容旧 ``node(state)`` 与新 Runtime 注入形式。

    支持：
        * ``node(state)``
        * ``node(state, runtime)``
        * ``node(state, *, runtime=...)``

    Node Factory 已经闭包绑定 Runtime 时，返回的函数通常仍是 ``node(state)``，
    因而不会重复注入。
    """

    if not callable(node):
        raise TypeError(f"Graph node is not callable: {node_name}")

    signature = _safe_signature(node)
    if signature is None:
        return node  # 某些扩展 Callable 无法安全读取签名，交由 LangGraph 调用。

    parameters = list(signature.parameters.values())
    runtime_parameter = signature.parameters.get("runtime")

    if runtime_parameter is not None:
        if runtime is None and runtime_parameter.default is inspect.Parameter.empty:
            raise RuntimeError(
                f"Graph node {node_name!r} requires RuntimeContext, but no "
                "runtime was supplied to build_graph()."
            )

        def runtime_bound(state: AgentState) -> Mapping[str, Any]:
            if runtime_parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                return node(state, runtime)
            return node(state, runtime=runtime)

        _copy_callable_metadata(runtime_bound, node, node_name=node_name)
        return runtime_bound

    required_positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and parameter.default is inspect.Parameter.empty
    ]

    if len(required_positional) <= 1:
        return node

    # 为迁移期显式支持 ``node(state, context)`` 一类二参数 Node。
    if len(required_positional) == 2:
        if runtime is None:
            raise RuntimeError(
                f"Graph node {node_name!r} requires a second runtime argument, "
                "but no RuntimeContext was supplied."
            )

        def positional_runtime_bound(state: AgentState) -> Mapping[str, Any]:
            return node(state, runtime)

        _copy_callable_metadata(
            positional_runtime_bound,
            node,
            node_name=node_name,
        )
        return positional_runtime_bound

    raise TypeError(
        f"Graph node {node_name!r} has unsupported signature {signature}. "
        "Expected node(state), node(state, runtime), or a Node Factory."
    )


# =========================================================
# Edge registration
# =========================================================


def _add_edges(graph_builder: StateGraph) -> None:
    """注册入口边、条件边和结束边。"""

    graph_builder.add_edge(START, MEMORY_LOAD_NODE)
    graph_builder.add_edge(MEMORY_LOAD_NODE, QUERY_ROUTER_NODE)

    graph_builder.add_conditional_edges(
        QUERY_ROUTER_NODE,
        route_after_query_router,
        {
            DIRECT_LLM_NODE: DIRECT_LLM_NODE,
            MENTION_EXTRACTION_NODE: MENTION_EXTRACTION_NODE,
            GENERATION_NODE: GENERATION_NODE,
        },
    )

    graph_builder.add_conditional_edges(
        MENTION_EXTRACTION_NODE,
        route_after_mention_extraction,
        {
            ENTITY_LINKING_NODE: ENTITY_LINKING_NODE,
            DIRECT_LLM_NODE: DIRECT_LLM_NODE,
            GENERATION_NODE: GENERATION_NODE,
        },
    )

    graph_builder.add_conditional_edges(
        ENTITY_LINKING_NODE,
        route_after_entity_linking,
        {
            ENTITY_GROUNDING_NODE: ENTITY_GROUNDING_NODE,
            GENERATION_NODE: GENERATION_NODE,
        },
    )

    graph_builder.add_conditional_edges(
        ENTITY_GROUNDING_NODE,
        route_after_entity_grounding,
        {
            KG_RETRIEVAL_NODE: KG_RETRIEVAL_NODE,
            GENERATION_NODE: GENERATION_NODE,
        },
    )

    graph_builder.add_conditional_edges(
        KG_RETRIEVAL_NODE,
        route_after_kg_retrieval,
        {
            SEMANTIC_SCORING_NODE: SEMANTIC_SCORING_NODE,
            GENERATION_NODE: GENERATION_NODE,
        },
    )

    graph_builder.add_conditional_edges(
        SEMANTIC_SCORING_NODE,
        route_after_semantic_scoring,
        {
            REASONING_NODE: REASONING_NODE,
            GENERATION_NODE: GENERATION_NODE,
        },
    )

    graph_builder.add_conditional_edges(
        REASONING_NODE,
        route_after_reasoning,
        {GENERATION_NODE: GENERATION_NODE},
    )

    graph_builder.add_edge(DIRECT_LLM_NODE, MEMORY_WRITE_NODE)
    graph_builder.add_edge(GENERATION_NODE, MEMORY_WRITE_NODE)
    graph_builder.add_edge(MEMORY_WRITE_NODE, END)


# =========================================================
# Validation helpers
# =========================================================


def _normalize_interrupt_nodes(
    value: Optional[Sequence[str]],
    *,
    parameter_name: str,
) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{parameter_name} must be a sequence of node names.")

    normalized: list[str] = []
    for item in value:
        node_name = str(item or "").strip()
        if not node_name:
            raise ValueError(f"{parameter_name} cannot contain empty names.")
        if node_name not in GRAPH_NODE_NAMES:
            raise ValueError(
                f"Unknown node in {parameter_name}: {node_name}"
            )
        if node_name not in normalized:
            normalized.append(node_name)
    return normalized


def _ensure_runtime_open(runtime: Optional["RuntimeContext"]) -> None:
    if runtime is None:
        return

    ensure_open = getattr(runtime, "ensure_open", None)
    if callable(ensure_open):
        ensure_open()
        return

    if bool(getattr(runtime, "is_closed", False)):
        raise RuntimeError("RuntimeContext has already been closed.")



def _copy_callable_metadata(
    wrapper: Callable[..., Any],
    original: Callable[..., Any],
    *,
    node_name: str,
) -> None:
    """复制调试信息，但不设置 ``__wrapped__``，避免框架误读旧签名。"""

    wrapper.__name__ = str(getattr(original, "__name__", node_name))
    wrapper.__qualname__ = str(
        getattr(original, "__qualname__", wrapper.__name__)
    )
    wrapper.__doc__ = getattr(original, "__doc__", None)
    wrapper.__module__ = str(
        getattr(original, "__module__", wrapper.__module__)
    )

def _safe_signature(callable_obj: Callable[..., Any]) -> Optional[inspect.Signature]:
    try:
        return inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return None


def _accepts_keyword(
    parameters: Mapping[str, inspect.Parameter],
    name: str,
) -> bool:
    parameter = parameters.get(name)
    if parameter is not None and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _accepts_positional(
    parameters: Mapping[str, inspect.Parameter],
) -> bool:
    return any(
        parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
        for parameter in parameters.values()
    )


def _log_graph_ready(
    runtime: Optional["RuntimeContext"],
    *,
    compiled: bool,
) -> None:
    if runtime is None:
        return
    logger = getattr(runtime, "logger", None)
    if logger is None:
        return

    log_method = getattr(logger, "debug", None)
    if callable(log_method):
        log_method(
            "KG-RAG graph %s | nodes=%s",
            "compiled" if compiled else "assembled",
            ",".join(GRAPH_NODE_NAMES),
        )


__all__ = [
    "GRAPH_NODE_NAMES",
    "NodeCallable",
    "NodeRegistry",
    "build_graph",
    "build_graph_builder",
]
