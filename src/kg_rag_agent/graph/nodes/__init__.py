# -*- coding: utf-8 -*-
"""KG-RAG LangGraph 节点的统一公共入口。

本模块只负责节点名称、节点函数和 Node Factory 的统一导出，不实现任何
业务逻辑，也不创建 RuntimeContext、LLM、图谱或向量库对象。

设计目标：
    1. 避免导入 ``graph.nodes`` 时一次性加载所有大型节点模块；
    2. 为旧式 ``node(state)`` 和新式 ``create_*_node(runtime)`` 同时提供兼容；
    3. 统一节点公开名称，减少 builder、测试和扩展代码中的重复字符串；
    4. 后续逐个替换节点文件时，不要求调用方同步修改导入路径。
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any, Final, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from kg_rag_agent.graph.state import AgentState
    from kg_rag_agent.runtime import RuntimeContext


NodeCallable = Callable[["AgentState"], Mapping[str, Any]]
NodeFactory = Callable[["RuntimeContext"], NodeCallable]


# ---------------------------------------------------------------------------
# Public node names
# ---------------------------------------------------------------------------

MEMORY_LOAD_NODE: Final[str] = "memory_load"
QUERY_ROUTER_NODE: Final[str] = "query_router"
DIRECT_LLM_NODE: Final[str] = "direct_llm"
MENTION_EXTRACTION_NODE: Final[str] = "mention_extraction"
ENTITY_LINKING_NODE: Final[str] = "entity_linking"
ENTITY_GROUNDING_NODE: Final[str] = "entity_grounding"
KG_RETRIEVAL_NODE: Final[str] = "kg_retrieval"
SEMANTIC_SCORING_NODE: Final[str] = "semantic_scoring"
REASONING_NODE: Final[str] = "reasoning"
GENERATION_NODE: Final[str] = "generation"
MEMORY_WRITE_NODE: Final[str] = "memory_write"

GRAPH_NODE_NAMES: Final[tuple[str, ...]] = (
    MEMORY_LOAD_NODE,
    QUERY_ROUTER_NODE,
    DIRECT_LLM_NODE,
    MENTION_EXTRACTION_NODE,
    ENTITY_LINKING_NODE,
    ENTITY_GROUNDING_NODE,
    KG_RETRIEVAL_NODE,
    SEMANTIC_SCORING_NODE,
    REASONING_NODE,
    GENERATION_NODE,
    MEMORY_WRITE_NODE,
)


# ``public attribute -> (relative module, implementation attribute)``
_NODE_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "memory_load_node": (".memory_load_node", "memory_load_node"),
    "query_router_node": (".query_router_node", "query_router_node"),
    "direct_llm_node": (".direct_llm_node", "direct_llm_node"),
    "mention_extraction_node": (
        ".mention_extraction_node",
        "mention_extraction_node",
    ),
    "entity_linking_node": (".entity_linking_node", "entity_linking_node"),
    "entity_grounding_node": (
        ".entity_grounding_node",
        "entity_grounding_node",
    ),
    "kg_retrieval_node": (".kg_retrieval_node", "kg_retrieval_node"),
    "semantic_scoring_node": (
        ".semantic_scoring_node",
        "semantic_scoring_node",
    ),
    "reasoning_node": (".reasoning_node", "reasoning_node"),
    "generation_node": (".generation_node", "generation_node"),
    "memory_write_node": (".memory_write_node", "memory_write_node"),
}

_FACTORY_EXPORTS: Final[dict[str, tuple[str, str, str]]] = {
    "create_memory_load_node": (
        ".memory_load_node",
        "create_memory_load_node",
        "memory_load_node",
    ),
    "create_query_router_node": (
        ".query_router_node",
        "create_query_router_node",
        "query_router_node",
    ),
    "create_direct_llm_node": (
        ".direct_llm_node",
        "create_direct_llm_node",
        "direct_llm_node",
    ),
    "create_mention_extraction_node": (
        ".mention_extraction_node",
        "create_mention_extraction_node",
        "mention_extraction_node",
    ),
    "create_entity_linking_node": (
        ".entity_linking_node",
        "create_entity_linking_node",
        "entity_linking_node",
    ),
    "create_entity_grounding_node": (
        ".entity_grounding_node",
        "create_entity_grounding_node",
        "entity_grounding_node",
    ),
    "create_kg_retrieval_node": (
        ".kg_retrieval_node",
        "create_kg_retrieval_node",
        "kg_retrieval_node",
    ),
    "create_semantic_scoring_node": (
        ".semantic_scoring_node",
        "create_semantic_scoring_node",
        "semantic_scoring_node",
    ),
    "create_reasoning_node": (
        ".reasoning_node",
        "create_reasoning_node",
        "reasoning_node",
    ),
    "create_generation_node": (
        ".generation_node",
        "create_generation_node",
        "generation_node",
    ),
    "create_memory_write_node": (
        ".memory_write_node",
        "create_memory_write_node",
        "memory_write_node",
    ),
}


# ---------------------------------------------------------------------------
# Lazy public imports
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    """按需加载节点函数或 Node Factory。

    对尚未完成 Runtime Factory 改造的旧节点，本模块会生成一个迁移期兼容
    Factory。该 Factory 不会把 Runtime 写入 AgentState：

    - 旧式 ``node(state)``：直接调用；
    - ``node(state, runtime=...)``：通过关键字注入 Runtime；
    - ``node(state, runtime)``：通过第二个位置参数注入 Runtime。

    当对应节点文件正式提供 ``create_*_node`` 后，会优先使用节点文件中的
    Factory，不再使用兼容包装。
    """

    if name in _NODE_EXPORTS:
        module_name, attribute_name = _NODE_EXPORTS[name]
        module = importlib.import_module(module_name, package=__name__)
        value = getattr(module, attribute_name)
        if not callable(value):
            raise TypeError(f"Node export is not callable: {name}")
        globals()[name] = value
        return value

    if name in _FACTORY_EXPORTS:
        module_name, factory_name, node_name = _FACTORY_EXPORTS[name]
        module = importlib.import_module(module_name, package=__name__)

        factory = getattr(module, factory_name, None)
        if callable(factory):
            globals()[name] = factory
            return factory

        node = getattr(module, node_name)
        if not callable(node):
            raise TypeError(f"Node export is not callable: {node_name}")

        compatibility_factory = _build_compatibility_factory(
            node=cast(Callable[..., Any], node),
            factory_name=factory_name,
        )
        globals()[name] = compatibility_factory
        return compatibility_factory

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return stable public names for IDE completion and introspection."""

    return sorted(set(globals()) | set(__all__))


# ---------------------------------------------------------------------------
# Compatibility factory
# ---------------------------------------------------------------------------


def _build_compatibility_factory(
    *,
    node: Callable[..., Any],
    factory_name: str,
) -> NodeFactory:
    """为旧节点生成迁移期 Runtime Factory。"""

    signature = inspect.signature(node)
    parameters = list(signature.parameters.values())
    runtime_parameter = signature.parameters.get("runtime")

    @wraps(node)
    def factory(runtime: "RuntimeContext") -> NodeCallable:
        if runtime is None:
            raise ValueError(f"{factory_name} requires a RuntimeContext")

        @wraps(node)
        def bound_node(state: "AgentState") -> Mapping[str, Any]:
            if runtime_parameter is not None:
                if runtime_parameter.kind in {
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                }:
                    result = node(state, runtime=runtime)
                else:
                    result = node(state, runtime)
            elif len(parameters) >= 2 and parameters[1].kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }:
                result = node(state, runtime)
            else:
                result = node(state)

            if result is None:
                return {}
            if not isinstance(result, Mapping):
                raise TypeError(
                    f"Graph node {node.__name__!r} must return a mapping, "
                    f"got {type(result).__name__}"
                )
            return result

        return bound_node

    factory.__name__ = factory_name
    factory.__qualname__ = factory_name
    factory.__doc__ = (
        f"Compatibility factory for {node.__module__}.{node.__name__}."
    )
    return cast(NodeFactory, factory)


__all__ = [
    # Node names
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
    "GRAPH_NODE_NAMES",
    # Node callables
    "memory_load_node",
    "query_router_node",
    "direct_llm_node",
    "mention_extraction_node",
    "entity_linking_node",
    "entity_grounding_node",
    "kg_retrieval_node",
    "semantic_scoring_node",
    "reasoning_node",
    "generation_node",
    "memory_write_node",
    # Runtime-aware factories
    "create_memory_load_node",
    "create_query_router_node",
    "create_direct_llm_node",
    "create_mention_extraction_node",
    "create_entity_linking_node",
    "create_entity_grounding_node",
    "create_kg_retrieval_node",
    "create_semantic_scoring_node",
    "create_reasoning_node",
    "create_generation_node",
    "create_memory_write_node",
]
