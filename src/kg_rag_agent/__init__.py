# -*- coding: utf-8 -*-
"""KG-RAG Agent 顶层公共 API。

包根目录采用懒加载导出。执行 ``import kg_rag_agent`` 时不会立即加载
LangGraph、FastAPI、LLM Provider、知识图谱、向量数据库或 Memory Store；
只有访问具体公共对象时，才导入其所属模块。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Final

__version__ = "0.1.0"

_AGENT_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "AgentService": ("kg_rag_agent.services", "AgentService"),
    "get_default_agent_service": (
        "kg_rag_agent.services",
        "get_default_agent_service",
    ),
    "ask": ("kg_rag_agent.services", "ask"),
    "invoke": ("kg_rag_agent.services", "invoke"),
    "KGRAGAgent": ("kg_rag_agent.agents", "KGRAGAgent"),
    "KGRAgent": ("kg_rag_agent.agents", "KGRAgent"),
    "create_agent": ("kg_rag_agent.agents", "create_agent"),
    "AgentRequest": ("kg_rag_agent.agents", "AgentRequest"),
    "AgentResult": ("kg_rag_agent.agents", "AgentResult"),
    "RequestOptions": ("kg_rag_agent.agents", "RequestOptions"),
}

_RUNTIME_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "RuntimeContext": ("kg_rag_agent.runtime", "RuntimeContext"),
    "RuntimeDependencyError": (
        "kg_rag_agent.runtime",
        "RuntimeDependencyError",
    ),
    "RuntimeBuildOptions": (
        "kg_rag_agent.runtime",
        "RuntimeBuildOptions",
    ),
    "RuntimeSettings": ("kg_rag_agent.runtime", "RuntimeSettings"),
    "RuntimeSettingsError": (
        "kg_rag_agent.runtime",
        "RuntimeSettingsError",
    ),
    "RuntimeFactory": ("kg_rag_agent.runtime", "RuntimeFactory"),
    "create_runtime": ("kg_rag_agent.runtime", "create_runtime"),
}

_MEMORY_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "MemoryManager": ("kg_rag_agent.memory", "MemoryManager"),
    "MemoryPolicy": ("kg_rag_agent.memory", "MemoryPolicy"),
    "MemoryType": ("kg_rag_agent.memory", "MemoryType"),
    "MemoryRecord": ("kg_rag_agent.memory", "MemoryRecord"),
    "MemoryContext": ("kg_rag_agent.memory", "MemoryContext"),
    "MemoryWriteResult": (
        "kg_rag_agent.memory",
        "MemoryWriteResult",
    ),
    "InMemoryMemoryStore": (
        "kg_rag_agent.memory",
        "InMemoryMemoryStore",
    ),
    "JSONMemoryStore": ("kg_rag_agent.memory", "JSONMemoryStore"),
}

_APP_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "APISettings": ("kg_rag_agent.app", "APISettings"),
    "AppDependencies": ("kg_rag_agent.app", "AppDependencies"),
    "create_app": ("kg_rag_agent.app", "create_app"),
}

_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    **_AGENT_EXPORTS,
    **_RUNTIME_EXPORTS,
    **_MEMORY_EXPORTS,
    **_APP_EXPORTS,
}


def __getattr__(name: str) -> Any:
    """按需解析并缓存公共对象。"""

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """让交互式补全能够发现懒加载公共对象。"""

    return sorted(set(globals()) | set(_EXPORTS))


__all__ = [
    "__version__",
    *_AGENT_EXPORTS,
    *_RUNTIME_EXPORTS,
    *_MEMORY_EXPORTS,
    *_APP_EXPORTS,
]
