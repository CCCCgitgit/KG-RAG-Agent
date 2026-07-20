# -*- coding: utf-8 -*-
"""
runtime/context.py

跨请求共享的运行时依赖容器。

RuntimeContext 只保存 Client、Store、Manager、Registry、配置和连接对象；
不得写入 AgentState，也不负责 LangGraph 的请求状态流转。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, MutableMapping, Optional, TYPE_CHECKING

from .settings import RuntimeSettings

if TYPE_CHECKING:
    from kg_rag_agent.entity_resolution.linker import EntityLinker
    from kg_rag_agent.kg.graph_loader import GraphLoader
    from kg_rag_agent.llm.llm_client import LLMClient
    from kg_rag_agent.llm.prompt_manager import PromptManager
    from kg_rag_agent.memory.manager import MemoryManager
    from kg_rag_agent.retrieval.embedding import EmbeddingClient
    from kg_rag_agent.retrieval.entity_vector_store import EntityVectorStore
    from kg_rag_agent.retrieval.hybrid_retriever import HybridRetriever
    from kg_rag_agent.retrieval.reranker import Reranker
    from kg_rag_agent.retrieval.vector_retriever import VectorRetriever


class RuntimeDependencyError(RuntimeError):
    """请求的运行时依赖尚未注册。"""


@dataclass(slots=True)
class RuntimeContext:
    """
    KG-RAG Agent 的共享运行时上下文。

    所有字段均为应用级共享对象。Node 应读取这些对象，但不得把它们复制到
    AgentState 中。测试可以直接注入 Fake 或 Mock 对象。
    """

    settings: RuntimeSettings
    logger: logging.Logger

    llm_client: Optional["LLMClient"] = None
    embedding_client: Optional["EmbeddingClient"] = None
    graph_loader: Optional["GraphLoader"] = None
    entity_vector_store: Optional["EntityVectorStore"] = None
    vector_retriever: Optional["VectorRetriever"] = None
    hybrid_retriever: Optional["HybridRetriever"] = None
    reranker: Optional["Reranker"] = None
    entity_linker: Optional["EntityLinker"] = None
    prompt_manager: Optional["PromptManager"] = None

    # 后续模块尚未实现时保持 None；不要提前创建第二套占位业务逻辑。
    tool_registry: Optional[Any] = None
    memory_manager: Optional["MemoryManager"] = None
    mcp_client_manager: Optional[Any] = None
    trace_manager: Optional[Any] = None

    extras: MutableMapping[str, Any] = field(default_factory=dict)
    _closed: bool = field(default=False, init=False, repr=False)

    def ensure_open(self) -> None:
        """已关闭的 RuntimeContext 不允许继续使用。"""

        if self._closed:
            raise RuntimeError("RuntimeContext has already been closed.")

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def graph_store(self) -> Optional["GraphLoader"]:
        """兼容文档中的 Graph Store 命名；当前实现由 GraphLoader 承担。"""

        return self.graph_loader

    @property
    def vector_store(self) -> Optional["EntityVectorStore"]:
        """当前默认的实体向量库。"""

        return self.entity_vector_store

    def get_graph(self) -> Any:
        """获取已缓存或延迟加载的知识图谱。"""

        self.ensure_open()
        loader = self.require("graph_loader")
        return loader.get_graph()

    def get(self, name: str, default: Any = None) -> Any:
        """按名称读取标准字段或 extras 中的扩展依赖。"""

        self.ensure_open()
        if hasattr(self, name) and name not in {"extras", "_closed"}:
            return getattr(self, name)
        return self.extras.get(name, default)

    def require(self, name: str) -> Any:
        """读取必需依赖；未注册时给出明确错误。"""

        marker = object()
        value = self.get(name, marker)
        if value is marker or value is None:
            raise RuntimeDependencyError(
                f"Runtime dependency is not available: {name}"
            )
        return value

    def register(
        self,
        name: str,
        value: Any,
        *,
        overwrite: bool = False,
    ) -> None:
        """注册扩展依赖，主要用于 Tool、Memory、MCP 和测试替身。"""

        self.ensure_open()
        normalized = str(name or "").strip()
        if not normalized:
            raise ValueError("dependency name must not be empty")
        if normalized.startswith("_"):
            raise ValueError("private dependency names are not allowed")

        if hasattr(self, normalized) and normalized not in {"extras", "_closed"}:
            current = getattr(self, normalized)
            if current is not None and not overwrite:
                raise KeyError(f"Runtime dependency already exists: {normalized}")
            setattr(self, normalized, value)
            return

        if normalized in self.extras and not overwrite:
            raise KeyError(f"Runtime dependency already exists: {normalized}")

        self.extras[normalized] = value

    def unregister(self, name: str) -> Any:
        """移除通过 extras 注册的扩展依赖。"""

        self.ensure_open()
        normalized = str(name or "").strip()
        if normalized in self.extras:
            return self.extras.pop(normalized)
        raise KeyError(f"Runtime extra dependency not found: {normalized}")

    def iter_dependencies(self) -> Iterator[tuple[str, Any]]:
        """遍历当前已注册且非空的依赖。"""

        standard_names = (
            "llm_client",
            "embedding_client",
            "graph_loader",
            "entity_vector_store",
            "vector_retriever",
            "hybrid_retriever",
            "reranker",
            "entity_linker",
            "prompt_manager",
            "tool_registry",
            "memory_manager",
            "mcp_client_manager",
            "trace_manager",
        )

        for name in standard_names:
            value = getattr(self, name)
            if value is not None:
                yield name, value

        for name, value in self.extras.items():
            if value is not None:
                yield name, value

    def summary(self) -> Dict[str, Any]:
        """返回不包含密钥和对象内容的诊断摘要。"""

        return {
            "closed": self._closed,
            "project_root": str(self.settings.project_root),
            "config_dir": str(self.settings.config_dir),
            "source_files": [str(path) for path in self.settings.source_files],
            "dependencies": sorted(name for name, _ in self.iter_dependencies()),
        }

    def close(self) -> None:
        """
        尝试释放运行时资源。

        当前多数项目组件没有 close 方法，因此本方法采用能力检测；同一对象即使被
        多个字段共享，也只关闭一次。异常只记录日志，不中断其余资源清理。
        """

        if self._closed:
            return

        seen: set[int] = set()
        dependencies = list(self.iter_dependencies())

        for name, dependency in reversed(dependencies):
            identity = id(dependency)
            if identity in seen:
                continue
            seen.add(identity)

            close_method = getattr(dependency, "close", None)
            shutdown_method = getattr(dependency, "shutdown", None)
            method = close_method if callable(close_method) else shutdown_method

            if not callable(method):
                continue

            try:
                method()
            except Exception as exc:  # 关闭阶段不能阻断其他资源释放
                self.logger.warning(
                    "Failed to close runtime dependency %s: %s",
                    name,
                    exc,
                )

        self._closed = True

    def __enter__(self) -> "RuntimeContext":
        self.ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
