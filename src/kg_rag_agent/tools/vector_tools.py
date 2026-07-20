# -*- coding: utf-8 -*-
"""
vector_tools.py

向量检索工具适配层。

职责：
    1. 将 retrieval/ 中已经实现好的能力包装成工具函数。
    2. 为后续 Tool Calling、脚本调试、服务层组合调用提供统一入口。
    3. 只做参数适配、结果包装，不实现底层向量算法。

注意：
    本文件属于 tools 层，不实现核心检索算法。

    Embedding 由：
        retrieval/embedding.py

    实体向量库由：
        retrieval/entity_vector_store.py

    通用向量召回由：
        retrieval/vector_retriever.py

    混合召回由：
        retrieval/hybrid_retriever.py

    重排由：
        retrieval/reranker.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from kg_rag_agent.runtime.context import RuntimeContext

from kg_rag_agent.retrieval.embedding import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingClient,
    embed_documents,
    embed_query,
)
from kg_rag_agent.retrieval.entity_vector_store import (
    EntityVectorStore,
    add_entities_to_vector_store,
    query_entity_vector_store,
)
from kg_rag_agent.retrieval.hybrid_retriever import (
    HybridRetriever,
    hybrid_retrieve,
)
from kg_rag_agent.retrieval.reranker import (
    Reranker,
    rerank_results,
)
from kg_rag_agent.retrieval.vector_retriever import (
    VectorRetriever,
    retrieve_vectors,
)


# =========================================================
# 1. 工具类
# =========================================================

class VectorTools:
    """
    向量检索工具适配器。

    说明：
        本类只包装 retrieval/ 的已有能力。
        不在这里重新实现 embedding、Chroma 查询、混合召回或重排算法。
    """

    def __init__(
        self,
        *,
        runtime: Optional["RuntimeContext"] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        entity_store: Optional[EntityVectorStore] = None,
        vector_retriever: Optional[VectorRetriever] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[Reranker] = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        local_files_only: bool = True,
        allow_hash_fallback: bool = False,
    ) -> None:
        self.runtime = runtime
        self.model_name = model_name
        self.local_files_only = bool(local_files_only)
        self.allow_hash_fallback = bool(allow_hash_fallback)

        self.embedding_client = embedding_client
        self.entity_store = entity_store
        self.vector_retriever = vector_retriever
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

    # =====================================================
    # 1.1 内部组件获取
    # =====================================================

    def get_embedder(self) -> EmbeddingClient:
        """
        获取 embedding client。
        """

        if self.embedding_client is None and self.runtime is not None:
            self.embedding_client = self.runtime.embedding_client
        if self.embedding_client is None:
            self.embedding_client = EmbeddingClient(
                model_name=self.model_name,
                local_files_only=self.local_files_only,
                allow_hash_fallback=self.allow_hash_fallback,
            )

        return self.embedding_client

    def get_entity_store(self) -> EntityVectorStore:
        """
        获取实体向量库。
        """

        if self.entity_store is None and self.runtime is not None:
            self.entity_store = self.runtime.entity_vector_store
        if self.entity_store is None:
            self.entity_store = EntityVectorStore(
                model_name=self.model_name,
                local_files_only=self.local_files_only,
                allow_hash_fallback=self.allow_hash_fallback,
            )

        return self.entity_store

    def get_vector_retriever(self) -> VectorRetriever:
        """
        获取通用向量召回器。
        """

        if self.vector_retriever is None and self.runtime is not None:
            self.vector_retriever = self.runtime.vector_retriever
        if self.vector_retriever is None:
            self.vector_retriever = VectorRetriever(
                model_name=self.model_name,
                local_files_only=self.local_files_only,
                allow_hash_fallback=self.allow_hash_fallback,
            )

        return self.vector_retriever

    def get_hybrid_retriever(self) -> HybridRetriever:
        """
        获取混合召回器。
        """

        if self.hybrid_retriever is None and self.runtime is not None:
            self.hybrid_retriever = self.runtime.hybrid_retriever
        if self.hybrid_retriever is None:
            self.hybrid_retriever = HybridRetriever(
                vector_retriever=self.get_vector_retriever(),
                entity_store=self.get_entity_store(),
            )

        return self.hybrid_retriever

    def get_reranker(self) -> Reranker:
        """
        获取重排器。
        """

        if self.reranker is None and self.runtime is not None:
            self.reranker = self.runtime.reranker
        if self.reranker is None:
            self.reranker = Reranker()

        return self.reranker

    # =====================================================
    # 1.2 Embedding 工具
    # =====================================================

    def embed_text(
        self,
        text: str,
    ) -> Dict[str, Any]:
        """
        将单条文本编码为向量。
        """

        vector = self.get_embedder().embed_query(text)

        return {
            "text": text,
            "embedding": vector,
            "dimension": len(vector),
            "model_name": self.model_name,
        }

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> Dict[str, Any]:
        """
        将多条文本编码为向量。
        """

        text_list = [str(item) for item in texts]
        vectors = self.get_embedder().embed_documents(text_list)

        return {
            "texts": text_list,
            "embeddings": vectors,
            "count": len(vectors),
            "dimension": len(vectors[0]) if vectors else 0,
            "model_name": self.model_name,
        }

    # =====================================================
    # 1.3 实体向量召回工具
    # =====================================================

    def search_entities(
        self,
        query: str,
        *,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        从实体向量库中召回实体候选。
        """

        results = self.get_entity_store().query(
            text=query,
            top_k=top_k,
            where=where,
        )

        return {
            "query": query,
            "tool": "entity_vector_search",
            "results": results,
            "count": len(results),
        }

    def add_entities(
        self,
        entities: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 128,
    ) -> Dict[str, Any]:
        """
        向实体向量库写入实体。

        注意：
            该工具主要用于离线构建或调试。
            在线问答流程不应该在用户提问时重建向量库。
        """

        ids = self.get_entity_store().add_entities(
            entities=entities,
            batch_size=batch_size,
        )

        return {
            "tool": "add_entities",
            "ids": ids,
            "count": len(ids),
        }

    # =====================================================
    # 1.4 通用向量召回工具
    # =====================================================

    def search_vectors(
        self,
        query: str,
        *,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        从通用文档向量库中召回结果。
        """

        results = self.get_vector_retriever().retrieve(
            query=query,
            top_k=top_k,
            where=where,
            min_score=min_score,
        )

        return {
            "query": query,
            "tool": "vector_search",
            "results": results,
            "count": len(results),
        }

    # =====================================================
    # 1.5 混合召回工具
    # =====================================================

    def hybrid_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        vector_top_k: Optional[int] = None,
        entity_top_k: Optional[int] = None,
        keyword_top_k: Optional[int] = None,
        vector_where: Optional[Dict[str, Any]] = None,
        entity_where: Optional[Dict[str, Any]] = None,
        keyword_corpus: Optional[Sequence[Dict[str, Any]]] = None,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        执行混合召回。

        混合召回通常包括：
            1. 文档向量召回
            2. 实体向量召回
            3. 关键词召回
            4. 分数融合、去重和截断
        """

        results = self.get_hybrid_retriever().retrieve(
            query=query,
            top_k=top_k,
            vector_top_k=vector_top_k,
            entity_top_k=entity_top_k,
            keyword_top_k=keyword_top_k,
            vector_where=vector_where,
            entity_where=entity_where,
            keyword_corpus=keyword_corpus,
            min_score=min_score,
        )

        return {
            "query": query,
            "tool": "hybrid_search",
            "results": results,
            "count": len(results),
        }

    # =====================================================
    # 1.6 重排工具
    # =====================================================

    def rerank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        *,
        top_k: Optional[int] = None,
        entities: Optional[Sequence[str]] = None,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        对候选结果进行重排。
        """

        results = self.get_reranker().rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
            entities=entities,
            min_score=min_score,
        )

        return {
            "query": query,
            "tool": "rerank",
            "results": results,
            "count": len(results),
        }

    def retrieve_and_rerank(
        self,
        query: str,
        *,
        top_k: int = 10,
        final_top_k: Optional[int] = None,
        entities: Optional[Sequence[str]] = None,
        keyword_corpus: Optional[Sequence[Dict[str, Any]]] = None,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        先混合召回，再重排。
        """

        retrieved = self.hybrid_search(
            query=query,
            top_k=top_k,
            keyword_corpus=keyword_corpus,
            min_score=min_score,
        )

        reranked = self.rerank(
            query=query,
            candidates=retrieved["results"],
            top_k=final_top_k,
            entities=entities,
            min_score=min_score,
        )

        return {
            "query": query,
            "tool": "retrieve_and_rerank",
            "retrieved": retrieved["results"],
            "reranked": reranked["results"],
            "retrieved_count": retrieved["count"],
            "reranked_count": reranked["count"],
        }

    # =====================================================
    # 1.7 状态信息
    # =====================================================

    def info(self) -> Dict[str, Any]:
        """
        返回工具基础信息。
        """

        info: Dict[str, Any] = {
            "tool": "VectorTools",
            "model_name": self.model_name,
            "local_files_only": self.local_files_only,
            "allow_hash_fallback": self.allow_hash_fallback,
        }

        try:
            info["entity_store_count"] = self.get_entity_store().count()
        except Exception as exc:
            info["entity_store_count"] = None
            info["entity_store_error"] = str(exc)

        try:
            info["vector_store_count"] = self.get_vector_retriever().count()
        except Exception as exc:
            info["vector_store_count"] = None
            info["vector_store_error"] = str(exc)

        return info

    def health_check(self) -> Dict[str, Any]:
        """返回轻量组件状态，不强制加载外部后端。"""

        status: Dict[str, Any] = {
            "ok": True,
            "tool": "VectorTools",
            "runtime_attached": self.runtime is not None,
        }
        for name, value in (
            ("embedding_client", self.embedding_client),
            ("entity_store", self.entity_store),
            ("vector_retriever", self.vector_retriever),
            ("hybrid_retriever", self.hybrid_retriever),
            ("reranker", self.reranker),
        ):
            status[name] = "ready" if value is not None else "lazy"
        return status

    def close(self) -> None:
        """仅关闭由本对象直接持有且支持 close 的组件。"""

        if self.runtime is not None:
            return
        seen: set[int] = set()
        for component in (
            self.reranker,
            self.hybrid_retriever,
            self.vector_retriever,
            self.entity_store,
            self.embedding_client,
        ):
            if component is None or id(component) in seen:
                continue
            seen.add(id(component))
            close = getattr(component, "close", None)
            if callable(close):
                close()


# =========================================================
# 2. 默认工具实例
# =========================================================

_DEFAULT_VECTOR_TOOLS: Optional[VectorTools] = None


def get_default_vector_tools(
    *,
    runtime: Optional["RuntimeContext"] = None,
    refresh: bool = False,
) -> VectorTools:
    """
    获取默认 VectorTools 实例。
    """

    global _DEFAULT_VECTOR_TOOLS

    if refresh or _DEFAULT_VECTOR_TOOLS is None:
        _DEFAULT_VECTOR_TOOLS = VectorTools(runtime=runtime)

    return _DEFAULT_VECTOR_TOOLS


# =========================================================
# 3. 函数式工具入口
# =========================================================

def embed_text_tool(
    text: str,
) -> Dict[str, Any]:
    """
    单文本 embedding 工具函数。
    """

    vector = embed_query(text)

    return {
        "text": text,
        "embedding": vector,
        "dimension": len(vector),
    }


def embed_texts_tool(
    texts: Sequence[str],
) -> Dict[str, Any]:
    """
    多文本 embedding 工具函数。
    """

    vectors = embed_documents(texts)

    return {
        "texts": list(texts),
        "embeddings": vectors,
        "count": len(vectors),
        "dimension": len(vectors[0]) if vectors else 0,
    }


def entity_search_tool(
    query: str,
    *,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    实体向量召回工具函数。
    """

    results = query_entity_vector_store(
        text=query,
        top_k=top_k,
    )

    return {
        "query": query,
        "tool": "entity_vector_search",
        "results": results,
        "count": len(results),
    }


def add_entities_tool(
    entities: Sequence[Dict[str, Any]],
    *,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """
    实体向量库写入工具函数。
    """

    ids = add_entities_to_vector_store(
        entities=entities,
        batch_size=batch_size,
    )

    return {
        "tool": "add_entities",
        "ids": ids,
        "count": len(ids),
    }


def vector_search_tool(
    query: str,
    *,
    top_k: int = 5,
    where: Optional[Dict[str, Any]] = None,
    min_score: float = 0.0,
) -> Dict[str, Any]:
    """
    通用向量召回工具函数。
    """

    results = retrieve_vectors(
        query=query,
        top_k=top_k,
        where=where,
        min_score=min_score,
    )

    return {
        "query": query,
        "tool": "vector_search",
        "results": results,
        "count": len(results),
    }


def hybrid_search_tool(
    query: str,
    *,
    top_k: int = 10,
    keyword_corpus: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    混合召回工具函数。
    """

    results = hybrid_retrieve(
        query=query,
        top_k=top_k,
        keyword_corpus=keyword_corpus,
    )

    return {
        "query": query,
        "tool": "hybrid_search",
        "results": results,
        "count": len(results),
    }


def rerank_tool(
    query: str,
    candidates: Sequence[Dict[str, Any]],
    *,
    top_k: Optional[int] = None,
    entities: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    重排工具函数。
    """

    results = rerank_results(
        query=query,
        candidates=candidates,
        top_k=top_k,
        entities=entities,
    )

    return {
        "query": query,
        "tool": "rerank",
        "results": results,
        "count": len(results),
    }


def retrieve_and_rerank_tool(
    query: str,
    *,
    top_k: int = 10,
    final_top_k: Optional[int] = None,
    entities: Optional[Sequence[str]] = None,
    keyword_corpus: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    先混合召回，再重排的工具函数。
    """

    return get_default_vector_tools().retrieve_and_rerank(
        query=query,
        top_k=top_k,
        final_top_k=final_top_k,
        entities=entities,
        keyword_corpus=keyword_corpus,
    )


__all__ = [
    "VectorTools",
    "get_default_vector_tools",
    "embed_text_tool",
    "embed_texts_tool",
    "entity_search_tool",
    "add_entities_tool",
    "vector_search_tool",
    "hybrid_search_tool",
    "rerank_tool",
    "retrieve_and_rerank_tool",
]