# -*- coding: utf-8 -*-
"""
kg_rag_agent.retrieval

语义召回层统一导出。

本模块主要负责：
    1. embedding 模型封装
    2. 实体向量库管理
    3. 通用向量召回
    4. 混合召回
    5. 召回结果重排序

retrieval 层不负责：
    1. 图结构查询
    2. LangGraph 节点调度
    3. 最终自然语言回答生成
"""

from .errors import (
    RetrievalError, RetrievalValidationError, RetrievalConfigurationError,
    RetrievalDependencyError, RetrievalBackendError,
)
from .schemas import RetrievalItem, VectorSearchItem, EntitySearchItem, RerankItem

from .embedding import (
    EmbeddingClient,
    get_default_embedder,
    embed_query,
    embed_documents,
    clear_embedding_cache,
    normalize_text_for_embedding,
    cosine_similarity,
    dot_product,
    l2_normalize,
    build_hash_embedding,
)

from .entity_vector_store import (
    EntityVectorStore,
    get_default_entity_vector_store,
    query_entity_vector_store,
    add_entities_to_vector_store,
    normalize_entity_record,
    build_entity_document,
    build_entity_metadata,
    normalize_query_results,
    make_entity_id,
)

from .vector_retriever import (
    VectorRetriever,
    get_default_vector_retriever,
    retrieve_vectors,
    clear_vector_retriever_cache,
    normalize_chroma_query_results,
)

from .hybrid_retriever import (
    HybridRetriever,
    get_default_hybrid_retriever,
    hybrid_retrieve,
    merge_retrieval_results,
    normalize_retrieval_item,
    keyword_score,
)

from .reranker import (
    Reranker,
    get_default_reranker,
    rerank_results,
    normalize_candidates,
    normalize_candidate,
    query_overlap_score,
    entity_match_score,
    source_priority_score,
    text_length_score,
)


__all__ = [
    # embedding
    "EmbeddingClient",
    "get_default_embedder",
    "embed_query",
    "embed_documents",
    "clear_embedding_cache",
    "normalize_text_for_embedding",
    "cosine_similarity",
    "dot_product",
    "l2_normalize",
    "build_hash_embedding",

    # entity vector store
    "EntityVectorStore",
    "get_default_entity_vector_store",
    "query_entity_vector_store",
    "add_entities_to_vector_store",
    "normalize_entity_record",
    "build_entity_document",
    "build_entity_metadata",
    "normalize_query_results",
    "make_entity_id",

    # vector retriever
    "VectorRetriever",
    "get_default_vector_retriever",
    "retrieve_vectors",
    "clear_vector_retriever_cache",
    "normalize_chroma_query_results",

    # hybrid retriever
    "HybridRetriever",
    "get_default_hybrid_retriever",
    "hybrid_retrieve",
    "merge_retrieval_results",
    "normalize_retrieval_item",
    "keyword_score",

    # reranker
    "Reranker",
    "get_default_reranker",
    "rerank_results",
    "normalize_candidates",
    "normalize_candidate",
    "query_overlap_score",
    "entity_match_score",
    "source_priority_score",
    "text_length_score",

    "RetrievalError", "RetrievalValidationError", "RetrievalConfigurationError",
    "RetrievalDependencyError", "RetrievalBackendError",
    "RetrievalItem", "VectorSearchItem", "EntitySearchItem", "RerankItem",
]