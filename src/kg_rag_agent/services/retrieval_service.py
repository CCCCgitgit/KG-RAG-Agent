# -*- coding: utf-8 -*-
"""Retrieval 业务服务。

服务层只编排已有 Embedding、Vector Store、Hybrid Retriever 和 Reranker；
共享组件由 RuntimeContext 统一创建和复用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from kg_rag_agent.retrieval.embedding import DEFAULT_EMBEDDING_MODEL, EmbeddingClient
from kg_rag_agent.retrieval.entity_vector_store import EntityVectorStore
from kg_rag_agent.retrieval.hybrid_retriever import HybridRetriever
from kg_rag_agent.retrieval.reranker import Reranker
from kg_rag_agent.retrieval.vector_retriever import VectorRetriever
from kg_rag_agent.runtime import (
    RuntimeBuildOptions,
    RuntimeContext,
    RuntimeSettings,
    create_runtime,
)


@dataclass(slots=True)
class RetrievalResult:
    """检索服务统一返回结构。"""

    query: str
    results: List[Dict[str, Any]]
    count: int
    retrieval_type: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RetrievalService:
    """实体、文档、混合召回与重排的统一服务入口。"""

    def __init__(
        self,
        *,
        runtime: Optional[RuntimeContext] = None,
        config: Optional[Mapping[str, Any]] = None,
        config_path: Optional[str | Path] = None,
        validate: bool = True,
        embedding_client: Optional[EmbeddingClient] = None,
        entity_store: Optional[EntityVectorStore] = None,
        vector_retriever: Optional[VectorRetriever] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[Reranker] = None,
    ) -> None:
        if runtime is not None and (config is not None or config_path is not None):
            raise ValueError(
                "runtime cannot be combined with config or config_path."
            )

        self._owns_runtime = runtime is None
        self.runtime = runtime or self._create_runtime(
            config=config,
            config_path=config_path,
            validate=validate,
            embedding_client=embedding_client,
            entity_store=entity_store,
            vector_retriever=vector_retriever,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
        )

        # 显式注入优先于 RuntimeFactory 创建结果，便于测试和组件复用。
        if embedding_client is not None:
            self.runtime.embedding_client = embedding_client
        if entity_store is not None:
            self.runtime.entity_vector_store = entity_store
        if vector_retriever is not None:
            self.runtime.vector_retriever = vector_retriever
        if hybrid_retriever is not None:
            self.runtime.hybrid_retriever = hybrid_retriever
        if reranker is not None:
            self.runtime.reranker = reranker

        self.config = self.runtime.settings.to_dict()
        self.retrieval_config = dict(self.runtime.settings.retrieval)

    @staticmethod
    def _create_runtime(
        *,
        config: Optional[Mapping[str, Any]],
        config_path: Optional[str | Path],
        validate: bool,
        embedding_client: Optional[EmbeddingClient],
        entity_store: Optional[EntityVectorStore],
        vector_retriever: Optional[VectorRetriever],
        hybrid_retriever: Optional[HybridRetriever],
        reranker: Optional[Reranker],
    ) -> RuntimeContext:
        options = RuntimeBuildOptions(
            create_llm_client=False,
            create_embedding_client=embedding_client is None,
            create_graph_loader=False,
            create_entity_vector_store=entity_store is None,
            create_vector_retriever=vector_retriever is None,
            create_hybrid_retriever=hybrid_retriever is None,
            create_reranker=reranker is None,
            create_entity_linker=False,
            create_prompt_manager=False,
        )

        if config_path is not None and config is not None:
            settings = RuntimeSettings.load(
                config_path=config_path,
                validate=validate,
            ).with_overrides(config, validate=validate)
            return create_runtime(settings=settings, options=options)

        return create_runtime(
            config=config,
            config_path=config_path,
            validate=validate,
            options=options,
        )

    @property
    def embedding_client(self) -> Optional[EmbeddingClient]:
        return self.runtime.embedding_client

    @embedding_client.setter
    def embedding_client(self, value: Optional[EmbeddingClient]) -> None:
        self.runtime.embedding_client = value

    @property
    def entity_store(self) -> Optional[EntityVectorStore]:
        return self.runtime.entity_vector_store

    @entity_store.setter
    def entity_store(self, value: Optional[EntityVectorStore]) -> None:
        self.runtime.entity_vector_store = value

    @property
    def vector_retriever(self) -> Optional[VectorRetriever]:
        return self.runtime.vector_retriever

    @vector_retriever.setter
    def vector_retriever(self, value: Optional[VectorRetriever]) -> None:
        self.runtime.vector_retriever = value

    @property
    def hybrid_retriever(self) -> Optional[HybridRetriever]:
        return self.runtime.hybrid_retriever

    @hybrid_retriever.setter
    def hybrid_retriever(self, value: Optional[HybridRetriever]) -> None:
        self.runtime.hybrid_retriever = value

    @property
    def reranker(self) -> Optional[Reranker]:
        return self.runtime.reranker

    @reranker.setter
    def reranker(self, value: Optional[Reranker]) -> None:
        self.runtime.reranker = value

    @staticmethod
    def _get_retrieval_config(config: Dict[str, Any]) -> Dict[str, Any]:
        if "retrieval" in config and isinstance(config["retrieval"], dict):
            return dict(config["retrieval"])
        return dict(config or {})

    def _get_value(self, key: str, default: Any = None) -> Any:
        return self.retrieval_config.get(key, default)

    def get_embedding_client(self) -> EmbeddingClient:
        return self.runtime.require("embedding_client")

    def get_entity_store(self) -> EntityVectorStore:
        return self.runtime.require("entity_vector_store")

    def get_vector_retriever(self) -> VectorRetriever:
        return self.runtime.require("vector_retriever")

    def get_hybrid_retriever(self) -> HybridRetriever:
        return self.runtime.require("hybrid_retriever")

    def get_reranker(self) -> Reranker:
        return self.runtime.require("reranker")

    def search_entities(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        text = _require_query(query)
        final_top_k = _positive_int(
            top_k if top_k is not None else self._get_value("entity_top_k", 5),
            "top_k",
        )
        results = self.get_entity_store().query(
            text=text,
            top_k=final_top_k,
            where=where,
        )
        return RetrievalResult(
            query=text,
            results=list(results or []),
            count=len(results or []),
            retrieval_type="entity_vector",
            metadata={"top_k": final_top_k, "where": where},
        )

    def search_vectors(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> RetrievalResult:
        text = _require_query(query)
        final_top_k = _positive_int(
            top_k if top_k is not None else self._get_value("doc_top_k", 8),
            "top_k",
        )
        final_min_score = _float_value(
            self._get_value("min_score", 0.0)
            if min_score is None
            else min_score,
            "min_score",
        )
        results = self.get_vector_retriever().retrieve(
            query=text,
            top_k=final_top_k,
            where=where,
            min_score=final_min_score,
        )
        return RetrievalResult(
            query=text,
            results=list(results or []),
            count=len(results or []),
            retrieval_type="vector",
            metadata={
                "top_k": final_top_k,
                "where": where,
                "min_score": final_min_score,
            },
        )

    def hybrid_search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        vector_top_k: Optional[int] = None,
        entity_top_k: Optional[int] = None,
        keyword_top_k: Optional[int] = None,
        vector_where: Optional[Dict[str, Any]] = None,
        entity_where: Optional[Dict[str, Any]] = None,
        keyword_corpus: Optional[Sequence[Dict[str, Any]]] = None,
        min_score: Optional[float] = None,
    ) -> RetrievalResult:
        text = _require_query(query)
        hybrid_config = self._get_value("hybrid", {}) or {}
        default_top_k = hybrid_config.get("top_k", self._get_value("top_k", 10))
        final_top_k = _positive_int(
            top_k if top_k is not None else default_top_k,
            "top_k",
        )
        final_min_score = _float_value(
            hybrid_config.get("min_score", self._get_value("min_score", 0.0))
            if min_score is None
            else min_score,
            "min_score",
        )
        results = self.get_hybrid_retriever().retrieve(
            query=text,
            top_k=final_top_k,
            vector_top_k=vector_top_k,
            entity_top_k=entity_top_k,
            keyword_top_k=keyword_top_k,
            vector_where=vector_where,
            entity_where=entity_where,
            keyword_corpus=keyword_corpus,
            min_score=final_min_score,
        )
        return RetrievalResult(
            query=text,
            results=list(results or []),
            count=len(results or []),
            retrieval_type="hybrid",
            metadata={
                "top_k": final_top_k,
                "vector_top_k": vector_top_k,
                "entity_top_k": entity_top_k,
                "keyword_top_k": keyword_top_k,
                "min_score": final_min_score,
            },
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        *,
        top_k: Optional[int] = None,
        entities: Optional[Sequence[str]] = None,
        min_score: Optional[float] = None,
    ) -> RetrievalResult:
        text = _require_query(query)
        reranker_config = self._get_value("reranker", {}) or {}
        final_top_k = top_k if top_k is not None else reranker_config.get("top_k")
        if final_top_k is not None:
            final_top_k = _positive_int(final_top_k, "top_k")
        final_min_score = _float_value(
            reranker_config.get("min_score", 0.0)
            if min_score is None
            else min_score,
            "min_score",
        )
        results = self.get_reranker().rerank(
            query=text,
            candidates=list(candidates or []),
            top_k=final_top_k,
            entities=entities,
            min_score=final_min_score,
        )
        return RetrievalResult(
            query=text,
            results=list(results or []),
            count=len(results or []),
            retrieval_type="rerank",
            metadata={
                "top_k": final_top_k,
                "entities": list(entities or []),
                "min_score": final_min_score,
            },
        )

    def retrieve_and_rerank(
        self,
        query: str,
        *,
        retrieve_top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        entities: Optional[Sequence[str]] = None,
        keyword_corpus: Optional[Sequence[Dict[str, Any]]] = None,
        min_score: Optional[float] = None,
    ) -> RetrievalResult:
        retrieved = self.hybrid_search(
            query=query,
            top_k=retrieve_top_k,
            keyword_corpus=keyword_corpus,
            min_score=min_score,
        )
        reranked = self.rerank(
            query=query,
            candidates=retrieved.results,
            top_k=rerank_top_k,
            entities=entities,
            min_score=min_score,
        )
        return RetrievalResult(
            query=retrieved.query,
            results=reranked.results,
            count=reranked.count,
            retrieval_type="hybrid_rerank",
            metadata={
                "retrieved_count": retrieved.count,
                "reranked_count": reranked.count,
                "retrieve_top_k": retrieve_top_k,
                "rerank_top_k": rerank_top_k,
                "entities": list(entities or []),
            },
        )

    def add_entities(
        self,
        entities: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 128,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        ids = self.get_entity_store().add_entities(
            entities=list(entities or []),
            batch_size=_positive_int(batch_size, "batch_size"),
            overwrite=bool(overwrite),
        )
        return {"type": "add_entities", "ids": ids, "count": len(ids)}

    def add_documents(
        self,
        documents: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 128,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        ids = self.get_vector_retriever().add_documents(
            documents=list(documents or []),
            batch_size=_positive_int(batch_size, "batch_size"),
            overwrite=bool(overwrite),
        )
        return {"type": "add_documents", "ids": ids, "count": len(ids)}

    def info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "service": "RetrievalService",
            "embedding_model": self._get_value(
                "embedding_model", DEFAULT_EMBEDDING_MODEL
            ),
            "entity_chroma_dir": self._resolved_config_path(
                "entity_chroma_dir", "data/demo/vector_store/chroma_entity_db"
            ),
            "doc_chroma_dir": self._resolved_config_path(
                "doc_chroma_dir", "data/demo/vector_store/chroma_doc_db"
            ),
            "entity_collection_name": self._get_value(
                "entity_collection_name", "kg_entities"
            ),
            "doc_collection_name": self._get_value(
                "doc_collection_name", "kg_documents"
            ),
            "runtime": self.runtime.summary(),
        }
        try:
            info["entity_store_count"] = self.get_entity_store().count()
        except Exception as exc:
            info["entity_store_count"] = None
            info["entity_store_error"] = str(exc)
        try:
            info["doc_store_count"] = self.get_vector_retriever().count()
        except Exception as exc:
            info["doc_store_count"] = None
            info["doc_store_error"] = str(exc)
        return info

    def health_check(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "ok": not self.runtime.is_closed,
            "service": "RetrievalService",
            "components": {},
        }
        checks = {
            "embedding": lambda: self.get_embedding_client(),
            "entity_store": lambda: self.get_entity_store().count(),
            "vector_retriever": lambda: self.get_vector_retriever().count(),
        }
        for name, check in checks.items():
            try:
                check()
                status["components"][name] = "ok"
            except Exception as exc:
                status["ok"] = False
                status["components"][name] = str(exc)
        return status

    def _resolved_config_path(self, key: str, default: str) -> str:
        value = self._get_value(key, default)
        return str(self.runtime.settings.resolve_path(value))

    def close(self) -> None:
        if self._owns_runtime:
            self.runtime.close()

    def __enter__(self) -> "RetrievalService":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


_DEFAULT_RETRIEVAL_SERVICE: Optional[RetrievalService] = None


def get_default_retrieval_service(
    *,
    refresh: bool = False,
    **kwargs: Any,
) -> RetrievalService:
    global _DEFAULT_RETRIEVAL_SERVICE
    if refresh and _DEFAULT_RETRIEVAL_SERVICE is not None:
        _DEFAULT_RETRIEVAL_SERVICE.close()
        _DEFAULT_RETRIEVAL_SERVICE = None
    if _DEFAULT_RETRIEVAL_SERVICE is None:
        _DEFAULT_RETRIEVAL_SERVICE = RetrievalService(**kwargs)
    elif kwargs:
        raise ValueError(
            "Default RetrievalService already exists; use refresh=True to rebuild it."
        )
    return _DEFAULT_RETRIEVAL_SERVICE


def search_entities(query: str, **kwargs: Any) -> RetrievalResult:
    return get_default_retrieval_service().search_entities(query=query, **kwargs)


def search_vectors(query: str, **kwargs: Any) -> RetrievalResult:
    return get_default_retrieval_service().search_vectors(query=query, **kwargs)


def hybrid_search(query: str, **kwargs: Any) -> RetrievalResult:
    return get_default_retrieval_service().hybrid_search(query=query, **kwargs)


def rerank(
    query: str,
    candidates: Sequence[Dict[str, Any]],
    **kwargs: Any,
) -> RetrievalResult:
    return get_default_retrieval_service().rerank(
        query=query,
        candidates=candidates,
        **kwargs,
    )


def retrieve_and_rerank(query: str, **kwargs: Any) -> RetrievalResult:
    return get_default_retrieval_service().retrieve_and_rerank(
        query=query,
        **kwargs,
    )


def _require_query(query: Any) -> str:
    text = str(query or "").strip()
    if not text:
        raise ValueError("query must not be empty.")
    return text


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be int.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be int.") from exc
    if number <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return number


def _float_value(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be float.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be float.") from exc


__all__ = [
    "RetrievalResult",
    "RetrievalService",
    "get_default_retrieval_service",
    "search_entities",
    "search_vectors",
    "hybrid_search",
    "rerank",
    "retrieve_and_rerank",
]
