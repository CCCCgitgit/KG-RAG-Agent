# -*- coding: utf-8 -*-
"""
runtime/factory.py

统一创建 RuntimeContext 及其共享组件。

当前工厂只复用项目已有实现，不复制 LLM、KG、Retrieval 或 Prompt 算法。
所有外部模型和数据库组件默认延迟加载，因此创建 RuntimeContext 不会立即下载
模型、连接 Chroma 或读取图谱大文件。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

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
from kg_rag_agent.utils.logger import setup_logger

from .context import RuntimeContext
from .settings import RuntimeSettings


@dataclass(frozen=True, slots=True)
class RuntimeBuildOptions:
    """控制 RuntimeFactory 创建哪些共享组件。"""

    create_llm_client: bool = True
    create_embedding_client: bool = True
    create_graph_loader: bool = True
    create_entity_vector_store: bool = True
    create_vector_retriever: bool = True
    create_hybrid_retriever: bool = True
    create_reranker: bool = True
    create_entity_linker: bool = True
    create_prompt_manager: bool = True
    create_memory_manager: bool = True

    # 仅用于显式启动预热。默认 False，避免导入或启动时加载大文件。
    preload_graph: bool = False


class RuntimeFactory:
    """根据 RuntimeSettings 构建共享运行时对象。"""

    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        options: Optional[RuntimeBuildOptions] = None,
    ) -> None:
        self.settings = settings
        self.options = options or RuntimeBuildOptions()

    def build(self) -> RuntimeContext:
        """构建 RuntimeContext。"""

        logger = setup_logger(
            config=self.settings.logging,
            reset_handlers=False,
        )

        context = RuntimeContext(
            settings=self.settings,
            logger=logger,
        )

        if self.options.create_prompt_manager:
            context.prompt_manager = self._build_prompt_manager()

        if self.options.create_llm_client:
            context.llm_client = self._build_llm_client()

        if self.options.create_memory_manager:
            context.memory_manager = self._build_memory_manager(
                llm_client=context.llm_client,
            )

        if self.options.create_embedding_client:
            context.embedding_client = self._build_embedding_client()

        if self.options.create_graph_loader:
            context.graph_loader = self._build_graph_loader()

        if self.options.create_entity_vector_store:
            context.entity_vector_store = self._build_entity_vector_store(
                embedding_client=context.embedding_client,
            )

        if self.options.create_vector_retriever:
            context.vector_retriever = self._build_vector_retriever(
                embedding_client=context.embedding_client,
            )

        if self.options.create_hybrid_retriever:
            context.hybrid_retriever = self._build_hybrid_retriever(
                vector_retriever=context.vector_retriever,
                entity_store=context.entity_vector_store,
            )

        if self.options.create_reranker:
            context.reranker = self._build_reranker()

        if self.options.create_entity_linker:
            context.entity_linker = self._build_entity_linker(
                entity_store=context.entity_vector_store,
            )

        if self.options.preload_graph and context.graph_loader is not None:
            context.graph_loader.get_graph()

        logger.debug(
            "RuntimeContext created with dependencies: %s",
            context.summary()["dependencies"],
        )

        return context

    def _build_llm_client(self) -> LLMClient:
        config = self.settings.model
        api_key_env = str(config.get("api_key_env", "") or "").strip()
        api_key = os.getenv(api_key_env) if api_key_env else None

        return LLMClient(
            provider=config.get("provider"),
            api_key=api_key,
            base_url=config.get("base_url"),
            model=config.get("model_name") or config.get("model"),
            timeout=_optional_float(config.get("timeout")),
            max_retries=_optional_int(config.get("max_retries")),
            default_temperature=_float_or_default(
                config.get("temperature"),
                0.2,
            ),
            default_max_tokens=_int_or_default(
                config.get("max_tokens"),
                1024,
            ),
            lazy_init=True,
        )

    def _build_embedding_client(self) -> EmbeddingClient:
        config = self.settings.retrieval

        return EmbeddingClient(
            model_name=str(
                config.get("embedding_model")
                or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ),
            device=config.get("device"),
            normalize_embeddings=bool(
                config.get("normalize_embeddings", True)
            ),
            batch_size=_int_or_default(config.get("batch_size"), 32),
            local_files_only=bool(config.get("local_files_only", True)),
            lazy_load=True,
            allow_hash_fallback=bool(
                config.get("allow_hash_fallback", False)
            ),
            hash_embedding_dim=_int_or_default(
                config.get("hash_embedding_dim"),
                384,
            ),
        )

    def _build_graph_loader(self) -> GraphLoader:
        kg_config = self.settings.kg
        graph_config = self.settings.graph
        graph_path = (
            kg_config.get("graph_path")
            or graph_config.get("graph_path")
            or "data/demo/kg/graph.pkl"
        )

        return GraphLoader(
            graph_path=str(self.settings.resolve_path(graph_path)),
            use_cache=bool(kg_config.get("use_cache", True)),
            validate=bool(kg_config.get("validate_graph", True)),
        )

    def _build_entity_vector_store(
        self,
        *,
        embedding_client: Optional[EmbeddingClient],
    ) -> EntityVectorStore:
        config = self.settings.retrieval
        chroma_dir = config.get(
            "entity_chroma_dir",
            "data/demo/vector_store/chroma_entity_db",
        )

        return EntityVectorStore(
            chroma_dir=str(self.settings.resolve_path(chroma_dir)),
            collection_name=str(
                config.get("entity_collection_name", "kg_entities")
            ),
            embedding_client=embedding_client,
            model_name=str(
                config.get("embedding_model")
                or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ),
            local_files_only=bool(config.get("local_files_only", True)),
            allow_hash_fallback=bool(
                config.get("allow_hash_fallback", False)
            ),
            create_if_missing=bool(
                config.get("create_entity_collection_if_missing", True)
            ),
            lazy_load=True,
        )

    def _build_vector_retriever(
        self,
        *,
        embedding_client: Optional[EmbeddingClient],
    ) -> VectorRetriever:
        config = self.settings.retrieval
        chroma_dir = config.get(
            "doc_chroma_dir",
            "data/demo/vector_store/chroma_doc_db",
        )

        return VectorRetriever(
            chroma_dir=str(self.settings.resolve_path(chroma_dir)),
            collection_name=str(
                config.get("doc_collection_name", "kg_documents")
            ),
            embedding_client=embedding_client,
            model_name=str(
                config.get("embedding_model")
                or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ),
            local_files_only=bool(config.get("local_files_only", True)),
            allow_hash_fallback=bool(
                config.get("allow_hash_fallback", False)
            ),
            create_if_missing=bool(
                config.get("create_doc_collection_if_missing", False)
            ),
            lazy_load=True,
        )

    def _build_hybrid_retriever(
        self,
        *,
        vector_retriever: Optional[VectorRetriever],
        entity_store: Optional[EntityVectorStore],
    ) -> HybridRetriever:
        config = self.settings.retrieval.get("hybrid", {}) or {}

        return HybridRetriever(
            vector_retriever=vector_retriever,
            entity_store=entity_store,
            enable_vector=bool(config.get("enable_vector", True)),
            enable_entity=bool(config.get("enable_entity", True)),
            enable_keyword=bool(config.get("enable_keyword", True)),
            vector_weight=_float_or_default(
                config.get("vector_weight"),
                0.60,
            ),
            entity_weight=_float_or_default(
                config.get("entity_weight"),
                0.30,
            ),
            keyword_weight=_float_or_default(
                config.get("keyword_weight"),
                0.10,
            ),
            fail_silently=bool(config.get("fail_silently", True)),
        )

    def _build_reranker(self) -> Reranker:
        retrieval_config = self.settings.retrieval
        config = retrieval_config.get("reranker", {}) or {}

        return Reranker(
            use_cross_encoder=bool(config.get("use_cross_encoder", False)),
            cross_encoder_model=str(
                config.get("cross_encoder_model")
                or "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
            ),
            local_files_only=bool(config.get("local_files_only", True)),
            device=config.get("device"),
            fail_silently=bool(config.get("fail_silently", True)),
            score_weight=_float_or_default(config.get("score_weight"), 0.45),
            query_overlap_weight=_float_or_default(
                config.get("query_overlap_weight"),
                0.25,
            ),
            entity_match_weight=_float_or_default(
                config.get("entity_match_weight"),
                0.15,
            ),
            source_weight=_float_or_default(
                config.get("source_weight"),
                0.10,
            ),
            length_weight=_float_or_default(
                config.get("length_weight"),
                0.05,
            ),
            lazy_load=True,
        )

    def _build_entity_linker(
        self,
        *,
        entity_store: Optional[EntityVectorStore],
    ) -> EntityLinker:
        config = self.settings.section("entity_linking")
        retrieval_config = self.settings.retrieval
        kg_config = self.settings.kg

        chroma_dir = config.get(
            "chroma_dir",
            retrieval_config.get(
                "entity_chroma_dir",
                "data/demo/vector_store/chroma_entity_db",
            ),
        )
        alias_path = config.get(
            "alias_path",
            kg_config.get("alias_path", "data/demo/processed/alias_map.json"),
        )

        linker = EntityLinker(
            chroma_dir=str(self.settings.resolve_path(chroma_dir)),
            collection_name=str(
                config.get(
                    "collection_name",
                    retrieval_config.get(
                        "entity_collection_name",
                        "kg_entities",
                    ),
                )
            ),
            model_name=str(
                config.get(
                    "model_name",
                    retrieval_config.get(
                        "embedding_model",
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    ),
                )
            ),
            alias_path=str(self.settings.resolve_path(alias_path)),
            auto_select_threshold=_float_or_default(
                config.get("auto_select_threshold"),
                0.72,
            ),
            margin_threshold=_float_or_default(
                config.get("margin_threshold"),
                0.05,
            ),
            local_files_only=bool(
                config.get(
                    "local_files_only",
                    retrieval_config.get("local_files_only", True),
                )
            ),
            lazy_load=True,
        )

        # EntityLinker 当前没有构造参数注入 EntityVectorStore。
        # 这里显式复用 Runtime 中的实例，避免后续第一次 link 时再创建第二套。
        if entity_store is not None:
            linker.vector_store = entity_store

        return linker

    def _build_prompt_manager(self) -> PromptManager:
        prompt_config_path = self.settings.config_dir / "prompt.yaml"

        return PromptManager(
            project_root=self.settings.project_root,
            config_path=prompt_config_path,
            prompt_dir=None,
            auto_load=True,
        )

    def _build_memory_manager(
        self,
        *,
        llm_client: Optional[LLMClient],
    ) -> MemoryManager:
        """创建受控 MemoryManager，并复用 Runtime 中的 LLM Client。"""

        return MemoryManager.from_config(
            self.settings.memory,
            project_root=self.settings.project_root,
            llm_client=llm_client,
        )


def create_runtime(
    *,
    settings: Optional[RuntimeSettings] = None,
    config: Optional[Mapping[str, Any]] = None,
    config_path: Optional[str | Path] = None,
    config_dir: Optional[str | Path] = None,
    project_root: Optional[str | Path] = None,
    apply_env: bool = True,
    validate: bool = True,
    ignore_missing: bool = True,
    options: Optional[RuntimeBuildOptions] = None,
) -> RuntimeContext:
    """
    创建项目共享 RuntimeContext。

    配置来源只能三选一：
        1. settings
        2. config 字典
        3. config_path / config_dir 文件配置
    """

    specified_sources = sum(
        source is not None
        for source in (settings, config, config_path)
    )
    if specified_sources > 1:
        raise ValueError(
            "Use only one of settings, config, or config_path."
        )

    if settings is None:
        if config is not None:
            settings = RuntimeSettings.from_mapping(
                config,
                project_root=project_root,
                config_dir=config_dir,
                apply_env=apply_env,
                validate=validate,
            )
        else:
            settings = RuntimeSettings.load(
                config_path=config_path,
                config_dir=config_dir,
                project_root=project_root,
                apply_env=apply_env,
                validate=validate,
                ignore_missing=ignore_missing,
            )

    return RuntimeFactory(settings, options=options).build()


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_default(value: Any, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _int_or_default(value: Any, default: int) -> int:
    parsed = _optional_int(value)
    return default if parsed is None else parsed
