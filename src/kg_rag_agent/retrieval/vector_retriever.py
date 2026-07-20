# -*- coding: utf-8 -*-
"""
vector_retriever.py

通用向量召回模块。

作用：
    1. 封装基于 Chroma 的通用向量检索能力。
    2. 支持 query -> Top-K 文档 / 事实 / 实体候选召回。
    3. 支持 metadata 过滤。
    4. 支持不同 collection。
    5. 为 hybrid_retriever.py、reranker.py、graph/nodes 提供统一语义召回接口。

本文件属于 retrieval 层：
    retrieval/
        vector_retriever.py

它不负责：
    1. 图结构查询。
    2. 实体最终落地。
    3. 多跳推理。
    4. 最终回答生成。

典型使用：
    retriever = VectorRetriever()
    results = retriever.retrieve("阿尔茨海默病和FDG-PET有什么关系？", top_k=5)
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ._validation import non_negative_float, optional_mapping, positive_int, require_non_empty_text
from .errors import RetrievalDependencyError

from .embedding import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingClient,
)


# =========================================================
# 1. 默认配置
# =========================================================

DEFAULT_CHROMA_DIR = "data/demo/vector_store/chroma_doc_db"
DEFAULT_COLLECTION_NAME = "kg_documents"

LEGACY_CHROMA_DIR = "data/vector_db/chroma_doc_db"


# =========================================================
# 2. 路径工具
# =========================================================

def get_project_root() -> Path:
    """
    获取项目根目录。

    当前文件：
        src/kg_rag_agent/retrieval/vector_retriever.py

    parents:
        0 -> retrieval/
        1 -> kg_rag_agent/
        2 -> src/
        3 -> 项目根目录
    """

    return Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path) -> Path:
    """
    解析项目路径。

    支持：
        1. 绝对路径
        2. 相对于当前工作目录
        3. 相对于项目根目录
    """

    input_path = Path(path)

    if input_path.is_absolute():
        return input_path

    cwd_path = Path.cwd() / input_path

    if cwd_path.exists():
        return cwd_path

    return get_project_root() / input_path


def resolve_chroma_dir(chroma_dir: Optional[str] = None) -> Path:
    """
    解析通用 Chroma 向量库目录。

    优先级：
        1. 显式传入 chroma_dir
        2. 新结构 data/demo/vector_store/chroma_doc_db
        3. 旧结构 data/vector_db/chroma_doc_db
    """

    if chroma_dir:
        return resolve_project_path(chroma_dir)

    new_path = resolve_project_path(DEFAULT_CHROMA_DIR)

    if new_path.exists():
        return new_path

    legacy_path = resolve_project_path(LEGACY_CHROMA_DIR)

    if legacy_path.exists():
        return legacy_path

    return new_path


# =========================================================
# 3. VectorRetriever
# =========================================================

class VectorRetriever:
    """
    通用向量召回器。

    用法：
        retriever = VectorRetriever()
        results = retriever.retrieve("query", top_k=5)
    """

    def __init__(
        self,
        chroma_dir: Optional[str] = None,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_client: Optional[EmbeddingClient] = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        local_files_only: bool = True,
        allow_hash_fallback: bool = False,
        create_if_missing: bool = False,
        lazy_load: bool = True,
    ) -> None:
        """
        Args:
            chroma_dir:
                Chroma 持久化目录。

            collection_name:
                collection 名称。

            embedding_client:
                可选外部传入 embedding client。

            model_name:
                embedding 模型名称。

            local_files_only:
                是否只加载本地模型。

            allow_hash_fallback:
                是否允许 hash embedding 兜底。

            create_if_missing:
                collection 不存在时是否创建。
                检索器默认 False，避免误创建空库。

            lazy_load:
                是否延迟加载 Chroma collection。
        """

        self.chroma_dir = str(resolve_chroma_dir(chroma_dir))
        self.collection_name = require_non_empty_text(collection_name, field="collection_name")
        self.model_name = require_non_empty_text(model_name, field="model_name")
        self.local_files_only = bool(local_files_only)
        self.allow_hash_fallback = bool(allow_hash_fallback)
        self.create_if_missing = bool(create_if_missing)
        self.lazy_load = bool(lazy_load)

        self.embedding_client = embedding_client or EmbeddingClient(
            model_name=model_name,
            local_files_only=local_files_only,
            allow_hash_fallback=allow_hash_fallback,
        )

        self.client: Optional[Any] = None
        self.collection: Optional[Any] = None

        if not self.lazy_load:
            self._ensure_collection_ready()

    # =====================================================
    # 3.1 Collection 加载
    # =====================================================

    def _ensure_collection_ready(self) -> None:
        """
        确保 Chroma collection 已加载。
        """

        if self.collection is not None:
            return

        self._load_client()
        self._load_collection()

    def _load_client(self) -> None:
        """
        加载 Chroma PersistentClient。
        """

        if self.client is not None:
            return

        chroma_path = Path(self.chroma_dir)

        if not chroma_path.exists() and not self.create_if_missing:
            raise FileNotFoundError(
                f"Chroma vector store directory not found: {chroma_path}"
            )

        chroma_path.mkdir(parents=True, exist_ok=True)

        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "chromadb is not installed. Please install chromadb first."
            ) from exc

        self.client = chromadb.PersistentClient(path=str(chroma_path))

    def _load_collection(self) -> None:
        """
        加载或创建 collection。
        """

        if self.client is None:
            self._load_client()

        if self.collection is not None:
            return

        assert self.client is not None

        if self.create_if_missing:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Generic vector store for KG-RAG Agent",
                },
            )
            return

        self.collection = self.client.get_collection(
            name=self.collection_name,
        )

    # =====================================================
    # 3.2 召回接口
    # =====================================================

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
        include_embeddings: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        根据 query 进行向量召回。

        Args:
            query:
                用户查询或语义查询文本。

            top_k:
                返回条数。

            where:
                Chroma metadata 过滤条件。

            min_score:
                最小相似度分数。

            include_embeddings:
                是否返回 embeddings。
                一般不需要，默认 False。

        Returns:
            List[Dict[str, Any]]:
                [
                    {
                        "rank": 1,
                        "id": "...",
                        "text": "...",
                        "score": 0.82,
                        "distance": 0.18,
                        "metadata": {...},
                        "source": "chroma"
                    }
                ]
        """

        query = normalize_query_text(query)

        if not query:
            return []

        top_k = positive_int(top_k, field="top_k")
        min_score = non_negative_float(min_score, field="min_score")

        self._ensure_collection_ready()

        assert self.collection is not None

        query_embedding = self.embedding_client.embed_query(query)

        include = ["documents", "metadatas", "distances"]

        if include_embeddings:
            include.append("embeddings")

        raw_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=optional_mapping(where, field="where"),
            include=include,
        )

        results = normalize_chroma_query_results(raw_results)

        if min_score > 0:
            results = [
                item for item in results
                if safe_float(item.get("score", 0.0), default=0.0) >= min_score
            ]

        return results

    def retrieve_by_embedding(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        根据已有向量进行召回。
        """

        if not embedding:
            return []

        top_k = positive_int(top_k, field="top_k")

        self._ensure_collection_ready()

        assert self.collection is not None

        embedding_list = [
            safe_float(value, default=0.0)
            for value in embedding
        ]

        raw_results = self.collection.query(
            query_embeddings=[embedding_list],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        results = normalize_chroma_query_results(raw_results)

        if min_score > 0:
            results = [
                item for item in results
                if safe_float(item.get("score", 0.0), default=0.0) >= min_score
            ]

        return results

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        LangChain 风格别名接口。
        """

        return self.retrieve(
            query,
            top_k=k,
            **kwargs,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        retrieve() 的别名。
        """

        return self.retrieve(
            query,
            top_k=top_k,
            **kwargs,
        )

    # =====================================================
    # 3.3 写入接口
    # =====================================================

    def add_texts(
        self,
        texts: Sequence[str],
        *,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
        batch_size: int = 128,
        overwrite: bool = True,
    ) -> List[str]:
        """
        批量写入文本。

        Args:
            texts:
                文本列表。

            metadatas:
                metadata 列表。

            ids:
                可选自定义 ID。

            batch_size:
                批量大小。

            overwrite:
                True 使用 upsert，False 使用 add。
        """

        clean_texts = [
            normalize_document_text(text)
            for text in texts
        ]

        records = []

        for idx, text in enumerate(clean_texts):
            if not text:
                continue

            metadata = {}

            if metadatas and idx < len(metadatas):
                metadata = metadatas[idx] or {}

            if not isinstance(metadata, dict):
                metadata = {}

            record_id = ""

            if ids and idx < len(ids):
                record_id = str(ids[idx] or "").strip()

            if not record_id:
                record_id = make_text_id(text, metadata)

            records.append(
                {
                    "id": record_id,
                    "text": text,
                    "metadata": make_chroma_metadata(metadata),
                }
            )

        all_ids: List[str] = []

        for batch in batch_iter(records, batch_size):
            batch_ids = [item["id"] for item in batch]
            batch_texts = [item["text"] for item in batch]
            batch_metadatas = [item["metadata"] for item in batch]

            embeddings = self.embedding_client.embed_documents(batch_texts)

            self._ensure_collection_ready()

            assert self.collection is not None

            if overwrite and hasattr(self.collection, "upsert"):
                self.collection.upsert(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    embeddings=embeddings,
                )
            else:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    embeddings=embeddings,
                )

            all_ids.extend(batch_ids)

        return all_ids

    def add_documents(
        self,
        documents: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 128,
        overwrite: bool = True,
    ) -> List[str]:
        """
        批量写入文档字典。

        支持字段：
            {
                "id": "...",
                "text": "...",
                "content": "...",
                "metadata": {...}
            }
        """

        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for doc in documents:
            if not isinstance(doc, dict):
                continue

            text = first_non_empty_str(
                doc,
                ["text", "content", "document", "page_content", "body"],
            )

            text = normalize_document_text(text)

            if not text:
                continue

            metadata = doc.get("metadata", {})

            if not isinstance(metadata, dict):
                metadata = {}

            extra_metadata = {
                key: value
                for key, value in doc.items()
                if key not in {
                    "id",
                    "doc_id",
                    "text",
                    "content",
                    "document",
                    "page_content",
                    "body",
                    "metadata",
                }
            }

            metadata.update(extra_metadata)

            doc_id = first_non_empty_str(
                doc,
                ["id", "doc_id", "document_id"],
            )

            if not doc_id:
                doc_id = make_text_id(text, metadata)

            texts.append(text)
            metadatas.append(metadata)
            ids.append(doc_id)

        return self.add_texts(
            texts,
            metadatas=metadatas,
            ids=ids,
            batch_size=batch_size,
            overwrite=overwrite,
        )

    # =====================================================
    # 3.4 管理接口
    # =====================================================

    def count(self) -> int:
        """
        返回 collection 记录数量。
        """

        self._ensure_collection_ready()

        assert self.collection is not None

        try:
            return int(self.collection.count())
        except Exception:
            return 0

    def get(
        self,
        ids: Optional[List[str]] = None,
        *,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        获取 collection 中的记录。
        """

        self._ensure_collection_ready()

        assert self.collection is not None

        kwargs: Dict[str, Any] = {
            "include": ["documents", "metadatas"],
        }

        if ids is not None:
            kwargs["ids"] = ids

        if where is not None:
            kwargs["where"] = where

        if limit is not None:
            kwargs["limit"] = int(limit)

        return self.collection.get(**kwargs)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        *,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        删除 collection 中的记录。
        """

        self._ensure_collection_ready()

        assert self.collection is not None

        kwargs: Dict[str, Any] = {}

        if ids is not None:
            kwargs["ids"] = ids

        if where is not None:
            kwargs["where"] = where

        if not kwargs:
            raise ValueError("delete requires ids or where.")

        self.collection.delete(**kwargs)

    def reset_collection(self) -> None:
        """
        删除并重建 collection。

        注意：
            这是危险操作，只应该在重建向量库时使用。
        """

        self._load_client()

        assert self.client is not None

        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "Generic vector store for KG-RAG Agent",
            },
        )

    def info(self) -> Dict[str, Any]:
        """
        返回检索器信息。
        """

        info = {
            "chroma_dir": self.chroma_dir,
            "collection_name": self.collection_name,
            "model_name": self.model_name,
            "local_files_only": self.local_files_only,
            "allow_hash_fallback": self.allow_hash_fallback,
            "create_if_missing": self.create_if_missing,
            "collection_loaded": self.collection is not None,
            "embedding": self.embedding_client.info(),
        }

        if self.collection is not None:
            info["count"] = self.count()

        return info


    def health_check(self, *, load_backend: bool = False) -> Dict[str, Any]:
        """返回检索器健康状态；默认不触发 Chroma 加载。"""
        if load_backend:
            try: self._ensure_collection_ready()
            except Exception as exc: return {"ok": False, "error": str(exc), **self.info()}
        return {"ok": True, **self.info()}

    def close(self) -> None:
        """释放当前实例持有的 Chroma 句柄。"""
        self.collection = None
        self.client = None

# =========================================================
# 4. 查询结果标准化
# =========================================================

def normalize_chroma_query_results(
    results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    标准化 Chroma query 返回结果。
    """

    if not isinstance(results, dict):
        return []

    ids = first_nested_list(results.get("ids"))
    documents = first_nested_list(results.get("documents"))
    metadatas = first_nested_list(results.get("metadatas"))
    distances = first_nested_list(results.get("distances"))
    embeddings = first_nested_list(results.get("embeddings"))

    max_len = max(
        len(ids),
        len(documents),
        len(metadatas),
        len(distances),
        len(embeddings),
    )

    normalized: List[Dict[str, Any]] = []

    for idx in range(max_len):
        item_id = safe_list_get(ids, idx, "")
        text = safe_list_get(documents, idx, "")
        metadata = safe_list_get(metadatas, idx, {})
        distance = safe_list_get(distances, idx, 1.0)
        embedding = safe_list_get(embeddings, idx, None)

        if not isinstance(metadata, dict):
            metadata = {}

        text = str(text or "").strip()

        if not text:
            continue

        normalized_item = {
            "rank": idx + 1,
            "id": str(item_id or ""),
            "text": text,
            "content": text,
            "score": distance_to_score(distance),
            "distance": safe_float(distance, default=1.0),
            "metadata": metadata,
            "source": "chroma",
        }

        if embedding is not None:
            normalized_item["embedding"] = embedding

        normalized.append(normalized_item)

    return normalized


def first_nested_list(value: Any) -> List[Any]:
    """
    Chroma 返回通常是二维列表：
        [["a", "b"]]

    本函数取第一层。
    """

    if value is None:
        return []

    if not isinstance(value, list):
        return []

    if not value:
        return []

    first = value[0]

    if isinstance(first, list):
        return first

    return value


def safe_list_get(
    items: List[Any],
    index: int,
    default: Any,
) -> Any:
    """
    安全读取 list[index]。
    """

    try:
        return items[index]
    except Exception:
        return default


def distance_to_score(distance: Any) -> float:
    """
    将 Chroma distance 转成 score。

    这里采用简单转换：
        score = 1 - distance

    并限制在 [0, 1]。
    """

    distance = safe_float(distance, default=1.0)

    return clip_score(1.0 - distance)


# =========================================================
# 5. 便捷函数
# =========================================================

_GLOBAL_RETRIEVER_CACHE: Dict[str, VectorRetriever] = {}


def get_default_vector_retriever(
    *,
    chroma_dir: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> VectorRetriever:
    """
    获取默认 VectorRetriever。
    """

    resolved_dir = str(resolve_chroma_dir(chroma_dir))

    cache_key = f"{resolved_dir}::{collection_name}::{model_name}"

    if cache_key in _GLOBAL_RETRIEVER_CACHE:
        return _GLOBAL_RETRIEVER_CACHE[cache_key]

    retriever = VectorRetriever(
        chroma_dir=resolved_dir,
        collection_name=collection_name,
        model_name=model_name,
    )

    _GLOBAL_RETRIEVER_CACHE[cache_key] = retriever

    return retriever


def retrieve_vectors(
    query: str,
    *,
    top_k: int = 5,
    chroma_dir: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    where: Optional[Dict[str, Any]] = None,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    函数式接口：执行向量召回。
    """

    retriever = get_default_vector_retriever(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
    )

    return retriever.retrieve(
        query,
        top_k=top_k,
        where=where,
        min_score=min_score,
    )


def clear_vector_retriever_cache() -> None:
    """
    清空全局 retriever 缓存。
    """

    _GLOBAL_RETRIEVER_CACHE.clear()


# =========================================================
# 6. 文本与 metadata 工具
# =========================================================

def normalize_query_text(value: Any) -> str:
    """
    查询文本清洗。
    """

    if value is None:
        return ""

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def normalize_document_text(value: Any) -> str:
    """
    文档文本清洗。
    """

    if value is None:
        return ""

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def make_text_id(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    根据文本和 metadata 生成稳定 ID。
    """

    raw = {
        "text": str(text or ""),
        "metadata": metadata or {},
    }

    raw_text = json.dumps(
        raw,
        ensure_ascii=False,
        sort_keys=True,
    )

    import hashlib

    digest = hashlib.md5(
        raw_text.encode("utf-8")
    ).hexdigest()[:16]

    return f"doc_{digest}"


def make_chroma_metadata(
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    将 metadata 转成 Chroma 支持的简单类型。

    Chroma metadata 支持：
        str / int / float / bool / None

    复杂对象转 JSON 字符串。
    """

    if not isinstance(metadata, dict):
        return {}

    result: Dict[str, Any] = {}

    for key, value in metadata.items():
        result[str(key)] = make_chroma_metadata_value(value)

    return result


def make_chroma_metadata_value(value: Any) -> Any:
    """
    metadata value 标准化。
    """

    if value is None:
        return ""

    if isinstance(value, (str, int, float, bool)):
        return value

    return json.dumps(value, ensure_ascii=False)


def first_non_empty_str(
    item: Dict[str, Any],
    keys: List[str],
) -> str:
    """
    从 dict 中取第一个非空字符串字段。
    """

    for key in keys:
        value = item.get(key)

        if value is None:
            continue

        value = str(value).strip()

        if value:
            return value

    return ""


# =========================================================
# 7. 批处理与数值工具
# =========================================================

def batch_iter(
    items: Sequence[Any],
    batch_size: int,
) -> Iterable[List[Any]]:
    """
    批量迭代。
    """

    batch_size = max(int(batch_size), 1)

    for start in range(0, len(items), batch_size):
        yield list(items[start:start + batch_size])


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    安全转 float。
    """

    try:
        if value is None:
            return default

        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return default

        return value

    except Exception:
        return default


def clip_score(score: float) -> float:
    """
    将分数限制在 [0, 1]。
    """

    if score < 0:
        return 0.0

    if score > 1:
        return 1.0

    return score


# =========================================================
# 8. 快速测试入口
# =========================================================

if __name__ == "__main__":
    retriever = VectorRetriever(
        create_if_missing=True,
        allow_hash_fallback=True,
    )

    docs = [
        {
            "id": "doc_1",
            "text": "阿尔茨海默病是一种神经退行性疾病。",
            "metadata": {
                "source": "demo",
                "type": "disease",
            },
        },
        {
            "id": "doc_2",
            "text": "FDG-PET 可以反映脑代谢信息。",
            "metadata": {
                "source": "demo",
                "type": "imaging",
            },
        },
    ]

    retriever.add_documents(docs)

    results = retriever.retrieve(
        "阿尔茨海默病和脑代谢有什么关系？",
        top_k=3,
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))