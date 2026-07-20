# -*- coding: utf-8 -*-
"""
entity_vector_store.py

实体向量库模块。

作用：
    1. 管理 Chroma 实体向量库。
    2. 支持实体写入、批量写入、查询、删除、统计。
    3. 统一实体文档、metadata、embedding 的格式。
    4. 为 entity_linker.py、vector_retriever.py、data_pipeline/vector_store_builder.py 提供底层能力。

本文件属于 retrieval 层：
    retrieval/
        entity_vector_store.py

它不负责：
    1. 用户问题路由。
    2. LangGraph 节点调度。
    3. 图结构路径查询。
    4. 最终回答生成。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ._validation import optional_mapping, positive_int, require_non_empty_text
from .errors import RetrievalDependencyError

from .embedding import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingClient,
    convert_vectors_to_list,
)


# =========================================================
# 1. 默认配置
# =========================================================

DEFAULT_CHROMA_DIR = "data/demo/vector_store/chroma_entity_db"
LEGACY_CHROMA_DIR = "data/vector_db/chroma_entity_db"
DEFAULT_COLLECTION_NAME = "kg_entities"


# =========================================================
# 2. 路径工具
# =========================================================

def get_project_root() -> Path:
    """
    获取项目根目录。

    当前文件：
        src/kg_rag_agent/retrieval/entity_vector_store.py

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
    解析 Chroma 实体向量库目录。

    优先级：
        1. 显式传入 chroma_dir
        2. 新结构 data/demo/vector_store/chroma_entity_db
        3. 旧结构 data/vector_db/chroma_entity_db
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
# 3. EntityVectorStore
# =========================================================

class EntityVectorStore:
    """
    实体向量库封装。

    用法：
        store = EntityVectorStore()
        store.add_entities([...])
        results = store.query("阿尔茨海默病", top_k=5)
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
        create_if_missing: bool = True,
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
                默认 embedding 模型名称。

            local_files_only:
                是否只加载本地模型。

            allow_hash_fallback:
                是否允许 hash embedding 兜底。

            create_if_missing:
                collection 不存在时是否创建。

            lazy_load:
                是否延迟加载 Chroma。
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

        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "chromadb is not installed. Please install chromadb first."
            ) from exc

        chroma_path = Path(self.chroma_dir)
        chroma_path.mkdir(parents=True, exist_ok=True)

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
                    "description": "Entity vector store for KG-RAG Agent",
                },
            )
            return

        self.collection = self.client.get_collection(
            name=self.collection_name,
        )

    # =====================================================
    # 3.2 写入实体
    # =====================================================

    def add_entity(
        self,
        entity: Dict[str, Any],
        *,
        overwrite: bool = True,
    ) -> str:
        """
        写入单个实体。

        Args:
            entity:
                实体字典。

                推荐字段：
                    {
                        "entity_id": "...",
                        "entity_name": "...",
                        "aliases": [...],
                        "description": "...",
                        "type": "...",
                        "metadata": {...}
                    }

            overwrite:
                Chroma upsert 表示覆盖；add 表示重复时报错。
                当前默认使用 upsert。

        Returns:
            str:
                实体向量库 ID。
        """

        ids, documents, metadatas = self._prepare_entities([entity])
        embeddings = self.embedding_client.embed_documents(documents)

        self._ensure_collection_ready()

        assert self.collection is not None

        if overwrite and hasattr(self.collection, "upsert"):
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        else:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

        return ids[0]

    def add_entities(
        self,
        entities: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 128,
        overwrite: bool = True,
    ) -> List[str]:
        """
        批量写入实体。
        """

        all_ids: List[str] = []

        for batch in batch_iter(list(entities), batch_size):
            ids, documents, metadatas = self._prepare_entities(batch)

            if not ids:
                continue

            embeddings = self.embedding_client.embed_documents(documents)

            self._ensure_collection_ready()

            assert self.collection is not None

            if overwrite and hasattr(self.collection, "upsert"):
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
            else:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )

            all_ids.extend(ids)

        return all_ids

    def _prepare_entities(
        self,
        entities: Sequence[Dict[str, Any]],
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """
        将实体对象转成 Chroma 写入格式。
        """

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for entity in entities:
            normalized = normalize_entity_record(entity)

            if not normalized:
                continue

            vector_id = normalized["vector_id"]
            document = build_entity_document(normalized)
            metadata = build_entity_metadata(normalized)

            ids.append(vector_id)
            documents.append(document)
            metadatas.append(metadata)

        return ids, documents, metadatas

    # =====================================================
    # 3.3 查询实体
    # =====================================================

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include_documents: bool = True,
        include_metadatas: bool = True,
        include_distances: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        根据文本查询实体候选。
        """

        text = normalize_query_text(text)

        if not text:
            return []

        self._ensure_collection_ready()

        assert self.collection is not None

        query_embedding = self.embedding_client.embed_query(text)

        include: List[str] = []

        if include_documents:
            include.append("documents")

        if include_metadatas:
            include.append("metadatas")

        if include_distances:
            include.append("distances")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=int(top_k),
            where=optional_mapping(where, field="where"),
            include=include,
        )

        return normalize_query_results(results)

    def query_by_embedding(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        根据向量查询实体候选。
        """

        self._ensure_collection_ready()

        assert self.collection is not None

        embedding_list = [
            safe_float(value, default=0.0)
            for value in embedding
        ]

        results = self.collection.query(
            query_embeddings=[embedding_list],
            n_results=int(top_k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return normalize_query_results(results)

    # =====================================================
    # 3.4 获取 / 删除 / 统计
    # =====================================================

    def get(
        self,
        ids: Optional[List[str]] = None,
        *,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        获取实体向量库中的记录。
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
        删除实体记录。
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

    def count(self) -> int:
        """
        返回 collection 中记录数量。
        """

        self._ensure_collection_ready()

        assert self.collection is not None

        try:
            return int(self.collection.count())
        except Exception:
            return 0

    def reset_collection(self) -> None:
        """
        删除并重建 collection。

        注意：
            这是危险操作，只应在重建向量库时使用。
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
                "description": "Entity vector store for KG-RAG Agent",
            },
        )

    def info(self) -> Dict[str, Any]:
        """
        返回向量库信息。
        """

        return {
            "chroma_dir": self.chroma_dir,
            "collection_name": self.collection_name,
            "model_name": self.model_name,
            "count": self.count() if self.collection is not None else None,
            "collection_loaded": self.collection is not None,
            "embedding": self.embedding_client.info(),
        }


    def health_check(self, *, load_backend: bool = False) -> Dict[str, Any]:
        """返回向量库健康状态；默认不触发 Chroma 加载。"""
        if load_backend:
            try: self._ensure_collection_ready()
            except Exception as exc: return {"ok": False, "error": str(exc), **self.info()}
        return {"ok": True, **self.info()}

    def close(self) -> None:
        """释放当前实例持有的 Chroma 句柄。"""
        self.collection = None
        self.client = None

# =========================================================
# 4. 实体记录标准化
# =========================================================

def normalize_entity_record(entity: Any) -> Dict[str, Any]:
    """
    将实体记录标准化。

    支持：
        1. str
        2. dict
    """

    if entity is None:
        return {}

    if isinstance(entity, str):
        entity_name = clean_entity_name(entity)

        if not entity_name:
            return {}

        entity_id = make_entity_id(entity_name)

        return {
            "vector_id": entity_id,
            "entity_id": entity_id,
            "entity_name": entity_name,
            "aliases": [],
            "description": "",
            "type": "",
            "metadata": {},
        }

    if not isinstance(entity, dict):
        return {}

    entity_name = first_non_empty_str(
        entity,
        [
            "entity_name",
            "entity",
            "name",
            "title",
            "label",
            "node",
            "node_key",
        ],
    )

    entity_name = clean_entity_name(entity_name)

    if not entity_name:
        return {}

    entity_id = first_non_empty_str(
        entity,
        [
            "entity_id",
            "id",
            "node_id",
            "qid",
            "key",
            "node_key",
        ],
    )

    if not entity_id:
        entity_id = make_entity_id(entity_name)

    vector_id = first_non_empty_str(
        entity,
        [
            "vector_id",
            "chroma_id",
        ],
    )

    if not vector_id:
        vector_id = make_entity_id(entity_id or entity_name)

    aliases = entity.get("aliases", entity.get("alias", []))

    if isinstance(aliases, str):
        aliases = [aliases]
    elif not isinstance(aliases, list):
        aliases = []

    aliases = [
        clean_entity_name(alias)
        for alias in aliases
        if clean_entity_name(alias)
    ]

    description = first_non_empty_str(
        entity,
        [
            "description",
            "desc",
            "summary",
            "text",
            "document",
        ],
    )

    entity_type = first_non_empty_str(
        entity,
        [
            "type",
            "entity_type",
            "category",
            "label_type",
        ],
    )

    metadata = entity.get("metadata", {})

    if not isinstance(metadata, dict):
        metadata = {}

    extra_metadata = {
        key: value
        for key, value in entity.items()
        if key not in {
            "vector_id",
            "chroma_id",
            "entity_id",
            "id",
            "node_id",
            "qid",
            "key",
            "node_key",
            "entity_name",
            "entity",
            "name",
            "title",
            "label",
            "aliases",
            "alias",
            "description",
            "desc",
            "summary",
            "text",
            "document",
            "type",
            "entity_type",
            "category",
            "label_type",
            "metadata",
        }
    }

    metadata.update(extra_metadata)

    return {
        "vector_id": str(vector_id),
        "entity_id": str(entity_id),
        "entity_name": entity_name,
        "aliases": aliases,
        "description": str(description or ""),
        "type": str(entity_type or ""),
        "metadata": metadata,
    }


def build_entity_document(entity: Dict[str, Any]) -> str:
    """
    构造用于 embedding 的实体文本。
    """

    entity_name = str(entity.get("entity_name", "") or "").strip()
    entity_id = str(entity.get("entity_id", "") or "").strip()
    aliases = entity.get("aliases", []) or []
    description = str(entity.get("description", "") or "").strip()
    entity_type = str(entity.get("type", "") or "").strip()

    parts: List[str] = []

    if entity_name:
        parts.append(f"实体名称：{entity_name}")

    if entity_id and entity_id != entity_name:
        parts.append(f"实体ID：{entity_id}")

    if aliases:
        parts.append("别名：" + "，".join(str(alias) for alias in aliases))

    if entity_type:
        parts.append(f"类型：{entity_type}")

    if description:
        parts.append(f"描述：{description}")

    return "\n".join(parts)


def build_entity_metadata(entity: Dict[str, Any]) -> Dict[str, Any]:
    """
    构造 Chroma metadata。

    Chroma metadata 只支持简单类型：
        str / int / float / bool / None

    因此复杂字段转 JSON 字符串。
    """

    metadata: Dict[str, Any] = {
        "entity_id": str(entity.get("entity_id", "") or ""),
        "entity_name": str(entity.get("entity_name", "") or ""),
        "type": str(entity.get("type", "") or ""),
        "aliases_json": json.dumps(
            entity.get("aliases", []) or [],
            ensure_ascii=False,
        ),
    }

    raw_metadata = entity.get("metadata", {}) or {}

    if isinstance(raw_metadata, dict):
        for key, value in raw_metadata.items():
            metadata[str(key)] = make_chroma_metadata_value(value)

    return metadata


def make_chroma_metadata_value(value: Any) -> Any:
    """
    将 metadata value 转成 Chroma 支持的类型。
    """

    if value is None:
        return ""

    if isinstance(value, (str, int, float, bool)):
        return value

    return json.dumps(value, ensure_ascii=False)


# =========================================================
# 5. 查询结果标准化
# =========================================================

def normalize_query_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    标准化 Chroma query 返回结果。
    """

    if not isinstance(results, dict):
        return []

    ids = first_nested_list(results.get("ids"))
    documents = first_nested_list(results.get("documents"))
    metadatas = first_nested_list(results.get("metadatas"))
    distances = first_nested_list(results.get("distances"))

    max_len = max(
        len(ids),
        len(documents),
        len(metadatas),
        len(distances),
    )

    normalized: List[Dict[str, Any]] = []

    for idx in range(max_len):
        chroma_id = safe_list_get(ids, idx, "")
        document = safe_list_get(documents, idx, "")
        metadata = safe_list_get(metadatas, idx, {})
        distance = safe_list_get(distances, idx, 1.0)

        if not isinstance(metadata, dict):
            metadata = {}

        entity_name = first_non_empty_str(
            metadata,
            [
                "entity_name",
                "entity",
                "name",
                "label",
                "title",
            ],
        )

        if not entity_name:
            entity_name = extract_entity_name_from_document(document)

        entity_id = first_non_empty_str(
            metadata,
            [
                "entity_id",
                "id",
                "node_id",
                "qid",
                "key",
            ],
        )

        if not entity_id:
            entity_id = str(chroma_id)

        aliases = parse_aliases_from_metadata(metadata)

        normalized.append(
            {
                "rank": idx + 1,
                "vector_id": str(chroma_id),
                "entity_id": str(entity_id),
                "entity_name": str(entity_name),
                "entity": str(entity_name),
                "document": str(document or ""),
                "aliases": aliases,
                "score": distance_to_score(distance),
                "distance": safe_float(distance, default=1.0),
                "source": "chroma",
                "metadata": metadata,
            }
        )

    return [
        item for item in normalized
        if str(item.get("entity_name", "")).strip()
    ]


def first_nested_list(value: Any) -> List[Any]:
    """
    Chroma 返回一般是二维列表：
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
    将 distance 转 score。

    简单采用：
        score = 1 - distance
    """

    distance = safe_float(distance, default=1.0)

    return clip_score(1.0 - distance)


def parse_aliases_from_metadata(metadata: Dict[str, Any]) -> List[str]:
    """
    从 metadata 中解析 aliases。
    """

    aliases_json = metadata.get("aliases_json")

    if aliases_json:
        try:
            aliases = json.loads(str(aliases_json))

            if isinstance(aliases, list):
                return [
                    str(alias)
                    for alias in aliases
                    if str(alias).strip()
                ]
        except Exception:
            pass

    aliases = metadata.get("aliases") or metadata.get("alias")

    if isinstance(aliases, list):
        return [
            str(alias)
            for alias in aliases
            if str(alias).strip()
        ]

    if isinstance(aliases, str):
        return [aliases]

    return []


def extract_entity_name_from_document(document: Any) -> str:
    """
    从 document 中提取实体名称。
    """

    text = str(document or "").strip()

    if not text:
        return ""

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("实体名称："):
            return line.replace("实体名称：", "", 1).strip()

        if line.startswith("Entity Name:"):
            return line.replace("Entity Name:", "", 1).strip()

    return text.splitlines()[0].strip()


# =========================================================
# 6. 便捷函数
# =========================================================

_GLOBAL_ENTITY_STORE: Optional[EntityVectorStore] = None


def get_default_entity_vector_store() -> EntityVectorStore:
    """
    获取全局默认实体向量库。
    """

    global _GLOBAL_ENTITY_STORE

    if _GLOBAL_ENTITY_STORE is None:
        _GLOBAL_ENTITY_STORE = EntityVectorStore()

    return _GLOBAL_ENTITY_STORE


def query_entity_vector_store(
    text: str,
    *,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    函数式接口：查询实体向量库。
    """

    store = get_default_entity_vector_store()

    return store.query(text, top_k=top_k)


def add_entities_to_vector_store(
    entities: Sequence[Dict[str, Any]],
    *,
    batch_size: int = 128,
) -> List[str]:
    """
    函数式接口：批量写入实体向量库。
    """

    store = get_default_entity_vector_store()

    return store.add_entities(
        entities,
        batch_size=batch_size,
    )


# =========================================================
# 7. 通用工具
# =========================================================

def clean_entity_name(value: Any) -> str:
    """
    清洗实体名称。
    """

    if value is None:
        return ""

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n\"'“”‘’《》<>[]【】()（）")

    return text


def normalize_query_text(value: Any) -> str:
    """
    清洗查询文本。
    """

    if value is None:
        return ""

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def make_entity_id(entity_name: str) -> str:
    """
    根据实体名称生成稳定 ID。
    """

    text = str(entity_name or "").strip()

    digest = hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()[:16]

    return f"ent_{digest}"


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
    store = EntityVectorStore(
        allow_hash_fallback=True,
    )

    test_entities = [
        {
            "entity_id": "E001",
            "entity_name": "阿尔茨海默病",
            "aliases": ["AD", "Alzheimer's disease"],
            "description": "一种常见的神经退行性疾病。",
            "type": "Disease",
        },
        {
            "entity_id": "E002",
            "entity_name": "FDG-PET",
            "aliases": ["PET"],
            "description": "一种用于观察脑代谢信息的医学影像模态。",
            "type": "Imaging",
        },
    ]

    ids = store.add_entities(test_entities)
    print("Added:", ids)

    results = store.query("Alzheimer", top_k=3)
    print(json.dumps(results, ensure_ascii=False, indent=2))