# -*- coding: utf-8 -*-
"""
vector_builder.py

实体向量库构建模块。

职责：
    1. 优先读取 data/processed/entities.json。
    2. 如果 entities.json 不存在，则从 data/kg/graph.pkl 中抽取实体节点。
    3. 将实体标准化为 EntityVectorStore 可写入的 record。
    4. 写入 data/vector_store/chroma_entity_db。
    5. 保存 data/vector_store/vector_store_stats.json。

注意：
    本文件属于 data_pipeline 层，只做离线向量库构建。
    不参与在线问答流程。
    不调用 LangGraph node、不调用 LLM、不生成最终回答。
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_KG_DIR = "data/kg"
DEFAULT_VECTOR_STORE_DIR = "data/vector_store"

DEFAULT_ENTITIES_FILE = "entities.json"
DEFAULT_GRAPH_FILE = "graph.pkl"
DEFAULT_CHROMA_ENTITY_DIR = "chroma_entity_db"
DEFAULT_VECTOR_STATS_FILE = "vector_store_stats.json"

DEFAULT_COLLECTION_NAME = "kg_entities"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# =========================================================
# 1. 路径工具
# =========================================================

def get_project_root() -> Path:
    """
    获取项目根目录。

    当前文件：
        src/kg_rag_agent/data_pipeline/vector_builder.py

    parents:
        0 -> data_pipeline/
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


def ensure_dir(path: str | Path) -> Path:
    """
    确保目录存在。
    """

    resolved = resolve_project_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_json_if_exists(path: str | Path, default: Any) -> Any:
    """
    如果 JSON 文件存在，则读取；否则返回 default。
    """

    resolved = resolve_project_path(path)

    if not resolved.exists():
        return default

    try:
        with resolved.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(data: Any, path: str | Path) -> None:
    """
    保存 JSON。
    """

    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    with resolved.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


# =========================================================
# 2. 路径配置
# =========================================================

@dataclass
class VectorBuilderPaths:
    """
    VectorBuilder 使用的路径集合。
    """

    processed_dir: Path
    kg_dir: Path
    vector_store_dir: Path

    entities_file: Path
    graph_file: Path
    chroma_entity_dir: Path
    stats_file: Path

    @classmethod
    def build(
        cls,
        *,
        processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
        kg_dir: str | Path = DEFAULT_KG_DIR,
        vector_store_dir: str | Path = DEFAULT_VECTOR_STORE_DIR,
        entities_file: str = DEFAULT_ENTITIES_FILE,
        graph_file: str = DEFAULT_GRAPH_FILE,
        chroma_entity_dir: str = DEFAULT_CHROMA_ENTITY_DIR,
        stats_file: str = DEFAULT_VECTOR_STATS_FILE,
    ) -> "VectorBuilderPaths":
        """
        构造路径集合。
        """

        processed_path = resolve_project_path(processed_dir)
        kg_path = resolve_project_path(kg_dir)
        vector_path = resolve_project_path(vector_store_dir)

        return cls(
            processed_dir=processed_path,
            kg_dir=kg_path,
            vector_store_dir=vector_path,
            entities_file=processed_path / entities_file,
            graph_file=kg_path / graph_file,
            chroma_entity_dir=vector_path / chroma_entity_dir,
            stats_file=vector_path / stats_file,
        )


# =========================================================
# 3. 实体加载
# =========================================================

def load_entities_json(path: str | Path) -> List[Dict[str, Any]]:
    """
    从 entities.json 加载实体。
    """

    data = load_json_if_exists(path, default=[])

    if not isinstance(data, list):
        return []

    entities: List[Dict[str, Any]] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        entity = normalize_entity_record(item)

        if entity:
            entities.append(entity)

    return entities


def load_graph_pickle(path: str | Path) -> Any:
    """
    加载 graph.pkl。
    """

    resolved = resolve_project_path(path)

    if not resolved.exists():
        raise FileNotFoundError(f"Graph file not found: {resolved}")

    with resolved.open("rb") as f:
        return pickle.load(f)


def extract_entities_from_graph(graph: Any) -> List[Dict[str, Any]]:
    """
    从图对象中抽取实体。

    支持：
        1. NetworkX graph
        2. dict graph
    """

    entities: List[Dict[str, Any]] = []

    if hasattr(graph, "nodes"):
        for node_key, attrs in graph.nodes(data=True):
            entity = normalize_node_record(
                node_key=node_key,
                attrs=dict(attrs or {}),
            )
            if entity:
                entities.append(entity)

        return deduplicate_entities(entities)

    if isinstance(graph, dict):
        nodes = graph.get("nodes", {})

        if isinstance(nodes, dict):
            for node_key, attrs in nodes.items():
                entity = normalize_node_record(
                    node_key=node_key,
                    attrs=dict(attrs or {}),
                )
                if entity:
                    entities.append(entity)

        elif isinstance(nodes, list):
            for item in nodes:
                if not isinstance(item, dict):
                    continue

                node_key = (
                    item.get("node_key")
                    or item.get("id")
                    or item.get("entity_id")
                    or item.get("name")
                    or item.get("entity_name")
                )

                entity = normalize_node_record(
                    node_key=node_key,
                    attrs=dict(item),
                )
                if entity:
                    entities.append(entity)

        return deduplicate_entities(entities)

    raise TypeError(
        f"Unsupported graph type: {type(graph)}. "
        "Expected NetworkX graph or dict graph."
    )


def load_entities_from_best_source(
    *,
    entities_file: str | Path,
    graph_file: str | Path,
) -> List[Dict[str, Any]]:
    """
    优先从 entities.json 加载实体。
    如果 entities.json 不存在或为空，则从 graph.pkl 抽取。
    """

    entities = load_entities_json(entities_file)

    if entities:
        return entities

    graph = load_graph_pickle(graph_file)
    return extract_entities_from_graph(graph)


# =========================================================
# 4. 实体标准化
# =========================================================

def normalize_entity_record(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    标准化实体 record。

    输出字段与 retrieval/entity_vector_store.py 兼容：
        entity_id
        entity_name
        name
        node_key
        type
        aliases
        description
        metadata
    """

    entity_id = first_non_empty(
        item,
        [
            "entity_id",
            "id",
            "node_id",
            "node_key",
            "name",
            "entity_name",
        ],
    )

    entity_name = first_non_empty(
        item,
        [
            "entity_name",
            "name",
            "label",
            "title",
            "node_key",
            "entity_id",
        ],
    )

    node_key = first_non_empty(
        item,
        [
            "node_key",
            "entity_id",
            "id",
            "entity_name",
            "name",
        ],
    )

    entity_type = first_non_empty(
        item,
        [
            "type",
            "entity_type",
            "label_type",
            "category",
        ],
        default="entity",
    )

    description = first_non_empty(
        item,
        [
            "description",
            "desc",
            "summary",
            "text",
        ],
        default="",
    )

    aliases = normalize_aliases(item.get("aliases") or item.get("alias") or [])

    if not entity_name and not entity_id:
        return {}

    if not entity_id:
        entity_id = entity_name

    if not entity_name:
        entity_name = entity_id

    if not node_key:
        node_key = entity_id

    metadata = item.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    normalized = {
        "entity_id": str(entity_id),
        "entity_name": str(entity_name),
        "name": str(entity_name),
        "node_key": str(node_key),
        "type": str(entity_type or "entity"),
        "aliases": aliases,
        "description": str(description or ""),
        "metadata": {
            **metadata,
            "node_key": str(node_key),
            "entity_id": str(entity_id),
            "entity_name": str(entity_name),
            "type": str(entity_type or "entity"),
            "source": metadata.get("source", "vector_builder"),
        },
    }

    return normalized


def normalize_node_record(
    *,
    node_key: Any,
    attrs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    将图节点标准化为实体 record。
    """

    node_key_str = str(node_key or "").strip()

    data = {
        **attrs,
        "node_key": attrs.get("node_key") or node_key_str,
    }

    return normalize_entity_record(data)


def normalize_aliases(value: Any) -> List[str]:
    """
    标准化 aliases。
    """

    aliases: List[str] = []

    if value is None:
        return aliases

    if isinstance(value, str):
        raw_parts = value.replace("；", ";").replace(",", ";").split(";")
        aliases.extend(
            part.strip()
            for part in raw_parts
            if part.strip()
        )
        return deduplicate_keep_order(aliases)

    if isinstance(value, (list, tuple, set)):
        for item in value:
            item_text = str(item).strip()

            if item_text:
                aliases.append(item_text)

        return deduplicate_keep_order(aliases)

    text = str(value).strip()

    if text:
        aliases.append(text)

    return deduplicate_keep_order(aliases)


def first_non_empty(
    item: Dict[str, Any],
    keys: Sequence[str],
    *,
    default: str = "",
) -> str:
    """
    从字典中取第一个非空字段。
    """

    for key in keys:
        value = item.get(key)

        if value is None:
            continue

        text = str(value).strip()

        if text:
            return text

    return default


def deduplicate_keep_order(items: Iterable[str]) -> List[str]:
    """
    保序去重。
    """

    seen: set[str] = set()
    result: List[str] = []

    for item in items:
        text = str(item).strip()

        if not text:
            continue

        key = text.lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(text)

    return result


def deduplicate_entities(entities: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按 entity_id / node_key / entity_name 对实体去重。
    """

    seen: set[str] = set()
    result: List[Dict[str, Any]] = []

    for entity in entities:
        if not entity:
            continue

        key = (
            str(entity.get("entity_id") or "").lower()
            or str(entity.get("node_key") or "").lower()
            or str(entity.get("entity_name") or "").lower()
        )

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)
        result.append(entity)

    return result


# =========================================================
# 5. 向量库构建
# =========================================================

def reset_directory(path: str | Path) -> Path:
    """
    删除并重建目录。
    """

    resolved = resolve_project_path(path)

    if resolved.exists():
        shutil.rmtree(resolved)

    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def safe_store_count(store: Any) -> Optional[int]:
    """
    安全获取向量库记录数量。
    """

    try:
        return int(store.count())
    except Exception:
        return None


def write_entities_to_vector_store(
    entities: List[Dict[str, Any]],
    *,
    chroma_dir: str | Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    local_files_only: bool = True,
    allow_hash_fallback: bool = False,
    batch_size: int = 128,
    reset: bool = True,
    store_factory: Optional[Any] = None,
    embedding_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    写入实体向量库。
    """

    from kg_rag_agent.retrieval import EntityVectorStore

    chroma_path = resolve_project_path(chroma_dir)

    if reset:
        reset_directory(chroma_path)
    else:
        chroma_path.mkdir(parents=True, exist_ok=True)

    if not entities:
        return {
            "ok": False,
            "reason": "no_entities",
            "chroma_dir": chroma_path.as_posix(),
            "collection_name": collection_name,
            "num_entities": 0,
            "num_written": 0,
            "collection_count": None,
        }

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    factory = store_factory or EntityVectorStore
    store_kwargs: Dict[str, Any] = {
        "chroma_dir": str(chroma_path),
        "collection_name": collection_name,
        "model_name": model_name,
        "local_files_only": local_files_only,
        "allow_hash_fallback": allow_hash_fallback,
        "create_if_missing": True,
        "lazy_load": False,
    }
    if embedding_client is not None:
        store_kwargs["embedding_client"] = embedding_client
    store = factory(**store_kwargs)

    ids = store.add_entities(
        entities,
        batch_size=batch_size,
        overwrite=True,
    )

    count = safe_store_count(store)

    return {
        "ok": True,
        "chroma_dir": chroma_path.as_posix(),
        "collection_name": collection_name,
        "model_name": model_name,
        "local_files_only": bool(local_files_only),
        "allow_hash_fallback": bool(allow_hash_fallback),
        "batch_size": int(batch_size),
        "reset": bool(reset),
        "num_entities": len(entities),
        "num_written": len(ids),
        "collection_count": count,
    }


# =========================================================
# 6. VectorBuilder 类
# =========================================================

class VectorBuilder:
    """
    离线实体向量库构建器。

    用法：
        builder = VectorBuilder()
        stats = builder.build(allow_hash_fallback=True)
    """

    def __init__(
        self,
        *,
        processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
        kg_dir: str | Path = DEFAULT_KG_DIR,
        vector_store_dir: str | Path = DEFAULT_VECTOR_STORE_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        store_factory: Optional[Any] = None,
        embedding_client: Optional[Any] = None,
    ) -> None:
        self.paths = VectorBuilderPaths.build(
            processed_dir=processed_dir,
            kg_dir=kg_dir,
            vector_store_dir=vector_store_dir,
        )
        self.collection_name = collection_name
        self.model_name = model_name
        self.store_factory = store_factory
        self.embedding_client = embedding_client

    def load_entities(self) -> List[Dict[str, Any]]:
        """
        加载实体。
        """

        return load_entities_from_best_source(
            entities_file=self.paths.entities_file,
            graph_file=self.paths.graph_file,
        )

    def save_entities_snapshot(
        self,
        entities: List[Dict[str, Any]],
    ) -> None:
        """
        保存 entities.json 快照。

        如果实体来自 graph.pkl，则这里会补出 entities.json。
        如果实体本来来自 entities.json，则相当于标准化覆盖保存。
        """

        save_json(
            entities,
            self.paths.entities_file,
        )

    def build(
        self,
        *,
        local_files_only: bool = True,
        allow_hash_fallback: bool = False,
        batch_size: int = 128,
        reset: bool = True,
        save_entities: bool = True,
    ) -> Dict[str, Any]:
        """
        构建实体向量库。
        """

        ensure_dir(self.paths.vector_store_dir)

        source = (
            "entities_json"
            if self.paths.entities_file.exists() and load_entities_json(self.paths.entities_file)
            else "graph_pickle"
        )
        entities = self.load_entities()

        if save_entities:
            self.save_entities_snapshot(entities)

        stats = write_entities_to_vector_store(
            entities,
            chroma_dir=self.paths.chroma_entity_dir,
            collection_name=self.collection_name,
            model_name=self.model_name,
            local_files_only=local_files_only,
            allow_hash_fallback=allow_hash_fallback,
            batch_size=batch_size,
            reset=reset,
            store_factory=self.store_factory,
            embedding_client=self.embedding_client,
        )

        stats.update(
            {
                "entities_file": self.paths.entities_file.as_posix(),
                "graph_file": self.paths.graph_file.as_posix(),
                "vector_store_dir": self.paths.vector_store_dir.as_posix(),
                "stats_file": self.paths.stats_file.as_posix(),
                "source": source,
            }
        )

        save_json(
            stats,
            self.paths.stats_file,
        )

        return stats


# =========================================================
# 7. 函数式入口
# =========================================================

def build_vector_store(
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    kg_dir: str | Path = DEFAULT_KG_DIR,
    vector_store_dir: str | Path = DEFAULT_VECTOR_STORE_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    local_files_only: bool = True,
    allow_hash_fallback: bool = False,
    batch_size: int = 128,
    reset: bool = True,
    store_factory: Optional[Any] = None,
    embedding_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    函数式实体向量库构建入口。
    """

    builder = VectorBuilder(
        processed_dir=processed_dir,
        kg_dir=kg_dir,
        vector_store_dir=vector_store_dir,
        collection_name=collection_name,
        model_name=model_name,
        store_factory=store_factory,
        embedding_client=embedding_client,
    )

    return builder.build(
        local_files_only=local_files_only,
        allow_hash_fallback=allow_hash_fallback,
        batch_size=batch_size,
        reset=reset,
    )


__all__ = [
    "VectorBuilder",
    "VectorBuilderPaths",
    "load_entities_json",
    "load_graph_pickle",
    "extract_entities_from_graph",
    "load_entities_from_best_source",
    "normalize_entity_record",
    "normalize_node_record",
    "normalize_aliases",
    "deduplicate_entities",
    "write_entities_to_vector_store",
    "build_vector_store",
]