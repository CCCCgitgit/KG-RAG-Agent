# -*- coding: utf-8 -*-
"""
graph_builder.py

知识图谱构建模块。

职责：
    1. 读取 data/processed/triples.csv。
    2. 可选读取 data/processed/entities.json 和 relations.json。
    3. 构建 NetworkX MultiDiGraph。
    4. 保存到 data/kg/graph.pkl。
    5. 保存 data/kg/graph_stats.json。

注意：
    本文件属于 data_pipeline 层，只做离线图谱构建。
    不参与在线问答流程。
    不调用 LangGraph node、不调用 LLM、不做最终回答生成。
"""

from __future__ import annotations

import csv
import json
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_KG_DIR = "data/kg"

DEFAULT_TRIPLES_FILE = "triples.csv"
DEFAULT_ENTITIES_FILE = "entities.json"
DEFAULT_RELATIONS_FILE = "relations.json"

DEFAULT_GRAPH_FILE = "graph.pkl"
DEFAULT_GRAPH_STATS_FILE = "graph_stats.json"


# =========================================================
# 1. 路径工具
# =========================================================

def get_project_root() -> Path:
    """
    获取项目根目录。

    当前文件：
        src/kg_rag_agent/data_pipeline/graph_builder.py

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
class GraphBuilderPaths:
    """
    GraphBuilder 使用的路径集合。
    """

    processed_dir: Path
    kg_dir: Path

    triples_file: Path
    entities_file: Path
    relations_file: Path

    graph_file: Path
    graph_stats_file: Path

    @classmethod
    def build(
        cls,
        *,
        processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
        kg_dir: str | Path = DEFAULT_KG_DIR,
        triples_file: str = DEFAULT_TRIPLES_FILE,
        entities_file: str = DEFAULT_ENTITIES_FILE,
        relations_file: str = DEFAULT_RELATIONS_FILE,
        graph_file: str = DEFAULT_GRAPH_FILE,
        graph_stats_file: str = DEFAULT_GRAPH_STATS_FILE,
    ) -> "GraphBuilderPaths":
        """
        构造路径集合。
        """

        processed_path = resolve_project_path(processed_dir)
        kg_path = resolve_project_path(kg_dir)

        return cls(
            processed_dir=processed_path,
            kg_dir=kg_path,
            triples_file=processed_path / triples_file,
            entities_file=processed_path / entities_file,
            relations_file=processed_path / relations_file,
            graph_file=kg_path / graph_file,
            graph_stats_file=kg_path / graph_stats_file,
        )


# =========================================================
# 3. 数据加载
# =========================================================

def iter_triples_csv(path: str | Path) -> Iterable[Dict[str, Any]]:
    """逐行迭代 triples.csv，避免构图前额外复制完整边列表。"""

    resolved = resolve_project_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Triples file not found: {resolved}")

    with resolved.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            head = first_non_empty(
                row, ["head_entity", "head", "source", "source_entity", "subject"]
            )
            relation = first_non_empty(
                row, ["relation_code", "relation", "predicate", "rel", "label"]
            )
            tail = first_non_empty(
                row, ["tail_entity", "tail", "target", "target_entity", "object"]
            )
            if not head or not relation or not tail:
                continue
            yield {
                "head_entity": head,
                "relation_code": relation,
                "tail_entity": tail,
                "head_id": first_non_empty(row, ["head_id"]),
                "tail_id": first_non_empty(row, ["tail_id"]),
                "internal_relation_id": first_non_empty(
                    row, ["internal_relation_id", "relation_id"]
                ),
                "time_id": first_non_empty(row, ["time_id", "time"], default="0"),
                "metadata": {
                    key: value
                    for key, value in row.items()
                    if key not in {
                        "head_entity", "head", "source", "source_entity", "subject",
                        "relation_code", "relation", "predicate", "rel", "label",
                        "tail_entity", "tail", "target", "target_entity", "object",
                    }
                },
            }


def load_triples_csv(path: str | Path) -> List[Dict[str, Any]]:
    """加载 triples.csv；大图构建优先使用 :func:`iter_triples_csv`。"""

    return list(iter_triples_csv(path))


def first_non_empty(
    row: Dict[str, Any],
    keys: Iterable[str],
    *,
    default: str = "",
) -> str:
    """
    从 row 中取第一个非空字段。
    """

    for key in keys:
        value = row.get(key)

        if value is None:
            continue

        text = str(value).strip()

        if text:
            return text

    return default


def build_entity_lookup(entities: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    构造实体 lookup。

    支持通过 name / entity_name / node_key / entity_id 查找。
    """

    lookup: Dict[str, Dict[str, Any]] = {}

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        keys = [
            entity.get("node_key"),
            entity.get("entity_id"),
            entity.get("entity_name"),
            entity.get("name"),
        ]

        aliases = entity.get("aliases", [])
        if isinstance(aliases, list):
            keys.extend(aliases)

        for key in keys:
            text = str(key or "").strip()
            if not text:
                continue

            lookup.setdefault(text, entity)
            lookup.setdefault(text.lower(), entity)

    return lookup


def build_relation_lookup(relations: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    构造关系 lookup。
    """

    lookup: Dict[str, Dict[str, Any]] = {}

    for relation in relations:
        if not isinstance(relation, dict):
            continue

        keys = [
            relation.get("relation_code"),
            relation.get("name"),
            relation.get("relation"),
            relation.get("internal_relation_id"),
        ]

        for key in keys:
            text = str(key or "").strip()
            if not text:
                continue

            lookup.setdefault(text, relation)
            lookup.setdefault(text.lower(), relation)

    return lookup


# =========================================================
# 4. 图构建
# =========================================================

def import_networkx() -> Any:
    """
    导入 networkx。

    这里放在函数内导入，避免模块导入阶段强依赖。
    """

    try:
        import networkx as nx
    except Exception as exc:
        raise ImportError(
            "networkx is required to build graph. "
            "Please install it with: pip install networkx"
        ) from exc

    return nx


def get_node_key(
    entity_name: str,
    *,
    entity_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """
    获取图中的节点 key。

    如果 entities.json 中有 node_key，则优先使用 node_key。
    否则直接使用 entity_name。
    """

    entity_name = str(entity_name or "").strip()

    if not entity_name:
        return ""

    entity_lookup = entity_lookup or {}

    entity = (
        entity_lookup.get(entity_name)
        or entity_lookup.get(entity_name.lower())
        or {}
    )

    node_key = str(
        entity.get("node_key")
        or entity.get("entity_id")
        or entity.get("entity_name")
        or entity.get("name")
        or entity_name
    ).strip()

    return node_key or entity_name


def build_node_attrs(
    entity_name: str,
    *,
    entity_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    构造节点属性。
    """

    entity_name = str(entity_name or "").strip()
    entity_lookup = entity_lookup or {}

    entity = (
        entity_lookup.get(entity_name)
        or entity_lookup.get(entity_name.lower())
        or {}
    )

    attrs = {
        "entity_id": str(entity.get("entity_id") or entity_name),
        "entity_name": str(entity.get("entity_name") or entity.get("name") or entity_name),
        "name": str(entity.get("name") or entity.get("entity_name") or entity_name),
        "type": str(entity.get("type") or "entity"),
        "aliases": entity.get("aliases", []),
        "description": str(entity.get("description") or ""),
        "metadata": entity.get("metadata", {}),
    }

    return attrs


def build_edge_attrs(
    triple: Dict[str, Any],
    *,
    relation_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    构造边属性。
    """

    relation_code = str(triple.get("relation_code") or "").strip()
    relation_lookup = relation_lookup or {}

    relation_record = (
        relation_lookup.get(relation_code)
        or relation_lookup.get(relation_code.lower())
        or {}
    )

    relation_name = str(
        relation_record.get("name")
        or relation_record.get("relation")
        or relation_record.get("relation_code")
        or relation_code
        or "related_to"
    ).strip()

    return {
        "relation": relation_name,
        "relation_code": relation_code,
        "predicate": relation_name,
        "label": relation_name,
        "head_entity": str(triple.get("head_entity") or ""),
        "tail_entity": str(triple.get("tail_entity") or ""),
        "head_id": str(triple.get("head_id") or ""),
        "tail_id": str(triple.get("tail_id") or ""),
        "internal_relation_id": str(
            triple.get("internal_relation_id")
            or relation_record.get("internal_relation_id")
            or ""
        ),
        "time_id": str(triple.get("time_id") or "0"),
        "weight": 1.0,
        "source": "processed_triples_csv",
        "metadata": triple.get("metadata", {}),
    }


def build_networkx_graph(
    triples: Iterable[Dict[str, Any]],
    *,
    entities: Optional[List[Dict[str, Any]]] = None,
    relations: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    """
    构建 NetworkX MultiDiGraph。
    """

    nx = import_networkx()

    graph = nx.MultiDiGraph()
    graph.graph["name"] = "KG-RAG Agent Knowledge Graph"
    graph.graph["builder"] = "data_pipeline.graph_builder"
    graph.graph["directed"] = True
    graph.graph["multigraph"] = True

    entity_lookup = build_entity_lookup(entities or [])
    relation_lookup = build_relation_lookup(relations or [])

    # 先加入 entities.json 中声明的实体，允许孤立节点存在。
    for entity in entities or []:
        if not isinstance(entity, dict):
            continue

        raw_name = str(
            entity.get("entity_name")
            or entity.get("name")
            or entity.get("node_key")
            or entity.get("entity_id")
            or ""
        ).strip()

        if not raw_name:
            continue

        node_key = get_node_key(
            raw_name,
            entity_lookup=entity_lookup,
        )

        if not node_key:
            continue

        graph.add_node(
            node_key,
            **build_node_attrs(
                raw_name,
                entity_lookup=entity_lookup,
            ),
        )

    edge_counter = 0

    for triple in triples:
        head_entity = str(triple.get("head_entity") or "").strip()
        tail_entity = str(triple.get("tail_entity") or "").strip()

        if not head_entity or not tail_entity:
            continue

        source_node = get_node_key(
            head_entity,
            entity_lookup=entity_lookup,
        )
        target_node = get_node_key(
            tail_entity,
            entity_lookup=entity_lookup,
        )

        if not source_node or not target_node:
            continue

        if not graph.has_node(source_node):
            graph.add_node(
                source_node,
                **build_node_attrs(
                    head_entity,
                    entity_lookup=entity_lookup,
                ),
            )

        if not graph.has_node(target_node):
            graph.add_node(
                target_node,
                **build_node_attrs(
                    tail_entity,
                    entity_lookup=entity_lookup,
                ),
            )

        edge_attrs = build_edge_attrs(
            triple,
            relation_lookup=relation_lookup,
        )

        graph.add_edge(
            source_node,
            target_node,
            key=f"edge_{edge_counter}",
            **edge_attrs,
        )

        edge_counter += 1

    return graph


# =========================================================
# 5. 保存与统计
# =========================================================

def save_graph_pickle(
    graph: Any,
    path: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    """
    保存 graph.pkl。
    """

    resolved = resolve_project_path(path)

    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Graph file already exists: {resolved}")

    resolved.parent.mkdir(parents=True, exist_ok=True)

    with resolved.open("wb") as f:
        pickle.dump(graph, f)

    return resolved


def get_graph_stats(graph: Any) -> Dict[str, Any]:
    """
    获取图统计。
    """

    relation_counter: Counter[str] = Counter()
    node_type_counter: Counter[str] = Counter()

    try:
        for _, attrs in graph.nodes(data=True):
            node_type_counter[str(attrs.get("type") or "entity")] += 1
    except Exception:
        pass

    try:
        for _, _, attrs in graph.edges(data=True):
            relation_counter[str(attrs.get("relation") or "related_to")] += 1
    except Exception:
        pass

    num_nodes = 0
    num_edges = 0
    is_directed = False
    is_multigraph = False

    try:
        num_nodes = int(graph.number_of_nodes())
    except Exception:
        pass

    try:
        num_edges = int(graph.number_of_edges())
    except Exception:
        pass

    try:
        is_directed = bool(graph.is_directed())
    except Exception:
        pass

    try:
        is_multigraph = bool(graph.is_multigraph())
    except Exception:
        pass

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "is_directed": is_directed,
        "is_multigraph": is_multigraph,
        "graph_type": type(graph).__name__,
        "num_node_types": len(node_type_counter),
        "num_relation_types": len(relation_counter),
        "top_node_types": node_type_counter.most_common(20),
        "top_relations": relation_counter.most_common(20),
    }


def save_graph_stats(
    graph: Any,
    path: str | Path,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    保存 graph_stats.json。
    """

    stats = get_graph_stats(graph)

    if extra:
        stats.update(extra)

    save_json(stats, path)
    return stats


# =========================================================
# 6. GraphBuilder 类
# =========================================================

class GraphBuilder:
    """
    离线知识图谱构建器。

    用法：
        builder = GraphBuilder()
        stats = builder.build()
    """

    def __init__(
        self,
        *,
        processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
        kg_dir: str | Path = DEFAULT_KG_DIR,
        triples_file: str = DEFAULT_TRIPLES_FILE,
        entities_file: str = DEFAULT_ENTITIES_FILE,
        relations_file: str = DEFAULT_RELATIONS_FILE,
        graph_file: str = DEFAULT_GRAPH_FILE,
        graph_stats_file: str = DEFAULT_GRAPH_STATS_FILE,
    ) -> None:
        self.paths = GraphBuilderPaths.build(
            processed_dir=processed_dir,
            kg_dir=kg_dir,
            triples_file=triples_file,
            entities_file=entities_file,
            relations_file=relations_file,
            graph_file=graph_file,
            graph_stats_file=graph_stats_file,
        )

    def load_inputs(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        加载 triples / entities / relations。
        """

        triples = load_triples_csv(self.paths.triples_file)

        entities = load_json_if_exists(
            self.paths.entities_file,
            default=[],
        )
        relations = load_json_if_exists(
            self.paths.relations_file,
            default=[],
        )

        if not isinstance(entities, list):
            entities = []

        if not isinstance(relations, list):
            relations = []

        return triples, entities, relations

    def build_graph(self) -> Any:
        """
        构建图对象。
        """

        triples, entities, relations = self.load_inputs()

        return build_networkx_graph(
            triples,
            entities=entities,
            relations=relations,
        )

    def build(
        self,
        *,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        """流式读取 triples.csv，构建并保存图。"""

        ensure_dir(self.paths.kg_dir)

        entities = load_json_if_exists(self.paths.entities_file, default=[])
        relations = load_json_if_exists(self.paths.relations_file, default=[])
        if not isinstance(entities, list):
            entities = []
        if not isinstance(relations, list):
            relations = []

        graph = build_networkx_graph(
            iter_triples_csv(self.paths.triples_file),
            entities=entities,
            relations=relations,
        )

        graph_path = save_graph_pickle(
            graph,
            self.paths.graph_file,
            overwrite=overwrite,
        )

        stats = save_graph_stats(
            graph,
            self.paths.graph_stats_file,
            extra={
                "graph_path": graph_path.as_posix(),
                "graph_stats_path": self.paths.graph_stats_file.as_posix(),
                "triples_path": self.paths.triples_file.as_posix(),
                "entities_path": self.paths.entities_file.as_posix(),
                "relations_path": self.paths.relations_file.as_posix(),
                "num_loaded_triples": int(graph.number_of_edges()),
                "num_loaded_entities": len(entities),
                "num_loaded_relations": len(relations),
            },
        )

        return stats


# =========================================================
# 7. 函数式入口
# =========================================================

def build_graph(
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    kg_dir: str | Path = DEFAULT_KG_DIR,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    函数式图构建入口。
    """

    builder = GraphBuilder(
        processed_dir=processed_dir,
        kg_dir=kg_dir,
    )

    return builder.build(overwrite=overwrite)


__all__ = [
    "GraphBuilder",
    "GraphBuilderPaths",
    "load_triples_csv",
    "iter_triples_csv",
    "build_networkx_graph",
    "save_graph_pickle",
    "get_graph_stats",
    "save_graph_stats",
    "build_graph",
]