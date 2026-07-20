# -*- coding: utf-8 -*-
"""
data_loader.py

离线数据加载与标准化模块。

职责：
    1. 读取 data/raw/ent2ids。
    2. 读取 data/raw/relation2ids。
    3. 读取 data/raw/path_graph。
    4. 生成 data/processed/ 下的标准中间文件：
        - entities.json
        - relations.json
        - triples.csv
        - entity_index.json
        - relation_index.json
        - alias_map.json
        - data_loader_stats.json

注意：
    本文件属于 data_pipeline 层，只做离线数据处理。
    不参与在线问答流程。
    不调用 LangGraph、LLM、KG 检索节点或向量检索节点。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =========================================================
# 1. 默认路径
# =========================================================

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_PROCESSED_DIR = "data/processed"

DEFAULT_ENTITY_FILE = "ent2ids"
DEFAULT_RELATION_FILE = "relation2ids"
DEFAULT_PATH_GRAPH_FILE = "path_graph"


# =========================================================
# 2. 通用路径与文件工具
# =========================================================

def get_project_root() -> Path:
    """
    获取项目根目录。

    当前文件：
        src/kg_rag_agent/data_pipeline/data_loader.py

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
        2. 相对于当前工作目录的路径
        3. 相对于项目根目录的路径
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


def load_json(path: str | Path) -> Any:
    """
    读取 JSON 文件。
    """

    resolved = resolve_project_path(path)

    with resolved.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    """
    保存 JSON 文件。
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


def read_non_empty_lines(path: str | Path) -> List[str]:
    """
    读取非空行。
    """

    resolved = resolve_project_path(path)

    with resolved.open("r", encoding="utf-8", errors="ignore") as f:
        return [
            line.strip()
            for line in f
            if line.strip()
        ]


# =========================================================
# 3. 数据结构
# =========================================================

@dataclass
class DataLoaderPaths:
    """
    DataLoader 使用的路径集合。
    """

    raw_dir: Path
    processed_dir: Path
    entity_file: Path
    relation_file: Path
    path_graph_file: Path

    @classmethod
    def build(
        cls,
        raw_dir: str | Path = DEFAULT_RAW_DIR,
        processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
        entity_file: str = DEFAULT_ENTITY_FILE,
        relation_file: str = DEFAULT_RELATION_FILE,
        path_graph_file: str = DEFAULT_PATH_GRAPH_FILE,
    ) -> "DataLoaderPaths":
        """
        构造标准路径集合。
        """

        raw_path = resolve_project_path(raw_dir)
        processed_path = resolve_project_path(processed_dir)

        return cls(
            raw_dir=raw_path,
            processed_dir=processed_path,
            entity_file=raw_path / entity_file,
            relation_file=raw_path / relation_file,
            path_graph_file=raw_path / path_graph_file,
        )


# =========================================================
# 4. 原始索引加载
# =========================================================

def load_id_map(path: str | Path) -> Dict[str, int]:
    """
    加载 ent2ids 或 relation2ids。

    兼容两种格式：

    1. JSON dict，一整行：
        {"Barack Obama": 0, "Michelle Obama": 1}

    2. 文本行：
        Barack Obama    0
        Michelle Obama  1
    """

    resolved = resolve_project_path(path)

    if not resolved.exists():
        raise FileNotFoundError(f"ID map file not found: {resolved}")

    text = resolved.read_text(
        encoding="utf-8",
        errors="ignore",
    ).strip()

    if not text:
        return {}

    # 优先尝试 JSON dict
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {
                str(key): int(value)
                for key, value in data.items()
            }
    except Exception:
        pass

    # 回退到逐行解析
    result: Dict[str, int] = {}

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 2:
            continue

        name = " ".join(parts[:-1]).strip()
        value = parts[-1].strip()

        if not name:
            continue

        try:
            result[name] = int(value)
        except ValueError:
            continue

    return result


def load_entity_index(path: str | Path) -> Dict[str, int]:
    """
    加载实体索引。
    """

    return load_id_map(path)


def load_relation_index(path: str | Path) -> Dict[str, int]:
    """
    加载关系索引。
    """

    return load_id_map(path)


# =========================================================
# 5. 原始 path_graph 加载
# =========================================================

def parse_path_graph_line(
    line: str,
    *,
    entity_index: Optional[Dict[str, int]] = None,
    relation_index: Optional[Dict[str, int]] = None,
) -> Optional[Dict[str, Any]]:
    """
    解析 path_graph 中的一行。

    预期格式：
        head_entity<TAB>relation_code<TAB>tail_entity<TAB>time_id

    示例：
        PRESIDENT    30    NIGERIA    0
    """

    parts = line.strip().split("\t")

    if len(parts) < 3:
        parts = line.strip().split()

    if len(parts) < 3:
        return None

    head_entity = str(parts[0]).strip()
    relation_code = str(parts[1]).strip()
    tail_entity = str(parts[2]).strip()

    time_id = "0"
    if len(parts) >= 4:
        time_id = str(parts[3]).strip() or "0"

    entity_index = entity_index or {}
    relation_index = relation_index or {}

    head_id = entity_index.get(head_entity)
    tail_id = entity_index.get(tail_entity)
    internal_relation_id = relation_index.get(relation_code)

    return {
        "head_entity": head_entity,
        "head_id": head_id,
        "relation_code": relation_code,
        "internal_relation_id": internal_relation_id,
        "tail_entity": tail_entity,
        "tail_id": tail_id,
        "time_id": time_id,
    }


def iter_path_graph(
    path: str | Path,
    *,
    entity_index: Optional[Dict[str, int]] = None,
    relation_index: Optional[Dict[str, int]] = None,
    max_rows: Optional[int] = None,
) -> Iterable[Dict[str, Any]]:
    """逐行迭代 path_graph，避免完整数据一次性进入内存。"""

    resolved = resolve_project_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"path_graph file not found: {resolved}")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be greater than 0 when provided.")

    count = 0
    with resolved.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            item = parse_path_graph_line(
                line,
                entity_index=entity_index,
                relation_index=relation_index,
            )
            if item is None:
                continue
            yield item
            count += 1
            if max_rows is not None and count >= max_rows:
                break


def load_path_graph(
    path: str | Path,
    *,
    entity_index: Optional[Dict[str, int]] = None,
    relation_index: Optional[Dict[str, int]] = None,
    max_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """加载 path_graph；大数据构建优先使用 :func:`iter_path_graph`。"""

    return list(
        iter_path_graph(
            path,
            entity_index=entity_index,
            relation_index=relation_index,
            max_rows=max_rows,
        )
    )


# =========================================================
# 6. processed 数据构造
# =========================================================

def build_entities(entity_index: Dict[str, int]) -> List[Dict[str, Any]]:
    """
    构造 entities.json。
    """

    entities: List[Dict[str, Any]] = []

    for name, entity_id in sorted(
        entity_index.items(),
        key=lambda item: item[1],
    ):
        entity_name = str(name).strip()

        entities.append(
            {
                "entity_id": str(entity_id),
                "entity_name": entity_name,
                "name": entity_name,
                "node_key": entity_name,
                "type": "entity",
                "aliases": build_aliases(entity_name),
                "description": "",
                "metadata": {
                    "source": "raw_ent2ids",
                    "raw_id": entity_id,
                },
            }
        )

    return entities


def build_relations(relation_index: Dict[str, int]) -> List[Dict[str, Any]]:
    """
    构造 relations.json。
    """

    relations: List[Dict[str, Any]] = []

    for relation_code, internal_relation_id in sorted(
        relation_index.items(),
        key=lambda item: item[1],
    ):
        relation_text = str(relation_code).strip()

        relations.append(
            {
                "relation_code": relation_text,
                "internal_relation_id": int(internal_relation_id),
                "name": relation_text,
                "description": "",
                "metadata": {
                    "source": "raw_relation2ids",
                },
            }
        )

    return relations


def build_aliases(entity_name: str) -> List[str]:
    """
    为实体构造轻量别名。

    不做复杂语义改写，只做安全的大小写兼容。
    """

    aliases: List[str] = []
    candidates = [
        entity_name,
        entity_name.lower(),
        entity_name.title(),
    ]

    for item in candidates:
        item = str(item).strip()
        if item and item not in aliases:
            aliases.append(item)

    return aliases


def build_alias_map(entities: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    构造 alias_map。

    格式：
        alias -> entity record
    """

    alias_map: Dict[str, Dict[str, Any]] = {}

    for entity in entities:
        entity_id = str(entity.get("entity_id", ""))
        entity_name = str(entity.get("entity_name") or entity.get("name") or "")
        node_key = str(entity.get("node_key") or entity_name)
        entity_type = str(entity.get("type", "entity") or "entity")
        aliases = entity.get("aliases", []) or []

        all_aliases = [
            entity_name,
            node_key,
            entity_id,
            *aliases,
        ]

        for alias in all_aliases:
            alias_text = str(alias).strip()

            if not alias_text:
                continue

            record = {
                "entity_id": entity_id,
                "entity_name": entity_name,
                "name": entity_name,
                "node_key": node_key,
                "type": entity_type,
                "aliases": aliases,
                "score": 1.0,
                "source": "alias_map",
            }

            alias_map[alias_text] = record
            alias_map[alias_text.lower()] = record

    return alias_map


def save_triples_csv(
    triples: Iterable[Dict[str, Any]],
    path: str | Path,
) -> int:
    """
    保存 triples.csv。
    """

    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "head_entity",
        "head_id",
        "relation_code",
        "internal_relation_id",
        "tail_entity",
        "tail_id",
        "time_id",
    ]

    count = 0

    with resolved.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for item in triples:
            writer.writerow(
                {
                    "head_entity": item.get("head_entity", ""),
                    "head_id": item.get("head_id", ""),
                    "relation_code": item.get("relation_code", ""),
                    "internal_relation_id": item.get("internal_relation_id", ""),
                    "tail_entity": item.get("tail_entity", ""),
                    "tail_id": item.get("tail_id", ""),
                    "time_id": item.get("time_id", ""),
                }
            )
            count += 1

    return count


# =========================================================
# 7. DataLoader 类
# =========================================================

class DataLoader:
    """
    原始数据加载器。

    用法：
        loader = DataLoader()
        result = loader.process()
    """

    def __init__(
        self,
        *,
        raw_dir: str | Path = DEFAULT_RAW_DIR,
        processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
        entity_file: str = DEFAULT_ENTITY_FILE,
        relation_file: str = DEFAULT_RELATION_FILE,
        path_graph_file: str = DEFAULT_PATH_GRAPH_FILE,
    ) -> None:
        self.paths = DataLoaderPaths.build(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            entity_file=entity_file,
            relation_file=relation_file,
            path_graph_file=path_graph_file,
        )

    def load_entity_index(self) -> Dict[str, int]:
        """
        读取实体索引。
        """

        return load_entity_index(self.paths.entity_file)

    def load_relation_index(self) -> Dict[str, int]:
        """
        读取关系索引。
        """

        return load_relation_index(self.paths.relation_file)

    def load_triples(
        self,
        *,
        max_rows: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        读取 path_graph 三元组。
        """

        entity_index = self.load_entity_index()
        relation_index = self.load_relation_index()

        return load_path_graph(
            self.paths.path_graph_file,
            entity_index=entity_index,
            relation_index=relation_index,
            max_rows=max_rows,
        )

    def process(
        self,
        *,
        max_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        执行完整离线数据加载与标准化。
        """

        processed_dir = ensure_dir(self.paths.processed_dir)

        entity_index = self.load_entity_index()
        relation_index = self.load_relation_index()

        entities = build_entities(entity_index)
        relations = build_relations(relation_index)

        triples = iter_path_graph(
            self.paths.path_graph_file,
            entity_index=entity_index,
            relation_index=relation_index,
            max_rows=max_rows,
        )

        alias_map = build_alias_map(entities)

        entity_index_path = processed_dir / "entity_index.json"
        relation_index_path = processed_dir / "relation_index.json"
        entities_path = processed_dir / "entities.json"
        relations_path = processed_dir / "relations.json"
        triples_path = processed_dir / "triples.csv"
        alias_map_path = processed_dir / "alias_map.json"
        stats_path = processed_dir / "data_loader_stats.json"

        save_json(entity_index, entity_index_path)
        save_json(relation_index, relation_index_path)
        save_json(entities, entities_path)
        save_json(relations, relations_path)
        save_json(alias_map, alias_map_path)
        triple_count = save_triples_csv(triples, triples_path)

        stats = {
            "raw_dir": self.paths.raw_dir.as_posix(),
            "processed_dir": processed_dir.as_posix(),
            "entity_file": self.paths.entity_file.as_posix(),
            "relation_file": self.paths.relation_file.as_posix(),
            "path_graph_file": self.paths.path_graph_file.as_posix(),
            "num_entities": len(entities),
            "num_relations": len(relations),
            "num_triples": triple_count,
            "num_aliases": len(alias_map),
            "max_rows": max_rows,
            "output_files": {
                "entity_index": entity_index_path.as_posix(),
                "relation_index": relation_index_path.as_posix(),
                "entities": entities_path.as_posix(),
                "relations": relations_path.as_posix(),
                "triples": triples_path.as_posix(),
                "alias_map": alias_map_path.as_posix(),
                "stats": stats_path.as_posix(),
            },
        }

        save_json(stats, stats_path)

        return stats


# =========================================================
# 8. 函数式入口
# =========================================================

def process_raw_data(
    *,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """
    函数式离线数据处理入口。
    """

    loader = DataLoader(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )

    return loader.process(max_rows=max_rows)


__all__ = [
    "DataLoader",
    "DataLoaderPaths",
    "load_id_map",
    "load_entity_index",
    "load_relation_index",
    "load_path_graph",
    "iter_path_graph",
    "parse_path_graph_line",
    "build_entities",
    "build_relations",
    "build_alias_map",
    "save_triples_csv",
    "process_raw_data",
]