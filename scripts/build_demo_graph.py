# -*- coding: utf-8 -*-
"""构建可直接运行的 Demo Profile。

产物统一写入 ``data/demo/``，不会与 Production 数据混用。
该脚本生成图谱、实体/关系索引、三元组、Alias、示例问题和 Build Manifest。
实体向量库可随后通过 ``python scripts/build_vector_store.py --profile demo`` 构建。
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = PROJECT_ROOT / "data" / "demo"
PROCESSED_DIR = PROFILE_ROOT / "processed"
KG_DIR = PROFILE_ROOT / "kg"
VECTOR_DIR = PROFILE_ROOT / "vector_store"
EXAMPLES_DIR = PROFILE_ROOT / "examples"
MANIFEST_PATH = PROFILE_ROOT / "build_manifest.json"

DEMO_ENTITIES: list[dict[str, Any]] = [
    {
        "node_key": "barack_obama",
        "entity_id": "barack_obama",
        "name": "Barack Obama",
        "type": "person",
        "aliases": ["Barack Obama", "Obama", "奥巴马", "贝拉克·奥巴马"],
        "description": "The 44th president of the United States.",
    },
    {
        "node_key": "michelle_obama",
        "entity_id": "michelle_obama",
        "name": "Michelle Obama",
        "type": "person",
        "aliases": ["Michelle Obama", "Michelle", "米歇尔·奥巴马", "米歇尔"],
        "description": "An American attorney and author, and former First Lady of the United States.",
    },
    {
        "node_key": "hawaii",
        "entity_id": "hawaii",
        "name": "Hawaii",
        "type": "place",
        "aliases": ["Hawaii", "夏威夷"],
        "description": "A state of the United States.",
    },
    {
        "node_key": "united_states",
        "entity_id": "united_states",
        "name": "United States",
        "type": "country",
        "aliases": ["United States", "USA", "U.S.", "America", "美国"],
        "description": "A country in North America.",
    },
    {
        "node_key": "lawyer",
        "entity_id": "lawyer",
        "name": "Lawyer",
        "type": "occupation",
        "aliases": ["Lawyer", "Attorney", "律师"],
        "description": "A legal professional.",
    },
    {
        "node_key": "president",
        "entity_id": "president",
        "name": "President",
        "type": "position",
        "aliases": ["President", "总统"],
        "description": "A head of state or head of government position.",
    },
]

DEMO_RELATIONS: list[dict[str, Any]] = [
    {"source": "barack_obama", "target": "michelle_obama", "relation": "spouse", "relation_zh": "配偶", "text": "Barack Obama is the spouse of Michelle Obama."},
    {"source": "michelle_obama", "target": "barack_obama", "relation": "spouse", "relation_zh": "配偶", "text": "Michelle Obama is the spouse of Barack Obama."},
    {"source": "barack_obama", "target": "hawaii", "relation": "born_in", "relation_zh": "出生于", "text": "Barack Obama was born in Hawaii."},
    {"source": "hawaii", "target": "united_states", "relation": "located_in", "relation_zh": "位于", "text": "Hawaii is located in the United States."},
    {"source": "michelle_obama", "target": "lawyer", "relation": "occupation", "relation_zh": "职业", "text": "Michelle Obama worked as a lawyer."},
    {"source": "barack_obama", "target": "president", "relation": "held_position", "relation_zh": "担任职位", "text": "Barack Obama held the position of President."},
    {"source": "president", "target": "united_states", "relation": "of_country", "relation_zh": "所属国家", "text": "The President position is associated with the United States."},
]

DEMO_QUESTIONS: list[dict[str, str]] = [
    {"query": "Barack Obama 和 Michelle Obama 有什么关系？", "expected": "二者是配偶关系。"},
    {"query": "Barack Obama 出生在哪里？", "expected": "Barack Obama 出生于 Hawaii。"},
    {"query": "Michelle Obama 的职业是什么？", "expected": "Michelle Obama 的职业相关信息是 Lawyer。"},
    {"query": "Hawaii 和 United States 有什么关系？", "expected": "Hawaii 位于 United States。"},
    {"query": "Barack Obama 和 United States 有什么间接关系？", "expected": "Barack Obama 担任 President，而 President 与 United States 相关。"},
]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, required: bool = True) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "required": required,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "sha256": _sha256(path) if exists else None,
    }


def build_demo_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(
        name="demo_kg",
        description="A minimal demo knowledge graph for KG-RAG Agent.",
        version="0.1.0",
    )
    for entity in DEMO_ENTITIES:
        graph.add_node(
            entity["node_key"],
            entity_id=entity["entity_id"],
            entity_name=entity["name"],
            name=entity["name"],
            label=entity["name"],
            type=entity["type"],
            aliases=list(entity["aliases"]),
            description=entity["description"],
            metadata={"profile": "demo"},
        )
    for relation in DEMO_RELATIONS:
        graph.add_edge(
            relation["source"],
            relation["target"],
            key=relation["relation"],
            relation=relation["relation"],
            relation_code=relation["relation"],
            predicate=relation["relation"],
            label=relation["relation"],
            relation_zh=relation["relation_zh"],
            score=1.0,
            weight=1.0,
            text=relation["text"],
            time_id="0",
        )
    return graph


def build_alias_map() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entity in DEMO_ENTITIES:
        payload = {
            "entity_id": entity["entity_id"],
            "entity_name": entity["name"],
            "name": entity["name"],
            "node_key": entity["node_key"],
            "type": entity["type"],
            "aliases": list(entity["aliases"]),
            "score": 1.0,
            "source": "alias_map",
        }
        for alias in [entity["name"], entity["entity_id"], entity["node_key"], *entity["aliases"]]:
            if alias:
                result.setdefault(alias, payload)
                result.setdefault(alias.casefold(), payload)
    return result


def build_graph_stats(graph: nx.MultiDiGraph) -> dict[str, Any]:
    node_types: dict[str, int] = {}
    relation_types: dict[str, int] = {}
    for _, attrs in graph.nodes(data=True):
        key = str(attrs.get("type") or "unknown")
        node_types[key] = node_types.get(key, 0) + 1
    for _, _, attrs in graph.edges(data=True):
        key = str(attrs.get("relation") or "unknown")
        relation_types[key] = relation_types.get(key, 0) + 1
    return {
        "name": graph.graph.get("name", ""),
        "description": graph.graph.get("description", ""),
        "profile": "demo",
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "is_directed": graph.is_directed(),
        "is_multigraph": graph.is_multigraph(),
        "node_types": node_types,
        "relation_types": relation_types,
    }


def main() -> None:
    for directory in (PROCESSED_DIR, KG_DIR, VECTOR_DIR, EXAMPLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    entity_index = {item["name"]: index for index, item in enumerate(DEMO_ENTITIES)}
    relation_names = list(dict.fromkeys(item["relation"] for item in DEMO_RELATIONS))
    relation_index = {name: index for index, name in enumerate(relation_names)}

    entities = [
        {
            "entity_id": item["entity_id"],
            "entity_name": item["name"],
            "name": item["name"],
            "node_key": item["node_key"],
            "type": item["type"],
            "aliases": list(item["aliases"]),
            "description": item["description"],
            "metadata": {"profile": "demo"},
        }
        for item in DEMO_ENTITIES
    ]
    relations = [
        {
            "relation_code": name,
            "internal_relation_id": index,
            "name": name,
            "relation": name,
        }
        for name, index in relation_index.items()
    ]

    graph = build_demo_graph()
    with (KG_DIR / "graph.pkl").open("wb") as handle:
        pickle.dump(graph, handle)

    _write_json(KG_DIR / "graph_stats.json", build_graph_stats(graph))
    _write_json(PROCESSED_DIR / "entities.json", entities)
    _write_json(PROCESSED_DIR / "relations.json", relations)
    _write_json(PROCESSED_DIR / "entity_index.json", entity_index)
    _write_json(PROCESSED_DIR / "relation_index.json", relation_index)
    _write_json(PROCESSED_DIR / "alias_map.json", build_alias_map())
    _write_json(
        PROCESSED_DIR / "data_loader_stats.json",
        {
            "profile": "demo",
            "total_lines": len(DEMO_RELATIONS),
            "valid_lines": len(DEMO_RELATIONS),
            "used_entity_count": len(DEMO_ENTITIES),
            "used_relation_count": len(relation_names),
            "used_time_id_count": 1,
            "min_time_id": 0,
            "max_time_id": 0,
        },
    )

    with (PROCESSED_DIR / "triples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "head_entity",
                "head_id",
                "relation_code",
                "internal_relation_id",
                "tail_entity",
                "tail_id",
                "time_id",
            ],
        )
        writer.writeheader()
        by_key = {item["node_key"]: item for item in DEMO_ENTITIES}
        for relation in DEMO_RELATIONS:
            head = by_key[relation["source"]]
            tail = by_key[relation["target"]]
            writer.writerow(
                {
                    "head_entity": head["name"],
                    "head_id": entity_index[head["name"]],
                    "relation_code": relation["relation"],
                    "internal_relation_id": relation_index[relation["relation"]],
                    "tail_entity": tail["name"],
                    "tail_id": entity_index[tail["name"]],
                    "time_id": 0,
                }
            )

    _write_json(EXAMPLES_DIR / "demo_questions.json", DEMO_QUESTIONS)

    artifacts = {
        "entities": _artifact(PROCESSED_DIR / "entities.json"),
        "relations": _artifact(PROCESSED_DIR / "relations.json"),
        "triples": _artifact(PROCESSED_DIR / "triples.csv"),
        "entity_index": _artifact(PROCESSED_DIR / "entity_index.json"),
        "relation_index": _artifact(PROCESSED_DIR / "relation_index.json"),
        "alias_map": _artifact(PROCESSED_DIR / "alias_map.json"),
        "graph": _artifact(KG_DIR / "graph.pkl"),
        "graph_stats": _artifact(KG_DIR / "graph_stats.json"),
        "demo_questions": _artifact(EXAMPLES_DIR / "demo_questions.json"),
        "vector_stats": _artifact(VECTOR_DIR / "vector_store_stats.json", required=False),
    }
    manifest = {
        "schema_version": "1.0",
        "build_id": "demo-static-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": "demo",
        "ok": all(item["exists"] for item in artifacts.values() if item["required"]),
        "paths": {
            "profile": "demo",
            "processed_dir": "data/demo/processed",
            "kg_dir": "data/demo/kg",
            "vector_store_dir": "data/demo/vector_store",
            "examples_dir": "data/demo/examples",
        },
        "artifacts": artifacts,
    }
    _write_json(MANIFEST_PATH, manifest)
    _write_json(PROCESSED_DIR / "build_manifest.json", manifest)
    _write_json(KG_DIR / "build_manifest.json", manifest)

    print(json.dumps({"ok": manifest["ok"], "profile": "demo", "paths": manifest["paths"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
