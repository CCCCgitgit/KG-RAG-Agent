# -*- coding: utf-8 -*-
"""
relation_search.py

直接关系查询模块。

作用：
    1. 查询两个对象之间是否存在直接关系。
    2. 支持 NetworkX Graph / DiGraph / MultiDiGraph。
    3. 支持简单 dict 图结构。
    4. 返回统一结构，供 kg_retrieval_node.py 转换为 evidence。

本文件属于 kg 底层能力层：
    kg/
        relation_search.py

它不负责：
    1. 用户问题解析。
    2. 实体链接。
    3. 路径搜索。
    4. 邻居搜索。
    5. 最终回答生成。

这些由 graph/nodes/ 和其他 kg 模块负责。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# =========================================================
# 1. 对外主函数
# =========================================================

def relation_search(
    graph: Any,
    source: Optional[str] = None,
    target: Optional[str] = None,
    *,
    head: Optional[str] = None,
    tail: Optional[str] = None,
    directed: bool = True,
    include_reverse: bool = True,
    max_results: int = 20,
) -> Dict[str, Any]:
    """
    查询两个节点之间的直接关系。

    Args:
        graph:
            图结构对象。
            通常是 networkx.Graph / DiGraph / MultiDiGraph。

        source:
            起始节点。

        target:
            目标节点。

        head:
            source 的别名参数。

        tail:
            target 的别名参数。

        directed:
            是否优先按有向关系查询。

        include_reverse:
            如果 source -> target 没查到，是否尝试 target -> source。

        max_results:
            最多返回多少条关系。

    Returns:
        Dict[str, Any]:
            {
                "source": "...",
                "target": "...",
                "found": true,
                "relations": [
                    {
                        "head": "...",
                        "relation": "...",
                        "tail": "...",
                        "direction": "forward",
                        "score": 1.0,
                        "metadata": {...}
                    }
                ]
            }
    """

    source_node = str(source or head or "").strip()
    target_node = str(target or tail or "").strip()

    if not source_node or not target_node:
        return _empty_result(
            source=source_node,
            target=target_node,
            reason="source or target is empty",
        )

    if graph is None:
        return _empty_result(
            source=source_node,
            target=target_node,
            reason="graph is None",
        )

    relations: List[Dict[str, Any]] = []

    # -----------------------------------------------------
    # 1. forward: source -> target
    # -----------------------------------------------------
    forward_edges = _get_edges_between(
        graph=graph,
        source=source_node,
        target=target_node,
    )

    for edge_attrs in forward_edges:
        relations.append(
            _build_relation_item(
                head=source_node,
                tail=target_node,
                attrs=edge_attrs,
                direction="forward",
            )
        )

    # -----------------------------------------------------
    # 2. reverse: target -> source
    # -----------------------------------------------------
    if include_reverse:
        reverse_edges = _get_edges_between(
            graph=graph,
            source=target_node,
            target=source_node,
        )

        for edge_attrs in reverse_edges:
            relations.append(
                _build_relation_item(
                    head=target_node,
                    tail=source_node,
                    attrs=edge_attrs,
                    direction="reverse",
                )
            )

    # -----------------------------------------------------
    # 3. 对无向图 / dict 图兜底
    # -----------------------------------------------------
    if not relations and not directed:
        undirected_edges = _get_undirected_edges_between(
            graph=graph,
            source=source_node,
            target=target_node,
        )

        for edge_attrs in undirected_edges:
            relations.append(
                _build_relation_item(
                    head=source_node,
                    tail=target_node,
                    attrs=edge_attrs,
                    direction="undirected",
                )
            )

    relations = _deduplicate_relations(relations)
    relations = relations[:max_results]

    return {
        "source": source_node,
        "target": target_node,
        "found": len(relations) > 0,
        "relations": relations,
        "num_relations": len(relations),
    }


# =========================================================
# 2. 兼容别名函数
# =========================================================

def search_relation(
    graph: Any,
    source: str,
    target: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    relation_search 的别名。
    """

    return relation_search(
        graph=graph,
        source=source,
        target=target,
        **kwargs,
    )


def find_relations(
    graph: Any,
    source: str,
    target: str,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    只返回 relations 列表的便捷函数。
    """

    result = relation_search(
        graph=graph,
        source=source,
        target=target,
        **kwargs,
    )

    return result.get("relations", [])


# =========================================================
# 3. 边查询
# =========================================================

def _get_edges_between(
    *,
    graph: Any,
    source: str,
    target: str,
) -> List[Dict[str, Any]]:
    """
    获取 source -> target 的边属性列表。

    支持：
        1. networkx Graph / DiGraph
        2. networkx MultiGraph / MultiDiGraph
        3. dict 邻接结构
    """

    # -----------------------------------------------------
    # 1. NetworkX 风格 get_edge_data
    # -----------------------------------------------------
    if hasattr(graph, "get_edge_data"):
        try:
            edge_data = graph.get_edge_data(source, target)

            if edge_data:
                return _normalize_edge_data(edge_data)

        except Exception:
            pass

    # -----------------------------------------------------
    # 2. NetworkX edges(data=True) 兜底
    # -----------------------------------------------------
    if hasattr(graph, "edges"):
        try:
            matched_edges = []

            for edge in graph.edges(data=True):
                if len(edge) != 3:
                    continue

                u, v, attrs = edge

                if str(u) == str(source) and str(v) == str(target):
                    if isinstance(attrs, dict):
                        matched_edges.append(dict(attrs))
                    else:
                        matched_edges.append({"value": attrs})

            if matched_edges:
                return matched_edges

        except Exception:
            pass

    # -----------------------------------------------------
    # 3. dict 图结构
    # -----------------------------------------------------
    if isinstance(graph, dict):
        return _get_edges_from_dict_graph(
            graph=graph,
            source=source,
            target=target,
        )

    return []


def _get_undirected_edges_between(
    *,
    graph: Any,
    source: str,
    target: str,
) -> List[Dict[str, Any]]:
    """
    无向兜底查询。

    同时尝试 source -> target 和 target -> source。
    """

    edges = _get_edges_between(
        graph=graph,
        source=source,
        target=target,
    )

    edges.extend(
        _get_edges_between(
            graph=graph,
            source=target,
            target=source,
        )
    )

    return edges


def _normalize_edge_data(edge_data: Any) -> List[Dict[str, Any]]:
    """
    将 NetworkX get_edge_data 返回值规范化为 List[Dict]。

    普通 DiGraph:
        {"relation": "spouse"}

    MultiDiGraph:
        {
            0: {"relation": "spouse"},
            1: {"relation": "colleague"}
        }
    """

    if edge_data is None:
        return []

    if not isinstance(edge_data, dict):
        return [{"value": edge_data}]

    if not edge_data:
        return []

    # 普通边属性
    if _looks_like_single_edge_attrs(edge_data):
        return [dict(edge_data)]

    # 多重边属性
    normalized = []

    for edge_key, attrs in edge_data.items():
        if isinstance(attrs, dict):
            item = dict(attrs)
            item.setdefault("edge_key", edge_key)
            normalized.append(item)
        else:
            normalized.append(
                {
                    "edge_key": edge_key,
                    "value": attrs,
                }
            )

    return normalized


def _looks_like_single_edge_attrs(edge_data: Dict[Any, Any]) -> bool:
    """
    判断 edge_data 是否像单条边的属性字典。
    """

    common_attr_keys = {
        "relation",
        "predicate",
        "rel",
        "type",
        "label",
        "weight",
        "score",
        "name",
        "text",
        "description",
    }

    return any(key in edge_data for key in common_attr_keys)


def _get_edges_from_dict_graph(
    *,
    graph: Dict[Any, Any],
    source: str,
    target: str,
) -> List[Dict[str, Any]]:
    """
    从 dict 图结构中查询边。

    兼容几种格式：

    1. 邻接字典：
        {
            "A": {
                "B": {"relation": "r"}
            }
        }

    2. 邻接列表：
        {
            "A": [
                {"target": "B", "relation": "r"}
            ]
        }

    3. 三元组列表：
        {
            "edges": [
                {"head": "A", "relation": "r", "tail": "B"}
            ]
        }
    """

    results: List[Dict[str, Any]] = []

    # -----------------------------------------------------
    # 1. graph["edges"] 三元组列表
    # -----------------------------------------------------
    edges = graph.get("edges")

    if isinstance(edges, list):
        for edge in edges:
            if not isinstance(edge, dict):
                continue

            head = _first_non_empty_str(
                edge,
                ["head", "source", "subject", "h"],
            )
            tail = _first_non_empty_str(
                edge,
                ["tail", "target", "object", "t"],
            )

            if str(head) == str(source) and str(tail) == str(target):
                results.append(dict(edge))

    # -----------------------------------------------------
    # 2. graph[source] 邻接结构
    # -----------------------------------------------------
    adjacency = graph.get(source)

    if isinstance(adjacency, dict):
        value = adjacency.get(target)

        if isinstance(value, dict):
            results.append(dict(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    results.append(dict(item))
                else:
                    results.append({"value": item})
        elif value is not None:
            results.append({"value": value})

    elif isinstance(adjacency, list):
        for item in adjacency:
            if not isinstance(item, dict):
                continue

            item_target = _first_non_empty_str(
                item,
                ["target", "tail", "object", "neighbor", "to"],
            )

            if str(item_target) == str(target):
                results.append(dict(item))

    return results


# =========================================================
# 4. Relation Item 构造
# =========================================================

def _build_relation_item(
    *,
    head: str,
    tail: str,
    attrs: Dict[str, Any],
    direction: str,
) -> Dict[str, Any]:
    """
    构造统一 relation item。
    """

    relation = _extract_relation_name(attrs)
    score = _extract_score(attrs)
    text = _extract_text(
        attrs=attrs,
        head=head,
        relation=relation,
        tail=tail,
    )

    return {
        "head": head,
        "relation": relation,
        "tail": tail,
        "direction": direction,
        "score": score,
        "text": text,
        "metadata": {
            "edge_attrs": attrs,
        },
    }


def _extract_relation_name(attrs: Dict[str, Any]) -> str:
    """
    从边属性中提取关系名称。
    """

    relation = _first_non_empty_str(
        attrs,
        [
            "relation",
            "predicate",
            "rel",
            "type",
            "label",
            "name",
        ],
    )

    if relation:
        return relation

    value = attrs.get("value")

    if value is not None:
        return str(value)

    return "related_to"


def _extract_score(attrs: Dict[str, Any]) -> float:
    """
    从边属性中提取分数。
    """

    for key in ["score", "weight", "confidence", "similarity"]:
        if key in attrs:
            try:
                return _clip_score(float(attrs[key]))
            except Exception:
                pass

    return 1.0


def _extract_text(
    *,
    attrs: Dict[str, Any],
    head: str,
    relation: str,
    tail: str,
) -> str:
    """
    构造关系文本。
    """

    text = _first_non_empty_str(
        attrs,
        [
            "text",
            "description",
            "sentence",
            "evidence",
        ],
    )

    if text:
        return text

    return f"{head} --{relation}--> {tail}"


# =========================================================
# 5. 去重与工具函数
# =========================================================

def _deduplicate_relations(
    relations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    对关系结果去重。
    """

    seen = set()
    unique: List[Dict[str, Any]] = []

    for item in relations:
        key = (
            str(item.get("head", "")),
            str(item.get("relation", "")),
            str(item.get("tail", "")),
            str(item.get("direction", "")),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    return unique


def _first_non_empty_str(
    item: Dict[str, Any],
    keys: List[str],
) -> str:
    """
    从 dict 中按顺序获取第一个非空字符串字段。
    """

    for key in keys:
        value = item.get(key)

        if value is None:
            continue

        value = str(value).strip()

        if value:
            return value

    return ""


def _clip_score(score: float) -> float:
    """
    将分数限制在 [0, 1]。
    """

    if score < 0:
        return 0.0

    if score > 1:
        return 1.0

    return score


def _empty_result(
    *,
    source: str,
    target: str,
    reason: str,
) -> Dict[str, Any]:
    """
    构造空结果。
    """

    return {
        "source": source,
        "target": target,
        "found": False,
        "relations": [],
        "num_relations": 0,
        "reason": reason,
    }


# =========================================================
# 6. 快速测试入口
# =========================================================

if __name__ == "__main__":
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_edge(
        "Barack Obama",
        "Michelle Obama",
        relation="spouse",
        score=1.0,
    )

    result = relation_search(
        graph=graph,
        source="Barack Obama",
        target="Michelle Obama",
    )

    from pprint import pprint

    pprint(result)