# -*- coding: utf-8 -*-
"""
neighbor_search.py

邻居查询模块。

作用：
    1. 查询某个对象的一跳邻居。
    2. 支持 NetworkX Graph / DiGraph / MultiDiGraph。
    3. 支持简单 dict 图结构。
    4. 返回统一邻居结构，供 kg_retrieval_node.py 转换为 evidence。

本文件属于 kg 底层能力层：
    kg/
        neighbor_search.py

它不负责：
    1. 用户问题解析。
    2. 实体链接。
    3. 直接关系查询。
    4. 多跳路径查询。
    5. 最终回答生成。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple


# =========================================================
# 1. 对外主函数
# =========================================================

def neighbor_search(
    graph: Any,
    entity: Optional[str] = None,
    *,
    node: Optional[str] = None,
    max_neighbors: int = 20,
    limit: Optional[int] = None,
    direction: str = "both",
    include_relation: bool = True,
) -> Dict[str, Any]:
    """
    查询某个节点的一跳邻居。

    Args:
        graph:
            图结构对象。
            通常是 networkx.Graph / DiGraph / MultiDiGraph。

        entity:
            当前节点名称 / node_key。

        node:
            entity 的别名参数。

        max_neighbors:
            最多返回多少个邻居。

        limit:
            max_neighbors 的别名参数。
            如果传入 limit，则优先使用 limit。

        direction:
            邻居方向：
                - "out"  ：只查出边邻居
                - "in"   ：只查入边邻居
                - "both" ：同时查出边和入边

        include_relation:
            是否返回边关系信息。

    Returns:
        Dict[str, Any]:
            {
                "entity": "A",
                "found": true,
                "neighbors": [
                    {
                        "source": "A",
                        "relation": "r",
                        "target": "B",
                        "direction": "out",
                        "score": 1.0,
                        "text": "A --r--> B",
                        "metadata": {...}
                    }
                ],
                "num_neighbors": 1
            }
    """

    entity_node = str(entity or node or "").strip()

    if limit is not None:
        max_neighbors = int(limit)

    max_neighbors = max(int(max_neighbors), 1)

    if not entity_node:
        return _empty_result(
            entity=entity_node,
            reason="entity is empty",
        )

    if graph is None:
        return _empty_result(
            entity=entity_node,
            reason="graph is None",
        )

    if not _graph_has_node(graph, entity_node):
        return _empty_result(
            entity=entity_node,
            reason="entity not found in graph",
        )

    direction = _normalize_direction(direction)

    neighbors: List[Dict[str, Any]] = []

    if direction in {"out", "both"}:
        neighbors.extend(
            _get_out_neighbors(
                graph=graph,
                entity=entity_node,
                include_relation=include_relation,
            )
        )

    if direction in {"in", "both"}:
        neighbors.extend(
            _get_in_neighbors(
                graph=graph,
                entity=entity_node,
                include_relation=include_relation,
            )
        )

    neighbors = _deduplicate_neighbors(neighbors)
    neighbors = sorted(
        neighbors,
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )[:max_neighbors]

    return {
        "entity": entity_node,
        "found": len(neighbors) > 0,
        "neighbors": neighbors,
        "num_neighbors": len(neighbors),
    }


# =========================================================
# 2. 兼容别名函数
# =========================================================

def search_neighbors(
    graph: Any,
    entity: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    neighbor_search 的别名。
    """

    return neighbor_search(
        graph=graph,
        entity=entity,
        **kwargs,
    )


def find_neighbors(
    graph: Any,
    entity: str,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    只返回 neighbors 列表的便捷函数。
    """

    result = neighbor_search(
        graph=graph,
        entity=entity,
        **kwargs,
    )

    return result.get("neighbors", [])


# =========================================================
# 3. 出边邻居查询
# =========================================================

def _get_out_neighbors(
    *,
    graph: Any,
    entity: str,
    include_relation: bool,
) -> List[Dict[str, Any]]:
    """
    查询 entity 的出边邻居。
    """

    results: List[Dict[str, Any]] = []

    # -----------------------------------------------------
    # 1. NetworkX successors
    # -----------------------------------------------------
    if hasattr(graph, "successors"):
        try:
            for target in graph.successors(entity):
                target = str(target)
                edge_attrs_list = _get_edge_attrs_between(
                    graph=graph,
                    source=entity,
                    target=target,
                )

                if not edge_attrs_list:
                    edge_attrs_list = [{}]

                for attrs in edge_attrs_list:
                    results.append(
                        _build_neighbor_item(
                            source=entity,
                            target=target,
                            attrs=attrs,
                            direction="out",
                            include_relation=include_relation,
                        )
                    )

            return results

        except Exception:
            pass

    # -----------------------------------------------------
    # 2. NetworkX neighbors
    # -----------------------------------------------------
    if hasattr(graph, "neighbors"):
        try:
            for target in graph.neighbors(entity):
                target = str(target)
                edge_attrs_list = _get_edge_attrs_between(
                    graph=graph,
                    source=entity,
                    target=target,
                )

                if not edge_attrs_list:
                    edge_attrs_list = [{}]

                for attrs in edge_attrs_list:
                    results.append(
                        _build_neighbor_item(
                            source=entity,
                            target=target,
                            attrs=attrs,
                            direction="out",
                            include_relation=include_relation,
                        )
                    )

            if results:
                return results

        except Exception:
            pass

    # -----------------------------------------------------
    # 3. dict 图结构
    # -----------------------------------------------------
    if isinstance(graph, dict):
        results.extend(
            _get_out_neighbors_from_dict_graph(
                graph=graph,
                entity=entity,
                include_relation=include_relation,
            )
        )

    return results


# =========================================================
# 4. 入边邻居查询
# =========================================================

def _get_in_neighbors(
    *,
    graph: Any,
    entity: str,
    include_relation: bool,
) -> List[Dict[str, Any]]:
    """
    查询 entity 的入边邻居。
    """

    results: List[Dict[str, Any]] = []

    # -----------------------------------------------------
    # 1. NetworkX predecessors
    # -----------------------------------------------------
    if hasattr(graph, "predecessors"):
        try:
            for source in graph.predecessors(entity):
                source = str(source)
                edge_attrs_list = _get_edge_attrs_between(
                    graph=graph,
                    source=source,
                    target=entity,
                )

                if not edge_attrs_list:
                    edge_attrs_list = [{}]

                for attrs in edge_attrs_list:
                    results.append(
                        _build_neighbor_item(
                            source=source,
                            target=entity,
                            attrs=attrs,
                            direction="in",
                            include_relation=include_relation,
                        )
                    )

            return results

        except Exception:
            pass

    # -----------------------------------------------------
    # 2. 无 predecessors 时，扫描 edges(data=True)
    # -----------------------------------------------------
    if hasattr(graph, "edges"):
        try:
            for edge in graph.edges(data=True):
                if len(edge) != 3:
                    continue

                source, target, attrs = edge

                if str(target) != str(entity):
                    continue

                if isinstance(attrs, dict):
                    edge_attrs = dict(attrs)
                else:
                    edge_attrs = {"value": attrs}

                results.append(
                    _build_neighbor_item(
                        source=str(source),
                        target=entity,
                        attrs=edge_attrs,
                        direction="in",
                        include_relation=include_relation,
                    )
                )

            if results:
                return results

        except Exception:
            pass

    # -----------------------------------------------------
    # 3. dict 图结构
    # -----------------------------------------------------
    if isinstance(graph, dict):
        results.extend(
            _get_in_neighbors_from_dict_graph(
                graph=graph,
                entity=entity,
                include_relation=include_relation,
            )
        )

    return results


# =========================================================
# 5. dict 图结构邻居查询
# =========================================================

def _get_out_neighbors_from_dict_graph(
    *,
    graph: Dict[Any, Any],
    entity: str,
    include_relation: bool,
) -> List[Dict[str, Any]]:
    """
    从 dict 图结构中查询出边邻居。

    支持：
        1. {"A": {"B": {"relation": "r"}}}
        2. {"A": [{"target": "B", "relation": "r"}]}
        3. {"edges": [{"head": "A", "relation": "r", "tail": "B"}]}
    """

    results: List[Dict[str, Any]] = []

    adjacency = graph.get(entity)

    if isinstance(adjacency, dict):
        for target, attrs in adjacency.items():
            if isinstance(attrs, dict):
                edge_attrs = dict(attrs)
            else:
                edge_attrs = {"value": attrs}

            results.append(
                _build_neighbor_item(
                    source=entity,
                    target=str(target),
                    attrs=edge_attrs,
                    direction="out",
                    include_relation=include_relation,
                )
            )

    elif isinstance(adjacency, list):
        for item in adjacency:
            if not isinstance(item, dict):
                continue

            target = _first_non_empty_str(
                item,
                ["target", "tail", "object", "neighbor", "to", "entity", "name"],
            )

            if not target:
                continue

            results.append(
                _build_neighbor_item(
                    source=entity,
                    target=target,
                    attrs=dict(item),
                    direction="out",
                    include_relation=include_relation,
                )
            )

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

            if head == entity and tail:
                results.append(
                    _build_neighbor_item(
                        source=head,
                        target=tail,
                        attrs=dict(edge),
                        direction="out",
                        include_relation=include_relation,
                    )
                )

    return results


def _get_in_neighbors_from_dict_graph(
    *,
    graph: Dict[Any, Any],
    entity: str,
    include_relation: bool,
) -> List[Dict[str, Any]]:
    """
    从 dict 图结构中查询入边邻居。
    """

    results: List[Dict[str, Any]] = []

    # -----------------------------------------------------
    # 1. 扫描 graph["edges"]
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

            if tail == entity and head:
                results.append(
                    _build_neighbor_item(
                        source=head,
                        target=tail,
                        attrs=dict(edge),
                        direction="in",
                        include_relation=include_relation,
                    )
                )

    # -----------------------------------------------------
    # 2. 扫描邻接结构
    # -----------------------------------------------------
    for possible_source, adjacency in graph.items():
        if possible_source == "edges":
            continue

        possible_source = str(possible_source)

        if isinstance(adjacency, dict):
            if entity in adjacency:
                attrs = adjacency[entity]

                if isinstance(attrs, dict):
                    edge_attrs = dict(attrs)
                else:
                    edge_attrs = {"value": attrs}

                results.append(
                    _build_neighbor_item(
                        source=possible_source,
                        target=entity,
                        attrs=edge_attrs,
                        direction="in",
                        include_relation=include_relation,
                    )
                )

        elif isinstance(adjacency, list):
            for item in adjacency:
                if not isinstance(item, dict):
                    continue

                target = _first_non_empty_str(
                    item,
                    ["target", "tail", "object", "neighbor", "to", "entity", "name"],
                )

                if target == entity:
                    results.append(
                        _build_neighbor_item(
                            source=possible_source,
                            target=entity,
                            attrs=dict(item),
                            direction="in",
                            include_relation=include_relation,
                        )
                    )

    return results


# =========================================================
# 6. 边属性提取
# =========================================================

def _get_edge_attrs_between(
    *,
    graph: Any,
    source: str,
    target: str,
) -> List[Dict[str, Any]]:
    """
    获取 source -> target 的边属性列表。
    """

    if hasattr(graph, "get_edge_data"):
        try:
            edge_data = graph.get_edge_data(source, target)

            if edge_data:
                return _normalize_edge_data(edge_data)

        except Exception:
            pass

    if hasattr(graph, "edges"):
        try:
            matched: List[Dict[str, Any]] = []

            for edge in graph.edges(data=True):
                if len(edge) != 3:
                    continue

                u, v, attrs = edge

                if str(u) == str(source) and str(v) == str(target):
                    if isinstance(attrs, dict):
                        matched.append(dict(attrs))
                    else:
                        matched.append({"value": attrs})

            if matched:
                return matched

        except Exception:
            pass

    if isinstance(graph, dict):
        return _get_edge_attrs_from_dict_graph(
            graph=graph,
            source=source,
            target=target,
        )

    return []


def _normalize_edge_data(edge_data: Any) -> List[Dict[str, Any]]:
    """
    标准化 NetworkX get_edge_data 返回值。

    普通 DiGraph:
        {"relation": "r"}

    MultiDiGraph:
        {
            0: {"relation": "r1"},
            1: {"relation": "r2"}
        }
    """

    if edge_data is None:
        return []

    if not isinstance(edge_data, dict):
        return [{"value": edge_data}]

    if not edge_data:
        return []

    if _looks_like_single_edge_attrs(edge_data):
        return [dict(edge_data)]

    normalized: List[Dict[str, Any]] = []

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


def _get_edge_attrs_from_dict_graph(
    *,
    graph: Dict[Any, Any],
    source: str,
    target: str,
) -> List[Dict[str, Any]]:
    """
    从 dict 图结构中获取边属性。
    """

    results: List[Dict[str, Any]] = []

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

            if item_target == target:
                results.append(dict(item))

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

            if head == source and tail == target:
                results.append(dict(edge))

    return results


# =========================================================
# 7. Neighbor Item 构造
# =========================================================

def _build_neighbor_item(
    *,
    source: str,
    target: str,
    attrs: Dict[str, Any],
    direction: str,
    include_relation: bool,
) -> Dict[str, Any]:
    """
    构造统一 neighbor item。
    """

    relation = _extract_relation_name(attrs) if include_relation else "related_to"
    score = _extract_score(attrs)
    text = _extract_text(
        attrs=attrs,
        source=source,
        relation=relation,
        target=target,
    )

    # 对于 in 方向，当前 entity 实际是 target。
    # kg_retrieval_node 会把 source_entity 作为当前实体，因此这里仍然保留 source/target。
    return {
        "source": source,
        "relation": relation,
        "target": target,
        "neighbor": target if direction == "out" else source,
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
    source: str,
    relation: str,
    target: str,
) -> str:
    """
    构造邻居文本。
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

    return f"{source} --{relation}--> {target}"


# =========================================================
# 8. 图结构判断
# =========================================================

def _graph_has_node(graph: Any, node: str) -> bool:
    """
    判断图中是否存在节点。
    """

    if graph is None:
        return False

    if hasattr(graph, "has_node"):
        try:
            return bool(graph.has_node(node))
        except Exception:
            pass

    if hasattr(graph, "nodes"):
        try:
            return node in graph.nodes
        except Exception:
            pass

        try:
            return node in graph.nodes()
        except Exception:
            pass

    if isinstance(graph, dict):
        if node in graph:
            return True

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

                if node in {head, tail}:
                    return True

    try:
        return node in graph
    except Exception:
        return False


# =========================================================
# 9. 去重与通用工具
# =========================================================

def _deduplicate_neighbors(
    neighbors: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    邻居结果去重。
    """

    seen = set()
    unique: List[Dict[str, Any]] = []

    for item in neighbors:
        key = (
            str(item.get("source", "")),
            str(item.get("relation", "")),
            str(item.get("target", "")),
            str(item.get("direction", "")),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    return unique


def _normalize_direction(direction: str) -> str:
    """
    规范化方向参数。
    """

    direction = str(direction or "both").strip().lower()

    if direction not in {"out", "in", "both"}:
        return "both"

    return direction


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
    entity: str,
    reason: str,
) -> Dict[str, Any]:
    """
    构造空结果。
    """

    return {
        "entity": entity,
        "found": False,
        "neighbors": [],
        "num_neighbors": 0,
        "reason": reason,
    }


# =========================================================
# 10. 快速测试入口
# =========================================================

if __name__ == "__main__":
    import networkx as nx

    graph = nx.DiGraph()

    graph.add_edge(
        "A",
        "B",
        relation="r1",
        score=0.9,
    )
    graph.add_edge(
        "C",
        "A",
        relation="r2",
        score=0.8,
    )

    result = neighbor_search(
        graph=graph,
        entity="A",
        direction="both",
        max_neighbors=10,
    )

    from pprint import pprint

    pprint(result)