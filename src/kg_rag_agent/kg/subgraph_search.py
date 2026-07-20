# -*- coding: utf-8 -*-
"""
subgraph_search.py

局部子图查询模块。

作用：
    1. 围绕一个或多个对象抽取局部相关结构。
    2. 支持 NetworkX Graph / DiGraph / MultiDiGraph。
    3. 支持简单 dict 图结构。
    4. 返回统一结构，供 evidence_builder.py 或 kg_retrieval_node.py 转换为 evidence。

本文件属于 kg 底层能力层：
    kg/
        subgraph_search.py

它不负责：
    1. 用户问题解析。
    2. 对象识别。
    3. 最终回答生成。
    4. LangGraph 节点调度。

典型用途：
    当用户只问某个对象的相关信息，或者直接关系 / 路径信息不足时，
    可以抽取该对象附近若干跳的局部结构作为补充材料。
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# =========================================================
# 1. 对外主函数
# =========================================================

def subgraph_search(
    graph: Any,
    entities: Optional[List[str]] = None,
    *,
    entity: Optional[str] = None,
    nodes: Optional[List[str]] = None,
    max_depth: int = 2,
    max_nodes: int = 50,
    max_edges: int = 100,
    direction: str = "both",
    include_center: bool = True,
) -> Dict[str, Any]:
    """
    围绕一个或多个对象抽取局部子图。

    Args:
        graph:
            图结构对象。
            通常是 networkx.Graph / DiGraph / MultiDiGraph。

        entities:
            中心对象列表。

        entity:
            单个中心对象，作为 entities 的便捷参数。

        nodes:
            entities 的别名参数。

        max_depth:
            最大扩展跳数。
            1 表示只取一跳邻居。
            2 表示取二跳邻居。

        max_nodes:
            最多返回多少个节点。

        max_edges:
            最多返回多少条边。

        direction:
            扩展方向：
                - "out"  ：只沿出边扩展
                - "in"   ：只沿入边扩展
                - "both" ：同时考虑出边和入边

        include_center:
            是否在结果 nodes 中包含中心对象。

    Returns:
        Dict[str, Any]:
            {
                "center_entities": ["A"],
                "found": true,
                "nodes": [
                    {"id": "A", "label": "A", "depth": 0, "metadata": {...}}
                ],
                "edges": [
                    {"head": "A", "relation": "r", "tail": "B", "score": 1.0}
                ],
                "triples": [
                    {"head": "A", "relation": "r", "tail": "B"}
                ],
                "num_nodes": 2,
                "num_edges": 1
            }
    """

    center_entities = _normalize_centers(
        entities=entities,
        entity=entity,
        nodes=nodes,
    )

    if not center_entities:
        return _empty_result(
            center_entities=[],
            reason="center entities are empty",
        )

    if graph is None:
        return _empty_result(
            center_entities=center_entities,
            reason="graph is None",
        )

    direction = _normalize_direction(direction)
    max_depth = max(int(max_depth), 0)
    max_nodes = max(int(max_nodes), 1)
    max_edges = max(int(max_edges), 1)

    valid_centers = [
        center for center in center_entities
        if _graph_has_node(graph, center)
    ]

    if not valid_centers:
        return _empty_result(
            center_entities=center_entities,
            reason="no center entity found in graph",
        )

    visited_nodes, node_depths = _collect_nodes_by_bfs(
        graph=graph,
        centers=valid_centers,
        max_depth=max_depth,
        max_nodes=max_nodes,
        direction=direction,
        include_center=include_center,
    )

    edges = _collect_edges_among_nodes(
        graph=graph,
        nodes=visited_nodes,
        max_edges=max_edges,
        direction=direction,
    )

    node_items = _build_node_items(
        graph=graph,
        nodes=visited_nodes,
        node_depths=node_depths,
        center_entities=valid_centers,
    )

    edge_items = _build_edge_items(edges)
    triples = _edges_to_triples(edge_items)

    return {
        "center_entities": valid_centers,
        "found": bool(node_items or edge_items),
        "nodes": node_items,
        "edges": edge_items,
        "triples": triples,
        "num_nodes": len(node_items),
        "num_edges": len(edge_items),
        "max_depth": max_depth,
        "direction": direction,
    }


# =========================================================
# 2. 兼容别名函数
# =========================================================

def search_subgraph(
    graph: Any,
    entities: List[str],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    subgraph_search 的别名。
    """

    return subgraph_search(
        graph=graph,
        entities=entities,
        **kwargs,
    )


def ego_graph_search(
    graph: Any,
    entity: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    单中心对象局部子图查询。
    """

    return subgraph_search(
        graph=graph,
        entity=entity,
        **kwargs,
    )


# =========================================================
# 3. 中心对象规范化
# =========================================================

def _normalize_centers(
    *,
    entities: Optional[List[str]],
    entity: Optional[str],
    nodes: Optional[List[str]],
) -> List[str]:
    """
    规范化中心对象列表。
    """

    raw_centers: List[Any] = []

    if entities:
        raw_centers.extend(entities)

    if nodes:
        raw_centers.extend(nodes)

    if entity:
        raw_centers.append(entity)

    centers: List[str] = []
    seen: Set[str] = set()

    for item in raw_centers:
        value = str(item or "").strip()

        if not value:
            continue

        if value in seen:
            continue

        seen.add(value)
        centers.append(value)

    return centers


# =========================================================
# 4. BFS 收集节点
# =========================================================

def _collect_nodes_by_bfs(
    *,
    graph: Any,
    centers: List[str],
    max_depth: int,
    max_nodes: int,
    direction: str,
    include_center: bool,
) -> Tuple[List[str], Dict[str, int]]:
    """
    从中心对象开始 BFS，收集局部节点。
    """

    visited: Set[str] = set()
    ordered_nodes: List[str] = []
    node_depths: Dict[str, int] = {}

    queue: deque[Tuple[str, int]] = deque()

    for center in centers:
        queue.append((center, 0))

        if include_center:
            visited.add(center)
            ordered_nodes.append(center)
            node_depths[center] = 0

    while queue and len(ordered_nodes) < max_nodes:
        current, depth = queue.popleft()

        if depth >= max_depth:
            continue

        neighbors = _iter_neighbors(
            graph=graph,
            node=current,
            direction=direction,
        )

        for neighbor in neighbors:
            neighbor = str(neighbor)

            if not neighbor:
                continue

            if neighbor in visited:
                continue

            visited.add(neighbor)
            ordered_nodes.append(neighbor)
            node_depths[neighbor] = depth + 1
            queue.append((neighbor, depth + 1))

            if len(ordered_nodes) >= max_nodes:
                break

    # 如果 include_center=False，但没有收集到任何邻居，允许返回空。
    return ordered_nodes, node_depths


# =========================================================
# 5. 邻居遍历
# =========================================================

def _iter_neighbors(
    *,
    graph: Any,
    node: str,
    direction: str,
) -> List[str]:
    """
    根据方向获取邻居。
    """

    neighbors: Set[str] = set()

    if direction in {"out", "both"}:
        for neighbor in _iter_out_neighbors(graph, node):
            neighbors.add(str(neighbor))

    if direction in {"in", "both"}:
        for neighbor in _iter_in_neighbors(graph, node):
            neighbors.add(str(neighbor))

    return list(neighbors)


def _iter_out_neighbors(graph: Any, node: str) -> List[str]:
    """
    获取出边邻居。
    """

    neighbors: Set[str] = set()

    if hasattr(graph, "successors"):
        try:
            for neighbor in graph.successors(node):
                neighbors.add(str(neighbor))
            return list(neighbors)
        except Exception:
            pass

    if hasattr(graph, "neighbors"):
        try:
            for neighbor in graph.neighbors(node):
                neighbors.add(str(neighbor))
            return list(neighbors)
        except Exception:
            pass

    if isinstance(graph, dict):
        for neighbor in _iter_out_neighbors_from_dict(graph, node):
            neighbors.add(str(neighbor))

    return list(neighbors)


def _iter_in_neighbors(graph: Any, node: str) -> List[str]:
    """
    获取入边邻居。
    """

    neighbors: Set[str] = set()

    if hasattr(graph, "predecessors"):
        try:
            for neighbor in graph.predecessors(node):
                neighbors.add(str(neighbor))
            return list(neighbors)
        except Exception:
            pass

    if hasattr(graph, "edges"):
        try:
            for edge in graph.edges(data=True):
                if len(edge) != 3:
                    continue

                source, target, _ = edge

                if str(target) == str(node):
                    neighbors.add(str(source))

            if neighbors:
                return list(neighbors)

        except Exception:
            pass

    if isinstance(graph, dict):
        for neighbor in _iter_in_neighbors_from_dict(graph, node):
            neighbors.add(str(neighbor))

    return list(neighbors)


def _iter_out_neighbors_from_dict(
    graph: Dict[Any, Any],
    node: str,
) -> List[str]:
    """
    从 dict 图结构中获取出边邻居。
    """

    neighbors: Set[str] = set()

    adjacency = graph.get(node)

    if isinstance(adjacency, dict):
        for target in adjacency.keys():
            neighbors.add(str(target))

    elif isinstance(adjacency, list):
        for item in adjacency:
            if not isinstance(item, dict):
                continue

            target = _first_non_empty_str(
                item,
                ["target", "tail", "object", "neighbor", "to", "entity", "name"],
            )

            if target:
                neighbors.add(str(target))

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

            if head == node and tail:
                neighbors.add(str(tail))

    return list(neighbors)


def _iter_in_neighbors_from_dict(
    graph: Dict[Any, Any],
    node: str,
) -> List[str]:
    """
    从 dict 图结构中获取入边邻居。
    """

    neighbors: Set[str] = set()

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

            if tail == node and head:
                neighbors.add(str(head))

    for possible_source, adjacency in graph.items():
        if possible_source == "edges":
            continue

        possible_source = str(possible_source)

        if isinstance(adjacency, dict):
            if node in adjacency:
                neighbors.add(possible_source)

        elif isinstance(adjacency, list):
            for item in adjacency:
                if not isinstance(item, dict):
                    continue

                target = _first_non_empty_str(
                    item,
                    ["target", "tail", "object", "neighbor", "to", "entity", "name"],
                )

                if target == node:
                    neighbors.add(possible_source)

    return list(neighbors)


# =========================================================
# 6. 收集节点之间的边
# =========================================================

def _collect_edges_among_nodes(
    *,
    graph: Any,
    nodes: List[str],
    max_edges: int,
    direction: str,
) -> List[Dict[str, Any]]:
    """
    收集局部节点集合内部的边。
    """

    node_set = set(nodes)
    edges: List[Dict[str, Any]] = []

    if not node_set:
        return []

    # -----------------------------------------------------
    # 1. NetworkX edges(data=True)
    # -----------------------------------------------------
    if hasattr(graph, "edges"):
        try:
            for edge in graph.edges(data=True):
                if len(edge) != 3:
                    continue

                source, target, attrs = edge
                source = str(source)
                target = str(target)

                if source not in node_set or target not in node_set:
                    continue

                if direction == "out":
                    # out 方向仍然允许中心向外形成的边。
                    pass

                if isinstance(attrs, dict):
                    edge_attrs = dict(attrs)
                else:
                    edge_attrs = {"value": attrs}

                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "attrs": edge_attrs,
                    }
                )

                if len(edges) >= max_edges:
                    return edges

            if edges:
                return edges

        except Exception:
            pass

    # -----------------------------------------------------
    # 2. dict 图结构
    # -----------------------------------------------------
    if isinstance(graph, dict):
        edges.extend(
            _collect_edges_from_dict_graph(
                graph=graph,
                node_set=node_set,
                max_edges=max_edges,
            )
        )

    return edges[:max_edges]


def _collect_edges_from_dict_graph(
    *,
    graph: Dict[Any, Any],
    node_set: Set[str],
    max_edges: int,
) -> List[Dict[str, Any]]:
    """
    从 dict 图结构中收集边。
    """

    edges: List[Dict[str, Any]] = []

    raw_edges = graph.get("edges")

    if isinstance(raw_edges, list):
        for edge in raw_edges:
            if not isinstance(edge, dict):
                continue

            source = _first_non_empty_str(
                edge,
                ["head", "source", "subject", "h"],
            )
            target = _first_non_empty_str(
                edge,
                ["tail", "target", "object", "t"],
            )

            if source in node_set and target in node_set:
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "attrs": dict(edge),
                    }
                )

                if len(edges) >= max_edges:
                    return edges

    for source, adjacency in graph.items():
        if source == "edges":
            continue

        source = str(source)

        if source not in node_set:
            continue

        if isinstance(adjacency, dict):
            for target, attrs in adjacency.items():
                target = str(target)

                if target not in node_set:
                    continue

                if isinstance(attrs, dict):
                    edge_attrs = dict(attrs)
                else:
                    edge_attrs = {"value": attrs}

                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "attrs": edge_attrs,
                    }
                )

                if len(edges) >= max_edges:
                    return edges

        elif isinstance(adjacency, list):
            for item in adjacency:
                if not isinstance(item, dict):
                    continue

                target = _first_non_empty_str(
                    item,
                    ["target", "tail", "object", "neighbor", "to", "entity", "name"],
                )

                if target not in node_set:
                    continue

                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "attrs": dict(item),
                    }
                )

                if len(edges) >= max_edges:
                    return edges

    return edges


# =========================================================
# 7. 结果构造
# =========================================================

def _build_node_items(
    *,
    graph: Any,
    nodes: List[str],
    node_depths: Dict[str, int],
    center_entities: List[str],
) -> List[Dict[str, Any]]:
    """
    构造 nodes 结果。
    """

    center_set = set(center_entities)
    node_items: List[Dict[str, Any]] = []

    for node in nodes:
        attrs = _get_node_attrs(graph, node)

        node_items.append(
            {
                "id": node,
                "label": _get_node_label(node, attrs),
                "depth": int(node_depths.get(node, 0)),
                "is_center": node in center_set,
                "metadata": attrs,
            }
        )

    return node_items


def _build_edge_items(
    edges: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    构造 edges 结果。
    """

    edge_items: List[Dict[str, Any]] = []

    for edge in edges:
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        attrs = edge.get("attrs", {})

        if not isinstance(attrs, dict):
            attrs = {}

        if not source or not target:
            continue

        relation = _extract_relation_name(attrs)
        score = _extract_score(attrs)
        text = _extract_text(
            attrs=attrs,
            source=source,
            relation=relation,
            target=target,
        )

        edge_items.append(
            {
                "head": source,
                "source": source,
                "relation": relation,
                "tail": target,
                "target": target,
                "score": score,
                "text": text,
                "metadata": {
                    "edge_attrs": attrs,
                },
            }
        )

    return _deduplicate_edges(edge_items)


def _edges_to_triples(
    edges: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    edges 转 triples。
    """

    triples: List[Dict[str, Any]] = []

    for edge in edges:
        head = str(edge.get("head", "") or edge.get("source", "")).strip()
        relation = str(edge.get("relation", "related_to")).strip()
        tail = str(edge.get("tail", "") or edge.get("target", "")).strip()

        if not head or not tail:
            continue

        triples.append(
            {
                "head": head,
                "relation": relation,
                "tail": tail,
            }
        )

    return triples


# =========================================================
# 8. 图结构适配工具
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


def _get_node_attrs(
    graph: Any,
    node: str,
) -> Dict[str, Any]:
    """
    获取节点属性。
    """

    if graph is None:
        return {}

    if hasattr(graph, "nodes"):
        try:
            attrs = graph.nodes[node]
            if isinstance(attrs, dict):
                return dict(attrs)
        except Exception:
            pass

    if isinstance(graph, dict):
        value = graph.get(node)

        if isinstance(value, dict):
            # 如果是邻接字典，里面不一定是节点属性。
            # 这里仅保留明显像节点属性的字段。
            node_attr_keys = {
                "id",
                "name",
                "label",
                "title",
                "entity_id",
                "entity_name",
                "type",
                "description",
                "aliases",
            }

            if any(key in value for key in node_attr_keys):
                return {
                    key: val for key, val in value.items()
                    if key in node_attr_keys
                }

    return {}


def _get_node_label(
    node: str,
    attrs: Dict[str, Any],
) -> str:
    """
    获取节点展示 label。
    """

    for key in ["label", "name", "entity_name", "title"]:
        value = attrs.get(key)

        if value:
            return str(value)

    return str(node)


# =========================================================
# 9. 边属性工具
# =========================================================

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
    构造边文本。
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
# 10. 去重与通用工具
# =========================================================

def _deduplicate_edges(
    edges: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    边去重。
    """

    seen = set()
    unique: List[Dict[str, Any]] = []

    for edge in edges:
        key = (
            str(edge.get("head", "")),
            str(edge.get("relation", "")),
            str(edge.get("tail", "")),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(edge)

    return unique


def _normalize_direction(direction: str) -> str:
    """
    规范化方向。
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
    center_entities: List[str],
    reason: str,
) -> Dict[str, Any]:
    """
    构造空结果。
    """

    return {
        "center_entities": center_entities,
        "found": False,
        "nodes": [],
        "edges": [],
        "triples": [],
        "num_nodes": 0,
        "num_edges": 0,
        "reason": reason,
    }


# =========================================================
# 11. 快速测试入口
# =========================================================

if __name__ == "__main__":
    import networkx as nx

    graph = nx.DiGraph()

    graph.add_node("A", label="Entity A")
    graph.add_node("B", label="Entity B")
    graph.add_node("C", label="Entity C")
    graph.add_node("D", label="Entity D")

    graph.add_edge("A", "B", relation="r1")
    graph.add_edge("B", "C", relation="r2")
    graph.add_edge("D", "A", relation="r3")

    result = subgraph_search(
        graph=graph,
        entity="A",
        max_depth=2,
        direction="both",
    )

    from pprint import pprint

    pprint(result)