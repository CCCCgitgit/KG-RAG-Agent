# -*- coding: utf-8 -*-
"""
path_search.py

多跳路径查询模块。

作用：
    1. 查询两个对象之间是否存在路径。
    2. 支持 NetworkX Graph / DiGraph / MultiDiGraph。
    3. 支持简单 dict 图结构。
    4. 返回统一路径结构，供 kg_retrieval_node.py 转换为 evidence。

本文件属于 kg 底层能力层：
    kg/
        path_search.py

它不负责：
    1. 用户问题解析。
    2. 实体链接。
    3. 直接关系查询。
    4. 邻居查询。
    5. 最终回答生成。
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# =========================================================
# 1. 对外主函数
# =========================================================

def path_search(
    graph: Any,
    source: Optional[str] = None,
    target: Optional[str] = None,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_paths: int = 5,
    max_path_length: int = 4,
    max_depth: Optional[int] = None,
    directed: bool = True,
    include_triples: bool = True,
) -> Dict[str, Any]:
    """
    查询 source 到 target 的多跳路径。

    Args:
        graph:
            图结构对象，通常是 networkx.Graph / DiGraph / MultiDiGraph。

        source:
            起始节点。

        target:
            目标节点。

        start:
            source 的别名参数。

        end:
            target 的别名参数。

        max_paths:
            最多返回多少条路径。

        max_path_length:
            路径最大节点数。
            例如 A -> B -> C 的 path_length 为 3。

        max_depth:
            max_path_length 的别名。
            如果传入 max_depth，则优先使用 max_depth。

        directed:
            是否按有向路径搜索。

        include_triples:
            是否将路径中的边转成 triples。

    Returns:
        Dict[str, Any]:
            {
                "source": "A",
                "target": "C",
                "found": true,
                "paths": [
                    {
                        "path": ["A", "B", "C"],
                        "triples": [
                            {"head": "A", "relation": "r1", "tail": "B"},
                            {"head": "B", "relation": "r2", "tail": "C"}
                        ],
                        "text": "A --r1--> B --r2--> C",
                        "score": 0.82,
                        "path_length": 3
                    }
                ],
                "num_paths": 1
            }
    """

    source_node = str(source or start or "").strip()
    target_node = str(target or end or "").strip()

    if max_depth is not None:
        max_path_length = int(max_depth)

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

    if source_node == target_node:
        path_item = _build_path_item(
            graph=graph,
            path=[source_node],
            include_triples=include_triples,
        )

        return {
            "source": source_node,
            "target": target_node,
            "found": True,
            "paths": [path_item],
            "num_paths": 1,
        }

    max_paths = max(int(max_paths), 1)
    max_path_length = max(int(max_path_length), 2)

    paths = _find_paths(
        graph=graph,
        source=source_node,
        target=target_node,
        max_paths=max_paths,
        max_path_length=max_path_length,
        directed=directed,
    )

    path_items = [
        _build_path_item(
            graph=graph,
            path=path,
            include_triples=include_triples,
        )
        for path in paths
    ]

    path_items = _deduplicate_path_items(path_items)
    path_items = sorted(
        path_items,
        key=lambda item: (
            item.get("path_length", 999999),
            -float(item.get("score", 0.0)),
        ),
    )[:max_paths]

    return {
        "source": source_node,
        "target": target_node,
        "found": len(path_items) > 0,
        "paths": path_items,
        "num_paths": len(path_items),
    }


# =========================================================
# 2. 兼容别名函数
# =========================================================

def search_path(
    graph: Any,
    source: str,
    target: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    path_search 的别名。
    """

    return path_search(
        graph=graph,
        source=source,
        target=target,
        **kwargs,
    )


def find_paths(
    graph: Any,
    source: str,
    target: str,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    只返回 paths 列表的便捷函数。
    """

    result = path_search(
        graph=graph,
        source=source,
        target=target,
        **kwargs,
    )

    return result.get("paths", [])


# =========================================================
# 3. 路径搜索主逻辑
# =========================================================

def _find_paths(
    *,
    graph: Any,
    source: str,
    target: str,
    max_paths: int,
    max_path_length: int,
    directed: bool,
) -> List[List[str]]:
    """
    查找路径。

    优先使用 NetworkX all_simple_paths；
    如果不可用，则使用自定义 BFS。
    """

    if _looks_like_networkx_graph(graph):
        paths = _find_paths_with_networkx(
            graph=graph,
            source=source,
            target=target,
            max_paths=max_paths,
            max_path_length=max_path_length,
            directed=directed,
        )

        if paths:
            return paths

    return _find_paths_with_bfs(
        graph=graph,
        source=source,
        target=target,
        max_paths=max_paths,
        max_path_length=max_path_length,
        directed=directed,
    )


def _find_paths_with_networkx(
    *,
    graph: Any,
    source: str,
    target: str,
    max_paths: int,
    max_path_length: int,
    directed: bool,
) -> List[List[str]]:
    """
    使用 NetworkX 查找简单路径。
    """

    try:
        if hasattr(graph, "has_node"):
            if not graph.has_node(source) or not graph.has_node(target):
                return []

        search_graph = graph

        if not directed and hasattr(graph, "to_undirected"):
            search_graph = graph.to_undirected()

        import networkx as nx

        cutoff = max_path_length - 1

        raw_paths = nx.all_simple_paths(
            search_graph,
            source=source,
            target=target,
            cutoff=cutoff,
        )

        paths: List[List[str]] = []

        for path in raw_paths:
            path = [str(node) for node in path]

            if len(path) <= max_path_length:
                paths.append(path)

            if len(paths) >= max_paths:
                break

        return paths

    except Exception:
        return []


def _find_paths_with_bfs(
    *,
    graph: Any,
    source: str,
    target: str,
    max_paths: int,
    max_path_length: int,
    directed: bool,
) -> List[List[str]]:
    """
    使用 BFS 查找简单路径。

    说明：
        这里是轻量实现，不追求最复杂的图算法。
        目标是保证在 NetworkX 不可用或图结构较简单时仍能工作。
    """

    if not _graph_has_node(graph, source):
        return []

    if not _graph_has_node(graph, target):
        return []

    paths: List[List[str]] = []
    queue: deque[List[str]] = deque()

    queue.append([source])

    while queue and len(paths) < max_paths:
        path = queue.popleft()
        current = path[-1]

        if len(path) > max_path_length:
            continue

        if current == target:
            paths.append(path)
            continue

        if len(path) == max_path_length:
            continue

        neighbors = _iter_neighbors(
            graph=graph,
            node=current,
            directed=directed,
        )

        for neighbor in neighbors:
            neighbor = str(neighbor)

            if neighbor in path:
                continue

            new_path = path + [neighbor]

            if len(new_path) <= max_path_length:
                queue.append(new_path)

    return paths


# =========================================================
# 4. 邻居遍历
# =========================================================

def _iter_neighbors(
    *,
    graph: Any,
    node: str,
    directed: bool,
) -> List[str]:
    """
    获取节点邻居。

    directed=True:
        优先使用 successors。

    directed=False:
        同时考虑 successors / predecessors / neighbors。
    """

    neighbors: Set[str] = set()

    # -----------------------------------------------------
    # 1. NetworkX DiGraph successors
    # -----------------------------------------------------
    if directed and hasattr(graph, "successors"):
        try:
            for neighbor in graph.successors(node):
                neighbors.add(str(neighbor))
            return list(neighbors)
        except Exception:
            pass

    # -----------------------------------------------------
    # 2. NetworkX Graph neighbors
    # -----------------------------------------------------
    if hasattr(graph, "neighbors"):
        try:
            for neighbor in graph.neighbors(node):
                neighbors.add(str(neighbor))
        except Exception:
            pass

    # -----------------------------------------------------
    # 3. 无向时补充 predecessors
    # -----------------------------------------------------
    if not directed and hasattr(graph, "predecessors"):
        try:
            for neighbor in graph.predecessors(node):
                neighbors.add(str(neighbor))
        except Exception:
            pass

    # -----------------------------------------------------
    # 4. dict 图结构
    # -----------------------------------------------------
    if isinstance(graph, dict):
        for neighbor in _iter_neighbors_from_dict_graph(
            graph=graph,
            node=node,
            directed=directed,
        ):
            neighbors.add(str(neighbor))

    return list(neighbors)


def _iter_neighbors_from_dict_graph(
    *,
    graph: Dict[Any, Any],
    node: str,
    directed: bool,
) -> List[str]:
    """
    从 dict 图结构中获取邻居。

    支持：
        1. {"A": {"B": {...}}}
        2. {"A": [{"target": "B"}]}
        3. {"edges": [{"head": "A", "tail": "B"}]}
    """

    neighbors: Set[str] = set()

    # -----------------------------------------------------
    # 1. graph[node] 邻接结构
    # -----------------------------------------------------
    adjacency = graph.get(node)

    if isinstance(adjacency, dict):
        for key in adjacency.keys():
            neighbors.add(str(key))

    elif isinstance(adjacency, list):
        for item in adjacency:
            if not isinstance(item, dict):
                continue

            target = _first_non_empty_str(
                item,
                ["target", "tail", "object", "neighbor", "to"],
            )

            if target:
                neighbors.add(str(target))

    # -----------------------------------------------------
    # 2. graph["edges"] 三元组列表
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

            if head == node and tail:
                neighbors.add(str(tail))

            if not directed and tail == node and head:
                neighbors.add(str(head))

    # -----------------------------------------------------
    # 3. 无向时扫描反向邻接
    # -----------------------------------------------------
    if not directed:
        for possible_source, adjacency_value in graph.items():
            if possible_source == "edges":
                continue

            if isinstance(adjacency_value, dict):
                if node in adjacency_value:
                    neighbors.add(str(possible_source))

            elif isinstance(adjacency_value, list):
                for item in adjacency_value:
                    if not isinstance(item, dict):
                        continue

                    target = _first_non_empty_str(
                        item,
                        ["target", "tail", "object", "neighbor", "to"],
                    )

                    if target == node:
                        neighbors.add(str(possible_source))

    return list(neighbors)


# =========================================================
# 5. Path Item 构造
# =========================================================

def _build_path_item(
    *,
    graph: Any,
    path: List[str],
    include_triples: bool,
) -> Dict[str, Any]:
    """
    构造统一 path item。
    """

    triples: List[Dict[str, Any]] = []

    if include_triples and len(path) >= 2:
        triples = _path_to_triples(
            graph=graph,
            path=path,
        )

    text = _path_to_text(
        path=path,
        triples=triples,
    )

    return {
        "path": path,
        "triples": triples,
        "text": text,
        "score": _score_path(path),
        "path_length": len(path),
        "metadata": {
            "num_hops": max(len(path) - 1, 0),
        },
    }


def _path_to_triples(
    *,
    graph: Any,
    path: List[str],
) -> List[Dict[str, Any]]:
    """
    将路径转成 triples。

    例如：
        ["A", "B", "C"]

    转成：
        [
            {"head": "A", "relation": "r1", "tail": "B"},
            {"head": "B", "relation": "r2", "tail": "C"}
        ]
    """

    triples: List[Dict[str, Any]] = []

    for idx in range(len(path) - 1):
        head = path[idx]
        tail = path[idx + 1]

        relation = _get_relation_between(
            graph=graph,
            source=head,
            target=tail,
        )

        triples.append(
            {
                "head": head,
                "relation": relation,
                "tail": tail,
            }
        )

    return triples


def _path_to_text(
    *,
    path: List[str],
    triples: List[Dict[str, Any]],
) -> str:
    """
    将路径转成文本。
    """

    if not path:
        return ""

    if len(path) == 1:
        return str(path[0])

    if triples:
        parts: List[str] = []

        for idx, triple in enumerate(triples):
            head = str(triple.get("head", ""))
            relation = str(triple.get("relation", "related_to"))
            tail = str(triple.get("tail", ""))

            if idx == 0:
                parts.append(head)

            parts.append(f"--{relation}-->")
            parts.append(tail)

        return " ".join(parts)

    return " -> ".join(str(node) for node in path)


# =========================================================
# 6. 边关系提取
# =========================================================

def _get_relation_between(
    *,
    graph: Any,
    source: str,
    target: str,
) -> str:
    """
    获取 source -> target 之间的关系名称。
    """

    attrs_list = _get_edge_attrs_between(
        graph=graph,
        source=source,
        target=target,
    )

    if not attrs_list:
        return "related_to"

    attrs = attrs_list[0]

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
            matched = []

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
# 7. 图结构判断
# =========================================================

def _looks_like_networkx_graph(graph: Any) -> bool:
    """
    判断是否像 NetworkX 图对象。
    """

    return (
        hasattr(graph, "nodes")
        and hasattr(graph, "edges")
        and (
            hasattr(graph, "neighbors")
            or hasattr(graph, "successors")
        )
    )


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
# 8. 去重与评分
# =========================================================

def _deduplicate_path_items(
    path_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    路径结果去重。
    """

    seen = set()
    unique: List[Dict[str, Any]] = []

    for item in path_items:
        path = item.get("path", [])

        if not isinstance(path, list):
            continue

        key = tuple(str(node) for node in path)

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    return unique


def _score_path(path: List[str]) -> float:
    """
    根据路径长度给基础分。

    节点越少，路径越直接，分数越高。
    """

    length = len(path)

    if length <= 1:
        return 1.0

    if length == 2:
        return 0.90

    if length == 3:
        return 0.82

    if length == 4:
        return 0.74

    if length == 5:
        return 0.66

    return 0.58


# =========================================================
# 9. 通用工具
# =========================================================

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
        "paths": [],
        "num_paths": 0,
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
    )
    graph.add_edge(
        "B",
        "C",
        relation="r2",
    )
    graph.add_edge(
        "A",
        "D",
        relation="r3",
    )
    graph.add_edge(
        "D",
        "C",
        relation="r4",
    )

    result = path_search(
        graph=graph,
        source="A",
        target="C",
        max_paths=3,
        max_path_length=4,
    )

    from pprint import pprint

    pprint(result)