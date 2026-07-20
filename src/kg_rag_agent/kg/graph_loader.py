# -*- coding: utf-8 -*-
"""
graph_loader.py

图结构加载模块。

作用：
    1. 从磁盘加载已经构建好的图结构，例如 data/demo/kg/graph.pkl。
    2. 对外提供统一的 load_graph() 函数。
    3. 支持 NetworkX Graph / DiGraph / MultiDiGraph 等对象。
    4. 支持简单缓存，避免每次查询重复加载大图。
    5. 提供基础图统计函数，方便调试和日志记录。

本文件属于 kg 层：
    kg/
        graph_loader.py

它不负责：
    1. 构建图。
    2. 清洗原始数据。
    3. 实体链接。
    4. 路径查询。
    5. 最终回答生成。

这些工作分别交给：
    data_pipeline/graph_builder.py
    kg/entity_linker.py
    kg/path_search.py
    graph/nodes/generation_node.py
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# =========================================================
# 1. 全局缓存
# =========================================================

_GRAPH_CACHE: Dict[str, Any] = {}
_GRAPH_STATS_CACHE: Dict[str, Dict[str, Any]] = {}


# =========================================================
# 2. 默认路径
# =========================================================

DEFAULT_GRAPH_PATH = "data/demo/kg/graph.pkl"
DEFAULT_GRAPH_STATS_PATH = "data/demo/kg/graph_stats.json"


# =========================================================
# 3. 对外主函数：load_graph
# =========================================================

def load_graph(
    graph_path: Optional[str] = None,
    *,
    use_cache: bool = True,
    validate: bool = True,
) -> Any:
    """
    加载图结构。

    Args:
        graph_path:
            图文件路径。
            默认使用 data/demo/kg/graph.pkl。

        use_cache:
            是否启用内存缓存。
            图通常较大，建议开启。

        validate:
            是否加载后做基础校验。

    Returns:
        Any:
            图结构对象。
            通常是 networkx.Graph / DiGraph / MultiDiGraph。

    Example:
        graph = load_graph()
        graph = load_graph("data/demo/kg/graph.pkl")
    """

    resolved_path = resolve_project_path(graph_path or DEFAULT_GRAPH_PATH)
    cache_key = str(resolved_path)

    if use_cache and cache_key in _GRAPH_CACHE:
        return _GRAPH_CACHE[cache_key]

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Graph file not found: {resolved_path}. "
            f"Please build graph first, for example by running scripts/build_kg.py."
        )

    graph = _load_pickle(resolved_path)

    if validate:
        validate_graph(graph)

    if use_cache:
        _GRAPH_CACHE[cache_key] = graph
        _GRAPH_STATS_CACHE[cache_key] = get_graph_stats(graph)

    return graph


# =========================================================
# 4. GraphLoader 类
# =========================================================

class GraphLoader:
    """
    图加载器。

    用法：
        loader = GraphLoader(graph_path="data/demo/kg/graph.pkl")
        graph = loader.load()

    设计原因：
        有些模块喜欢函数式调用 load_graph()；
        有些模块喜欢对象式调用 GraphLoader().load()。
        两种方式都支持，方便后续工程扩展。
    """

    def __init__(
        self,
        graph_path: Optional[str] = None,
        *,
        use_cache: bool = True,
        validate: bool = True,
    ) -> None:
        self.graph_path = graph_path or DEFAULT_GRAPH_PATH
        self.use_cache = use_cache
        self.validate = validate
        self.graph: Optional[Any] = None

    def load(self) -> Any:
        """
        加载图结构。
        """

        self.graph = load_graph(
            graph_path=self.graph_path,
            use_cache=self.use_cache,
            validate=self.validate,
        )

        return self.graph

    def get_graph(self) -> Any:
        """
        获取图结构。

        如果尚未加载，则自动加载。
        """

        if self.graph is None:
            self.graph = self.load()

        return self.graph

    def stats(self) -> Dict[str, Any]:
        """
        获取图统计信息。
        """

        graph = self.get_graph()

        return get_graph_stats(graph)


# =========================================================
# 5. 路径解析
# =========================================================

def resolve_project_path(path: str | Path) -> Path:
    """
    解析项目路径。

    支持：
        1. 绝对路径
        2. 相对于项目根目录的路径
        3. 相对于当前工作目录的路径

    项目结构大致为：
        kg_rag_agent/
            data/
            src/
                kg_rag_agent/
                    kg/
                        graph_loader.py

    当前文件位置：
        src/kg_rag_agent/kg/graph_loader.py

    因此项目根目录一般是 parents[3]。
    """

    input_path = Path(path)

    if input_path.is_absolute():
        return input_path

    # 优先按当前工作目录解析
    cwd_path = Path.cwd() / input_path

    if cwd_path.exists():
        return cwd_path

    # 再按项目根目录解析
    project_root = get_project_root()
    project_path = project_root / input_path

    return project_path


def get_project_root() -> Path:
    """
    获取项目根目录。

    当前文件：
        src/kg_rag_agent/kg/graph_loader.py

    parents:
        0 -> kg/
        1 -> kg_rag_agent/
        2 -> src/
        3 -> 项目根目录
    """

    return Path(__file__).resolve().parents[3]


# =========================================================
# 6. Pickle 加载
# =========================================================

def _load_pickle(path: Path) -> Any:
    """
    加载 pickle 文件。
    """

    try:
        with open(path, "rb") as file:
            graph = pickle.load(file)

        return graph

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load graph pickle file: {path}. Detail: {exc}"
        ) from exc


def save_graph(
    graph: Any,
    graph_path: Optional[str] = None,
    *,
    overwrite: bool = True,
) -> Path:
    """
    保存图结构到 pickle 文件。

    主要供 data_pipeline/graph_builder.py 或脚本调用。

    Args:
        graph:
            图结构对象。

        graph_path:
            保存路径。
            默认 data/demo/kg/graph.pkl。

        overwrite:
            如果文件存在，是否覆盖。

    Returns:
        Path:
            实际保存路径。
    """

    resolved_path = resolve_project_path(graph_path or DEFAULT_GRAPH_PATH)

    if resolved_path.exists() and not overwrite:
        raise FileExistsError(
            f"Graph file already exists: {resolved_path}"
        )

    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with open(resolved_path, "wb") as file:
        pickle.dump(graph, file)

    cache_key = str(resolved_path)
    _GRAPH_CACHE[cache_key] = graph
    _GRAPH_STATS_CACHE[cache_key] = get_graph_stats(graph)

    return resolved_path


# =========================================================
# 7. 图结构校验
# =========================================================

def validate_graph(graph: Any) -> None:
    """
    对图结构做基础校验。

    只做轻量检查：
        1. graph 不能为空。
        2. graph 应该能获取节点。
        3. graph 至少有一个节点。
    """

    if graph is None:
        raise ValueError("Loaded graph is None.")

    num_nodes = get_num_nodes(graph)

    if num_nodes <= 0:
        raise ValueError("Loaded graph has no nodes.")

    # 边可以为 0，因为某些测试图可能只有孤立节点。
    # 所以这里不强制 num_edges > 0。


def is_networkx_graph(graph: Any) -> bool:
    """
    判断是否像 NetworkX 图对象。

    不强制 import networkx，避免增加硬依赖判断。
    """

    return (
        hasattr(graph, "nodes")
        and hasattr(graph, "edges")
        and (
            hasattr(graph, "has_node")
            or hasattr(graph, "neighbors")
        )
    )


# =========================================================
# 8. 图统计信息
# =========================================================

def get_graph_stats(graph: Any) -> Dict[str, Any]:
    """
    获取图统计信息。

    Returns:
        {
            "num_nodes": 123,
            "num_edges": 456,
            "is_directed": true,
            "is_multigraph": false,
            "graph_type": "DiGraph"
        }
    """

    stats = {
        "num_nodes": get_num_nodes(graph),
        "num_edges": get_num_edges(graph),
        "is_directed": is_directed_graph(graph),
        "is_multigraph": is_multigraph(graph),
        "graph_type": type(graph).__name__,
    }

    return stats


def get_num_nodes(graph: Any) -> int:
    """
    获取节点数量。
    """

    if graph is None:
        return 0

    if hasattr(graph, "number_of_nodes"):
        try:
            return int(graph.number_of_nodes())
        except Exception:
            pass

    if hasattr(graph, "nodes"):
        try:
            return len(graph.nodes)
        except Exception:
            pass

        try:
            return len(list(graph.nodes()))
        except Exception:
            pass

    if isinstance(graph, dict):
        return len(graph)

    try:
        return len(graph)
    except Exception:
        return 0


def get_num_edges(graph: Any) -> int:
    """
    获取边数量。
    """

    if graph is None:
        return 0

    if hasattr(graph, "number_of_edges"):
        try:
            return int(graph.number_of_edges())
        except Exception:
            pass

    if hasattr(graph, "edges"):
        try:
            return len(graph.edges)
        except Exception:
            pass

        try:
            return len(list(graph.edges()))
        except Exception:
            pass

    return 0


def is_directed_graph(graph: Any) -> bool:
    """
    判断是否有向图。
    """

    if graph is None:
        return False

    if hasattr(graph, "is_directed"):
        try:
            return bool(graph.is_directed())
        except Exception:
            pass

    return False


def is_multigraph(graph: Any) -> bool:
    """
    判断是否多重图。
    """

    if graph is None:
        return False

    if hasattr(graph, "is_multigraph"):
        try:
            return bool(graph.is_multigraph())
        except Exception:
            pass

    return False


# =========================================================
# 9. 图节点与边的通用访问
# =========================================================

def has_node(graph: Any, node: str) -> bool:
    """
    判断图中是否存在某个节点。
    """

    if graph is None:
        return False

    if hasattr(graph, "has_node"):
        try:
            return bool(graph.has_node(node))
        except Exception:
            pass

    try:
        return node in graph
    except Exception:
        return False


def get_nodes(graph: Any) -> list[Any]:
    """
    获取所有节点。
    """

    if graph is None:
        return []

    if hasattr(graph, "nodes"):
        try:
            return list(graph.nodes())
        except TypeError:
            try:
                return list(graph.nodes)
            except Exception:
                return []

    if isinstance(graph, dict):
        return list(graph.keys())

    try:
        return list(graph)
    except Exception:
        return []


def get_edges(graph: Any) -> list[Any]:
    """
    获取所有边。
    """

    if graph is None:
        return []

    if hasattr(graph, "edges"):
        try:
            return list(graph.edges(data=True))
        except TypeError:
            try:
                return list(graph.edges())
            except Exception:
                return []

    return []


def get_node_attrs(graph: Any, node: Any) -> Dict[str, Any]:
    """
    获取节点属性。

    NetworkX:
        graph.nodes[node]

    dict:
        graph[node]
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
            return dict(value)

    return {}


def get_edge_attrs(
    graph: Any,
    source: Any,
    target: Any,
) -> Dict[str, Any]:
    """
    获取边属性。

    主要支持 NetworkX。
    """

    if graph is None:
        return {}

    if hasattr(graph, "get_edge_data"):
        try:
            data = graph.get_edge_data(source, target)

            if isinstance(data, dict):
                # 普通 DiGraph:
                #   {"relation": "..."}
                # MultiDiGraph:
                #   {0: {"relation": "..."}, 1: {...}}
                if _looks_like_multiedge_data(data):
                    merged = {}

                    for _, edge_attrs in data.items():
                        if isinstance(edge_attrs, dict):
                            merged.update(edge_attrs)

                    return merged

                return dict(data)

        except Exception:
            pass

    return {}


def _looks_like_multiedge_data(data: Dict[Any, Any]) -> bool:
    """
    判断 get_edge_data 返回值是否像 MultiGraph 的多边数据。
    """

    if not data:
        return False

    return all(
        isinstance(value, dict)
        for value in data.values()
    ) and not any(
        key in data
        for key in ["relation", "predicate", "weight", "label"]
    )


# =========================================================
# 10. graph_stats.json
# =========================================================

def save_graph_stats(
    graph: Any,
    stats_path: Optional[str] = None,
) -> Path:
    """
    保存图统计信息到 JSON。
    """

    resolved_path = resolve_project_path(
        stats_path or DEFAULT_GRAPH_STATS_PATH
    )

    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    stats = get_graph_stats(graph)

    with open(resolved_path, "w", encoding="utf-8") as file:
        json.dump(
            stats,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return resolved_path


def load_graph_stats(
    stats_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    加载 graph_stats.json。
    """

    resolved_path = resolve_project_path(
        stats_path or DEFAULT_GRAPH_STATS_PATH
    )

    if not resolved_path.exists():
        return {}

    with open(resolved_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        return {}

    return data


# =========================================================
# 11. 缓存控制
# =========================================================

def clear_graph_cache() -> None:
    """
    清空图缓存。
    """

    _GRAPH_CACHE.clear()
    _GRAPH_STATS_CACHE.clear()


def get_cached_graph_paths() -> list[str]:
    """
    返回当前缓存中的图路径。
    """

    return list(_GRAPH_CACHE.keys())


# =========================================================
# 12. 快速测试入口
# =========================================================

if __name__ == "__main__":
    graph = load_graph()
    stats = get_graph_stats(graph)

    print("Graph loaded successfully.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))