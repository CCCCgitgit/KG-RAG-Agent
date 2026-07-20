# -*- coding: utf-8 -*-
"""
kg_tools.py

KG 工具适配层。

职责：
    1. 将 kg/ 底层图查询能力包装成可复用的工具函数。
    2. 为后续 Tool Calling、脚本调试、服务层组合调用提供统一入口。
    3. 只做参数适配、图对象加载、结果包装。

注意：
    本文件属于 tools 层，不实现 KG 查询算法。

    直接关系查询由：
        kg/relation_search.py

    路径查询由：
        kg/path_search.py

    邻居查询由：
        kg/neighbor_search.py

    局部子图查询由：
        kg/subgraph_search.py

    evidence 构造由：
        kg/evidence_builder.py
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, TYPE_CHECKING

from .errors import ToolValidationError

if TYPE_CHECKING:
    from kg_rag_agent.runtime.context import RuntimeContext

from kg_rag_agent.kg.evidence_builder import (
    build_evidence_from_results,
    build_evidence_text,
)
from kg_rag_agent.kg.graph_loader import get_graph_stats, load_graph
from kg_rag_agent.kg.neighbor_search import neighbor_search
from kg_rag_agent.kg.path_search import path_search
from kg_rag_agent.kg.relation_search import relation_search
from kg_rag_agent.kg.subgraph_search import subgraph_search


# =========================================================
# 1. 工具类
# =========================================================

class KGTools:
    """
    KG 工具适配器。

    说明：
        本类只负责把 kg/ 中已经实现好的底层能力包装成更方便调用的接口。
        不在这里重新实现图搜索算法。
    """

    def __init__(
        self,
        *,
        runtime: Optional["RuntimeContext"] = None,
        graph: Optional[Any] = None,
        graph_path: Optional[str] = None,
        use_cache: bool = True,
        validate_graph: bool = True,
        max_evidence: int = 30,
    ) -> None:
        self.runtime = runtime
        self._graph = graph
        self.graph_path = graph_path
        self.use_cache = bool(use_cache)
        self.validate_graph = bool(validate_graph)
        self.max_evidence = _positive_int(max_evidence, field="max_evidence")

    # =====================================================
    # 1.1 Graph 加载
    # =====================================================

    def get_graph(self) -> Any:
        """
        获取图对象。

        如果初始化时没有传入 graph，则从 graph_path 或默认路径加载。
        """

        if self._graph is not None:
            return self._graph

        if self.runtime is not None:
            self._graph = self.runtime.get_graph()
            return self._graph

        self._graph = load_graph(
            graph_path=self.graph_path,
            use_cache=self.use_cache,
            validate=self.validate_graph,
        )
        return self._graph

    def graph_info(self) -> Dict[str, Any]:
        """
        返回图基础统计信息。
        """

        graph = self.get_graph()
        return get_graph_stats(graph)

    # =====================================================
    # 1.2 直接关系工具
    # =====================================================

    def relation(
        self,
        source: str,
        target: str,
        *,
        include_reverse: bool = True,
        max_results: int = 20,
        include_evidence: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        查询两个实体之间的直接关系。
        """

        max_results = _positive_int(max_results, field="max_results")
        result = relation_search(
            graph=self.get_graph(),
            source=source,
            target=target,
            include_reverse=include_reverse,
            max_results=max_results,
            **kwargs,
        )

        return self._wrap_single_result(
            result=result,
            result_type="relation",
            include_evidence=include_evidence,
        )

    # =====================================================
    # 1.3 路径工具
    # =====================================================

    def path(
        self,
        source: str,
        target: str,
        *,
        max_paths: int = 5,
        max_path_length: int = 4,
        include_evidence: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        查询两个实体之间的多跳路径。
        """

        max_paths = _positive_int(max_paths, field="max_paths")
        max_path_length = _positive_int(max_path_length, field="max_path_length")
        result = path_search(
            graph=self.get_graph(),
            source=source,
            target=target,
            max_paths=max_paths,
            max_path_length=max_path_length,
            **kwargs,
        )

        return self._wrap_single_result(
            result=result,
            result_type="path",
            include_evidence=include_evidence,
        )

    # =====================================================
    # 1.4 邻居工具
    # =====================================================

    def neighbors(
        self,
        entity: str,
        *,
        max_neighbors: int = 20,
        direction: str = "both",
        include_evidence: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        查询实体的一跳邻居。
        """

        max_neighbors = _positive_int(max_neighbors, field="max_neighbors")
        result = neighbor_search(
            graph=self.get_graph(),
            entity=entity,
            max_neighbors=max_neighbors,
            direction=direction,
            **kwargs,
        )

        return self._wrap_single_result(
            result=result,
            result_type="neighbor",
            include_evidence=include_evidence,
        )

    # =====================================================
    # 1.5 局部子图工具
    # =====================================================

    def subgraph(
        self,
        entities: Sequence[str],
        *,
        max_depth: int = 2,
        max_nodes: int = 50,
        max_edges: int = 100,
        direction: str = "both",
        include_evidence: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        围绕一个或多个实体抽取局部子图。
        """

        entity_list = _normalize_entities(entities)
        max_depth = _positive_int(max_depth, field="max_depth")
        max_nodes = _positive_int(max_nodes, field="max_nodes")
        max_edges = _positive_int(max_edges, field="max_edges")

        result = subgraph_search(
            graph=self.get_graph(),
            entities=entity_list,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_edges=max_edges,
            direction=direction,
            **kwargs,
        )

        return self._wrap_single_result(
            result=result,
            result_type="subgraph",
            include_evidence=include_evidence,
        )

    # =====================================================
    # 1.6 综合检索工具
    # =====================================================

    def retrieve(
        self,
        entities: Sequence[Any],
        *,
        max_neighbors: int = 10,
        max_paths: int = 5,
        max_path_length: int = 4,
        include_subgraph: bool = True,
        subgraph_depth: int = 1,
        max_evidence: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        围绕实体集合执行综合 KG 检索。

        返回字段：
            raw_evidence:
                relation / path / neighbor / subgraph 原始结果。

            evidence:
                由 evidence_builder.py 统一格式化后的证据。

            evidence_text:
                面向 reasoning / generation 的文本化证据。
        """

        entity_list = _normalize_entities(entities)
        if not entity_list:
            raise ToolValidationError("entities must contain at least one entity")
        max_neighbors = _positive_int(max_neighbors, field="max_neighbors")
        max_paths = _positive_int(max_paths, field="max_paths")
        max_path_length = _positive_int(max_path_length, field="max_path_length")
        subgraph_depth = _positive_int(subgraph_depth, field="subgraph_depth")
        max_evidence = _positive_int(
            max_evidence or self.max_evidence,
            field="max_evidence",
        )

        relation_results: List[Dict[str, Any]] = []
        path_results: List[Dict[str, Any]] = []
        neighbor_results: List[Dict[str, Any]] = []
        subgraph_results: List[Dict[str, Any]] = []

        graph = self.get_graph()

        # 两两实体关系与路径
        for source, target in _pairwise_entities(entity_list):
            relation_results.append(
                relation_search(
                    graph=graph,
                    source=source,
                    target=target,
                    include_reverse=True,
                )
            )
            path_results.append(
                path_search(
                    graph=graph,
                    source=source,
                    target=target,
                    max_paths=max_paths,
                    max_path_length=max_path_length,
                )
            )

        # 单实体邻居
        for entity in entity_list:
            neighbor_results.append(
                neighbor_search(
                    graph=graph,
                    entity=entity,
                    max_neighbors=max_neighbors,
                    direction="both",
                )
            )

        # 局部子图补充
        if include_subgraph and entity_list:
            subgraph_results.append(
                subgraph_search(
                    graph=graph,
                    entities=entity_list,
                    max_depth=subgraph_depth,
                    max_nodes=max(20, max_neighbors * max(len(entity_list), 1)),
                    max_edges=max(40, max_neighbors * max(len(entity_list), 1) * 2),
                    direction="both",
                )
            )

        evidence = build_evidence_from_results(
            relation_results=relation_results,
            path_results=path_results,
            neighbor_results=neighbor_results,
            subgraph_results=subgraph_results,
            max_evidence=max_evidence,
        )

        evidence_text = build_evidence_text(
            evidence,
            max_items=max_evidence,
            include_score=True,
        )

        return {
            "entities": entity_list,
            "raw_evidence": {
                "relations": relation_results,
                "paths": path_results,
                "neighbors": neighbor_results,
                "subgraphs": subgraph_results,
            },
            "evidence": evidence,
            "evidence_text": evidence_text,
            "num_evidence": len(evidence),
        }

    # =====================================================
    # 1.7 内部包装函数
    # =====================================================

    def _wrap_single_result(
        self,
        *,
        result: Dict[str, Any],
        result_type: str,
        include_evidence: bool,
    ) -> Dict[str, Any]:
        """
        将单类 KG 查询结果包装为统一工具输出。
        """

        output: Dict[str, Any] = {
            "result_type": result_type,
            "result": result,
        }

        if not include_evidence:
            return output

        relation_results: List[Any] = []
        path_results: List[Any] = []
        neighbor_results: List[Any] = []
        subgraph_results: List[Any] = []

        if result_type == "relation":
            relation_results.append(result)
        elif result_type == "path":
            path_results.append(result)
        elif result_type == "neighbor":
            neighbor_results.append(result)
        elif result_type == "subgraph":
            subgraph_results.append(result)

        evidence = build_evidence_from_results(
            relation_results=relation_results,
            path_results=path_results,
            neighbor_results=neighbor_results,
            subgraph_results=subgraph_results,
            max_evidence=self.max_evidence,
        )

        output["evidence"] = evidence
        output["evidence_text"] = build_evidence_text(evidence)
        output["num_evidence"] = len(evidence)

        return output


# =========================================================
# 2. 默认工具实例
# =========================================================

_DEFAULT_KG_TOOLS: Optional[KGTools] = None


def get_default_kg_tools(
    *,
    runtime: Optional["RuntimeContext"] = None,
    graph_path: Optional[str] = None,
    refresh: bool = False,
) -> KGTools:
    """
    获取默认 KGTools 实例。
    """

    global _DEFAULT_KG_TOOLS

    if refresh or _DEFAULT_KG_TOOLS is None:
        _DEFAULT_KG_TOOLS = KGTools(
            runtime=runtime,
            graph_path=graph_path,
        )

    return _DEFAULT_KG_TOOLS


# =========================================================
# 3. 函数式工具入口
# =========================================================

def relation_tool(
    source: str,
    target: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    直接关系工具函数。
    """

    return get_default_kg_tools().relation(
        source=source,
        target=target,
        **kwargs,
    )


def path_tool(
    source: str,
    target: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    多跳路径工具函数。
    """

    return get_default_kg_tools().path(
        source=source,
        target=target,
        **kwargs,
    )


def neighbor_tool(
    entity: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    邻居查询工具函数。
    """

    return get_default_kg_tools().neighbors(
        entity=entity,
        **kwargs,
    )


def subgraph_tool(
    entities: Sequence[str],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    局部子图工具函数。
    """

    return get_default_kg_tools().subgraph(
        entities=entities,
        **kwargs,
    )


def kg_retrieval_tool(
    entities: Sequence[Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    综合 KG 检索工具函数。
    """

    return get_default_kg_tools().retrieve(
        entities=entities,
        **kwargs,
    )


# =========================================================
# 4. 辅助函数
# =========================================================

def _normalize_entities(entities: Sequence[Any] | Any) -> List[str]:
    """
    将字符串、dict、GroundedEntity 风格对象统一转成实体 node_key/name 列表。
    """

    if entities is None:
        return []

    if isinstance(entities, (str, dict)):
        raw_items: Iterable[Any] = [entities]
    else:
        raw_items = entities

    normalized: List[str] = []
    seen = set()

    for item in raw_items:
        value = _extract_entity_text(item)

        if not value:
            continue

        if value in seen:
            continue

        seen.add(value)
        normalized.append(value)

    return normalized


def _extract_entity_text(item: Any) -> str:
    """
    从不同实体结构中抽取可用于 KG 查询的实体文本。
    """

    if item is None:
        return ""

    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):
        for key in (
            "node_key",
            "entity_id",
            "entity",
            "entity_name",
            "name",
            "text",
            "mention",
        ):
            value = item.get(key)
            if value:
                return str(value).strip()
        return ""

    for attr in (
        "node_key",
        "entity_id",
        "entity",
        "entity_name",
        "name",
        "text",
        "mention",
    ):
        value = getattr(item, attr, None)
        if value:
            return str(value).strip()

    return str(item).strip()


def _pairwise_entities(entities: Sequence[str]) -> List[tuple[str, str]]:
    """
    生成实体两两组合。
    """

    pairs: List[tuple[str, str]] = []

    for i, source in enumerate(entities):
        for target in entities[i + 1:]:
            if source and target and source != target:
                pairs.append((source, target))

    return pairs



def _positive_int(value: Any, *, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ToolValidationError(f"{field} must be an integer") from exc
    if parsed <= 0:
        raise ToolValidationError(f"{field} must be > 0")
    return parsed

__all__ = [
    "KGTools",
    "get_default_kg_tools",
    "relation_tool",
    "path_tool",
    "neighbor_tool",
    "subgraph_tool",
    "kg_retrieval_tool",
]