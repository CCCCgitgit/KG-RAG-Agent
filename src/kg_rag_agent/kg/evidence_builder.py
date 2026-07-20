# -*- coding: utf-8 -*-
"""
evidence_builder.py

证据构造模块。

作用：
    1. 将 relation_search / path_search / neighbor_search / subgraph_search 的结果统一转换为 evidence。
    2. 提供 evidence_id 生成、去重、排序、文本格式化能力。
    3. 为 graph/nodes/kg_retrieval_node.py、reasoning_node.py、generation_node.py 提供统一材料格式。

本文件属于 kg 底层能力层：
    kg/
        evidence_builder.py

它不负责：
    1. 用户问题解析。
    2. 对象识别。
    3. 图查询本身。
    4. 最终回答生成。

统一 evidence 格式：
    {
        "evidence_id": "ev_xxx",
        "evidence_type": "relation | path | neighbor | subgraph | text",
        "source_entity": "...",
        "target_entity": "...",
        "relation": "...",
        "path": [...],
        "triples": [...],
        "text": "...",
        "score": 0.9,
        "metadata": {...}
    }
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


EvidenceItem = Dict[str, Any]


# =========================================================
# 1. EvidenceBuilder
# =========================================================

class EvidenceBuilder:
    """
    evidence 构造器。

    用法：
        builder = EvidenceBuilder()
        evidence = builder.from_relation_result(result)
        evidence_text = builder.build_evidence_text(evidence)
    """

    def __init__(
        self,
        *,
        max_evidence: int = 30,
        min_score: float = 0.0,
        deduplicate: bool = True,
    ) -> None:
        self.max_evidence = int(max_evidence)
        self.min_score = float(min_score)
        self.deduplicate = bool(deduplicate)

    # =====================================================
    # 1.1 通用构造
    # =====================================================

    def build(
        self,
        *,
        evidence_type: str,
        source_entity: str = "",
        target_entity: str = "",
        relation: str = "",
        path: Optional[List[str]] = None,
        triples: Optional[List[Dict[str, Any]]] = None,
        text: str = "",
        score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceItem:
        """
        构造单条 evidence。
        """

        evidence_type = str(evidence_type or "text").strip()
        source_entity = str(source_entity or "").strip()
        target_entity = str(target_entity or "").strip()
        relation = str(relation or "").strip()
        path = path or []
        triples = triples or []
        text = str(text or "").strip()

        if not text:
            text = build_text_from_parts(
                source_entity=source_entity,
                relation=relation,
                target_entity=target_entity,
                path=path,
                triples=triples,
            )

        score = clip_score(score)

        evidence_id = make_evidence_id(
            evidence_type=evidence_type,
            source_entity=source_entity,
            target_entity=target_entity,
            relation=relation,
            text=text,
            path=path,
            triples=triples,
        )

        return {
            "evidence_id": evidence_id,
            "evidence_type": evidence_type,
            "source_entity": source_entity,
            "target_entity": target_entity,
            "relation": relation,
            "path": path,
            "triples": triples,
            "text": text,
            "score": score,
            "metadata": metadata or {},
        }

    # =====================================================
    # 1.2 relation_search 结果转 evidence
    # =====================================================

    def from_relation_result(
        self,
        result: Any,
        *,
        default_source: str = "",
        default_target: str = "",
    ) -> List[EvidenceItem]:
        """
        将 relation_search 返回结果转换为 evidence。

        支持格式：
            {
                "relations": [
                    {
                        "head": "A",
                        "relation": "r",
                        "tail": "B",
                        "score": 1.0,
                        "text": "..."
                    }
                ]
            }

        或直接传入 list。
        """

        items = extract_list_from_result(
            result,
            keys=["relations", "relation", "results", "data"],
        )

        evidence: List[EvidenceItem] = []

        for item in items:
            if item is None:
                continue

            if isinstance(item, str):
                relation = item.strip()

                evidence.append(
                    self.build(
                        evidence_type="relation",
                        source_entity=default_source,
                        target_entity=default_target,
                        relation=relation,
                        triples=[
                            {
                                "head": default_source,
                                "relation": relation,
                                "tail": default_target,
                            }
                        ],
                        text=build_relation_text(
                            head=default_source,
                            relation=relation,
                            tail=default_target,
                        ),
                        score=0.9,
                        metadata={
                            "raw_result": item,
                        },
                    )
                )

                continue

            if not isinstance(item, dict):
                continue

            head = first_non_empty_str(
                item,
                ["head", "source", "subject", "h"],
            ) or default_source

            tail = first_non_empty_str(
                item,
                ["tail", "target", "object", "t"],
            ) or default_target

            relation = first_non_empty_str(
                item,
                ["relation", "predicate", "rel", "edge", "type", "label"],
            ) or "related_to"

            text = first_non_empty_str(
                item,
                ["text", "description", "sentence", "evidence"],
            )

            if not text:
                text = build_relation_text(
                    head=head,
                    relation=relation,
                    tail=tail,
                )

            triples = item.get("triples")

            if not isinstance(triples, list):
                triples = [
                    {
                        "head": head,
                        "relation": relation,
                        "tail": tail,
                    }
                ]

            evidence.append(
                self.build(
                    evidence_type="relation",
                    source_entity=head,
                    target_entity=tail,
                    relation=relation,
                    triples=triples,
                    text=text,
                    score=extract_score(item, default=0.9),
                    metadata={
                        "raw_result": item,
                        "direction": item.get("direction", ""),
                    },
                )
            )

        return self.postprocess(evidence)

    # =====================================================
    # 1.3 path_search 结果转 evidence
    # =====================================================

    def from_path_result(
        self,
        result: Any,
        *,
        default_source: str = "",
        default_target: str = "",
    ) -> List[EvidenceItem]:
        """
        将 path_search 返回结果转换为 evidence。
        """

        items = extract_list_from_result(
            result,
            keys=["paths", "path", "results", "data"],
        )

        evidence: List[EvidenceItem] = []

        for item in items:
            if item is None:
                continue

            if isinstance(item, (list, tuple)):
                path = [str(node) for node in item]
                triples: List[Dict[str, Any]] = []
                text = " -> ".join(path)

                evidence.append(
                    self.build(
                        evidence_type="path",
                        source_entity=default_source or first_or_empty(path),
                        target_entity=default_target or last_or_empty(path),
                        relation="path",
                        path=path,
                        triples=triples,
                        text=text,
                        score=score_path(path),
                        metadata={
                            "raw_result": item,
                            "path_length": len(path),
                        },
                    )
                )

                continue

            if not isinstance(item, dict):
                continue

            path = item.get("path") or item.get("nodes") or item.get("node_path") or []

            if not isinstance(path, list):
                path = []

            path = [str(node) for node in path]

            triples = item.get("triples") or item.get("edges") or []

            if not isinstance(triples, list):
                triples = []

            text = first_non_empty_str(
                item,
                ["text", "description", "sentence", "evidence"],
            )

            if not text:
                if triples:
                    text = triples_to_text(triples)
                elif path:
                    text = " -> ".join(path)
                else:
                    text = ""

            source = (
                first_non_empty_str(item, ["source", "head", "start"])
                or default_source
                or first_or_empty(path)
            )

            target = (
                first_non_empty_str(item, ["target", "tail", "end"])
                or default_target
                or last_or_empty(path)
            )

            evidence.append(
                self.build(
                    evidence_type="path",
                    source_entity=source,
                    target_entity=target,
                    relation="path",
                    path=path,
                    triples=triples,
                    text=text,
                    score=extract_score(item, default=score_path(path)),
                    metadata={
                        "raw_result": item,
                        "path_length": len(path),
                        "num_hops": max(len(path) - 1, 0),
                    },
                )
            )

        return self.postprocess(evidence)

    # =====================================================
    # 1.4 neighbor_search 结果转 evidence
    # =====================================================

    def from_neighbor_result(
        self,
        result: Any,
        *,
        default_source: str = "",
    ) -> List[EvidenceItem]:
        """
        将 neighbor_search 返回结果转换为 evidence。
        """

        items = extract_list_from_result(
            result,
            keys=["neighbors", "neighbor", "results", "data"],
        )

        evidence: List[EvidenceItem] = []

        for item in items:
            if item is None:
                continue

            if isinstance(item, str):
                target = item.strip()

                if not target:
                    continue

                evidence.append(
                    self.build(
                        evidence_type="neighbor",
                        source_entity=default_source,
                        target_entity=target,
                        relation="related_to",
                        triples=[
                            {
                                "head": default_source,
                                "relation": "related_to",
                                "tail": target,
                            }
                        ],
                        text=build_relation_text(
                            head=default_source,
                            relation="related_to",
                            tail=target,
                        ),
                        score=0.65,
                        metadata={
                            "raw_result": item,
                        },
                    )
                )

                continue

            if not isinstance(item, dict):
                continue

            source = first_non_empty_str(
                item,
                ["source", "head", "subject", "h"],
            ) or default_source

            target = first_non_empty_str(
                item,
                ["target", "tail", "object", "t", "neighbor", "entity", "name"],
            )

            if not target:
                continue

            relation = first_non_empty_str(
                item,
                ["relation", "predicate", "rel", "edge", "type", "label"],
            ) or "related_to"

            text = first_non_empty_str(
                item,
                ["text", "description", "sentence", "evidence"],
            )

            if not text:
                text = build_relation_text(
                    head=source,
                    relation=relation,
                    tail=target,
                )

            triples = item.get("triples")

            if not isinstance(triples, list):
                triples = [
                    {
                        "head": source,
                        "relation": relation,
                        "tail": target,
                    }
                ]

            evidence.append(
                self.build(
                    evidence_type="neighbor",
                    source_entity=source,
                    target_entity=target,
                    relation=relation,
                    triples=triples,
                    text=text,
                    score=extract_score(item, default=0.65),
                    metadata={
                        "raw_result": item,
                        "direction": item.get("direction", ""),
                    },
                )
            )

        return self.postprocess(evidence)

    # =====================================================
    # 1.5 subgraph 结果转 evidence
    # =====================================================

    def from_subgraph_result(
        self,
        result: Any,
        *,
        default_source: str = "",
    ) -> List[EvidenceItem]:
        """
        将 subgraph_search 返回结果转换为 evidence。

        支持：
            {
                "triples": [...],
                "nodes": [...],
                "edges": [...]
            }
        """

        if result is None:
            return []

        evidence: List[EvidenceItem] = []

        if isinstance(result, dict):
            triples = result.get("triples") or result.get("edges") or []

            if isinstance(triples, list):
                for triple in triples:
                    if not isinstance(triple, dict):
                        continue

                    head = first_non_empty_str(
                        triple,
                        ["head", "source", "subject", "h"],
                    )

                    relation = first_non_empty_str(
                        triple,
                        ["relation", "predicate", "rel", "r", "type", "label"],
                    ) or "related_to"

                    tail = first_non_empty_str(
                        triple,
                        ["tail", "target", "object", "t"],
                    )

                    if not head and default_source:
                        head = default_source

                    if not head or not tail:
                        continue

                    text = first_non_empty_str(
                        triple,
                        ["text", "description", "sentence", "evidence"],
                    )

                    if not text:
                        text = build_relation_text(
                            head=head,
                            relation=relation,
                            tail=tail,
                        )

                    evidence.append(
                        self.build(
                            evidence_type="subgraph",
                            source_entity=head,
                            target_entity=tail,
                            relation=relation,
                            triples=[
                                {
                                    "head": head,
                                    "relation": relation,
                                    "tail": tail,
                                }
                            ],
                            text=text,
                            score=extract_score(triple, default=0.6),
                            metadata={
                                "raw_result": triple,
                            },
                        )
                    )

        elif isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    evidence.extend(
                        self.from_subgraph_result(
                            item,
                            default_source=default_source,
                        )
                    )

        return self.postprocess(evidence)

    # =====================================================
    # 1.6 postprocess
    # =====================================================

    def postprocess(
        self,
        evidence: List[EvidenceItem],
    ) -> List[EvidenceItem]:
        """
        evidence 后处理。

        包括：
            1. 空文本过滤
            2. 分数过滤
            3. 去重
            4. 排序
            5. 截断
        """

        cleaned: List[EvidenceItem] = []

        for item in evidence:
            normalized = normalize_evidence_item(item)

            if not normalized:
                continue

            if float(normalized.get("score", 0.0)) < self.min_score:
                continue

            cleaned.append(normalized)

        if self.deduplicate:
            cleaned = deduplicate_evidence(cleaned)

        cleaned = sort_evidence(cleaned)

        return cleaned[: self.max_evidence]

    # =====================================================
    # 1.7 文本构造
    # =====================================================

    def build_evidence_text(
        self,
        evidence: List[EvidenceItem],
        *,
        max_items: Optional[int] = None,
        include_score: bool = True,
    ) -> str:
        """
        构造 evidence_text。
        """

        return build_evidence_text(
            evidence,
            max_items=max_items,
            include_score=include_score,
        )


# =========================================================
# 2. 函数式接口
# =========================================================

def build_evidence_item(
    *,
    evidence_type: str,
    source_entity: str = "",
    target_entity: str = "",
    relation: str = "",
    path: Optional[List[str]] = None,
    triples: Optional[List[Dict[str, Any]]] = None,
    text: str = "",
    score: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> EvidenceItem:
    """
    函数式构造 evidence。
    """

    builder = EvidenceBuilder()

    return builder.build(
        evidence_type=evidence_type,
        source_entity=source_entity,
        target_entity=target_entity,
        relation=relation,
        path=path,
        triples=triples,
        text=text,
        score=score,
        metadata=metadata,
    )


def build_evidence_from_results(
    *,
    relation_results: Optional[List[Any]] = None,
    path_results: Optional[List[Any]] = None,
    neighbor_results: Optional[List[Any]] = None,
    subgraph_results: Optional[List[Any]] = None,
    max_evidence: int = 30,
    min_score: float = 0.0,
) -> List[EvidenceItem]:
    """
    从多类查询结果统一构建 evidence。
    """

    builder = EvidenceBuilder(
        max_evidence=max_evidence,
        min_score=min_score,
    )

    evidence: List[EvidenceItem] = []

    for result in relation_results or []:
        evidence.extend(builder.from_relation_result(result))

    for result in path_results or []:
        evidence.extend(builder.from_path_result(result))

    for result in neighbor_results or []:
        evidence.extend(builder.from_neighbor_result(result))

    for result in subgraph_results or []:
        evidence.extend(builder.from_subgraph_result(result))

    return builder.postprocess(evidence)


# =========================================================
# 3. evidence 标准化
# =========================================================

def normalize_evidence_item(item: Any) -> EvidenceItem:
    """
    标准化单条 evidence。
    """

    if item is None:
        return {}

    if isinstance(item, str):
        text = item.strip()

        if not text:
            return {}

        return build_evidence_item(
            evidence_type="text",
            text=text,
            score=0.5,
            metadata={
                "raw_result": item,
            },
        )

    if not isinstance(item, dict):
        return {}

    evidence_type = str(item.get("evidence_type") or item.get("type") or "text").strip()
    source_entity = str(item.get("source_entity") or item.get("source") or item.get("head") or "").strip()
    target_entity = str(item.get("target_entity") or item.get("target") or item.get("tail") or "").strip()
    relation = str(item.get("relation") or item.get("predicate") or item.get("rel") or "").strip()

    path = item.get("path", [])
    if not isinstance(path, list):
        path = []

    path = [str(node) for node in path]

    triples = item.get("triples", [])
    if not isinstance(triples, list):
        triples = []

    text = str(item.get("text") or item.get("description") or item.get("sentence") or "").strip()

    if not text:
        text = build_text_from_parts(
            source_entity=source_entity,
            target_entity=target_entity,
            relation=relation,
            path=path,
            triples=triples,
        )

    if not text:
        return {}

    score = extract_score(item, default=0.0)

    evidence_id = str(item.get("evidence_id", "") or "").strip()

    if not evidence_id:
        evidence_id = make_evidence_id(
            evidence_type=evidence_type,
            source_entity=source_entity,
            target_entity=target_entity,
            relation=relation,
            text=text,
            path=path,
            triples=triples,
        )

    metadata = item.get("metadata", {})

    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "evidence_id": evidence_id,
        "evidence_type": evidence_type,
        "source_entity": source_entity,
        "target_entity": target_entity,
        "relation": relation,
        "path": path,
        "triples": triples,
        "text": text,
        "score": clip_score(score),
        "metadata": metadata,
    }


def normalize_evidence_list(items: Iterable[Any]) -> List[EvidenceItem]:
    """
    批量标准化 evidence。
    """

    evidence: List[EvidenceItem] = []

    for item in items:
        normalized = normalize_evidence_item(item)

        if normalized:
            evidence.append(normalized)

    return evidence


# =========================================================
# 4. evidence 去重、排序、过滤
# =========================================================

def deduplicate_evidence(
    evidence: List[EvidenceItem],
) -> List[EvidenceItem]:
    """
    evidence 去重。

    优先依据：
        1. text
        2. source + relation + target
        3. path
    """

    best_by_key: Dict[str, EvidenceItem] = {}

    for item in evidence:
        key = evidence_dedup_key(item)

        if not key:
            continue

        old = best_by_key.get(key)

        if old is None:
            best_by_key[key] = item
            continue

        old_score = safe_float(old.get("score", 0.0), default=0.0)
        new_score = safe_float(item.get("score", 0.0), default=0.0)

        if new_score > old_score:
            best_by_key[key] = item

    return list(best_by_key.values())


def evidence_dedup_key(item: EvidenceItem) -> str:
    """
    构造 evidence 去重 key。

    优先使用结构化字段。
    这样 relation/path/neighbor 等 evidence 即使自动生成了 text，
    也不会退化成 text key，便于下游测试和调试判断来源类型。
    """

    evidence_type = normalize_text(item.get("evidence_type", "")) or "text"

    source = normalize_text(item.get("source_entity", ""))
    relation = normalize_text(item.get("relation", ""))
    target = normalize_text(item.get("target_entity", ""))

    if source or relation or target:
        return f"{evidence_type}::{source}|{relation}|{target}"

    path = item.get("path", [])

    if isinstance(path, list) and path:
        return f"{evidence_type}::" + "|".join(
            normalize_text(node) for node in path
        )

    text = normalize_text(item.get("text", ""))

    if text:
        return f"text::{text}"

    return ""


def sort_evidence(
    evidence: List[EvidenceItem],
) -> List[EvidenceItem]:
    """
    evidence 排序。

    优先级：
        1. evidence_type
        2. score
    """

    return sorted(
        evidence,
        key=lambda item: (
            evidence_type_priority(str(item.get("evidence_type", ""))),
            safe_float(item.get("score", 0.0), default=0.0),
        ),
        reverse=True,
    )


def evidence_type_priority(evidence_type: str) -> int:
    """
    evidence 类型优先级。
    """

    evidence_type = str(evidence_type or "").lower()

    priority = {
        "relation": 5,
        "path": 4,
        "subgraph": 3,
        "neighbor": 2,
        "text": 1,
    }

    return priority.get(evidence_type, 0)


def filter_evidence_by_score(
    evidence: List[EvidenceItem],
    *,
    min_score: float,
) -> List[EvidenceItem]:
    """
    按分数过滤 evidence。
    """

    return [
        item for item in evidence
        if safe_float(item.get("score", 0.0), default=0.0) >= min_score
    ]


# =========================================================
# 5. evidence 文本构造
# =========================================================

def build_evidence_text(
    evidence: List[EvidenceItem],
    *,
    max_items: Optional[int] = None,
    include_score: bool = True,
) -> str:
    """
    将 evidence 列表构造成文本。
    """

    if not evidence:
        return ""

    items = evidence[:max_items] if max_items is not None else evidence

    lines: List[str] = []

    for idx, item in enumerate(items, start=1):
        evidence_type = str(item.get("evidence_type", "unknown"))
        score = safe_float(item.get("score", 0.0), default=0.0)
        text = str(item.get("text", "")).strip()

        if not text:
            continue

        if include_score:
            lines.append(
                f"[E{idx}] ({evidence_type}, score={score:.3f}) {text}"
            )
        else:
            lines.append(f"[E{idx}] {text}")

    return "\n".join(lines)


def build_text_from_parts(
    *,
    source_entity: str = "",
    relation: str = "",
    target_entity: str = "",
    path: Optional[List[str]] = None,
    triples: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    根据结构化字段构造 evidence text。
    """

    source_entity = str(source_entity or "").strip()
    relation = str(relation or "").strip()
    target_entity = str(target_entity or "").strip()
    path = path or []
    triples = triples or []

    if triples:
        text = triples_to_text(triples)
        if text:
            return text

    if path:
        return " -> ".join(str(node) for node in path)

    if source_entity or relation or target_entity:
        return build_relation_text(
            head=source_entity,
            relation=relation or "related_to",
            tail=target_entity,
        )

    return ""


def build_relation_text(
    *,
    head: str,
    relation: str,
    tail: str,
) -> str:
    """
    构造关系文本。
    """

    head = str(head or "").strip()
    relation = str(relation or "related_to").strip()
    tail = str(tail or "").strip()

    if head and relation and tail:
        return f"{head} --{relation}--> {tail}"

    if head and tail:
        return f"{head} 与 {tail} 存在相关联系。"

    if head:
        return head

    if tail:
        return tail

    return ""


def triples_to_text(
    triples: List[Dict[str, Any]],
) -> str:
    """
    将 triples 转成文本。
    """

    texts: List[str] = []

    for triple in triples:
        if not isinstance(triple, dict):
            continue

        head = first_non_empty_str(
            triple,
            ["head", "source", "subject", "h"],
        )

        relation = first_non_empty_str(
            triple,
            ["relation", "predicate", "rel", "r", "type"],
        ) or "related_to"

        tail = first_non_empty_str(
            triple,
            ["tail", "target", "object", "t"],
        )

        text = build_relation_text(
            head=head,
            relation=relation,
            tail=tail,
        )

        if text:
            texts.append(text)

    return "; ".join(texts)


# =========================================================
# 6. evidence_id
# =========================================================

def make_evidence_id(
    *,
    evidence_type: str,
    source_entity: str = "",
    target_entity: str = "",
    relation: str = "",
    text: str = "",
    path: Optional[List[str]] = None,
    triples: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    生成稳定 evidence_id。
    """

    raw = {
        "evidence_type": evidence_type,
        "source_entity": source_entity,
        "target_entity": target_entity,
        "relation": relation,
        "text": text,
        "path": path or [],
        "triples": triples or [],
    }

    raw_text = json.dumps(
        raw,
        ensure_ascii=False,
        sort_keys=True,
    )

    digest = hashlib.md5(raw_text.encode("utf-8")).hexdigest()[:12]

    return f"ev_{digest}"


# =========================================================
# 7. 结果提取工具
# =========================================================

def extract_list_from_result(
    result: Any,
    *,
    keys: List[str],
) -> List[Any]:
    """
    从不同格式 result 中提取列表。
    """

    if result is None:
        return []

    if isinstance(result, list):
        return result

    if isinstance(result, tuple):
        return list(result)

    if isinstance(result, dict):
        for key in keys:
            value = result.get(key)

            if isinstance(value, list):
                return value

        return [result]

    return [result]


def first_non_empty_str(
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


def first_or_empty(items: List[Any]) -> str:
    """
    返回列表第一个元素字符串。
    """

    if not items:
        return ""

    return str(items[0])


def last_or_empty(items: List[Any]) -> str:
    """
    返回列表最后一个元素字符串。
    """

    if not items:
        return ""

    return str(items[-1])


# =========================================================
# 8. 分数工具
# =========================================================

def extract_score(
    item: Dict[str, Any],
    *,
    default: float = 0.0,
) -> float:
    """
    从 dict 中提取 score。
    """

    for key in ["score", "similarity", "confidence", "weight"]:
        if key in item:
            return clip_score(safe_float(item.get(key), default=default))

    if "distance" in item:
        distance = safe_float(item.get("distance"), default=1.0)
        return clip_score(1.0 - distance)

    return clip_score(default)


def score_path(path: List[str]) -> float:
    """
    根据路径长度给基础分。
    """

    if not path:
        return 0.5

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


def clip_score(score: float) -> float:
    """
    将分数限制在 [0, 1]。
    """

    if score < 0:
        return 0.0

    if score > 1:
        return 1.0

    return score


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    安全转 float。
    """

    try:
        if value is None:
            return default

        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return default

        return value

    except Exception:
        return default


# =========================================================
# 9. 文本工具
# =========================================================

def normalize_text(value: Any) -> str:
    """
    文本标准化，用于去重。
    """

    if value is None:
        return ""

    text = str(value).strip().lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)

    return text


# =========================================================
# 10. 快速测试入口
# =========================================================

if __name__ == "__main__":
    builder = EvidenceBuilder()

    relation_result = {
        "relations": [
            {
                "head": "A",
                "relation": "friend_of",
                "tail": "B",
                "score": 0.95,
            }
        ]
    }

    path_result = {
        "paths": [
            {
                "path": ["A", "C", "B"],
                "triples": [
                    {"head": "A", "relation": "knows", "tail": "C"},
                    {"head": "C", "relation": "knows", "tail": "B"},
                ],
            }
        ]
    }

    neighbor_result = {
        "neighbors": [
            {
                "source": "A",
                "relation": "related_to",
                "target": "D",
            }
        ]
    }

    evidence = []
    evidence.extend(builder.from_relation_result(relation_result))
    evidence.extend(builder.from_path_result(path_result))
    evidence.extend(builder.from_neighbor_result(neighbor_result))

    evidence = builder.postprocess(evidence)

    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    print()
    print(builder.build_evidence_text(evidence))