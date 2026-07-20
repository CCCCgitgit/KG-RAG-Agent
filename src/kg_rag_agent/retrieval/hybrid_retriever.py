# -*- coding: utf-8 -*-
"""
hybrid_retriever.py

混合召回模块。

作用：
    1. 融合多种召回方式：
        - 通用文档向量召回
        - 实体向量召回
        - 关键词规则召回
    2. 对不同来源结果统一标准化。
    3. 对召回结果进行去重、分数融合、排序、截断。
    4. 为后续复杂 RAG、证据增强、候选补充提供统一入口。

本文件属于 retrieval 层：
    retrieval/
        hybrid_retriever.py

它不负责：
    1. 图结构路径查询。
    2. 实体最终落地。
    3. LangGraph 节点调度。
    4. 最终回答生成。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ._validation import non_negative_float, positive_int, validate_weights
from .errors import RetrievalConfigurationError

from .entity_vector_store import EntityVectorStore
from .vector_retriever import VectorRetriever


# =========================================================
# 1. 默认配置
# =========================================================

DEFAULT_VECTOR_WEIGHT = 0.60
DEFAULT_ENTITY_WEIGHT = 0.30
DEFAULT_KEYWORD_WEIGHT = 0.10


# =========================================================
# 2. HybridRetriever
# =========================================================

class HybridRetriever:
    """
    混合召回器。

    用法：
        retriever = HybridRetriever()
        results = retriever.retrieve("阿尔茨海默病和FDG-PET有什么关系？", top_k=10)
    """

    def __init__(
        self,
        *,
        vector_retriever: Optional[VectorRetriever] = None,
        entity_store: Optional[EntityVectorStore] = None,
        enable_vector: bool = True,
        enable_entity: bool = True,
        enable_keyword: bool = True,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        entity_weight: float = DEFAULT_ENTITY_WEIGHT,
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
        fail_silently: bool = True,
    ) -> None:
        """
        Args:
            vector_retriever:
                通用文档向量召回器。

            entity_store:
                实体向量库。

            enable_vector:
                是否启用文档向量召回。

            enable_entity:
                是否启用实体向量召回。

            enable_keyword:
                是否启用关键词召回。

            vector_weight / entity_weight / keyword_weight:
                不同来源的融合权重。

            fail_silently:
                某一路召回失败时是否静默跳过。
                默认 True，保证主流程稳定。
        """

        self.vector_retriever = vector_retriever
        self.entity_store = entity_store

        self.enable_vector = bool(enable_vector)
        self.enable_entity = bool(enable_entity)
        self.enable_keyword = bool(enable_keyword)

        weights = validate_weights(vector_weight=vector_weight, entity_weight=entity_weight, keyword_weight=keyword_weight)
        self.vector_weight = weights["vector_weight"]
        self.entity_weight = weights["entity_weight"]
        self.keyword_weight = weights["keyword_weight"]

        self.fail_silently = bool(fail_silently)

        if not (self.enable_vector or self.enable_entity or self.enable_keyword):
            raise RetrievalConfigurationError("at least one retrieval channel must be enabled")

    # =====================================================
    # 2.1 主召回接口
    # =====================================================

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        vector_top_k: Optional[int] = None,
        entity_top_k: Optional[int] = None,
        keyword_top_k: Optional[int] = None,
        vector_where: Optional[Dict[str, Any]] = None,
        entity_where: Optional[Dict[str, Any]] = None,
        keyword_corpus: Optional[Sequence[Dict[str, Any]]] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        执行混合召回。

        Args:
            query:
                查询文本。

            top_k:
                最终返回数量。

            vector_top_k:
                文档向量召回数量。
                默认 top_k * 2。

            entity_top_k:
                实体召回数量。
                默认 top_k。

            keyword_top_k:
                关键词召回数量。
                默认 top_k。

            vector_where:
                文档向量库 metadata 过滤条件。

            entity_where:
                实体向量库 metadata 过滤条件。

            keyword_corpus:
                关键词召回候选语料。
                每项格式建议：
                    {
                        "id": "...",
                        "text": "...",
                        "metadata": {...}
                    }

            min_score:
                最小融合分数。

        Returns:
            List[Dict[str, Any]]:
                [
                    {
                        "id": "...",
                        "text": "...",
                        "score": 0.82,
                        "source": "hybrid",
                        "retrieval_sources": ["vector", "keyword"],
                        "metadata": {...}
                    }
                ]
        """

        query = normalize_query(query)

        if not query:
            return []

        top_k = positive_int(top_k, field="top_k")
        vector_top_k = positive_int(vector_top_k, field="vector_top_k", default=top_k * 2)
        entity_top_k = positive_int(entity_top_k, field="entity_top_k", default=top_k)
        keyword_top_k = positive_int(keyword_top_k, field="keyword_top_k", default=top_k)
        min_score = non_negative_float(min_score, field="min_score")

        all_results: List[Dict[str, Any]] = []

        # -------------------------------------------------
        # 1. 文档向量召回
        # -------------------------------------------------
        if self.enable_vector:
            vector_results = self._retrieve_vector(
                query=query,
                top_k=vector_top_k,
                where=vector_where,
            )
            all_results.extend(vector_results)

        # -------------------------------------------------
        # 2. 实体向量召回
        # -------------------------------------------------
        if self.enable_entity:
            entity_results = self._retrieve_entity(
                query=query,
                top_k=entity_top_k,
                where=entity_where,
            )
            all_results.extend(entity_results)

        # -------------------------------------------------
        # 3. 关键词召回
        # -------------------------------------------------
        if self.enable_keyword and keyword_corpus:
            keyword_results = self._retrieve_keyword(
                query=query,
                corpus=keyword_corpus,
                top_k=keyword_top_k,
            )
            all_results.extend(keyword_results)

        # -------------------------------------------------
        # 4. 融合与后处理
        # -------------------------------------------------
        merged = merge_retrieval_results(
            all_results,
            vector_weight=self.vector_weight,
            entity_weight=self.entity_weight,
            keyword_weight=self.keyword_weight,
        )

        if min_score > 0:
            merged = [
                item for item in merged
                if safe_float(item.get("score", 0.0), default=0.0) >= min_score
            ]

        merged = sorted(
            merged,
            key=lambda item: safe_float(item.get("score", 0.0), default=0.0),
            reverse=True,
        )

        return merged[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        retrieve() 的别名。
        """

        return self.retrieve(
            query,
            top_k=top_k,
            **kwargs,
        )

    # =====================================================
    # 2.2 文档向量召回
    # =====================================================

    def _retrieve_vector(
        self,
        *,
        query: str,
        top_k: int,
        where: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        调用 VectorRetriever。
        """

        try:
            retriever = self.vector_retriever or VectorRetriever()

            raw_results = retriever.retrieve(
                query,
                top_k=top_k,
                where=where,
            )

            return [
                normalize_retrieval_item(
                    item,
                    source_type="vector",
                    default_score=0.0,
                )
                for item in raw_results
            ]

        except Exception:
            if self.fail_silently:
                return []
            raise

    # =====================================================
    # 2.3 实体向量召回
    # =====================================================

    def _retrieve_entity(
        self,
        *,
        query: str,
        top_k: int,
        where: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        调用 EntityVectorStore。
        """

        try:
            store = self.entity_store or EntityVectorStore(
                create_if_missing=False,
            )

            raw_results = store.query(
                query,
                top_k=top_k,
                where=where,
            )

            normalized: List[Dict[str, Any]] = []

            for item in raw_results:
                entity_name = str(
                    item.get("entity_name")
                    or item.get("entity")
                    or ""
                ).strip()

                document = str(item.get("document", "") or "").strip()

                text = document or entity_name

                metadata = item.get("metadata", {}) or {}

                if not isinstance(metadata, dict):
                    metadata = {}

                metadata = dict(metadata)
                metadata["entity_id"] = item.get("entity_id", "")
                metadata["entity_name"] = entity_name
                metadata["aliases"] = item.get("aliases", [])

                normalized.append(
                    {
                        "id": str(item.get("vector_id") or item.get("entity_id") or make_result_id(text)),
                        "text": text,
                        "content": text,
                        "score": safe_float(item.get("score", 0.0), default=0.0),
                        "raw_score": safe_float(item.get("score", 0.0), default=0.0),
                        "distance": item.get("distance", None),
                        "source": "entity",
                        "source_type": "entity",
                        "retrieval_sources": ["entity"],
                        "metadata": metadata,
                        "raw_result": item,
                    }
                )

            return normalized

        except Exception:
            if self.fail_silently:
                return []
            raise

    # =====================================================
    # 2.4 关键词召回
    # =====================================================

    def _retrieve_keyword(
        self,
        *,
        query: str,
        corpus: Sequence[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        简单关键词召回。

        不依赖第三方 BM25 包。
        适合作为轻量补充召回。
        """

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        results: List[Dict[str, Any]] = []

        for idx, doc in enumerate(corpus):
            if isinstance(doc, str):
                text = doc
                metadata: Dict[str, Any] = {}
                doc_id = make_result_id(text)
            elif isinstance(doc, dict):
                text = first_non_empty_str(
                    doc,
                    ["text", "content", "document", "page_content", "body"],
                )
                metadata = doc.get("metadata", {}) or {}

                if not isinstance(metadata, dict):
                    metadata = {}

                doc_id = first_non_empty_str(
                    doc,
                    ["id", "doc_id", "document_id"],
                ) or make_result_id(text)
            else:
                continue

            text = str(text or "").strip()

            if not text:
                continue

            score = keyword_score(
                query_tokens=query_tokens,
                text=text,
            )

            if score <= 0:
                continue

            results.append(
                {
                    "id": doc_id,
                    "text": text,
                    "content": text,
                    "score": score,
                    "raw_score": score,
                    "source": "keyword",
                    "source_type": "keyword",
                    "retrieval_sources": ["keyword"],
                    "metadata": metadata,
                    "raw_result": doc,
                }
            )

        results = sorted(
            results,
            key=lambda item: safe_float(item.get("score", 0.0), default=0.0),
            reverse=True,
        )

        return results[:top_k]

    # =====================================================
    # 2.5 信息
    # =====================================================

    def info(self) -> Dict[str, Any]:
        """
        返回 HybridRetriever 信息。
        """

        return {
            "enable_vector": self.enable_vector,
            "enable_entity": self.enable_entity,
            "enable_keyword": self.enable_keyword,
            "vector_weight": self.vector_weight,
            "entity_weight": self.entity_weight,
            "keyword_weight": self.keyword_weight,
            "fail_silently": self.fail_silently,
            "has_vector_retriever": self.vector_retriever is not None,
            "has_entity_store": self.entity_store is not None,
        }


    def health_check(self) -> Dict[str, Any]:
        """返回混合召回配置状态，不触发后端查询。"""
        return {"ok": True, **self.info()}

# =========================================================
# 3. 结果标准化
# =========================================================

def normalize_retrieval_item(
    item: Any,
    *,
    source_type: str,
    default_score: float = 0.0,
) -> Dict[str, Any]:
    """
    将不同来源召回结果统一成标准结构。
    """

    if item is None:
        return {}

    if isinstance(item, str):
        text = item.strip()

        if not text:
            return {}

        return {
            "id": make_result_id(text),
            "text": text,
            "content": text,
            "score": clip_score(default_score),
            "raw_score": clip_score(default_score),
            "source": source_type,
            "source_type": source_type,
            "retrieval_sources": [source_type],
            "metadata": {},
            "raw_result": item,
        }

    if not isinstance(item, dict):
        return {}

    text = first_non_empty_str(
        item,
        ["text", "content", "document", "page_content", "body"],
    )

    if not text:
        entity_name = first_non_empty_str(
            item,
            ["entity_name", "entity", "name", "title", "label"],
        )
        text = entity_name

    text = str(text or "").strip()

    if not text:
        return {}

    item_id = first_non_empty_str(
        item,
        ["id", "doc_id", "document_id", "vector_id", "entity_id"],
    )

    if not item_id:
        item_id = make_result_id(text)

    metadata = item.get("metadata", {}) or {}

    if not isinstance(metadata, dict):
        metadata = {}

    score = extract_score(item, default=default_score)

    return {
        "id": item_id,
        "text": text,
        "content": text,
        "score": score,
        "raw_score": score,
        "distance": item.get("distance", None),
        "source": source_type,
        "source_type": source_type,
        "retrieval_sources": [source_type],
        "metadata": metadata,
        "raw_result": item,
    }


# =========================================================
# 4. 结果融合
# =========================================================

def merge_retrieval_results(
    results: List[Dict[str, Any]],
    *,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    entity_weight: float = DEFAULT_ENTITY_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
) -> List[Dict[str, Any]]:
    """
    融合不同来源的召回结果。

    逻辑：
        1. 按 dedup key 合并。
        2. 同一条结果如果被多个来源召回，分数累加。
        3. metadata 合并。
        4. retrieval_sources 记录来源。
    """

    weights = validate_weights(vector_weight=vector_weight, entity_weight=entity_weight, keyword_weight=keyword_weight)
    weight_map = {"vector": weights["vector_weight"], "entity": weights["entity_weight"], "keyword": weights["keyword_weight"]}

    merged: Dict[str, Dict[str, Any]] = {}

    for item in results:
        normalized = normalize_retrieval_item(
            item,
            source_type=str(item.get("source_type") or item.get("source") or "unknown"),
            default_score=0.0,
        )

        if not normalized:
            continue

        key = retrieval_dedup_key(normalized)

        if not key:
            continue

        source_type = str(normalized.get("source_type", "unknown"))
        source_weight = weight_map.get(source_type, 0.20)

        raw_score = safe_float(normalized.get("score", 0.0), default=0.0)
        weighted_score = raw_score * source_weight

        if key not in merged:
            new_item = dict(normalized)
            new_item["score"] = weighted_score
            new_item["raw_scores"] = {
                source_type: raw_score,
            }
            new_item["retrieval_sources"] = [source_type]
            new_item["source"] = "hybrid"
            merged[key] = new_item
            continue

        old = merged[key]

        old["score"] = safe_float(old.get("score", 0.0), default=0.0) + weighted_score

        raw_scores = old.get("raw_scores", {})
        if not isinstance(raw_scores, dict):
            raw_scores = {}

        raw_scores[source_type] = max(
            safe_float(raw_scores.get(source_type, 0.0), default=0.0),
            raw_score,
        )
        old["raw_scores"] = raw_scores

        sources = old.get("retrieval_sources", [])
        if not isinstance(sources, list):
            sources = []

        if source_type not in sources:
            sources.append(source_type)

        old["retrieval_sources"] = sources

        # 保留更长、更完整的 text
        old_text = str(old.get("text", "") or "")
        new_text = str(normalized.get("text", "") or "")

        if len(new_text) > len(old_text):
            old["text"] = new_text
            old["content"] = new_text

        # 合并 metadata
        old_metadata = old.get("metadata", {})
        new_metadata = normalized.get("metadata", {})

        if not isinstance(old_metadata, dict):
            old_metadata = {}

        if not isinstance(new_metadata, dict):
            new_metadata = {}

        old_metadata.update(new_metadata)
        old["metadata"] = old_metadata

    # 多来源命中奖励
    final_results: List[Dict[str, Any]] = []

    for item in merged.values():
        sources = item.get("retrieval_sources", [])

        if isinstance(sources, list) and len(sources) > 1:
            item["score"] = safe_float(item.get("score", 0.0), default=0.0) * (
                1.0 + 0.08 * (len(sources) - 1)
            )

        item["score"] = clip_score(safe_float(item.get("score", 0.0), default=0.0))
        final_results.append(item)

    return final_results


def retrieval_dedup_key(item: Dict[str, Any]) -> str:
    """
    构造召回结果去重 key。

    优先级：
        1. metadata.entity_id
        2. id
        3. normalized text
    """

    metadata = item.get("metadata", {})

    if isinstance(metadata, dict):
        entity_id = str(metadata.get("entity_id", "") or "").strip()

        if entity_id:
            return f"entity::{entity_id}"

    item_id = str(item.get("id", "") or "").strip()

    if item_id:
        return f"id::{item_id}"

    text = normalize_text(item.get("text", ""))

    if text:
        return f"text::{text}"

    return ""


# =========================================================
# 5. 关键词评分
# =========================================================

def keyword_score(
    *,
    query_tokens: List[str],
    text: str,
) -> float:
    """
    简单关键词匹配分数。

    分数考虑：
        1. token 命中数量
        2. token 覆盖率
        3. 连续短语命中
    """

    if not query_tokens or not text:
        return 0.0

    text_norm = normalize_text(text)
    text_tokens = tokenize(text_norm)

    if not text_tokens:
        return 0.0

    query_set = set(query_tokens)
    text_set = set(text_tokens)

    overlap = query_set & text_set

    if not overlap:
        # 中文短语兜底：query token 可能是整段中文
        phrase_hits = sum(
            1 for token in query_tokens
            if token and token in text_norm
        )

        if phrase_hits <= 0:
            return 0.0

        return clip_score(0.25 + 0.1 * phrase_hits)

    coverage = len(overlap) / max(len(query_set), 1)
    density = len(overlap) / max(len(text_set), 1)

    score = 0.65 * coverage + 0.25 * density

    query_phrase = " ".join(query_tokens)

    if query_phrase and query_phrase in text_norm:
        score += 0.10

    return clip_score(score)


def tokenize(text: Any) -> List[str]:
    """
    简单 token 切分。

    兼容：
        - 英文单词
        - 数字
        - 中文连续片段
    """

    text = normalize_text(text)

    if not text:
        return []

    english_tokens = re.findall(r"[a-zA-Z0-9_]+", text)
    chinese_tokens = re.findall(r"[\u4e00-\u9fff]+", text)

    tokens = english_tokens + chinese_tokens

    # 对较长中文片段增加 2-gram 辅助
    extra_tokens: List[str] = []

    for token in chinese_tokens:
        if len(token) >= 4:
            for i in range(len(token) - 1):
                extra_tokens.append(token[i:i + 2])

    tokens.extend(extra_tokens)

    seen = set()
    unique_tokens: List[str] = []

    for token in tokens:
        token = token.strip()

        if not token:
            continue

        if token in STOP_TOKENS:
            continue

        if token in seen:
            continue

        seen.add(token)
        unique_tokens.append(token)

    return unique_tokens


STOP_TOKENS = {
    "的",
    "了",
    "和",
    "与",
    "或",
    "是",
    "有",
    "在",
    "对",
    "中",
    "什么",
    "哪些",
    "哪个",
    "怎么",
    "如何",
    "关系",
    "介绍",
    "说明",
    "解释",
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "in",
    "on",
    "for",
    "with",
    "about",
    "what",
    "how",
    "why",
}


# =========================================================
# 6. 便捷函数
# =========================================================

_GLOBAL_HYBRID_RETRIEVER: Optional[HybridRetriever] = None


def get_default_hybrid_retriever() -> HybridRetriever:
    """
    获取默认 HybridRetriever。
    """

    global _GLOBAL_HYBRID_RETRIEVER

    if _GLOBAL_HYBRID_RETRIEVER is None:
        _GLOBAL_HYBRID_RETRIEVER = HybridRetriever()

    return _GLOBAL_HYBRID_RETRIEVER


def hybrid_retrieve(
    query: str,
    *,
    top_k: int = 10,
    keyword_corpus: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    函数式接口：混合召回。
    """

    retriever = get_default_hybrid_retriever()

    return retriever.retrieve(
        query,
        top_k=top_k,
        keyword_corpus=keyword_corpus,
    )


# =========================================================
# 7. 通用工具
# =========================================================

def normalize_query(value: Any) -> str:
    """
    查询文本标准化。
    """

    if value is None:
        return ""

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def normalize_text(value: Any) -> str:
    """
    文本标准化。
    """

    if value is None:
        return ""

    text = str(value).strip().lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)

    return text


def make_result_id(text: str) -> str:
    """
    根据文本生成稳定 ID。
    """

    digest = hashlib.md5(
        str(text or "").encode("utf-8")
    ).hexdigest()[:16]

    return f"ret_{digest}"


def extract_score(
    item: Dict[str, Any],
    *,
    default: float = 0.0,
) -> float:
    """
    从结果中提取分数。
    """

    for key in ["score", "similarity", "confidence"]:
        if key in item:
            return clip_score(safe_float(item.get(key), default=default))

    if "distance" in item:
        distance = safe_float(item.get("distance"), default=1.0)
        return clip_score(1.0 - distance)

    return clip_score(default)


def first_non_empty_str(
    item: Dict[str, Any],
    keys: List[str],
) -> str:
    """
    从 dict 中获取第一个非空字符串字段。
    """

    for key in keys:
        value = item.get(key)

        if value is None:
            continue

        value = str(value).strip()

        if value:
            return value

    return ""


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


def clip_score(score: float) -> float:
    """
    分数限制到 [0, 1]。
    """

    if score < 0:
        return 0.0

    if score > 1:
        return 1.0

    return score


# =========================================================
# 8. 快速测试入口
# =========================================================

if __name__ == "__main__":
    corpus = [
        {
            "id": "d1",
            "text": "阿尔茨海默病是一种神经退行性疾病。",
            "metadata": {
                "source": "demo",
            },
        },
        {
            "id": "d2",
            "text": "FDG-PET 可以反映脑代谢信息。",
            "metadata": {
                "source": "demo",
            },
        },
    ]

    retriever = HybridRetriever(
        enable_vector=False,
        enable_entity=False,
        enable_keyword=True,
    )

    results = retriever.retrieve(
        "阿尔茨海默病和FDG-PET有什么关系",
        top_k=5,
        keyword_corpus=corpus,
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))