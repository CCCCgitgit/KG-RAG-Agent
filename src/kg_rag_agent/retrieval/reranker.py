# -*- coding: utf-8 -*-
"""
reranker.py

召回结果重排序模块。

作用：
    1. 对 vector_retriever / hybrid_retriever 返回的候选结果进行二次排序。
    2. 支持轻量规则重排：
        - 原始召回分数
        - 查询词覆盖
        - 实体词匹配
        - 文本长度惩罚
        - 来源权重
    3. 支持可选 CrossEncoder 重排。
    4. 输出统一排序后的结果。

本文件属于 retrieval 层：
    retrieval/
        reranker.py

它不负责：
    1. 向量库查询。
    2. 图结构查询。
    3. 实体落地。
    4. 最终回答生成。
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ._validation import non_negative_float, positive_int, require_non_empty_text, validate_weights
from .errors import RetrievalDependencyError


# =========================================================
# 1. 默认配置
# =========================================================

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

DEFAULT_SCORE_WEIGHT = 0.45
DEFAULT_QUERY_OVERLAP_WEIGHT = 0.25
DEFAULT_ENTITY_MATCH_WEIGHT = 0.15
DEFAULT_SOURCE_WEIGHT = 0.10
DEFAULT_LENGTH_WEIGHT = 0.05


# =========================================================
# 2. Reranker
# =========================================================

class Reranker:
    """
    召回结果重排序器。

    用法：
        reranker = Reranker()
        results = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(
        self,
        *,
        use_cross_encoder: bool = False,
        cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
        local_files_only: bool = True,
        device: Optional[str] = None,
        fail_silently: bool = True,
        score_weight: float = DEFAULT_SCORE_WEIGHT,
        query_overlap_weight: float = DEFAULT_QUERY_OVERLAP_WEIGHT,
        entity_match_weight: float = DEFAULT_ENTITY_MATCH_WEIGHT,
        source_weight: float = DEFAULT_SOURCE_WEIGHT,
        length_weight: float = DEFAULT_LENGTH_WEIGHT,
        lazy_load: bool = True,
    ) -> None:
        """
        Args:
            use_cross_encoder:
                是否启用 CrossEncoder 重排。

            cross_encoder_model:
                CrossEncoder 模型名称或本地路径。

            local_files_only:
                是否只加载本地模型。

            device:
                运行设备，例如 cpu / cuda。

            fail_silently:
                CrossEncoder 加载或推理失败时是否自动回退规则重排。

            score_weight:
                原始召回分数权重。

            query_overlap_weight:
                查询词覆盖权重。

            entity_match_weight:
                实体词匹配权重。

            source_weight:
                来源权重。

            length_weight:
                文本长度合理性权重。

            lazy_load:
                是否延迟加载 CrossEncoder。
        """

        self.use_cross_encoder = bool(use_cross_encoder)
        self.cross_encoder_model = require_non_empty_text(cross_encoder_model, field="cross_encoder_model")
        self.local_files_only = bool(local_files_only)
        self.device = device
        self.fail_silently = bool(fail_silently)

        weights = validate_weights(score_weight=score_weight, query_overlap_weight=query_overlap_weight, entity_match_weight=entity_match_weight, source_weight=source_weight, length_weight=length_weight)
        self.score_weight = weights["score_weight"]
        self.query_overlap_weight = weights["query_overlap_weight"]
        self.entity_match_weight = weights["entity_match_weight"]
        self.source_weight = weights["source_weight"]
        self.length_weight = weights["length_weight"]

        self.lazy_load = bool(lazy_load)

        self.cross_encoder: Optional[Any] = None

        if self.use_cross_encoder and not self.lazy_load:
            self._ensure_cross_encoder_loaded()

    # =====================================================
    # 2.1 主接口
    # =====================================================

    def rerank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        *,
        top_k: Optional[int] = None,
        entities: Optional[Sequence[str]] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        对候选结果进行重排序。

        Args:
            query:
                用户查询。

            candidates:
                召回候选列表。

            top_k:
                最终返回数量。
                None 表示不截断。

            entities:
                可选的关键实体名称列表，用于增强 entity_match_score。

            min_score:
                最小 rerank_score 阈值。

        Returns:
            List[Dict[str, Any]]:
                排序后的候选列表，每项会增加：
                    - rerank_score
                    - rerank_rank
                    - rerank_method
                    - rerank_detail
        """

        query = normalize_query(query)

        if not query:
            return []

        normalized_candidates = normalize_candidates(candidates)

        if not normalized_candidates:
            return []

        if self.use_cross_encoder:
            try:
                reranked = self._rerank_with_cross_encoder(
                    query=query,
                    candidates=normalized_candidates,
                    entities=entities,
                )
            except Exception:
                if not self.fail_silently:
                    raise

                reranked = self._rerank_with_rules(
                    query=query,
                    candidates=normalized_candidates,
                    entities=entities,
                )
        else:
            reranked = self._rerank_with_rules(
                query=query,
                candidates=normalized_candidates,
                entities=entities,
            )

        min_score = non_negative_float(min_score, field="min_score")

        if min_score > 0:
            reranked = [
                item for item in reranked
                if safe_float(item.get("rerank_score", 0.0), default=0.0) >= min_score
            ]

        reranked = sorted(
            reranked,
            key=lambda item: safe_float(item.get("rerank_score", 0.0), default=0.0),
            reverse=True,
        )

        for idx, item in enumerate(reranked, start=1):
            item["rerank_rank"] = idx

        if top_k is not None:
            return reranked[: positive_int(top_k, field="top_k")]

        return reranked

    def rank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        rerank() 的别名。
        """

        return self.rerank(
            query=query,
            candidates=candidates,
            **kwargs,
        )

    # =====================================================
    # 2.2 规则重排
    # =====================================================

    def _rerank_with_rules(
        self,
        *,
        query: str,
        candidates: List[Dict[str, Any]],
        entities: Optional[Sequence[str]],
    ) -> List[Dict[str, Any]]:
        """
        轻量规则重排。
        """

        query_tokens = tokenize(query)
        entity_terms = [
            normalize_text(entity)
            for entity in (entities or [])
            if normalize_text(entity)
        ]

        reranked: List[Dict[str, Any]] = []

        for candidate in candidates:
            text = str(candidate.get("text", "") or "").strip()
            metadata = candidate.get("metadata", {}) or {}

            if not isinstance(metadata, dict):
                metadata = {}

            base_score = extract_score(candidate, default=0.0)
            query_overlap = query_overlap_score(query_tokens, text)
            entity_match = entity_match_score(entity_terms, text, metadata)
            source_score = source_priority_score(candidate)
            length_score = text_length_score(text)

            rerank_score = (
                self.score_weight * base_score
                + self.query_overlap_weight * query_overlap
                + self.entity_match_weight * entity_match
                + self.source_weight * source_score
                + self.length_weight * length_score
            )

            rerank_score = clip_score(rerank_score)

            new_item = dict(candidate)
            new_item["rerank_score"] = rerank_score
            new_item["rerank_method"] = "rule"
            new_item["rerank_detail"] = {
                "base_score": base_score,
                "query_overlap_score": query_overlap,
                "entity_match_score": entity_match,
                "source_score": source_score,
                "length_score": length_score,
                "weights": {
                    "score_weight": self.score_weight,
                    "query_overlap_weight": self.query_overlap_weight,
                    "entity_match_weight": self.entity_match_weight,
                    "source_weight": self.source_weight,
                    "length_weight": self.length_weight,
                },
            }

            reranked.append(new_item)

        return reranked

    # =====================================================
    # 2.3 CrossEncoder 重排
    # =====================================================

    def _rerank_with_cross_encoder(
        self,
        *,
        query: str,
        candidates: List[Dict[str, Any]],
        entities: Optional[Sequence[str]],
    ) -> List[Dict[str, Any]]:
        """
        使用 CrossEncoder 重排。

        说明：
            CrossEncoder 输出可能不是 [0,1]。
            这里会做 min-max 归一化后，再与规则分数融合。
        """

        self._ensure_cross_encoder_loaded()

        if self.cross_encoder is None:
            return self._rerank_with_rules(
                query=query,
                candidates=candidates,
                entities=entities,
            )

        pairs = [
            [
                query,
                str(candidate.get("text", "") or ""),
            ]
            for candidate in candidates
        ]

        raw_scores = self.cross_encoder.predict(pairs)

        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()

        raw_scores = [
            safe_float(score, default=0.0)
            for score in raw_scores
        ]

        normalized_cross_scores = min_max_normalize(raw_scores)

        rule_results = self._rerank_with_rules(
            query=query,
            candidates=candidates,
            entities=entities,
        )

        final_results: List[Dict[str, Any]] = []

        for idx, item in enumerate(rule_results):
            rule_score = safe_float(item.get("rerank_score", 0.0), default=0.0)
            cross_score = normalized_cross_scores[idx] if idx < len(normalized_cross_scores) else 0.0

            final_score = clip_score(
                0.65 * cross_score + 0.35 * rule_score
            )

            new_item = dict(item)
            new_item["rerank_score"] = final_score
            new_item["rerank_method"] = "cross_encoder"
            new_item["rerank_detail"] = dict(item.get("rerank_detail", {}) or {})
            new_item["rerank_detail"]["cross_encoder_raw_score"] = (
                raw_scores[idx] if idx < len(raw_scores) else 0.0
            )
            new_item["rerank_detail"]["cross_encoder_score"] = cross_score
            new_item["rerank_detail"]["rule_score"] = rule_score

            final_results.append(new_item)

        return final_results

    def _ensure_cross_encoder_loaded(self) -> None:
        """
        加载 CrossEncoder 模型。
        """

        if self.cross_encoder is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RetrievalDependencyError(
                "sentence-transformers is not installed. "
                "Please install sentence-transformers first."
            ) from exc

        kwargs: Dict[str, Any] = {}

        if self.device:
            kwargs["device"] = self.device

        try:
            self.cross_encoder = CrossEncoder(
                self.cross_encoder_model,
                local_files_only=self.local_files_only,
                **kwargs,
            )
        except TypeError:
            self.cross_encoder = CrossEncoder(
                self.cross_encoder_model,
                **kwargs,
            )

    # =====================================================
    # 2.4 信息
    # =====================================================

    def info(self) -> Dict[str, Any]:
        """
        返回 Reranker 信息。
        """

        return {
            "use_cross_encoder": self.use_cross_encoder,
            "cross_encoder_model": self.cross_encoder_model,
            "local_files_only": self.local_files_only,
            "device": self.device,
            "fail_silently": self.fail_silently,
            "cross_encoder_loaded": self.cross_encoder is not None,
            "weights": {
                "score_weight": self.score_weight,
                "query_overlap_weight": self.query_overlap_weight,
                "entity_match_weight": self.entity_match_weight,
                "source_weight": self.source_weight,
                "length_weight": self.length_weight,
            },
        }


    def health_check(self) -> Dict[str, Any]:
        """返回重排器健康状态，不触发模型加载。"""
        return {"ok": True, **self.info()}

    def close(self) -> None:
        """释放当前实例持有的 CrossEncoder 引用。"""
        self.cross_encoder = None

# =========================================================
# 3. 候选标准化
# =========================================================

def normalize_candidates(
    candidates: Sequence[Any],
) -> List[Dict[str, Any]]:
    """
    标准化候选列表。
    """

    normalized: List[Dict[str, Any]] = []

    for idx, candidate in enumerate(candidates):
        item = normalize_candidate(candidate)

        if not item:
            continue

        item.setdefault("rank", idx + 1)
        normalized.append(item)

    return normalized


def normalize_candidate(candidate: Any) -> Dict[str, Any]:
    """
    标准化单个候选。
    """

    if candidate is None:
        return {}

    if isinstance(candidate, str):
        text = candidate.strip()

        if not text:
            return {}

        return {
            "id": make_candidate_id(text),
            "text": text,
            "content": text,
            "score": 0.5,
            "source": "unknown",
            "metadata": {},
        }

    if not isinstance(candidate, dict):
        return {}

    text = first_non_empty_str(
        candidate,
        ["text", "content", "document", "page_content", "body"],
    )

    if not text:
        text = first_non_empty_str(
            candidate,
            ["entity_name", "entity", "name", "title", "label"],
        )

    text = str(text or "").strip()

    if not text:
        return {}

    item_id = first_non_empty_str(
        candidate,
        ["id", "doc_id", "document_id", "vector_id", "entity_id", "evidence_id"],
    )

    if not item_id:
        item_id = make_candidate_id(text)

    metadata = candidate.get("metadata", {}) or {}

    if not isinstance(metadata, dict):
        metadata = {}

    normalized = dict(candidate)
    normalized["id"] = item_id
    normalized["text"] = text
    normalized["content"] = text
    normalized["score"] = extract_score(candidate, default=0.0)
    normalized["metadata"] = metadata

    return normalized


# =========================================================
# 4. 规则打分函数
# =========================================================

def query_overlap_score(
    query_tokens: List[str],
    text: str,
) -> float:
    """
    查询 token 与文本的覆盖分数。
    """

    if not query_tokens or not text:
        return 0.0

    text_tokens = tokenize(text)

    if not text_tokens:
        return 0.0

    query_set = set(query_tokens)
    text_set = set(text_tokens)

    overlap = query_set & text_set

    if not overlap:
        text_norm = normalize_text(text)
        phrase_hits = sum(
            1 for token in query_tokens
            if token and token in text_norm
        )

        if phrase_hits <= 0:
            return 0.0

        return clip_score(0.2 + 0.1 * phrase_hits)

    coverage = len(overlap) / max(len(query_set), 1)
    density = len(overlap) / max(len(text_set), 1)

    return clip_score(0.75 * coverage + 0.25 * density)


def entity_match_score(
    entity_terms: List[str],
    text: str,
    metadata: Dict[str, Any],
) -> float:
    """
    实体词匹配分数。
    """

    if not entity_terms:
        return 0.0

    text_norm = normalize_text(text)

    metadata_text = normalize_text(
        " ".join(
            str(value)
            for value in metadata.values()
            if isinstance(value, (str, int, float))
        )
    )

    combined = f"{text_norm} {metadata_text}"

    hits = 0

    for term in entity_terms:
        if term and term in combined:
            hits += 1

    return clip_score(hits / max(len(entity_terms), 1))


def source_priority_score(candidate: Dict[str, Any]) -> float:
    """
    根据来源给基础优先级。
    """

    source = str(candidate.get("source_type") or candidate.get("source") or "").lower()

    retrieval_sources = candidate.get("retrieval_sources", [])

    if isinstance(retrieval_sources, list):
        retrieval_sources = [str(item).lower() for item in retrieval_sources]
    else:
        retrieval_sources = []

    if len(retrieval_sources) >= 2:
        return 1.0

    if "entity" in retrieval_sources or source == "entity":
        return 0.90

    if "vector" in retrieval_sources or source == "vector":
        return 0.80

    if "keyword" in retrieval_sources or source == "keyword":
        return 0.65

    if source == "chroma":
        return 0.75

    return 0.50


def text_length_score(text: str) -> float:
    """
    文本长度合理性评分。

    太短信息不足，太长可能噪声多。
    """

    text = str(text or "").strip()
    length = len(text)

    if length <= 0:
        return 0.0

    if 40 <= length <= 500:
        return 1.0

    if 15 <= length < 40:
        return 0.75

    if 500 < length <= 1200:
        return 0.75

    if length < 15:
        return 0.45

    return 0.55


# =========================================================
# 5. 便捷函数
# =========================================================

_GLOBAL_RERANKER: Optional[Reranker] = None


def get_default_reranker() -> Reranker:
    """
    获取默认 Reranker。
    """

    global _GLOBAL_RERANKER

    if _GLOBAL_RERANKER is None:
        _GLOBAL_RERANKER = Reranker()

    return _GLOBAL_RERANKER


def rerank_results(
    query: str,
    candidates: Sequence[Dict[str, Any]],
    *,
    top_k: Optional[int] = None,
    entities: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    函数式接口：重排候选结果。
    """

    reranker = get_default_reranker()

    return reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=top_k,
        entities=entities,
    )


# =========================================================
# 6. 通用文本工具
# =========================================================

def normalize_query(value: Any) -> str:
    """
    查询文本清洗。
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


def tokenize(value: Any) -> List[str]:
    """
    简单 token 切分。

    支持：
        - 英文
        - 数字
        - 中文连续片段
        - 中文 2-gram 辅助
    """

    text = normalize_text(value)

    if not text:
        return []

    english_tokens = re.findall(r"[a-zA-Z0-9_]+", text)
    chinese_tokens = re.findall(r"[\u4e00-\u9fff]+", text)

    tokens = english_tokens + chinese_tokens

    extra_tokens: List[str] = []

    for token in chinese_tokens:
        if len(token) >= 4:
            for idx in range(len(token) - 1):
                extra_tokens.append(token[idx:idx + 2])

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
    "中",
    "对",
    "为",
    "什么",
    "哪些",
    "哪个",
    "怎么",
    "如何",
    "关系",
    "介绍",
    "说明",
    "解释",
    "分析",
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
# 7. 数值与字段工具
# =========================================================

def extract_score(
    item: Dict[str, Any],
    *,
    default: float = 0.0,
) -> float:
    """
    从候选中抽取基础分数。
    """

    for key in ["rerank_score", "score", "similarity", "confidence"]:
        if key in item:
            return clip_score(safe_float(item.get(key), default=default))

    if "distance" in item:
        distance = safe_float(item.get("distance"), default=1.0)
        return clip_score(1.0 - distance)

    return clip_score(default)


def min_max_normalize(values: Sequence[float]) -> List[float]:
    """
    min-max 归一化。
    """

    if not values:
        return []

    values = [
        safe_float(value, default=0.0)
        for value in values
    ]

    min_value = min(values)
    max_value = max(values)

    if abs(max_value - min_value) < 1e-12:
        return [0.5 for _ in values]

    return [
        (value - min_value) / (max_value - min_value)
        for value in values
    ]


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
    将分数限制到 [0, 1]。
    """

    if score < 0:
        return 0.0

    if score > 1:
        return 1.0

    return score


def first_non_empty_str(
    item: Dict[str, Any],
    keys: List[str],
) -> str:
    """
    从 dict 中取第一个非空字符串字段。
    """

    for key in keys:
        value = item.get(key)

        if value is None:
            continue

        value = str(value).strip()

        if value:
            return value

    return ""


def make_candidate_id(text: str) -> str:
    """
    根据文本生成稳定候选 ID。
    """

    import hashlib

    digest = hashlib.md5(
        str(text or "").encode("utf-8")
    ).hexdigest()[:16]

    return f"cand_{digest}"


# =========================================================
# 8. 快速测试入口
# =========================================================

if __name__ == "__main__":
    query = "阿尔茨海默病和FDG-PET有什么关系"

    candidates = [
        {
            "id": "1",
            "text": "阿尔茨海默病是一种神经退行性疾病。",
            "score": 0.72,
            "source": "vector",
        },
        {
            "id": "2",
            "text": "FDG-PET 可以反映脑代谢信息，并常用于阿尔茨海默病相关研究。",
            "score": 0.68,
            "source": "vector",
        },
        {
            "id": "3",
            "text": "这是一个无关文本。",
            "score": 0.80,
            "source": "keyword",
        },
    ]

    reranker = Reranker()
    results = reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=3,
        entities=["阿尔茨海默病", "FDG-PET"],
    )

    from pprint import pprint

    pprint(results)