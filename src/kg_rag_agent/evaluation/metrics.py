# -*- coding: utf-8 -*-
"""
metrics.py

KG-RAG Agent 评估指标模块。

职责：
    1. 提供问答结果的基础评估指标。
    2. 提供 route / answerability / evidence / latency 等统计。
    3. 为 evaluation/evaluator.py 和 scripts/evaluate.py 提供可复用指标函数。

注意：
    evaluation 层只做评估。
    不参与线上问答主流程。
    不实现 KG 查询、向量检索、实体链接或 LLM 调用。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence


# =========================================================
# 1. 文本标准化
# =========================================================

def normalize_text(text: Any) -> str:
    """
    标准化文本，用于简单匹配评估。
    """

    if text is None:
        return ""

    value = str(text).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def remove_punctuation(text: str) -> str:
    """
    移除常见中英文标点。
    """

    return re.sub(
        r"[，。！？；：“”‘’、,.!?;:\"'()\[\]{}<>《》【】]",
        "",
        text,
    )


def normalize_for_match(text: Any) -> str:
    """
    用于答案匹配的文本标准化。
    """

    return remove_punctuation(normalize_text(text)).strip()


# =========================================================
# 2. 基础安全工具
# =========================================================

def safe_divide(
    numerator: float,
    denominator: float,
    *,
    default: float = 0.0,
) -> float:
    """
    安全除法。
    """

    if denominator == 0:
        return default

    return numerator / denominator


def mean(values: Sequence[float]) -> float:
    """
    均值。
    """

    if not values:
        return 0.0

    return sum(values) / len(values)


def percentile(
    values: Sequence[float],
    q: float,
) -> float:
    """
    简单百分位数。

    Args:
        values:
            数值列表。

        q:
            百分位，范围 0-100。
    """

    if not values:
        return 0.0

    sorted_values = sorted(values)

    if q <= 0:
        return float(sorted_values[0])

    if q >= 100:
        return float(sorted_values[-1])

    position = (len(sorted_values) - 1) * (q / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(sorted_values[int(position)])

    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]

    weight = position - lower

    return float(lower_value * (1 - weight) + upper_value * weight)


def count_by_key(
    items: Iterable[Dict[str, Any]],
    key: str,
) -> Dict[str, int]:
    """
    按某个字段计数。
    """

    counter: Counter[str] = Counter()

    for item in items:
        value = str(item.get(key) or "")
        counter[value] += 1

    return dict(counter)


# =========================================================
# 3. 单条样例评估
# =========================================================

def exact_match(
    prediction: Any,
    reference: Any,
) -> bool:
    """
    精确匹配。

    会做基础大小写、空格和标点标准化。
    """

    pred = normalize_for_match(prediction)
    ref = normalize_for_match(reference)

    if not pred or not ref:
        return False

    return pred == ref


def contains_match(
    prediction: Any,
    reference: Any,
) -> bool:
    """
    包含匹配。

    适合开放式中文回答，只判断参考答案是否出现在模型回答中。
    """

    pred = normalize_for_match(prediction)
    ref = normalize_for_match(reference)

    if not pred or not ref:
        return False

    return ref in pred


def keyword_recall(
    prediction: Any,
    keywords: Optional[Sequence[Any]],
) -> float:
    """
    关键词召回率。

    keywords 为空时返回 0。
    """

    if not keywords:
        return 0.0

    pred = normalize_for_match(prediction)

    if not pred:
        return 0.0

    normalized_keywords = [
        normalize_for_match(keyword)
        for keyword in keywords
        if normalize_for_match(keyword)
    ]

    if not normalized_keywords:
        return 0.0

    hit = 0

    for keyword in normalized_keywords:
        if keyword in pred:
            hit += 1

    return safe_divide(hit, len(normalized_keywords))


def route_match(
    predicted_route: Any,
    expected_route: Any,
) -> bool:
    """
    路由匹配。
    """

    expected = normalize_text(expected_route)
    predicted = normalize_text(predicted_route)

    if not expected:
        return False

    return expected == predicted


def answerability_match(
    predicted_answerability: Any,
    expected_answerability: Any,
) -> bool:
    """
    answerability 匹配。
    """

    expected = normalize_text(expected_answerability)
    predicted = normalize_text(predicted_answerability)

    if not expected:
        return False

    return expected == predicted


def evaluate_case_metrics(
    *,
    prediction: Any,
    reference: Any = "",
    keywords: Optional[Sequence[Any]] = None,
    predicted_route: Any = "",
    expected_route: Any = "",
    predicted_answerability: Any = "",
    expected_answerability: Any = "",
) -> Dict[str, Any]:
    """
    评估单条样例的基础指标。
    """

    has_reference = bool(normalize_text(reference))
    has_expected_route = bool(normalize_text(expected_route))
    has_expected_answerability = bool(normalize_text(expected_answerability))

    result = {
        "has_reference": has_reference,
        "exact_match": exact_match(prediction, reference) if has_reference else None,
        "contains_match": contains_match(prediction, reference) if has_reference else None,
        "keyword_recall": keyword_recall(prediction, keywords),
        "has_expected_route": has_expected_route,
        "route_match": route_match(predicted_route, expected_route) if has_expected_route else None,
        "has_expected_answerability": has_expected_answerability,
        "answerability_match": (
            answerability_match(
                predicted_answerability,
                expected_answerability,
            )
            if has_expected_answerability
            else None
        ),
    }

    return result


# =========================================================
# 4. 批量结果汇总
# =========================================================

def summarize_boolean_metric(
    items: Iterable[Dict[str, Any]],
    key: str,
) -> Dict[str, Any]:
    """
    汇总布尔指标。

    只统计值为 True / False 的样例，跳过 None。
    """

    total = 0
    hit = 0

    for item in items:
        value = item.get(key)

        if value is None:
            continue

        if isinstance(value, bool):
            total += 1
            if value:
                hit += 1

    return {
        "total": total,
        "hit": hit,
        "score": round(safe_divide(hit, total), 4) if total else None,
    }


def summarize_numeric_metric(
    items: Iterable[Dict[str, Any]],
    key: str,
) -> Dict[str, Any]:
    """
    汇总数值指标。
    """

    values: List[float] = []

    for item in items:
        value = item.get(key)

        if value is None:
            continue

        try:
            values.append(float(value))
        except Exception:
            continue

    if not values:
        return {
            "count": 0,
            "avg": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }

    return {
        "count": len(values),
        "avg": round(mean(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "p50": round(percentile(values, 50), 4),
        "p90": round(percentile(values, 90), 4),
        "p95": round(percentile(values, 95), 4),
    }


def summarize_eval_results(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    汇总批量评估结果。

    兼容 scripts/evaluate.py 输出格式。
    """

    total = len(results)

    if total == 0:
        return {
            "total": 0,
            "message": "No evaluation results.",
        }

    num_errors = 0
    num_with_answer = 0
    num_with_evidence = 0
    num_with_grounded_entities = 0

    for item in results:
        if item.get("has_error"):
            num_errors += 1

        answer = str(item.get("answer") or item.get("final_answer") or "").strip()
        if answer:
            num_with_answer += 1

        if int(item.get("num_evidence") or 0) > 0:
            num_with_evidence += 1

        if int(item.get("num_grounded_entities") or 0) > 0:
            num_with_grounded_entities += 1

    return {
        "total": total,
        "num_errors": num_errors,
        "error_rate": round(safe_divide(num_errors, total), 4),
        "num_with_answer": num_with_answer,
        "answer_rate": round(safe_divide(num_with_answer, total), 4),
        "num_with_evidence": num_with_evidence,
        "evidence_rate": round(safe_divide(num_with_evidence, total), 4),
        "num_with_grounded_entities": num_with_grounded_entities,
        "grounded_entity_rate": round(
            safe_divide(num_with_grounded_entities, total),
            4,
        ),
        "route_counts": count_by_key(results, "route"),
        "answerability_counts": count_by_key(results, "answerability"),
        "boolean_metrics": {
            "exact_match": summarize_boolean_metric(results, "exact_match"),
            "contains_match": summarize_boolean_metric(results, "contains_match"),
            "route_match": summarize_boolean_metric(results, "route_match"),
            "answerability_match": summarize_boolean_metric(results, "answerability_match"),
        },
        "numeric_metrics": {
            "keyword_recall": summarize_numeric_metric(results, "keyword_recall"),
            "latency_ms": summarize_numeric_metric(results, "latency_ms"),
            "num_evidence": summarize_numeric_metric(results, "num_evidence"),
            "num_grounded_entities": summarize_numeric_metric(results, "num_grounded_entities"),
            "num_warnings": summarize_numeric_metric(results, "num_warnings"),
        },
    }


# =========================================================
# 5. 结果增强
# =========================================================

def attach_case_metrics(
    item: Dict[str, Any],
) -> Dict[str, Any]:
    """
    给单条 evaluate 结果补充指标字段。

    输入兼容字段：
        answer / final_answer
        expected_answer
        expected_keywords / keywords
        route / expected_route
        answerability / expected_answerability
    """

    prediction = item.get("answer") or item.get("final_answer") or ""
    reference = item.get("expected_answer") or item.get("reference") or ""

    keywords = (
        item.get("expected_keywords")
        or item.get("keywords")
        or item.get("reference_keywords")
        or []
    )

    metrics = evaluate_case_metrics(
        prediction=prediction,
        reference=reference,
        keywords=keywords,
        predicted_route=item.get("route", ""),
        expected_route=item.get("expected_route", ""),
        predicted_answerability=item.get("answerability", ""),
        expected_answerability=item.get("expected_answerability", ""),
    )

    return {
        **item,
        **metrics,
    }


def attach_metrics_to_results(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    批量补充评估指标。
    """

    return [
        attach_case_metrics(item)
        for item in results
    ]


# =========================================================
# 6. 对外统一入口
# =========================================================

def evaluate_results(
    results: List[Dict[str, Any]],
    *,
    attach_metrics: bool = True,
) -> Dict[str, Any]:
    """
    评估已有结果列表。

    返回：
        {
            "summary": ...,
            "results": ...
        }
    """

    evaluated_results = (
        attach_metrics_to_results(results)
        if attach_metrics
        else results
    )

    summary = summarize_eval_results(evaluated_results)

    return {
        "summary": summary,
        "results": evaluated_results,
    }


__all__ = [
    "normalize_text",
    "normalize_for_match",
    "remove_punctuation",
    "safe_divide",
    "mean",
    "percentile",
    "count_by_key",
    "exact_match",
    "contains_match",
    "keyword_recall",
    "route_match",
    "answerability_match",
    "evaluate_case_metrics",
    "summarize_boolean_metric",
    "summarize_numeric_metric",
    "summarize_eval_results",
    "attach_case_metrics",
    "attach_metrics_to_results",
    "evaluate_results",
]