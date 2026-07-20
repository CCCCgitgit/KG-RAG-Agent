# -*- coding: utf-8 -*-
"""KG-RAG Agent 评估层统一导出。"""

from .dataset_loader import (
    DEFAULT_DATASET_PATH,
    EvaluationDatasetLoader,
    load_cases,
    save_cases,
)
from .evaluator import Evaluator, evaluate_cases, evaluate_file
from .manifest import build_manifest, file_sha256, stable_hash, utc_now_iso
from .metrics import (
    answerability_match,
    attach_case_metrics,
    attach_metrics_to_results,
    contains_match,
    count_by_key,
    evaluate_case_metrics,
    evaluate_results,
    exact_match,
    keyword_recall,
    mean,
    normalize_for_match,
    normalize_text,
    percentile,
    remove_punctuation,
    route_match,
    safe_divide,
    summarize_boolean_metric,
    summarize_eval_results,
    summarize_numeric_metric,
)
from .reporter import EvaluationReporter
from .runner import AgentServiceProtocol, EvaluationRunner
from .schemas import (
    EvaluationCase,
    EvaluationRecord,
    EvaluationRunResult,
    EvaluationSummary,
)

__all__ = [
    "DEFAULT_DATASET_PATH",
    "EvaluationCase",
    "EvaluationRecord",
    "EvaluationSummary",
    "EvaluationRunResult",
    "EvaluationDatasetLoader",
    "EvaluationRunner",
    "EvaluationReporter",
    "AgentServiceProtocol",
    "Evaluator",
    "load_cases",
    "save_cases",
    "evaluate_cases",
    "evaluate_file",
    "build_manifest",
    "file_sha256",
    "stable_hash",
    "utc_now_iso",
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
