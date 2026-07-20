# -*- coding: utf-8 -*-
"""旧测试样例接口兼容；新代码使用 schemas 与 dataset_loader。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .dataset_loader import DEFAULT_DATASET_PATH, EvaluationDatasetLoader
from .schemas import EvaluationCase

DEFAULT_SAMPLE_QUESTIONS_PATH = DEFAULT_DATASET_PATH
DEFAULT_SAMPLE_OUTPUTS_PATH = "data/demo/examples/sample_outputs.json"


def default_test_cases() -> List[EvaluationCase]:
    return [
        EvaluationCase(
            case_id="case_direct_0001",
            query="你好，请简单介绍一下你能做什么。",
            expected_route="direct_llm",
            category="direct_llm",
            difficulty="easy",
        ),
        EvaluationCase(
            case_id="case_kg_0001",
            query="Barack Obama 和 Michelle Obama 有什么关系？",
            expected_route="kg_rag",
            expected_entities=["Barack Obama", "Michelle Obama"],
            category="kg_relation",
        ),
        EvaluationCase(
            case_id="case_clarify_0001",
            query="它们之间是什么关系？",
            expected_route="clarify",
            category="clarify",
            difficulty="easy",
        ),
    ]


def default_sample_outputs() -> List[Dict[str, Any]]:
    return [
        {
            "case_id": "case_direct_0001",
            "query": "你好，请简单介绍一下你能做什么。",
            "prediction": "我可以进行普通问答和基于知识图谱的检索问答。",
            "route": "direct_llm",
            "answerability": "",
            "has_error": False,
        }
    ]


def load_test_cases(
    path: str | Path = DEFAULT_SAMPLE_QUESTIONS_PATH,
    *,
    limit: Optional[int] = None,
    fallback_to_default: bool = True,
) -> List[EvaluationCase]:
    loader = EvaluationDatasetLoader()
    try:
        return loader.load(path, limit=limit)
    except FileNotFoundError:
        if fallback_to_default:
            cases = default_test_cases()
            return cases[:limit] if limit is not None else cases
        raise


def save_test_cases(
    cases: Iterable[EvaluationCase | Mapping[str, Any]],
    path: str | Path = DEFAULT_SAMPLE_QUESTIONS_PATH,
) -> Path:
    return EvaluationDatasetLoader().save(cases, path)


def save_sample_outputs(
    outputs: Optional[List[Dict[str, Any]]] = None,
    path: str | Path = DEFAULT_SAMPLE_OUTPUTS_PATH,
) -> Path:
    import json
    loader = EvaluationDatasetLoader()
    resolved = loader.resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(
            {"version": "1.0.0", "outputs": outputs or default_sample_outputs()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return resolved


def ensure_default_test_files(
    *,
    questions_path: str | Path = DEFAULT_SAMPLE_QUESTIONS_PATH,
    outputs_path: str | Path = DEFAULT_SAMPLE_OUTPUTS_PATH,
    overwrite: bool = False,
) -> Dict[str, Any]:
    loader = EvaluationDatasetLoader()
    questions = loader.resolve_path(questions_path)
    outputs = loader.resolve_path(outputs_path)
    wrote_questions = overwrite or not questions.exists()
    wrote_outputs = overwrite or not outputs.exists()
    if wrote_questions:
        save_test_cases(default_test_cases(), questions)
    if wrote_outputs:
        save_sample_outputs(default_sample_outputs(), outputs)
    return {
        "ok": True,
        "questions_path": questions.as_posix(),
        "outputs_path": outputs.as_posix(),
        "wrote_questions": wrote_questions,
        "wrote_outputs": wrote_outputs,
    }


def filter_cases(
    cases: Iterable[EvaluationCase],
    *,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    expected_route: Optional[str] = None,
) -> List[EvaluationCase]:
    return [
        case
        for case in cases
        if (category is None or case.category == category)
        and (difficulty is None or case.difficulty == difficulty)
        and (expected_route is None or case.expected_route == expected_route)
    ]


def list_case_ids(cases: Iterable[EvaluationCase]) -> List[str]:
    return [case.case_id for case in cases]


def find_case_by_id(
    cases: Iterable[EvaluationCase],
    case_id: str,
) -> Optional[EvaluationCase]:
    target = str(case_id)
    return next((case for case in cases if case.case_id == target), None)


def cases_to_dicts(
    cases: Iterable[EvaluationCase | Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for index, item in enumerate(cases):
        case = item if isinstance(item, EvaluationCase) else EvaluationCase.from_dict(item, index=index)
        result.append(case.to_dict())
    return result


__all__ = [
    "EvaluationCase",
    "DEFAULT_SAMPLE_QUESTIONS_PATH",
    "DEFAULT_SAMPLE_OUTPUTS_PATH",
    "default_test_cases",
    "default_sample_outputs",
    "load_test_cases",
    "save_test_cases",
    "save_sample_outputs",
    "ensure_default_test_files",
    "filter_cases",
    "list_case_ids",
    "find_case_by_id",
    "cases_to_dicts",
]
