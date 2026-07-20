from __future__ import annotations

import json

import pytest

from kg_rag_agent.evaluation.dataset_loader import EvaluationDatasetLoader
from kg_rag_agent.evaluation.metrics import (
    contains_match,
    evaluate_case_metrics,
    exact_match,
    keyword_recall,
)
from kg_rag_agent.evaluation.schemas import EvaluationCase, EvaluationRecord


def test_evaluation_case_aliases_and_validation():
    case = EvaluationCase.from_dict(
        {
            "id": "case-1",
            "question": "Who is Alice?",
            "expected_answer": "A person",
            "expected_keywords": ["person"],
        }
    )
    assert case.case_id == "case-1"
    assert case.query == "Who is Alice?"
    assert case.reference_answer == "A person"
    with pytest.raises(ValueError):
        EvaluationCase(case_id="", query="question")


def test_evaluation_dataset_json_and_jsonl_roundtrip(tmp_path):
    loader = EvaluationDatasetLoader(project_root=tmp_path)
    cases = [EvaluationCase(case_id="c1", query="q1")]
    json_path = loader.save(cases, tmp_path / "cases.json")
    jsonl_path = loader.save(cases, tmp_path / "cases.jsonl")
    assert loader.load(json_path)[0].case_id == "c1"
    assert loader.load(jsonl_path)[0].query == "q1"

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        json.dumps({"cases": [{"id": "x", "query": "a"}, {"id": "x", "query": "b"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        loader.load(duplicate)


def test_evaluation_metrics_are_stable():
    assert exact_match("Hello, World!", "hello world") is True
    assert contains_match("Alice works at Example Corp", "Example Corp") is True
    assert keyword_recall("Alice and Bob", ["Alice", "Bob"]) == 1.0

    case = EvaluationCase(
        case_id="c1",
        query="q",
        reference_answer="Alice",
        expected_route="kg_rag",
        keywords=["Alice"],
    )
    record = EvaluationRecord(
        case_id="c1",
        query="q",
        prediction="Alice",
        route="kg_rag",
    )
    metrics = evaluate_case_metrics(
        prediction=record.prediction,
        reference=case.reference_answer,
        keywords=case.keywords,
        predicted_route=record.route,
        expected_route=case.expected_route,
    )
    assert metrics["exact_match"] is True
    assert metrics["route_match"] is True
