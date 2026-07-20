# -*- coding: utf-8 -*-
"""向后兼容的 Evaluator 门面；新代码优先使用 EvaluationRunner。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .dataset_loader import EvaluationDatasetLoader, load_cases
from .reporter import EvaluationReporter
from .runner import EvaluationRunner
from .schemas import EvaluationCase, EvaluationRunResult


class Evaluator:
    def __init__(
        self,
        *,
        service: Any = None,
        agent_service: Any = None,
        config_path: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        auto_build_graph: bool = True,
        validate: bool = True,
        include_raw_state: bool = False,
        continue_on_error: bool = True,
    ) -> None:
        selected = agent_service or service
        if selected is None:
            from kg_rag_agent.services import AgentService
            selected = AgentService(
                config=config,
                config_path=config_path,
                auto_build_graph=auto_build_graph,
                validate=validate,
            )
        self.service = selected
        self.runner = EvaluationRunner(
            agent_service=selected,
            include_raw_state=include_raw_state,
            continue_on_error=continue_on_error,
        )

    def evaluate_case(self, case: EvaluationCase | Mapping[str, Any]) -> Dict[str, Any]:
        normalized = case if isinstance(case, EvaluationCase) else EvaluationCase.from_dict(case)
        return self.runner.run_case(normalized).to_dict(
            include_raw_state=self.runner.include_raw_state
        )

    def evaluate_cases(
        self,
        cases: Iterable[EvaluationCase | Mapping[str, Any]],
        **kwargs: Any,
    ) -> EvaluationRunResult:
        return self.runner.run(cases, **kwargs)

    def evaluate_file(
        self,
        input_path: str | Path,
        *,
        output_dir: str | Path | None = None,
        limit: Optional[int] = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> EvaluationRunResult:
        loader = EvaluationDatasetLoader()
        cases = loader.load(input_path, limit=limit, strict=strict)
        result = self.runner.run(cases, **kwargs)
        if output_dir is not None:
            EvaluationReporter().save(
                result,
                output_dir=output_dir,
                include_raw_state=self.runner.include_raw_state,
            )
        return result


def evaluate_cases(
    cases: Iterable[EvaluationCase | Mapping[str, Any]],
    *,
    service: Any = None,
    **kwargs: Any,
) -> EvaluationRunResult:
    return Evaluator(service=service).evaluate_cases(cases, **kwargs)


def evaluate_file(
    input_path: str | Path,
    *,
    service: Any = None,
    **kwargs: Any,
) -> EvaluationRunResult:
    return Evaluator(service=service).evaluate_file(input_path, **kwargs)


def load_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    return [case.to_dict() for case in EvaluationDatasetLoader().load(path)]


def normalize_case(record: Mapping[str, Any], *, index: int = 0) -> EvaluationCase:
    return EvaluationCase.from_dict(record, index=index)


__all__ = [
    "EvaluationCase",
    "Evaluator",
    "load_cases",
    "load_json_or_jsonl",
    "normalize_case",
    "evaluate_cases",
    "evaluate_file",
]
