# -*- coding: utf-8 -*-
"""调用正式 AgentService 和 evaluation 层执行评估。"""
from __future__ import annotations

import argparse
from typing import Any, Optional

try:
    from _script_runtime import positive_int, print_json
except ImportError:  # pragma: no cover
    from scripts._script_runtime import positive_int, print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate KG-RAG Agent.")
    parser.add_argument("--input", default="data/demo/examples/demo_questions.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dataset-name", default="kg-rag-evaluation")
    parser.add_argument("--dataset-version", default="1.0.0")
    parser.add_argument("--include-raw-state", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    return parser


def run(
    *,
    input_path: str,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    limit: Optional[int] = None,
    run_id: Optional[str] = None,
    dataset_name: str = "kg-rag-evaluation",
    dataset_version: str = "1.0.0",
    include_raw_state: bool = False,
    strict: bool = True,
    write_csv: bool = True,
    validate: bool = True,
) -> dict[str, Any]:
    from kg_rag_agent.evaluation import (
        EvaluationDatasetLoader,
        EvaluationReporter,
        EvaluationRunner,
    )
    from kg_rag_agent.services import AgentService

    service = AgentService(config_path=config_path, validate=validate)
    try:
        cases = EvaluationDatasetLoader().load(input_path, limit=limit, strict=strict)
        runner = EvaluationRunner(
            agent_service=service,
            include_raw_state=include_raw_state,
            continue_on_error=True,
        )
        result = runner.run(
            cases,
            run_id=run_id,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
        )
        target = EvaluationReporter().save(
            result,
            output_dir=output_dir,
            include_raw_state=include_raw_state,
            write_csv=write_csv,
        )
        return {
            "ok": result.summary.failed == 0,
            "output_dir": target.as_posix(),
            "summary": result.summary.to_dict(),
        }
    finally:
        service.close()


def main() -> None:
    args = build_parser().parse_args()
    result = run(
        input_path=args.input,
        output_dir=args.output_dir,
        config_path=args.config,
        limit=args.limit,
        run_id=args.run_id,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        include_raw_state=args.include_raw_state,
        strict=args.strict,
        write_csv=not args.no_csv,
        validate=not args.no_validate,
    )
    print_json(result)
    if not result.get("ok", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
