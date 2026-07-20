# -*- coding: utf-8 -*-
"""Evaluation 结果报告输出。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from .schemas import EvaluationRunResult


class EvaluationReporter:
    """将一次运行保存为稳定的机器可读和人工可读文件。"""

    def __init__(self, *, output_root: str | Path = "outputs/evaluation") -> None:
        self.output_root = Path(output_root).expanduser().resolve()

    def save(
        self,
        result: EvaluationRunResult,
        *,
        output_dir: str | Path | None = None,
        include_raw_state: bool = False,
        write_csv: bool = True,
    ) -> Path:
        target = self._resolve_run_dir(result.summary.run_id, output_dir)
        target.mkdir(parents=True, exist_ok=True)
        self._write_json(target / "manifest.json", result.manifest)
        self._write_json(target / "summary.json", result.summary.to_dict())
        self._write_jsonl(
            target / "records.jsonl",
            (
                record.to_dict(include_raw_state=include_raw_state)
                for record in result.records
            ),
        )
        self._write_jsonl(
            target / "errors.jsonl",
            (
                record.to_dict(include_raw_state=include_raw_state)
                for record in result.records
                if record.has_error
            ),
        )
        (target / "report.md").write_text(
            self._render_markdown(result),
            encoding="utf-8",
        )
        if write_csv:
            self._write_csv(target / "records.csv", result)
        result.summary.output_dir = target.as_posix()
        self._write_json(target / "summary.json", result.summary.to_dict())
        return target

    def _resolve_run_dir(self, run_id: str, output_dir: str | Path | None) -> Path:
        if output_dir is None:
            return self.output_root / run_id
        candidate = Path(output_dir).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (self.output_root / candidate).resolve()

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _write_jsonl(path: Path, items: Iterable[Mapping[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as file:
            for item in items:
                file.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _write_csv(path: Path, result: EvaluationRunResult) -> None:
        fieldnames = [
            "case_id",
            "query",
            "prediction",
            "reference_answer",
            "route",
            "expected_route",
            "answerability",
            "expected_answerability",
            "latency_ms",
            "has_error",
            "error_stage",
            "error_message",
            "num_mentions",
            "num_grounded_entities",
            "num_evidence",
            "num_citations",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for record in result.records:
                data = record.to_dict(include_raw_state=False)
                writer.writerow({key: data.get(key, "") for key in fieldnames})

    @staticmethod
    def _render_markdown(result: EvaluationRunResult) -> str:
        summary = result.summary
        metrics = summary.metrics
        lines = [
            f"# Evaluation Report: {summary.run_id}",
            "",
            f"- Total: {summary.total}",
            f"- Success: {summary.success}",
            f"- Failed: {summary.failed}",
            f"- Duration: {summary.duration_ms:.3f} ms",
            f"- Average latency: {summary.avg_latency_ms:.3f} ms",
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
            "```",
            "",
            "## Failures",
            "",
        ]
        failures = [record for record in result.records if record.has_error]
        if not failures:
            lines.append("No failed cases.")
        else:
            for record in failures:
                lines.append(
                    f"- `{record.case_id}` [{record.error_stage or 'unknown'}]: "
                    f"{record.error_message or 'unknown error'}"
                )
        lines.append("")
        return "\n".join(lines)


__all__ = ["EvaluationReporter"]
