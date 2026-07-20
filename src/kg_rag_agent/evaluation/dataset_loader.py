# -*- coding: utf-8 -*-
"""Evaluation 数据集的唯一加载与保存入口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .schemas import EvaluationCase


DEFAULT_DATASET_PATH = "data/demo/examples/demo_questions.json"
_CONTAINER_KEYS = ("cases", "questions", "data", "items", "examples", "test_cases")


class EvaluationDatasetLoader:
    """加载、校验和保存 JSON / JSONL 评估集。"""

    def __init__(self, *, project_root: str | Path | None = None) -> None:
        self.project_root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[3]
        )

    def resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        cwd_candidate = (Path.cwd() / candidate).resolve()
        if cwd_candidate.exists():
            return cwd_candidate
        return (self.project_root / candidate).resolve()

    def load(
        self,
        path: str | Path = DEFAULT_DATASET_PATH,
        *,
        limit: Optional[int] = None,
        strict: bool = True,
    ) -> List[EvaluationCase]:
        resolved = self.resolve_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {resolved}")
        if not resolved.is_file():
            raise IsADirectoryError(f"Evaluation dataset path is not a file: {resolved}")
        suffix = resolved.suffix.lower()
        if suffix == ".jsonl":
            records = self._read_jsonl(resolved, strict=strict)
        elif suffix == ".json":
            records = self._read_json(resolved)
        else:
            raise ValueError("Evaluation dataset must use .json or .jsonl.")
        return self.parse_records(records, limit=limit, strict=strict)

    def parse_records(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        limit: Optional[int] = None,
        strict: bool = True,
    ) -> List[EvaluationCase]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0.")
        cases: List[EvaluationCase] = []
        seen_ids = set()
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                if strict:
                    raise TypeError(f"Evaluation item at index {index} must be an object.")
                continue
            try:
                case = EvaluationCase.from_dict(record, index=index)
            except (TypeError, ValueError):
                if strict:
                    raise
                continue
            if case.case_id in seen_ids:
                raise ValueError(f"Duplicate evaluation case_id: {case.case_id}")
            seen_ids.add(case.case_id)
            cases.append(case)
            if limit is not None and len(cases) >= limit:
                break
        return cases

    def save(
        self,
        cases: Iterable[EvaluationCase | Mapping[str, Any]],
        path: str | Path,
        *,
        dataset_name: str = "kg-rag-evaluation",
        dataset_version: str = "1.0.0",
    ) -> Path:
        resolved = self.resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        normalized: List[EvaluationCase] = []
        for index, item in enumerate(cases):
            normalized.append(
                item if isinstance(item, EvaluationCase)
                else EvaluationCase.from_dict(item, index=index)
            )
        self._ensure_unique_ids(normalized)
        if resolved.suffix.lower() == ".jsonl":
            with resolved.open("w", encoding="utf-8") as file:
                for case in normalized:
                    file.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
        elif resolved.suffix.lower() == ".json":
            payload = {
                "dataset_name": dataset_name,
                "dataset_version": dataset_version,
                "cases": [case.to_dict() for case in normalized],
            }
            resolved.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            raise ValueError("Evaluation dataset must use .json or .jsonl.")
        return resolved

    @staticmethod
    def _ensure_unique_ids(cases: Iterable[EvaluationCase]) -> None:
        seen = set()
        for case in cases:
            if case.case_id in seen:
                raise ValueError(f"Duplicate evaluation case_id: {case.case_id}")
            seen.add(case.case_id)

    @staticmethod
    def _read_json(path: Path) -> List[Mapping[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, Mapping):
            for key in _CONTAINER_KEYS:
                value = data.get(key)
                if isinstance(value, list):
                    return value
            if any(key in data for key in ("query", "question", "input", "user_query")):
                return [data]
        raise ValueError(
            "Unsupported evaluation JSON format. Expected a list, one case object, "
            "or an object containing cases/questions/data/items/examples/test_cases."
        )

    @staticmethod
    def _read_jsonl(path: Path, *, strict: bool) -> List[Mapping[str, Any]]:
        records: List[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    item = json.loads(text)
                except json.JSONDecodeError as exc:
                    if strict:
                        raise ValueError(
                            f"Invalid JSONL at line {line_number}: {exc.msg}"
                        ) from exc
                    continue
                if not isinstance(item, Mapping):
                    if strict:
                        raise TypeError(
                            f"JSONL item at line {line_number} must be an object."
                        )
                    continue
                records.append(item)
        return records


def load_cases(
    path: str | Path = DEFAULT_DATASET_PATH,
    *,
    limit: Optional[int] = None,
    strict: bool = True,
    project_root: str | Path | None = None,
) -> List[EvaluationCase]:
    return EvaluationDatasetLoader(project_root=project_root).load(
        path,
        limit=limit,
        strict=strict,
    )


def save_cases(
    cases: Iterable[EvaluationCase | Mapping[str, Any]],
    path: str | Path,
    *,
    dataset_name: str = "kg-rag-evaluation",
    dataset_version: str = "1.0.0",
    project_root: str | Path | None = None,
) -> Path:
    return EvaluationDatasetLoader(project_root=project_root).save(
        cases,
        path,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
    )


__all__ = [
    "DEFAULT_DATASET_PATH",
    "EvaluationDatasetLoader",
    "load_cases",
    "save_cases",
]
