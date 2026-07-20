# -*- coding: utf-8 -*-
"""Evaluation 业务服务。

当前版本保留现有评估数据结构和文件格式，只调整依赖方向：评估服务调用
AgentService，不再构造独立的简化 KG-RAG 流程。后续 evaluation/ 专项重构时，
这些数据结构再统一迁移到 evaluation/schemas.py。
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from kg_rag_agent.agents.schemas import AgentResult, RequestOptions
from kg_rag_agent.services.agent_service import AgentService


@dataclass(slots=True)
class EvaluationCase:
    query: str
    expected_answer: str = ""
    case_id: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationRecord:
    case_id: str
    query: str
    expected_answer: str
    answer: str
    request_id: str = ""
    route: str = ""
    answerability: str = ""
    semantic_score: float = 0.0
    latency_ms: float = 0.0
    has_error: bool = False
    error_message: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    traces: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvaluationSummary:
    total: int
    success: int
    failed: int
    avg_latency_ms: float
    started_at: str
    finished_at: str
    output_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvaluationResult:
    summary: EvaluationSummary
    records: List[EvaluationRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "records": [record.to_dict() for record in self.records],
        }


class EvaluationService:
    """评估样例加载、Agent 调用和报告保存服务。"""

    def __init__(
        self,
        *,
        agent_service: Optional[AgentService] = None,
        output_dir: str | Path = "outputs/evaluation",
        **agent_service_kwargs: Any,
    ) -> None:
        if agent_service is not None and agent_service_kwargs:
            raise ValueError(
                "agent_service cannot be combined with AgentService construction arguments."
            )

        self._owns_agent_service = agent_service is None
        self.agent_service = agent_service or AgentService(**agent_service_kwargs)
        self.output_dir = self._resolve_output_dir(output_dir)

    def _resolve_output_dir(self, output_dir: str | Path) -> Path:
        path = Path(output_dir).expanduser()
        if path.is_absolute():
            return path.resolve()
        try:
            root = self.agent_service.runtime.settings.project_root
        except Exception:
            root = Path.cwd()
        return (root / path).resolve()

    def load_cases(self, file_path: str | Path) -> List[EvaluationCase]:
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            try:
                path = self.agent_service.runtime.settings.resolve_path(path)
            except Exception:
                path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(f"Evaluation case file not found: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"Evaluation case path is not a file: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            return self._load_json_cases(path)
        if suffix == ".jsonl":
            return self._load_jsonl_cases(path)
        raise ValueError(
            f"Unsupported evaluation case file type: {suffix}. "
            "Only .json and .jsonl are supported."
        )

    def _load_json_cases(self, path: Path) -> List[EvaluationCase]:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            raw_cases = data.get("cases", [])
        elif isinstance(data, list):
            raw_cases = data
        else:
            raise ValueError(
                "Evaluation JSON must be a list or a dict with 'cases'."
            )

        if not isinstance(raw_cases, list):
            raise TypeError("Evaluation 'cases' must be a list.")
        return [
            self._parse_case(item, index=index)
            for index, item in enumerate(raw_cases)
        ]

    def _load_jsonl_cases(self, path: Path) -> List[EvaluationCase]:
        cases: List[EvaluationCase] = []
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    item = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at line {line_number}: {exc.msg}"
                    ) from exc
                cases.append(
                    self._parse_case(item, index=len(cases))
                )
        return cases

    def _parse_case(
        self,
        item: Mapping[str, Any],
        *,
        index: int,
    ) -> EvaluationCase:
        if not isinstance(item, Mapping):
            raise TypeError(
                f"Evaluation case must be dict, got: {type(item)}"
            )

        query = str(
            item.get("query")
            or item.get("question")
            or item.get("input")
            or ""
        ).strip()
        if not query:
            raise ValueError(f"Evaluation case query is empty at index {index}.")

        expected_answer = str(
            item.get("expected_answer")
            or item.get("expected")
            or item.get("answer")
            or item.get("reference")
            or ""
        )
        case_id = str(
            item.get("case_id")
            or item.get("id")
            or f"case_{index:04d}"
        )

        tags = item.get("tags", []) or []
        if isinstance(tags, str):
            tags = [tags]
        if not isinstance(tags, (list, tuple)):
            raise TypeError(f"Evaluation case tags must be list at index {index}.")

        metadata = item.get("metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            raise TypeError(
                f"Evaluation case metadata must be dict at index {index}."
            )

        return EvaluationCase(
            query=query,
            expected_answer=expected_answer,
            case_id=case_id,
            tags=[str(tag) for tag in tags],
            metadata=dict(metadata),
        )

    def evaluate_cases(
        self,
        cases: Sequence[EvaluationCase],
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_options: Optional[Mapping[str, Any]] = None,
        config_overrides: Optional[Mapping[str, Any]] = None,
        include_traces: bool = False,
        include_raw_state: bool = False,
    ) -> EvaluationResult:
        started_at = _utc_now()
        records = [
            self._evaluate_one_case(
                case,
                user_id=user_id,
                session_id=session_id,
                request_options=request_options,
                config_overrides=config_overrides,
                include_traces=include_traces,
                include_raw_state=include_raw_state,
            )
            for case in cases
        ]
        finished_at = _utc_now()

        success = sum(1 for record in records if not record.has_error)
        failed = len(records) - success
        avg_latency_ms = (
            sum(record.latency_ms for record in records) / len(records)
            if records
            else 0.0
        )
        return EvaluationResult(
            summary=EvaluationSummary(
                total=len(records),
                success=success,
                failed=failed,
                avg_latency_ms=round(avg_latency_ms, 3),
                started_at=started_at,
                finished_at=finished_at,
            ),
            records=records,
        )

    def evaluate_file(
        self,
        file_path: str | Path,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_options: Optional[Mapping[str, Any]] = None,
        config_overrides: Optional[Mapping[str, Any]] = None,
        include_traces: bool = False,
        include_raw_state: bool = False,
    ) -> EvaluationResult:
        return self.evaluate_cases(
            self.load_cases(file_path),
            user_id=user_id,
            session_id=session_id,
            request_options=request_options,
            config_overrides=config_overrides,
            include_traces=include_traces,
            include_raw_state=include_raw_state,
        )

    def _evaluate_one_case(
        self,
        case: EvaluationCase,
        *,
        user_id: Optional[str],
        session_id: Optional[str],
        request_options: Optional[Mapping[str, Any]],
        config_overrides: Optional[Mapping[str, Any]],
        include_traces: bool,
        include_raw_state: bool,
    ) -> EvaluationRecord:
        start_time = time.perf_counter()
        try:
            result = self.agent_service.ask(
                query=case.query,
                user_id=user_id,
                session_id=session_id,
                metadata={
                    "evaluation": True,
                    "case_id": case.case_id,
                    **case.metadata,
                },
                request_options=request_options,
                config_overrides=config_overrides,
                include_raw_state=include_raw_state,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return self._build_record_from_result(
                case=case,
                result=result,
                latency_ms=latency_ms,
                include_traces=include_traces,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return EvaluationRecord(
                case_id=case.case_id,
                query=case.query,
                expected_answer=case.expected_answer,
                answer="",
                latency_ms=round(latency_ms, 3),
                has_error=True,
                error_message=str(exc),
                tags=list(case.tags),
                metadata=dict(case.metadata),
            )

    @staticmethod
    def _build_record_from_result(
        *,
        case: EvaluationCase,
        result: AgentResult,
        latency_ms: float,
        include_traces: bool,
    ) -> EvaluationRecord:
        return EvaluationRecord(
            case_id=case.case_id,
            query=case.query,
            expected_answer=case.expected_answer,
            answer=result.answer,
            request_id=result.request_id,
            route=result.route,
            answerability=result.answerability,
            semantic_score=result.semantic_score,
            latency_ms=round(latency_ms, 3),
            has_error=result.has_error,
            error_message=result.error_message,
            citations=list(result.citations or []),
            warnings=list(result.warnings or []),
            traces=(
                list(result.traces or [])
                if include_traces
                else []
            ),
            tags=list(case.tags),
            metadata=dict(case.metadata),
        )

    def save_result(
        self,
        result: EvaluationResult,
        *,
        output_path: Optional[str | Path] = None,
        save_csv: bool = True,
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.output_dir / f"evaluation_{timestamp}.json"
        else:
            path = Path(output_path).expanduser()
            if not path.is_absolute():
                path = self.output_dir / path
            path = path.resolve()

        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)

        data = result.to_dict()
        data["summary"]["output_path"] = str(path)
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

        result.summary.output_path = str(path)
        if save_csv:
            self.save_records_csv(result.records, path.with_suffix(".csv"))
        return path

    def save_records_csv(
        self,
        records: Sequence[EvaluationRecord],
        output_path: str | Path,
    ) -> Path:
        path = Path(output_path).expanduser()
        if not path.is_absolute():
            path = self.output_dir / path
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "case_id",
            "query",
            "expected_answer",
            "answer",
            "request_id",
            "route",
            "answerability",
            "semantic_score",
            "latency_ms",
            "has_error",
            "error_message",
            "tags",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                row = record.to_dict()
                row["tags"] = ",".join(record.tags)
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        return path

    def info(self) -> Dict[str, Any]:
        return {
            "service": "EvaluationService",
            "output_dir": str(self.output_dir),
            "agent_service": self.agent_service.__class__.__name__,
        }

    def close(self) -> None:
        if self._owns_agent_service:
            self.agent_service.close()

    def __enter__(self) -> "EvaluationService":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


_DEFAULT_EVALUATION_SERVICE: Optional[EvaluationService] = None


def get_default_evaluation_service(
    *,
    refresh: bool = False,
    **kwargs: Any,
) -> EvaluationService:
    global _DEFAULT_EVALUATION_SERVICE
    if refresh and _DEFAULT_EVALUATION_SERVICE is not None:
        _DEFAULT_EVALUATION_SERVICE.close()
        _DEFAULT_EVALUATION_SERVICE = None
    if _DEFAULT_EVALUATION_SERVICE is None:
        _DEFAULT_EVALUATION_SERVICE = EvaluationService(**kwargs)
    elif kwargs:
        raise ValueError(
            "Default EvaluationService already exists; use refresh=True to rebuild it."
        )
    return _DEFAULT_EVALUATION_SERVICE


def evaluate_file(
    file_path: str | Path,
    **kwargs: Any,
) -> EvaluationResult:
    return get_default_evaluation_service().evaluate_file(
        file_path=file_path,
        **kwargs,
    )


def evaluate_cases(
    cases: Sequence[EvaluationCase],
    **kwargs: Any,
) -> EvaluationResult:
    return get_default_evaluation_service().evaluate_cases(
        cases=cases,
        **kwargs,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "EvaluationCase",
    "EvaluationRecord",
    "EvaluationSummary",
    "EvaluationResult",
    "EvaluationService",
    "get_default_evaluation_service",
    "evaluate_file",
    "evaluate_cases",
]
