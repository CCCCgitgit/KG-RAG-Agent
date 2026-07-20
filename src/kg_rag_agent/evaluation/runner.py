# -*- coding: utf-8 -*-
"""Evaluation 唯一执行器。"""

from __future__ import annotations

import copy
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol

from .manifest import build_manifest, utc_now_iso
from .metrics import attach_case_metrics, summarize_eval_results
from .schemas import (
    EvaluationCase,
    EvaluationRecord,
    EvaluationRunResult,
    EvaluationSummary,
)


class AgentServiceProtocol(Protocol):
    def ask(self, query: str, **kwargs: Any) -> Any: ...


class EvaluationRunner:
    """逐条调用正式 AgentService，并隔离单条失败。"""

    def __init__(
        self,
        *,
        agent_service: AgentServiceProtocol,
        include_raw_state: bool = False,
        continue_on_error: bool = True,
    ) -> None:
        if agent_service is None:
            raise ValueError("agent_service is required.")
        self.agent_service = agent_service
        self.include_raw_state = bool(include_raw_state)
        self.continue_on_error = bool(continue_on_error)

    def run(
        self,
        cases: Iterable[EvaluationCase | Mapping[str, Any]],
        *,
        run_id: Optional[str] = None,
        dataset_name: str = "kg-rag-evaluation",
        dataset_version: str = "1.0.0",
        request_options: Optional[Mapping[str, Any]] = None,
        user_id: str = "evaluation",
        session_prefix: str = "eval",
        manifest_extra: Optional[Mapping[str, Any]] = None,
    ) -> EvaluationRunResult:
        normalized_cases = self._normalize_cases(cases)
        normalized_run_id = str(run_id or self._new_run_id()).strip()
        if not normalized_run_id:
            raise ValueError("run_id must not be empty.")
        started_at = utc_now_iso()
        started_clock = time.perf_counter()
        records: List[EvaluationRecord] = []
        for index, case in enumerate(normalized_cases):
            try:
                record = self.run_case(
                    case,
                    request_options=request_options,
                    user_id=user_id,
                    session_id=f"{session_prefix}-{normalized_run_id}-{index:04d}",
                )
            except Exception:
                if not self.continue_on_error:
                    raise
                record = self._unexpected_error_record(case)
            records.append(record)
        finished_at = utc_now_iso()
        duration_ms = round((time.perf_counter() - started_clock) * 1000.0, 3)
        flat_records = [record.to_dict(include_raw_state=False) for record in records]
        for item, record in zip(flat_records, records):
            item.update(record.metrics)
        aggregate = summarize_eval_results(flat_records)
        failed = sum(1 for item in records if item.has_error)
        summary = EvaluationSummary(
            run_id=normalized_run_id,
            total=len(records),
            success=len(records) - failed,
            failed=failed,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            metrics=aggregate,
        )
        manifest = build_manifest(
            run_id=normalized_run_id,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            started_at=started_at,
            finished_at=finished_at,
            cases=normalized_cases,
            request_options=request_options,
            agent_service=self.agent_service,
            extra=manifest_extra,
        )
        return EvaluationRunResult(
            summary=summary,
            records=records,
            manifest=manifest,
        )

    def run_case(
        self,
        case: EvaluationCase,
        *,
        request_options: Optional[Mapping[str, Any]] = None,
        user_id: str = "evaluation",
        session_id: Optional[str] = None,
    ) -> EvaluationRecord:
        started = time.perf_counter()
        request_id = f"eval-{case.case_id}-{uuid.uuid4().hex[:10]}"
        try:
            result = self.agent_service.ask(
                case.query,
                user_id=user_id,
                session_id=session_id or f"eval-{case.case_id}",
                request_id=request_id,
                metadata={
                    "evaluation": True,
                    "case_id": case.case_id,
                    "category": case.category,
                    "difficulty": case.difficulty,
                },
                request_options=dict(request_options or {}),
                include_raw_state=True,
            )
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            return self._record_from_result(case, result, latency_ms=latency_ms)
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            if not self.continue_on_error:
                raise
            return self._error_record(
                case,
                request_id=request_id,
                latency_ms=latency_ms,
                message=str(exc),
                stage="dependency",
            )

    def _record_from_result(
        self,
        case: EvaluationCase,
        result: Any,
        *,
        latency_ms: float,
    ) -> EvaluationRecord:
        data = self._result_to_dict(result)
        raw_state = data.get("raw_state") if isinstance(data.get("raw_state"), Mapping) else {}
        prediction = str(
            data.get("answer")
            or data.get("final_answer")
            or raw_state.get("final_answer")
            or ""
        )
        route = str(data.get("route") or raw_state.get("route") or "")
        answerability = str(
            data.get("answerability")
            or raw_state.get("answerability")
            or ""
        )
        citations = self._dict_list(data.get("citations") or raw_state.get("citations"))
        warnings = [str(item) for item in (data.get("warnings") or raw_state.get("warnings") or [])]
        has_error = bool(data.get("has_error") or raw_state.get("has_error"))
        error_message = str(data.get("error_message") or raw_state.get("error_message") or "")
        metrics = attach_case_metrics({
            "answer": prediction,
            "expected_answer": case.reference_answer,
            "expected_keywords": case.keywords,
            "route": route,
            "expected_route": case.expected_route,
            "answerability": answerability,
            "expected_answerability": case.expected_answerability,
        })
        metric_keys = (
            "has_reference",
            "exact_match",
            "contains_match",
            "keyword_recall",
            "has_expected_route",
            "route_match",
            "has_expected_answerability",
            "answerability_match",
        )
        state_summary = self._state_summary(raw_state)
        return EvaluationRecord(
            case_id=case.case_id,
            query=case.query,
            prediction=prediction,
            reference_answer=case.reference_answer,
            request_id=str(data.get("request_id") or raw_state.get("request_id") or ""),
            route=route,
            expected_route=case.expected_route,
            answerability=answerability,
            expected_answerability=case.expected_answerability,
            semantic_score=self._safe_float(
                data.get("semantic_score", raw_state.get("semantic_score", 0.0))
            ),
            latency_ms=latency_ms,
            has_error=has_error,
            error_message=error_message,
            error_stage=self._infer_error_stage(raw_state, has_error),
            num_mentions=state_summary["num_mentions"],
            num_grounded_entities=state_summary["num_grounded_entities"],
            num_evidence=state_summary["num_evidence"],
            num_citations=len(citations),
            num_warnings=len(warnings),
            citations=citations,
            warnings=warnings,
            metrics={key: metrics.get(key) for key in metric_keys},
            state_summary=state_summary,
            raw_state=copy.deepcopy(dict(raw_state)) if self.include_raw_state else None,
            category=case.category,
            difficulty=case.difficulty,
            tags=list(case.tags),
            metadata=copy.deepcopy(case.metadata),
        )

    def _error_record(
        self,
        case: EvaluationCase,
        *,
        request_id: str,
        latency_ms: float,
        message: str,
        stage: str,
    ) -> EvaluationRecord:
        metrics = attach_case_metrics({
            "answer": "",
            "expected_answer": case.reference_answer,
            "expected_keywords": case.keywords,
            "route": "error",
            "expected_route": case.expected_route,
            "answerability": "unanswerable",
            "expected_answerability": case.expected_answerability,
        })
        return EvaluationRecord(
            case_id=case.case_id,
            query=case.query,
            prediction="",
            reference_answer=case.reference_answer,
            request_id=request_id,
            route="error",
            expected_route=case.expected_route,
            answerability="unanswerable",
            expected_answerability=case.expected_answerability,
            latency_ms=latency_ms,
            has_error=True,
            error_message=message or "unknown error",
            error_stage=stage,
            metrics={
                key: metrics.get(key)
                for key in (
                    "has_reference",
                    "exact_match",
                    "contains_match",
                    "keyword_recall",
                    "has_expected_route",
                    "route_match",
                    "has_expected_answerability",
                    "answerability_match",
                )
            },
            category=case.category,
            difficulty=case.difficulty,
            tags=list(case.tags),
            metadata=copy.deepcopy(case.metadata),
        )

    def _unexpected_error_record(self, case: EvaluationCase) -> EvaluationRecord:
        return self._error_record(
            case,
            request_id="",
            latency_ms=0.0,
            message="Unexpected evaluation runner failure.",
            stage="evaluation",
        )

    @staticmethod
    def _normalize_cases(
        cases: Iterable[EvaluationCase | Mapping[str, Any]],
    ) -> List[EvaluationCase]:
        result: List[EvaluationCase] = []
        seen = set()
        for index, item in enumerate(cases):
            case = item if isinstance(item, EvaluationCase) else EvaluationCase.from_dict(item, index=index)
            if case.case_id in seen:
                raise ValueError(f"Duplicate evaluation case_id: {case.case_id}")
            seen.add(case.case_id)
            result.append(case)
        return result

    @staticmethod
    def _result_to_dict(result: Any) -> Dict[str, Any]:
        to_dict = getattr(result, "to_dict", None)
        if callable(to_dict):
            try:
                value = to_dict(include_raw_state=True)
            except TypeError:
                value = to_dict()
            return dict(value) if isinstance(value, Mapping) else {"answer": str(value)}
        if isinstance(result, Mapping):
            return dict(result)
        return {"answer": str(result)}

    @staticmethod
    def _dict_list(value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [copy.deepcopy(dict(item)) for item in value if isinstance(item, Mapping)]

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _state_summary(raw_state: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "num_mentions": EvaluationRunner._list_size(raw_state, "mentions"),
            "num_entity_candidates": EvaluationRunner._list_size(raw_state, "entity_candidates"),
            "num_linked_entities": EvaluationRunner._list_size(raw_state, "linked_entities"),
            "num_grounded_entities": EvaluationRunner._list_size(raw_state, "grounded_entities"),
            "num_evidence": max(
                EvaluationRunner._list_size(raw_state, "selected_evidence"),
                EvaluationRunner._list_size(raw_state, "evidence"),
            ),
            "fallback_used": bool(raw_state.get("fallback_used", False)),
        }

    @staticmethod
    def _list_size(raw_state: Mapping[str, Any], key: str) -> int:
        value = raw_state.get(key)
        return len(value) if isinstance(value, list) else 0

    @staticmethod
    def _infer_error_stage(raw_state: Mapping[str, Any], has_error: bool) -> str:
        if not has_error:
            return ""
        for key in ("error_stage", "failed_stage", "current_stage", "last_node"):
            value = raw_state.get(key)
            if value:
                return str(value)
        return "unknown"

    @staticmethod
    def _new_run_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"eval-{timestamp}-{uuid.uuid4().hex[:8]}"


__all__ = ["AgentServiceProtocol", "EvaluationRunner"]
