# -*- coding: utf-8 -*-
"""Memory Write LangGraph 节点。

本节点在最终答案生成后更新受控会话 Memory，并按策略写入长期 Memory。
它只负责一次请求结束阶段的 Memory 协调，不创建 Store、Retriever、LLM Client
或其他共享运行时对象。

职责边界：
    * MemoryManager 必须由 RuntimeContext 创建并复用；
    * AgentState 只保存候选和可序列化写入结果；
    * 当前轮对话可以进入短期消息窗口和会话摘要；
    * 长期 Memory 只写入显式候选或明确的“请记住”请求；
    * Memory 写入默认 fail-open，不得因为辅助记忆失败而丢失最终答案。
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..state import AgentState, make_error, utc_now

if TYPE_CHECKING:
    from kg_rag_agent.runtime import RuntimeContext


_DEFAULT_FAIL_OPEN = True
_DEFAULT_TRACK_CONVERSATION = True
_DEFAULT_SUMMARY_ENABLED = True
_DEFAULT_SUMMARY_MIN_MESSAGES = 2
_DEFAULT_SUMMARY_USE_LLM = False
_DEFAULT_CAPTURE_EXPLICIT_MEMORY = True
_MAX_CANDIDATE_CHARS = 20_000

_EXPLICIT_MEMORY_PATTERNS = (
    re.compile(
        r"^\s*(?:请|帮我|麻烦你)?\s*记住\s*(?:这件事|以下内容|这一点|：|:)?\s*(?P<content>.+?)\s*$",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"^\s*(?:please\s+)?remember(?:\s+that)?\s+(?P<content>.+?)\s*$",
        re.IGNORECASE | re.DOTALL,
    ),
)


def create_memory_write_node(
    runtime: Optional["RuntimeContext"] = None,
):
    """创建已绑定 RuntimeContext 的 Memory Write Node。"""

    _ensure_runtime_open(runtime)
    manager = _resolve_memory_manager(runtime)

    def _node(state: AgentState) -> AgentState:
        return memory_write_node(
            state,
            runtime=runtime,
            memory_manager=manager,
        )

    _node.__name__ = "memory_write_node"
    _node.__qualname__ = "memory_write_node"
    _node.__doc__ = memory_write_node.__doc__
    return _node


def memory_write_node(
    state: AgentState,
    *,
    runtime: Optional["RuntimeContext"] = None,
    memory_manager: Optional[Any] = None,
) -> AgentState:
    """记录当前轮对话并写入允许的长期 Memory 候选。"""

    if bool(state.get("memory_written")):
        return {
            "memory_written": True,
            "traces": [
                {
                    "stage": "memory",
                    "message": "Memory already written; skipped duplicate write.",
                    "timestamp": utc_now(),
                    "payload": {"skipped": True},
                }
            ],
        }

    user_id = _normalize_identifier(state.get("user_id"))
    project_id = _normalize_identifier(state.get("project_id"))
    session_id = _normalize_identifier(state.get("session_id"))
    query = _normalize_text(state.get("normalized_query") or state.get("query"))
    final_answer = _normalize_text(state.get("final_answer"))

    try:
        _ensure_runtime_open(runtime)
        config = _get_memory_config(state=state, runtime=runtime)
        fail_open = _as_bool(config.get("fail_open", _DEFAULT_FAIL_OPEN))
        manager = memory_manager or _resolve_memory_manager(runtime)

        if manager is None:
            message = "MemoryManager is not registered in RuntimeContext."
            if fail_open:
                return _build_unavailable_update(message)
            return make_error(
                stage="memory",
                message=message,
                detail={"operation": "write_memory"},
            )

        if not _manager_enabled(manager):
            return _build_disabled_update(manager)

        if not session_id:
            message = "session_id is required before writing Memory."
            if fail_open:
                return _build_unavailable_update(message)
            return make_error(
                stage="memory",
                message=message,
                detail={"operation": "write_memory"},
            )

        warnings: List[str] = []
        operation_errors: List[Dict[str, Any]] = []
        turn_recorded = False
        summary_text = ""

        if _as_bool(
            config.get("track_conversation", _DEFAULT_TRACK_CONVERSATION)
        ):
            messages = _build_turn_messages(query=query, final_answer=final_answer)
            if messages:
                try:
                    _invoke_add_messages(
                        manager=manager,
                        session_id=session_id,
                        user_id=user_id,
                        project_id=project_id,
                        messages=messages,
                    )
                    turn_recorded = True
                except Exception as exc:
                    _record_operation_error(
                        operation_errors,
                        warnings,
                        operation="add_messages",
                        exc=exc,
                    )
                    _log_failure(runtime, exc, operation="add_messages")

        if _as_bool(config.get("summary_enabled", _DEFAULT_SUMMARY_ENABLED)):
            try:
                recent_count = _get_recent_message_count(
                    manager,
                    session_id,
                    user_id=user_id,
                    project_id=project_id,
                )
                minimum = _safe_positive_int(
                    config.get(
                        "summary_min_messages",
                        _DEFAULT_SUMMARY_MIN_MESSAGES,
                    ),
                    default=_DEFAULT_SUMMARY_MIN_MESSAGES,
                )
                if recent_count >= minimum:
                    summary_text = _invoke_summarize_session(
                        manager=manager,
                        session_id=session_id,
                        user_id=user_id,
                        project_id=project_id,
                        use_llm=_as_bool(
                            config.get(
                                "summary_use_llm",
                                _DEFAULT_SUMMARY_USE_LLM,
                            )
                        ),
                        clear_summarized_messages=_as_bool(
                            config.get("clear_summarized_messages", False)
                        ),
                    )
            except Exception as exc:
                _record_operation_error(
                    operation_errors,
                    warnings,
                    operation="summarize_session",
                    exc=exc,
                )
                _log_failure(runtime, exc, operation="summarize_session")

        candidates = _normalize_candidates(state.get("memory_candidates"))
        if _as_bool(
            config.get(
                "capture_explicit_memory",
                _DEFAULT_CAPTURE_EXPLICIT_MEMORY,
            )
        ):
            explicit = _extract_explicit_memory_candidate(query)
            if explicit is not None and not _candidate_exists(candidates, explicit):
                candidates.append(explicit)

        write_result: Dict[str, Any] = _empty_write_result()
        if candidates:
            try:
                raw_result = _invoke_write_candidates(
                    manager=manager,
                    candidates=candidates,
                    user_id=user_id,
                    project_id=project_id,
                    session_id=session_id,
                )
                write_result = _normalize_write_result(raw_result)
            except Exception as exc:
                _record_operation_error(
                    operation_errors,
                    warnings,
                    operation="write_candidates",
                    exc=exc,
                )
                _log_failure(runtime, exc, operation="write_candidates")

        if operation_errors and not fail_open:
            return make_error(
                stage="memory",
                message="Memory write failed.",
                detail={
                    "operation": "write_memory",
                    "errors": operation_errors,
                    "turn_recorded": turn_recorded,
                    "candidate_count": len(candidates),
                },
            )

        return {
            "memory_write_result": write_result,
            "memory_written": True,
            "warnings": warnings,
            "traces": [
                {
                    "stage": "memory",
                    "message": "Memory write stage completed.",
                    "timestamp": utc_now(),
                    "payload": {
                        "turn_recorded": turn_recorded,
                        "summary_updated": bool(summary_text),
                        "summary_length": len(summary_text),
                        "candidate_count": len(candidates),
                        "written_count": write_result.get("written_count", 0),
                        "skipped_count": len(write_result.get("skipped", [])),
                        "error_count": len(operation_errors),
                        "user_id_present": bool(user_id),
                        "project_id_present": bool(project_id),
                        "session_id_present": bool(session_id),
                    },
                }
            ],
        }

    except Exception as exc:
        _log_failure(runtime, exc, operation="write_memory")
        config = _get_memory_config_safely(state=state, runtime=runtime)
        fail_open = _as_bool(config.get("fail_open", _DEFAULT_FAIL_OPEN))
        if fail_open:
            return {
                "memory_write_result": _empty_write_result(),
                "memory_written": True,
                "warnings": [f"Memory write skipped after error: {exc}"],
                "traces": [
                    {
                        "stage": "memory",
                        "message": "Memory write failed open; final answer preserved.",
                        "timestamp": utc_now(),
                        "payload": {
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    }
                ],
            }

        return make_error(
            stage="memory",
            message=str(exc),
            detail={
                "operation": "write_memory",
                "error_type": type(exc).__name__,
                "user_id_present": bool(user_id),
                "project_id_present": bool(project_id),
                "session_id_present": bool(session_id),
            },
        )


# ---------------------------------------------------------------------------
# Runtime and configuration
# ---------------------------------------------------------------------------


def _ensure_runtime_open(runtime: Optional["RuntimeContext"]) -> None:
    if runtime is None:
        return
    ensure_open = getattr(runtime, "ensure_open", None)
    if callable(ensure_open):
        ensure_open()


def _resolve_memory_manager(
    runtime: Optional["RuntimeContext"],
) -> Optional[Any]:
    if runtime is None:
        return None

    get_method = getattr(runtime, "get", None)
    if callable(get_method):
        try:
            value = get_method("memory_manager", None)
        except TypeError:
            try:
                value = get_method("memory_manager")
            except Exception:
                value = None
        except Exception:
            value = None
        if value is not None:
            return value

    value = getattr(runtime, "memory_manager", None)
    if value is not None:
        return value

    extras = getattr(runtime, "extras", None)
    if isinstance(extras, Mapping):
        return extras.get("memory_manager")

    return None


def _get_memory_config(
    *,
    state: AgentState,
    runtime: Optional["RuntimeContext"],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    legacy = state.get("config")
    if isinstance(legacy, Mapping):
        legacy_memory = legacy.get("memory")
        if isinstance(legacy_memory, Mapping):
            result.update(copy.deepcopy(dict(legacy_memory)))

    if runtime is not None:
        settings = getattr(runtime, "settings", None)
        if settings is not None:
            value = getattr(settings, "memory", None)
            if callable(value):
                value = value()
            if isinstance(value, Mapping):
                result.update(copy.deepcopy(dict(value)))
            else:
                section = getattr(settings, "section", None)
                if callable(section):
                    try:
                        value = section("memory")
                    except Exception:
                        value = None
                    if isinstance(value, Mapping):
                        result.update(copy.deepcopy(dict(value)))

    return result


def _get_memory_config_safely(
    *,
    state: AgentState,
    runtime: Optional["RuntimeContext"],
) -> Dict[str, Any]:
    try:
        return _get_memory_config(state=state, runtime=runtime)
    except Exception:
        return {}


def _manager_enabled(manager: Any) -> bool:
    value = getattr(manager, "enabled", None)
    if callable(value):
        value = value()
    if value is None:
        policy = getattr(manager, "policy", None)
        value = getattr(policy, "enabled", True)
    return bool(value)


# ---------------------------------------------------------------------------
# MemoryManager operations
# ---------------------------------------------------------------------------


def _invoke_add_messages(
    *,
    manager: Any,
    session_id: str,
    user_id: str,
    project_id: str,
    messages: List[Dict[str, Any]],
) -> None:
    method = getattr(manager, "add_messages", None)
    if not callable(method):
        raise TypeError("MemoryManager must expose add_messages().")
    try:
        method(
            session_id=session_id,
            user_id=user_id,
            project_id=project_id,
            messages=messages,
        )
    except TypeError as exc:
        if not _looks_like_signature_error(exc):
            raise
        method(session_id=session_id, messages=messages)


def _get_recent_message_count(
    manager: Any,
    session_id: str,
    *,
    user_id: str,
    project_id: str,
) -> int:
    method = getattr(manager, "get_recent_messages", None)
    if not callable(method):
        return 0
    try:
        value = method(
            session_id,
            user_id=user_id,
            project_id=project_id,
        )
    except TypeError as exc:
        if not _looks_like_signature_error(exc):
            raise
        value = method(session_id)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return len(value)
    return 0


def _invoke_summarize_session(
    *,
    manager: Any,
    session_id: str,
    user_id: str,
    project_id: str,
    use_llm: bool,
    clear_summarized_messages: bool,
) -> str:
    method = getattr(manager, "summarize_session", None)
    if not callable(method):
        raise TypeError("MemoryManager must expose summarize_session().")
    try:
        result = method(
            session_id=session_id,
            user_id=user_id,
            project_id=project_id,
            use_llm=use_llm,
            clear_summarized_messages=clear_summarized_messages,
        )
    except TypeError as exc:
        if not _looks_like_signature_error(exc):
            raise
        result = method(
            session_id=session_id,
            use_llm=use_llm,
            clear_summarized_messages=clear_summarized_messages,
        )
    return _normalize_text(result)


def _invoke_write_candidates(
    *,
    manager: Any,
    candidates: List[Dict[str, Any]],
    user_id: str,
    project_id: str,
    session_id: str,
) -> Any:
    method = getattr(manager, "write_candidates", None)
    if not callable(method):
        raise TypeError("MemoryManager must expose write_candidates().")
    return method(
        candidates,
        user_id=user_id,
        project_id=project_id,
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# Candidate and result normalization
# ---------------------------------------------------------------------------


def _build_turn_messages(
    *,
    query: str,
    final_answer: str,
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    timestamp = utc_now()
    if query:
        result.append(
            {
                "role": "user",
                "content": query,
                "timestamp": timestamp,
            }
        )
    if final_answer:
        result.append(
            {
                "role": "assistant",
                "content": final_answer,
                "timestamp": timestamp,
            }
        )
    return result


def _normalize_candidates(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        return []

    result: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        content = _normalize_text(item.get("content"))
        if not content:
            continue
        if len(content) > _MAX_CANDIDATE_CHARS:
            content = content[:_MAX_CANDIDATE_CHARS].rstrip()

        candidate: Dict[str, Any] = {
            "content": content,
            "memory_type": _normalize_text(item.get("memory_type")) or "note",
            "source": _normalize_text(item.get("source")) or "agent",
            "confidence": _bounded_confidence(item.get("confidence", 1.0)),
            "metadata": copy.deepcopy(dict(item.get("metadata") or {}))
            if isinstance(item.get("metadata"), Mapping)
            else {},
        }
        expires_at = _normalize_text(item.get("expires_at"))
        if expires_at:
            candidate["expires_at"] = expires_at
        result.append(candidate)
    return result


def _extract_explicit_memory_candidate(query: str) -> Optional[Dict[str, Any]]:
    if not query:
        return None
    if query.lstrip().startswith(("不要记住", "别记住", "do not remember", "don't remember")):
        return None

    for pattern in _EXPLICIT_MEMORY_PATTERNS:
        match = pattern.match(query)
        if not match:
            continue
        content = _normalize_text(match.group("content"))
        if not content or len(content) < 2:
            return None
        if len(content) > _MAX_CANDIDATE_CHARS:
            content = content[:_MAX_CANDIDATE_CHARS].rstrip()
        return {
            "content": content,
            "memory_type": _infer_memory_type(content),
            "source": "explicit_user_request",
            "confidence": 1.0,
            "metadata": {"captured_by": "memory_write_node"},
        }
    return None


def _infer_memory_type(content: str) -> str:
    lowered = content.casefold()
    if any(token in content for token in ("喜欢", "偏好", "习惯")) or any(
        token in lowered for token in ("prefer", "favorite", "favourite")
    ):
        return "preference"
    if any(token in content for token in ("不要", "必须", "只能", "不能", "要求")) or any(
        token in lowered for token in ("must", "must not", "never", "always")
    ):
        return "constraint"
    if any(token in content for token in ("项目", "工程")) or "project" in lowered:
        return "project"
    if any(token in content for token in ("任务", "待办")) or any(
        token in lowered for token in ("task", "todo", "to-do")
    ):
        return "task"
    return "note"


def _candidate_exists(
    candidates: List[Dict[str, Any]],
    candidate: Dict[str, Any],
) -> bool:
    target = _canonical_text(candidate.get("content"))
    target_type = _normalize_text(candidate.get("memory_type"))
    return any(
        _canonical_text(item.get("content")) == target
        and _normalize_text(item.get("memory_type")) == target_type
        for item in candidates
    )


def _normalize_write_result(value: Any) -> Dict[str, Any]:
    if value is None:
        raw: Dict[str, Any] = {}
    elif isinstance(value, Mapping):
        raw = copy.deepcopy(dict(value))
    else:
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            converted = to_dict()
            if not isinstance(converted, Mapping):
                raise TypeError("MemoryWriteResult.to_dict() must return a mapping.")
            raw = copy.deepcopy(dict(converted))
        elif is_dataclass(value):
            converted = asdict(value)
            raw = converted if isinstance(converted, dict) else {}
        else:
            data = getattr(value, "__dict__", None)
            if not isinstance(data, Mapping):
                raise TypeError(
                    "Memory write result must be a mapping, dataclass, or expose to_dict()."
                )
            raw = copy.deepcopy(dict(data))

    created = _normalize_record_list(raw.get("created"))
    updated = _normalize_record_list(raw.get("updated"))
    skipped = _normalize_record_list(raw.get("skipped"))
    written_count = _safe_non_negative_int(
        raw.get("written_count"),
        default=len(created) + len(updated),
    )
    return {
        "created": created,
        "updated": updated,
        "skipped": skipped,
        "written_count": written_count,
    }


def _normalize_record_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        return []
    result: List[Dict[str, Any]] = []
    for item in value:
        normalized = _to_serializable_mapping(item)
        if normalized:
            result.append(normalized)
    return result


def _to_serializable_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return copy.deepcopy(dict(converted)) if isinstance(converted, Mapping) else {}
    if is_dataclass(value):
        converted = asdict(value)
        return converted if isinstance(converted, dict) else {}
    data = getattr(value, "__dict__", None)
    return copy.deepcopy(dict(data)) if isinstance(data, Mapping) else {}


# ---------------------------------------------------------------------------
# State updates and diagnostics
# ---------------------------------------------------------------------------


def _build_disabled_update(manager: Any) -> AgentState:
    return {
        "memory_write_result": _empty_write_result(),
        "memory_written": True,
        "traces": [
            {
                "stage": "memory",
                "message": "Memory is disabled by policy; skipped Memory write.",
                "timestamp": utc_now(),
                "payload": {
                    "enabled": False,
                    "manager": type(manager).__name__,
                },
            }
        ],
    }


def _build_unavailable_update(reason: str) -> AgentState:
    return {
        "memory_write_result": _empty_write_result(),
        "memory_written": True,
        "warnings": [reason],
        "traces": [
            {
                "stage": "memory",
                "message": "Memory write unavailable; final answer preserved.",
                "timestamp": utc_now(),
                "payload": {"reason": reason},
            }
        ],
    }


def _empty_write_result() -> Dict[str, Any]:
    return {
        "created": [],
        "updated": [],
        "skipped": [],
        "written_count": 0,
    }


def _record_operation_error(
    errors: List[Dict[str, Any]],
    warnings: List[str],
    *,
    operation: str,
    exc: Exception,
) -> None:
    errors.append(
        {
            "operation": operation,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    )
    warnings.append(f"Memory {operation} failed: {exc}")


def _log_failure(
    runtime: Optional["RuntimeContext"],
    exc: Exception,
    *,
    operation: str,
) -> None:
    logger = getattr(runtime, "logger", None) if runtime is not None else None
    if logger is None:
        return
    warning = getattr(logger, "warning", None)
    if callable(warning):
        warning(
            "Memory operation %s failed: %s",
            operation,
            exc,
        )


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_identifier(value: Any) -> str:
    return str(value or "").strip()


def _canonical_text(value: Any) -> str:
    return " ".join(_normalize_text(value).casefold().split())


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _safe_positive_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _safe_non_negative_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default




def _looks_like_signature_error(exc: TypeError) -> bool:
    text = str(exc).lower()
    markers = (
        "unexpected keyword argument",
        "required positional argument",
        "positional arguments but",
        "takes ",
    )
    return any(marker in text for marker in markers)

def _bounded_confidence(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 1.0


__all__ = [
    "create_memory_write_node",
    "memory_write_node",
]
