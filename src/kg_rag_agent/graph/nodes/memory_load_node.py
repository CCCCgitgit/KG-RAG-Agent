# -*- coding: utf-8 -*-
"""Memory Load LangGraph 节点。

本节点在主流程开始前读取受控 Memory 上下文，并将可序列化结果写入
``AgentState``。它只负责读取，不写入长期 Memory，也不创建 Store、Retriever、
LLM Client 或其他共享对象。

职责边界：
    * MemoryManager 必须由 RuntimeContext 创建并复用；
    * Memory Store、Manager、连接对象不得写入 AgentState；
    * Memory 失败默认采用 fail-open，不阻断 KG-RAG 主流程；
    * 用户、项目和会话隔离由 MemoryPolicy 与 MemoryManager 统一执行；
    * 外部请求不能覆盖 Memory Namespace 或系统级权限。
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..state import AgentState, make_error, utc_now

if TYPE_CHECKING:
    from kg_rag_agent.runtime import RuntimeContext


_DEFAULT_FAIL_OPEN = True
_MAX_MEMORY_TEXT_CHARS = 200_000


def create_memory_load_node(
    runtime: Optional["RuntimeContext"] = None,
):
    """创建已绑定 RuntimeContext 的 Memory Load Node。"""

    _ensure_runtime_open(runtime)
    manager = _resolve_memory_manager(runtime)

    def _node(state: AgentState) -> AgentState:
        return memory_load_node(
            state,
            runtime=runtime,
            memory_manager=manager,
        )

    _node.__name__ = "memory_load_node"
    _node.__qualname__ = "memory_load_node"
    _node.__doc__ = memory_load_node.__doc__
    return _node


def memory_load_node(
    state: AgentState,
    *,
    runtime: Optional["RuntimeContext"] = None,
    memory_manager: Optional[Any] = None,
) -> AgentState:
    """加载当前请求可访问的 Memory 上下文，并返回部分 State 更新。"""

    if bool(state.get("memory_loaded")):
        return {
            "memory_loaded": True,
            "traces": [
                {
                    "stage": "memory",
                    "message": "Memory context already loaded; skipped duplicate load.",
                    "timestamp": utc_now(),
                    "payload": {"skipped": True},
                }
            ],
        }

    query = _normalize_text(
        state.get("normalized_query") or state.get("query") or ""
    )
    user_id = _normalize_identifier(state.get("user_id"))
    project_id = _normalize_identifier(state.get("project_id"))
    session_id = _normalize_identifier(state.get("session_id"))

    try:
        _ensure_runtime_open(runtime)
        config = _get_memory_config(state=state, runtime=runtime)
        fail_open = _as_bool(config.get("fail_open", _DEFAULT_FAIL_OPEN))

        manager = memory_manager or _resolve_memory_manager(runtime)
        if manager is None:
            return _build_unavailable_update(
                reason="MemoryManager is not registered in RuntimeContext.",
                warning=True,
            )

        if not _manager_enabled(manager):
            return _build_disabled_update(manager)

        if not session_id:
            # build_initial_state 正常会生成 session_id；这里仍做防御性保护。
            message = "session_id is required before loading Memory context."
            if fail_open:
                return _build_unavailable_update(reason=message, warning=True)
            return make_error(
                stage="memory",
                message=message,
                detail={"operation": "load_context"},
            )

        memory_types = _normalize_memory_types(config.get("load_memory_types"))
        context = _invoke_load_context(
            manager=manager,
            query=query,
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
            memory_types=memory_types,
        )
        context_data = _normalize_context(context)

        return _build_success_update(
            context=context_data,
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
        )

    except Exception as exc:
        _log_failure(runtime, exc)
        config = _get_memory_config_safely(state=state, runtime=runtime)
        fail_open = _as_bool(config.get("fail_open", _DEFAULT_FAIL_OPEN))

        if fail_open:
            return _build_failure_open_update(exc)

        return make_error(
            stage="memory",
            message=str(exc),
            detail={
                "operation": "load_context",
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


# ---------------------------------------------------------------------------
# MemoryManager invocation and serialization
# ---------------------------------------------------------------------------


def _manager_enabled(manager: Any) -> bool:
    value = getattr(manager, "enabled", None)
    if callable(value):
        value = value()
    if value is None:
        policy = getattr(manager, "policy", None)
        value = getattr(policy, "enabled", True)
    return bool(value)


def _invoke_load_context(
    *,
    manager: Any,
    query: str,
    user_id: str,
    project_id: str,
    session_id: str,
    memory_types: tuple[Any, ...],
) -> Any:
    method = getattr(manager, "load_context", None)
    if not callable(method):
        raise TypeError("MemoryManager must expose load_context().")

    kwargs: Dict[str, Any] = {
        "query": query,
        "user_id": user_id,
        "project_id": project_id,
        "session_id": session_id,
    }
    if memory_types:
        kwargs["memory_types"] = memory_types

    try:
        return method(**kwargs)
    except TypeError as exc:
        # 兼容简化 Fake/Mock Manager，但不掩盖真实方法内部的 TypeError。
        if not _looks_like_signature_error(exc):
            raise
        return method(
            query=query,
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
        )


def _normalize_context(value: Any) -> Dict[str, Any]:
    if value is None:
        raw: Dict[str, Any] = {}
    elif isinstance(value, Mapping):
        raw = copy.deepcopy(dict(value))
    else:
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            converted = to_dict()
            if not isinstance(converted, Mapping):
                raise TypeError("MemoryContext.to_dict() must return a mapping.")
            raw = copy.deepcopy(dict(converted))
        elif is_dataclass(value):
            converted = asdict(value)
            raw = converted if isinstance(converted, dict) else {}
        else:
            data = getattr(value, "__dict__", None)
            if not isinstance(data, Mapping):
                raise TypeError(
                    "Memory context must be a mapping, dataclass, or expose to_dict()."
                )
            raw = copy.deepcopy(dict(data))

    recent_messages = _normalize_messages(raw.get("recent_messages"))
    summary = _normalize_text(raw.get("summary"))
    memories = _normalize_memories(raw.get("memories"))
    text = _normalize_text(raw.get("text"))
    if len(text) > _MAX_MEMORY_TEXT_CHARS:
        text = text[:_MAX_MEMORY_TEXT_CHARS].rstrip()

    estimated_tokens = _safe_non_negative_int(
        raw.get("estimated_tokens"),
        default=max(0, len(text) // 4),
    )

    return {
        "recent_messages": recent_messages,
        "summary": summary,
        "memories": memories,
        "text": text,
        "estimated_tokens": estimated_tokens,
    }


def _normalize_messages(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []

    result: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        content = _normalize_text(item.get("content"))
        if not content:
            continue
        normalized: Dict[str, Any] = {
            "role": _normalize_text(item.get("role")) or "unknown",
            "content": content,
        }
        if item.get("timestamp") is not None:
            normalized["timestamp"] = _normalize_text(item.get("timestamp"))
        result.append(normalized)
    return result


def _normalize_memories(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
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


def _normalize_memory_types(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("memory.load_memory_types must be a sequence.")

    result: List[str] = []
    for item in value:
        normalized = _normalize_text(getattr(item, "value", item))
        if normalized and normalized not in result:
            result.append(normalized)
    return tuple(result)


# ---------------------------------------------------------------------------
# State updates
# ---------------------------------------------------------------------------


def _build_success_update(
    *,
    context: Dict[str, Any],
    user_id: str,
    project_id: str,
    session_id: str,
) -> AgentState:
    memories = context.get("memories") or []
    recent = context.get("recent_messages") or []
    text = _normalize_text(context.get("text"))

    return {
        "memory_context": context,
        "memory_text": text,
        "memory_loaded": True,
        "traces": [
            {
                "stage": "memory",
                "message": "Memory context loaded.",
                "timestamp": utc_now(),
                "payload": {
                    "user_id_present": bool(user_id),
                    "project_id_present": bool(project_id),
                    "session_id": session_id,
                    "memory_count": len(memories),
                    "recent_message_count": len(recent),
                    "has_summary": bool(context.get("summary")),
                    "estimated_tokens": context.get("estimated_tokens", 0),
                },
            }
        ],
    }


def _build_disabled_update(manager: Any) -> AgentState:
    return {
        "memory_context": _empty_context(),
        "memory_text": "",
        "memory_loaded": True,
        "traces": [
            {
                "stage": "memory",
                "message": "Memory is disabled by policy; continued without Memory context.",
                "timestamp": utc_now(),
                "payload": {
                    "enabled": False,
                    "manager": type(manager).__name__,
                },
            }
        ],
    }


def _build_unavailable_update(*, reason: str, warning: bool) -> AgentState:
    update: AgentState = {
        "memory_context": _empty_context(),
        "memory_text": "",
        "memory_loaded": False,
        "traces": [
            {
                "stage": "memory",
                "message": reason,
                "timestamp": utc_now(),
                "payload": {"available": False},
            }
        ],
    }
    if warning:
        update["warnings"] = [reason]
    return update


def _build_failure_open_update(exc: Exception) -> AgentState:
    message = (
        "Memory context could not be loaded; continued without Memory enhancement. "
        f"({type(exc).__name__}: {exc})"
    )
    return {
        "memory_context": _empty_context(),
        "memory_text": "",
        "memory_loaded": False,
        "warnings": [message],
        "traces": [
            {
                "stage": "memory",
                "message": "Memory load failed in fail-open mode.",
                "timestamp": utc_now(),
                "payload": {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "fail_open": True,
                },
            }
        ],
    }


def _empty_context() -> Dict[str, Any]:
    return {
        "recent_messages": [],
        "summary": "",
        "memories": [],
        "text": "",
        "estimated_tokens": 0,
    }


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_identifier(value: Any) -> str:
    return str(value or "").strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _safe_non_negative_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(number, 0)


def _looks_like_signature_error(exc: TypeError) -> bool:
    text = str(exc).lower()
    markers = (
        "unexpected keyword argument",
        "required positional argument",
        "positional arguments but",
        "takes ",
    )
    return any(marker in text for marker in markers)


def _log_failure(runtime: Optional["RuntimeContext"], exc: Exception) -> None:
    logger = getattr(runtime, "logger", None) if runtime is not None else None
    if logger is None:
        return
    method = getattr(logger, "warning", None)
    if callable(method):
        method("Memory load failed: %s", exc)


__all__ = [
    "create_memory_load_node",
    "memory_load_node",
]
