# -*- coding: utf-8 -*-
"""Direct LLM LangGraph 节点。

本节点处理不需要 KG、向量检索或外部工具增强的普通问题。它只负责：

1. 从 AgentState 读取当前问题和有限对话上下文；
2. 从 RuntimeContext 获取 PromptManager 与共享 LLMClient；
3. 应用经过白名单校验的请求级生成参数；
4. 调用统一 ``llm/`` 层并写入 ``final_answer``；
5. 在失败时返回可用的自然语言兜底结果。

本节点不得创建知识图谱、向量库、ToolRegistry 或其他领域对象，也不得把
RuntimeContext、Client 或连接对象写入 AgentState。
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple

from ..state import AgentState, get_request_option, utc_now

if TYPE_CHECKING:
    from kg_rag_agent.runtime import RuntimeContext


DEFAULT_SYSTEM_PROMPT = (
    "你是一个可靠、自然、专业的中文智能助手。"
    "请根据用户问题和有限对话上下文直接回答，表达清晰、准确、简洁。"
    "不得编造事实；信息不足时，应自然说明需要补充哪些信息。"
    "不得向用户暴露内部路由、检索、图结构、节点流程、状态字段或系统实现细节。"
)

DEFAULT_EMPTY_QUERY_ANSWER = "请先输入一个具体问题，我再继续帮你回答。"
DEFAULT_FAILURE_ANSWER = (
    "抱歉，我刚刚没有成功生成回答。你可以换一种问法，或者补充更多上下文后再试。"
)

_ALLOWED_ROLES = frozenset({"user", "assistant", "tool"})
_PROMPT_NAMES = (
    "direct_llm.system",
    "direct_llm_system_prompt",
    "direct_llm",
)


def create_direct_llm_node(
    runtime: Optional["RuntimeContext"] = None,
):
    """创建已绑定 RuntimeContext 的 Direct LLM Node。"""

    if runtime is not None:
        ensure_open = getattr(runtime, "ensure_open", None)
        if callable(ensure_open):
            ensure_open()

    def _node(state: AgentState) -> AgentState:
        return direct_llm_node(state, runtime=runtime)

    _node.__name__ = "direct_llm_node"
    _node.__qualname__ = "direct_llm_node"
    _node.__doc__ = direct_llm_node.__doc__
    return _node


def direct_llm_node(
    state: AgentState,
    *,
    runtime: Optional["RuntimeContext"] = None,
) -> AgentState:
    """直接调用共享 LLM 生成最终回答，并返回部分 AgentState 更新。"""

    query = _normalize_text(
        state.get("normalized_query") or state.get("query") or ""
    )

    if not query:
        return _build_empty_query_update(state)

    prompt_warnings: List[str] = []

    try:
        _ensure_runtime_open(runtime)

        generation_config = _get_config_section(
            state=state,
            runtime=runtime,
            name="generation",
        )
        model_config = _get_config_section(
            state=state,
            runtime=runtime,
            name="model",
        )

        language = _normalize_language(
            get_request_option(
                state,
                "language",
                generation_config.get("language", "zh"),
            )
        )

        temperature = _bounded_float(
            get_request_option(
                state,
                "temperature",
                generation_config.get(
                    "temperature",
                    model_config.get("temperature", 0.2),
                ),
            ),
            default=0.2,
            minimum=0.0,
            maximum=2.0,
        )
        max_tokens = _bounded_int(
            get_request_option(
                state,
                "max_tokens",
                generation_config.get(
                    "max_tokens",
                    model_config.get("max_tokens", 1024),
                ),
            ),
            default=1024,
            minimum=1,
            maximum=65536,
        )

        max_history_messages = _bounded_int(
            generation_config.get("max_history_messages", 20),
            default=20,
            minimum=0,
            maximum=100,
        )
        max_context_chars = _bounded_int(
            generation_config.get("max_context_chars", 12000),
            default=12000,
            minimum=256,
            maximum=200000,
        )

        system_prompt, prompt_source, prompt_warning = _resolve_system_prompt(
            state=state,
            runtime=runtime,
        )
        if prompt_warning:
            prompt_warnings.append(prompt_warning)
        system_prompt = _apply_language_instruction(system_prompt, language)

        messages = _build_messages(
            query=query,
            system_prompt=system_prompt,
            messages=state.get("messages"),
            chat_history=state.get("chat_history"),
            max_history_messages=max_history_messages,
            max_context_chars=max_context_chars,
        )

        llm = _resolve_llm_client(
            state=state,
            runtime=runtime,
            model_config=model_config,
        )
        raw_result = _invoke_llm(
            llm=llm,
            messages=messages,
            query=query,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        answer, response_metadata = _extract_answer_and_metadata(raw_result)
        answer = _normalize_answer(answer)
        if not answer:
            raise ValueError("LLM returned an empty answer.")

        return _build_success_update(
            state=state,
            answer=answer,
            prompt_source=prompt_source,
            response_metadata=response_metadata,
            temperature=temperature,
            max_tokens=max_tokens,
            language=language,
            warnings=prompt_warnings,
        )

    except Exception as exc:
        _log_failure(runtime, exc)
        return _build_failure_update(
            state=state,
            query=query,
            exc=exc,
            warnings=prompt_warnings,
        )


# =========================================================
# Runtime、配置与 Prompt
# =========================================================


def _ensure_runtime_open(runtime: Optional["RuntimeContext"]) -> None:
    if runtime is None:
        return
    ensure_open = getattr(runtime, "ensure_open", None)
    if callable(ensure_open):
        ensure_open()


def _get_config_section(
    *,
    state: AgentState,
    runtime: Optional["RuntimeContext"],
    name: str,
) -> Dict[str, Any]:
    """读取系统配置段；Runtime 配置优先，State.config 仅用于迁移兼容。"""

    merged = _legacy_config_section(state.get("config"), name)

    if runtime is None:
        return merged

    settings = getattr(runtime, "settings", None)
    if settings is None:
        return merged

    runtime_section: Dict[str, Any] = {}
    section_method = getattr(settings, "section", None)
    if callable(section_method):
        value = section_method(name)
        if isinstance(value, Mapping):
            runtime_section = copy.deepcopy(dict(value))
    elif isinstance(settings, Mapping):
        runtime_section = _legacy_config_section(settings, name)
    else:
        get_method = getattr(settings, "get", None)
        if callable(get_method):
            direct = get_method(name, {})
            nested = get_method(f"graph.{name}", {})
            if isinstance(direct, Mapping):
                runtime_section.update(copy.deepcopy(dict(direct)))
            if isinstance(nested, Mapping):
                runtime_section.update(copy.deepcopy(dict(nested)))

    merged.update(runtime_section)
    return merged


def _legacy_config_section(config: Any, name: str) -> Dict[str, Any]:
    if not isinstance(config, Mapping):
        return {}

    result: Dict[str, Any] = {}
    direct = config.get(name)
    graph = config.get("graph")

    if isinstance(direct, Mapping):
        result.update(copy.deepcopy(dict(direct)))
    if isinstance(graph, Mapping):
        nested = graph.get(name)
        if isinstance(nested, Mapping):
            result.update(copy.deepcopy(dict(nested)))

    return result


def _resolve_system_prompt(
    *,
    state: AgentState,
    runtime: Optional["RuntimeContext"],
) -> Tuple[str, str, str]:
    """返回 ``(prompt, source, warning)``。"""

    manager = getattr(runtime, "prompt_manager", None) if runtime else None
    if manager is not None:
        try:
            for name in _PROMPT_NAMES:
                prompt = _prompt_manager_get(manager, name)
                if prompt:
                    return prompt, f"prompt_manager:{name}", ""
        except Exception as exc:
            warning = (
                "PromptManager could not load the direct LLM prompt; "
                "configuration fallback was used."
            )
            _log_debug(runtime, f"Direct LLM prompt fallback: {exc}")
        else:
            warning = ""
    else:
        warning = ""

    prompt_config = _get_config_section(
        state=state,
        runtime=runtime,
        name="prompt",
    )
    legacy_config = state.get("config")
    legacy_prompts: Mapping[str, Any] = {}
    if isinstance(legacy_config, Mapping):
        candidate = legacy_config.get("prompts")
        if isinstance(candidate, Mapping):
            legacy_prompts = candidate

    configured = (
        prompt_config.get("direct_llm_system_prompt")
        or prompt_config.get("system_prompt")
        or legacy_prompts.get("direct_llm_system_prompt")
        or legacy_prompts.get("system_prompt")
    )
    if _normalize_text(configured):
        return _normalize_text(configured), "config", warning

    return DEFAULT_SYSTEM_PROMPT, "default", warning


def _prompt_manager_get(manager: Any, name: str) -> str:
    for method_name in ("get_prompt", "get"):
        method = getattr(manager, method_name, None)
        if not callable(method):
            continue
        try:
            value = method(name, default="")
        except TypeError:
            value = method(name)
        normalized = _normalize_text(value)
        if normalized:
            return normalized
    return ""


def _apply_language_instruction(system_prompt: str, language: str) -> str:
    prompt = _normalize_text(system_prompt) or DEFAULT_SYSTEM_PROMPT
    if language == "zh":
        return prompt
    if language == "en":
        return f"{prompt}\n请使用英文回答。"
    return f"{prompt}\n请使用 {language} 回答。"


# =========================================================
# LLM 调用
# =========================================================


def _resolve_llm_client(
    *,
    state: AgentState,
    runtime: Optional["RuntimeContext"],
    model_config: Mapping[str, Any],
) -> Any:
    """优先使用 Runtime 共享 Client；无 Runtime 时兼容旧独立调用。"""

    if runtime is not None:
        llm = getattr(runtime, "llm_client", None)
        if llm is not None:
            return llm

        require = getattr(runtime, "require", None)
        if callable(require):
            try:
                return require("llm_client")
            except Exception:
                pass

    from kg_rag_agent.llm.llm_client import LLMClient

    provider = model_config.get("provider")
    base_url = model_config.get("base_url")
    model = model_config.get("model_name") or model_config.get("model")
    timeout = model_config.get("timeout")
    max_retries = model_config.get("max_retries")

    kwargs: Dict[str, Any] = {
        "default_temperature": _bounded_float(
            model_config.get("temperature", 0.2),
            default=0.2,
            minimum=0.0,
            maximum=2.0,
        ),
        "default_max_tokens": _bounded_int(
            model_config.get("max_tokens", 1024),
            default=1024,
            minimum=1,
            maximum=65536,
        ),
    }
    if provider:
        kwargs["provider"] = str(provider)
    if base_url:
        kwargs["base_url"] = str(base_url)
    if model:
        kwargs["model"] = str(model)
    if timeout is not None:
        kwargs["timeout"] = _bounded_float(
            timeout,
            default=60.0,
            minimum=0.1,
            maximum=3600.0,
        )
    if max_retries is not None:
        kwargs["max_retries"] = _bounded_int(
            max_retries,
            default=2,
            minimum=0,
            maximum=20,
        )

    return LLMClient(**kwargs)


def _invoke_llm(
    *,
    llm: Any,
    messages: List[Dict[str, str]],
    query: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Any:
    """兼容统一 LLMClient、测试替身和迁移期客户端。"""

    prompt = _build_plain_prompt(
        query=query,
        system_prompt=system_prompt,
        messages=messages,
    )

    method = getattr(llm, "chat_with_metadata", None)
    if callable(method):
        try:
            return method(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except TypeError:
            pass

    method = getattr(llm, "chat", None)
    if callable(method):
        for kwargs in (
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            {
                "prompt": query,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        ):
            try:
                return method(**kwargs)
            except TypeError:
                continue
        try:
            return method(prompt)
        except TypeError:
            pass

    method = getattr(llm, "generate_with_metadata", None)
    if callable(method):
        try:
            return method(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except TypeError:
            return method(prompt)

    method = getattr(llm, "generate", None)
    if callable(method):
        try:
            return method(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except TypeError:
            return method(prompt)

    method = getattr(llm, "invoke", None)
    if callable(method):
        try:
            return method(messages)
        except TypeError:
            return method(prompt)

    if callable(llm):
        return llm(messages)

    raise AttributeError("LLM client has no supported generation method.")


# =========================================================
# 对话上下文
# =========================================================


def _build_messages(
    *,
    query: str,
    system_prompt: str,
    messages: Any,
    chat_history: Any,
    max_history_messages: int,
    max_context_chars: int,
) -> List[Dict[str, str]]:
    """构造受限、去重且无外部 system 注入的模型消息列表。"""

    merged: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()

    for source in (chat_history, messages):
        if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
            continue
        for raw in source:
            normalized = _normalize_message(raw)
            if normalized is None:
                continue
            key = (normalized["role"], normalized["content"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)

    if max_history_messages >= 0:
        merged = merged[-max_history_messages:] if max_history_messages else []

    merged = _trim_context_chars(merged, max_context_chars=max_context_chars)

    if not _latest_user_message_matches(merged, query):
        merged.append({"role": "user", "content": query})

    return [{"role": "system", "content": system_prompt}, *merged]


def _normalize_message(raw: Any) -> Optional[Dict[str, str]]:
    if isinstance(raw, Mapping):
        role = _normalize_text(raw.get("role")).lower()
        content = _normalize_text(raw.get("content"))
    else:
        role = _normalize_text(getattr(raw, "role", "")).lower()
        content = _normalize_text(getattr(raw, "content", ""))

    # 外部 system 消息不能覆盖项目的统一 System Prompt。
    if role not in _ALLOWED_ROLES or not content:
        return None

    return {"role": role, "content": content}


def _trim_context_chars(
    messages: List[Dict[str, str]],
    *,
    max_context_chars: int,
) -> List[Dict[str, str]]:
    if max_context_chars <= 0:
        return []

    selected_reversed: List[Dict[str, str]] = []
    used = 0

    for message in reversed(messages):
        content = message["content"]
        remaining = max_context_chars - used
        if remaining <= 0:
            break

        if len(content) > remaining:
            content = content[-remaining:]

        selected_reversed.append(
            {"role": message["role"], "content": content}
        )
        used += len(content)

    selected_reversed.reverse()
    return selected_reversed


def _latest_user_message_matches(
    messages: Sequence[Mapping[str, str]],
    query: str,
) -> bool:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        return _normalize_text(message.get("content")) == query
    return False


def _build_plain_prompt(
    *,
    query: str,
    system_prompt: str,
    messages: Sequence[Mapping[str, str]],
) -> str:
    history_lines: List[str] = []
    for message in messages:
        role = message.get("role", "")
        content = _normalize_text(message.get("content"))
        if not content or role == "system":
            continue
        label = {
            "user": "用户",
            "assistant": "助手",
            "tool": "工具结果",
        }.get(role, role)
        history_lines.append(f"{label}：{content}")

    history = "\n".join(history_lines)
    return (
        f"{system_prompt}\n\n"
        f"以下是有限对话上下文：\n{history}\n\n"
        f"请回答用户的最新问题：\n{query}"
    ).strip()


# =========================================================
# 响应解析与状态更新
# =========================================================


def _extract_answer_and_metadata(raw_result: Any) -> Tuple[str, Dict[str, Any]]:
    if raw_result is None:
        return "", {}

    metadata: Dict[str, Any] = {}

    if is_dataclass(raw_result):
        data = asdict(raw_result)
        answer = _extract_answer_from_mapping(data)
        metadata = _sanitize_response_metadata(data)
        return answer, metadata

    if isinstance(raw_result, str):
        return raw_result, {}

    if isinstance(raw_result, Mapping):
        data = dict(raw_result)
        return _extract_answer_from_mapping(data), _sanitize_response_metadata(data)

    to_dict = getattr(raw_result, "to_dict", None)
    if callable(to_dict):
        try:
            data = to_dict()
            if isinstance(data, Mapping):
                mapping = dict(data)
                return (
                    _extract_answer_from_mapping(mapping),
                    _sanitize_response_metadata(mapping),
                )
        except Exception:
            pass

    content = getattr(raw_result, "content", None)
    if content is not None:
        return str(content), _metadata_from_object(raw_result)

    choices = getattr(raw_result, "choices", None)
    if isinstance(choices, Sequence) and choices:
        first = choices[0]
        message = getattr(first, "message", None)
        content = getattr(message, "content", None)
        if content is not None:
            return str(content), _metadata_from_object(raw_result)
        text = getattr(first, "text", None)
        if text is not None:
            return str(text), _metadata_from_object(raw_result)

    return str(raw_result), _metadata_from_object(raw_result)


def _extract_answer_from_mapping(data: Mapping[str, Any]) -> str:
    for key in ("content", "answer", "text", "output", "response"):
        value = data.get(key)
        if value is not None and _normalize_text(value):
            return str(value)

    message = data.get("message")
    if isinstance(message, Mapping):
        value = message.get("content")
        if value is not None:
            return str(value)

    choices = data.get("choices")
    if isinstance(choices, Sequence) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            nested = first.get("message")
            if isinstance(nested, Mapping) and nested.get("content") is not None:
                return str(nested.get("content"))
            if first.get("text") is not None:
                return str(first.get("text"))

    return ""


def _sanitize_response_metadata(data: Mapping[str, Any]) -> Dict[str, Any]:
    allowed = (
        "model",
        "provider",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_seconds",
        "request_id",
    )
    result: Dict[str, Any] = {}
    for key in allowed:
        value = data.get(key)
        if value is not None and value != "":
            result[key] = copy.deepcopy(value)
    return result


def _metadata_from_object(value: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key in (
        "model",
        "provider",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_seconds",
        "request_id",
    ):
        item = getattr(value, key, None)
        if item is not None and item != "":
            result[key] = copy.deepcopy(item)
    return result


def _build_success_update(
    *,
    state: AgentState,
    answer: str,
    prompt_source: str,
    response_metadata: Mapping[str, Any],
    temperature: float,
    max_tokens: int,
    language: str,
    warnings: Sequence[str],
) -> AgentState:
    metadata = _copy_metadata(state.get("metadata"))
    direct_metadata: Dict[str, Any] = {
        "model_called": True,
        "answer_length": len(answer),
        "prompt_source": prompt_source,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "language": language,
    }
    direct_metadata.update(_sanitize_response_metadata(response_metadata))
    metadata["direct_llm"] = direct_metadata

    return AgentState(
        final_answer=answer,
        citations=[],
        need_clarification=False,
        clarifying_question="",
        has_error=False,
        error_stage="unknown",
        error_message="",
        error_detail={},
        metadata=metadata,
        traces=[
            {
                "stage": "direct_llm",
                "message": "Direct LLM answer generated.",
                "timestamp": utc_now(),
                "payload": {
                    "answer_length": len(answer),
                    "prompt_source": prompt_source,
                    "model": direct_metadata.get("model", ""),
                    "provider": direct_metadata.get("provider", ""),
                },
            }
        ],
        warnings=[str(item) for item in warnings if _normalize_text(item)],
    )


def _build_empty_query_update(state: AgentState) -> AgentState:
    metadata = _copy_metadata(state.get("metadata"))
    metadata["direct_llm"] = {
        "model_called": False,
        "reason": "empty_query",
    }

    return AgentState(
        final_answer=DEFAULT_EMPTY_QUERY_ANSWER,
        citations=[],
        need_clarification=True,
        clarifying_question=DEFAULT_EMPTY_QUERY_ANSWER,
        has_error=False,
        error_stage="unknown",
        error_message="",
        error_detail={},
        metadata=metadata,
        traces=[
            {
                "stage": "direct_llm",
                "message": "Empty query detected; clarification requested.",
                "timestamp": utc_now(),
                "payload": {},
            }
        ],
        warnings=[],
    )


def _build_failure_update(
    *,
    state: AgentState,
    query: str,
    exc: Exception,
    warnings: Sequence[str],
) -> AgentState:
    metadata = _copy_metadata(state.get("metadata"))
    metadata["direct_llm"] = {
        "model_called": False,
        "error_type": type(exc).__name__,
    }

    combined_warnings = [
        str(item) for item in warnings if _normalize_text(item)
    ]
    combined_warnings.append(
        "Direct LLM generation failed; a fallback answer was returned."
    )

    return AgentState(
        has_error=True,
        error_stage="direct_llm",
        error_message=str(exc),
        error_detail={
            "query_length": len(query),
            "error_type": type(exc).__name__,
        },
        final_answer=DEFAULT_FAILURE_ANSWER,
        citations=[],
        need_clarification=False,
        clarifying_question="",
        metadata=metadata,
        traces=[
            {
                "stage": "direct_llm",
                "message": "Direct LLM generation failed; fallback returned.",
                "timestamp": utc_now(),
                "payload": {
                    "query_length": len(query),
                    "error_type": type(exc).__name__,
                },
            }
        ],
        warnings=combined_warnings,
    )


# =========================================================
# 通用小工具
# =========================================================


def _normalize_answer(value: Any) -> str:
    answer = str(value or "").strip()
    if not answer:
        return ""

    # 仅移除模型偶尔包裹整段回答的空代码围栏；正常代码回答保持不变。
    if answer.startswith("```text") and answer.endswith("```"):
        answer = answer[len("```text") : -3].strip()
    return answer


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_language(value: Any) -> str:
    language = _normalize_text(value).lower()
    if not language:
        return "zh"
    if language in {"zh", "zh-cn", "zh_cn", "chinese", "中文", "简体中文"}:
        return "zh"
    if language in {"en", "en-us", "en_us", "english", "英文"}:
        return "en"
    return language[:32]


def _copy_metadata(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return copy.deepcopy(dict(value))


def _bounded_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(number, maximum))


def _bounded_float(
    value: Any,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(number, maximum))


def _log_failure(runtime: Optional["RuntimeContext"], exc: Exception) -> None:
    logger = getattr(runtime, "logger", None) if runtime else None
    if logger is None:
        return
    method = getattr(logger, "exception", None)
    if callable(method):
        method("Direct LLM node failed: %s", exc)


def _log_debug(runtime: Optional["RuntimeContext"], message: str) -> None:
    logger = getattr(runtime, "logger", None) if runtime else None
    if logger is None:
        return
    method = getattr(logger, "debug", None)
    if callable(method):
        method("%s", message)


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "create_direct_llm_node",
    "direct_llm_node",
]
