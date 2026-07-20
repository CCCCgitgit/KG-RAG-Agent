# -*- coding: utf-8 -*-
"""Query Router LangGraph 节点。

本节点只负责判断当前问题进入哪条处理路径，不抽取实体、不查询知识图谱，
也不生成最终回答。

正式依赖边界：
    * 系统级配置、LLMClient 和 PromptManager 由 RuntimeContext 注入；
    * AgentState 只保存当前请求及路由结果；
    * 下一节点由 ``graph/edges.py`` 根据 ``state["route"]`` 决定。

支持的路由：
    * ``kg_rag``：需要结构化事实或实体关系增强；
    * ``direct_llm``：可直接由语言模型回答；
    * ``clarify``：问题为空、指代不清或缺少必要对象；
    * ``error``：节点异常，由统一错误状态产生。
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from typing import Any, Dict, Optional, TYPE_CHECKING, Tuple

from ..state import AgentState, RouteType, make_error, utc_now

if TYPE_CHECKING:
    from kg_rag_agent.runtime import RuntimeContext


VALID_ROUTES = frozenset({"kg_rag", "direct_llm", "clarify", "error"})
LLM_ROUTER_ROUTES = frozenset({"kg_rag", "direct_llm", "clarify"})

DEFAULT_CLARIFYING_QUESTION = (
    "请补充一个更具体的问题，例如你想了解哪个对象、对象之间的什么关系，"
    "或者希望我完成什么任务。"
)

DEFAULT_ROUTER_SYSTEM_PROMPT = """
你是对话系统的内部路由判断器。
你的任务是判断用户问题应进入 kg_rag、direct_llm 或 clarify。
只允许输出符合要求的 JSON 对象，不得输出 Markdown、代码围栏或解释性文字。

路由标准：
1. kg_rag：涉及具体对象之间的事实关系、对象属性、归属、类别、路径、多跳联系或结构化事实查询。
2. direct_llm：写作、改写、翻译、代码解释、一般概念解释、日常交流以及无需额外事实检索的问题。
3. clarify：问题为空、过短、缺少明确对象、指代不清或无法可靠判断意图。

reason 必须是简短自然语言，不得出现“知识图谱”“KG”“KG-RAG”“实体链接”“检索流程”等内部技术词。
""".strip()

DEFAULT_ROUTER_USER_TEMPLATE = """
用户问题：
{query}

请只输出：
{{
  "route": "kg_rag | direct_llm | clarify",
  "reason": "一句简短中文原因"
}}
""".strip()

_DEFAULT_REASONS: Dict[str, str] = {
    "kg_rag": "这个问题涉及具体对象或事实关系，需要结合更多事实信息回答。",
    "direct_llm": "这个问题可以直接根据语言理解和已有上下文回答。",
    "clarify": "问题缺少明确对象或必要条件，需要用户补充。",
    "error": "处理问题时出现异常。",
}

_VAGUE_QUERIES = frozenset(
    {
        "这个呢",
        "这个是什么",
        "这是什么",
        "什么意思",
        "然后呢",
        "怎么办",
        "说说",
        "介绍一下",
        "它呢",
        "他呢",
        "她呢",
        "这个",
        "那个",
    }
)

_DIRECT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"写一段",
        r"帮我写",
        r"写一个",
        r"改写",
        r"润色",
        r"翻译",
        r"总结(?:这段|以下|上面|本文|文章|邮件|代码)",
        r"概括",
        r"扩写",
        r"缩写",
        r"优化这段",
        r"修改这段",
        r"解释这段代码",
        r"分析这段代码",
        r"生成代码",
        r"写代码",
        r"\bpython\b",
        r"\bjava\b",
        r"c\+\+",
        r"报错",
        r"\bbug\b",
        r"邮件",
        r"简历",
        r"cover\s*letter",
        r"论文润色",
        r"摘要怎么写",
        r"致谢怎么写",
        r"语法",
        r"计算",
    )
)

_RELATION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(.+?)(?:和|与)(.+?)(?:有什么关系|是什么关系|的关系|之间的联系)",
        r"(.+?)到(.+?)(?:的路径|是否存在路径|怎么连接)",
        r"(.+?)(?:通过什么路径|连接到|是否连接|是否认识)(.+)",
        r"(.+?)(?:有哪些关系|有哪些邻居|相关实体|相关的实体)",
        r"(.+?)(?:属于什么|属于哪一类|的上位实体|的下位实体)",
        r"(?:查询|查找|寻找).*(?:关系|路径|邻居|三元组)",
        r"\b(?:relation|path|neighbor|triple|connected|related)\b",
    )
)

_STRUCTURED_FACT_KEYWORDS = (
    "关系",
    "路径",
    "邻居",
    "三元组",
    "上位实体",
    "下位实体",
    "属于哪一类",
    "连接到",
    "是否连接",
    "相关实体",
)

_FORBIDDEN_REASON_TERMS = (
    "知识图谱",
    "kg-rag",
    "kgrag",
    "kg_rag",
    "knowledge graph",
    "图谱检索",
    "实体链接",
    "实体抽取",
    "检索流程",
    "进入流程",
    "进入检索",
    "llm router",
    "rule router",
    "traceback",
    "exception",
)


def create_query_router_node(
    runtime: Optional["RuntimeContext"] = None,
):
    """创建已绑定 RuntimeContext 的 Query Router Node。"""

    if runtime is not None:
        ensure_open = getattr(runtime, "ensure_open", None)
        if callable(ensure_open):
            ensure_open()

    def _node(state: AgentState) -> AgentState:
        return query_router_node(state, runtime=runtime)

    _node.__name__ = "query_router_node"
    _node.__qualname__ = "query_router_node"
    _node.__doc__ = query_router_node.__doc__
    return _node


def query_router_node(
    state: AgentState,
    *,
    runtime: Optional["RuntimeContext"] = None,
) -> AgentState:
    """判断当前问题应进入的处理路径，并返回部分 AgentState 更新。"""

    try:
        query = _normalize_query(state.get("query", ""))

        if not query:
            return _build_route_update(
                state=state,
                normalized_query="",
                route="clarify",
                reason="用户输入为空，需要补充问题。",
                router_type="rule",
                use_llm_requested=False,
                clarifying_question="请先输入一个具体问题。",
            )

        router_config = _get_router_config(state=state, runtime=runtime)
        use_llm = _as_bool(router_config.get("use_llm", False))
        warnings: list[str] = []
        fallback_reason = ""

        if use_llm:
            try:
                route, reason = _route_with_llm(
                    query=query,
                    runtime=runtime,
                    router_config=router_config,
                )
                router_type = "llm"
            except Exception as exc:
                _log_llm_fallback(runtime, exc)
                route, reason = _route_with_rules(query)
                router_type = "rule_fallback"
                fallback_reason = _fallback_reason_code(exc)
                warnings.append(
                    "LLM router unavailable or returned an invalid result; "
                    "rule routing was used."
                )
        else:
            route, reason = _route_with_rules(query)
            router_type = "rule"

        route, reason, normalization_warning = _normalize_route(route, reason)
        if normalization_warning:
            warnings.append(normalization_warning)

        return _build_route_update(
            state=state,
            normalized_query=query,
            route=route,
            reason=reason,
            router_type=router_type,
            use_llm_requested=use_llm,
            fallback_reason=fallback_reason,
            warnings=warnings,
        )

    except Exception as exc:
        return make_error(
            stage="query_router",
            message=str(exc),
            detail={
                "query_length": len(str(state.get("query", "") or "")),
            },
        )


def _build_route_update(
    *,
    state: AgentState,
    normalized_query: str,
    route: RouteType,
    reason: str,
    router_type: str,
    use_llm_requested: bool,
    fallback_reason: str = "",
    clarifying_question: str = "",
    warnings: Optional[list[str]] = None,
) -> AgentState:
    """构造 Query Router 的标准状态更新。"""

    safe_metadata = _copy_metadata(state.get("metadata"))
    router_metadata: Dict[str, Any] = {
        "route": route,
        "reason": reason,
        "router_type": router_type,
        "use_llm_requested": bool(use_llm_requested),
        "query_length": len(normalized_query),
    }
    if fallback_reason:
        router_metadata["fallback_reason"] = fallback_reason
    safe_metadata["query_router"] = router_metadata

    needs_clarification = route == "clarify"
    question = ""
    if needs_clarification:
        question = str(clarifying_question or DEFAULT_CLARIFYING_QUESTION).strip()

    trace_payload: Dict[str, Any] = {
        "route": route,
        "router_type": router_type,
        "use_llm_requested": bool(use_llm_requested),
    }
    if fallback_reason:
        trace_payload["fallback_reason"] = fallback_reason

    return AgentState(
        normalized_query=normalized_query,
        route=route,
        route_reason=reason,
        need_clarification=needs_clarification,
        clarifying_question=question,
        has_error=False,
        error_stage="unknown",
        error_message="",
        error_detail={},
        metadata=safe_metadata,
        traces=[
            {
                "stage": "query_router",
                "message": "Query routing completed.",
                "timestamp": utc_now(),
                "payload": trace_payload,
            }
        ],
        warnings=list(warnings or []),
    )


def _get_router_config(
    *,
    state: AgentState,
    runtime: Optional["RuntimeContext"],
) -> Dict[str, Any]:
    """读取 Router 系统配置；Runtime 配置优先于迁移期 State 配置。"""

    merged = _legacy_router_config(state.get("config"))

    if runtime is None:
        return merged

    settings = getattr(runtime, "settings", None)
    if settings is None:
        return merged

    runtime_config: Dict[str, Any] = {}
    section = getattr(settings, "section", None)
    if callable(section):
        value = section("router")
        if isinstance(value, Mapping):
            runtime_config = copy.deepcopy(dict(value))
    elif isinstance(settings, Mapping):
        runtime_config = _legacy_router_config(settings)
    else:
        get_method = getattr(settings, "get", None)
        if callable(get_method):
            direct = get_method("router", {})
            nested = get_method("graph.router", {})
            if isinstance(direct, Mapping):
                runtime_config.update(copy.deepcopy(dict(direct)))
            if isinstance(nested, Mapping):
                runtime_config.update(copy.deepcopy(dict(nested)))

    merged.update(runtime_config)
    return merged


def _legacy_router_config(config: Any) -> Dict[str, Any]:
    """兼容旧 ``config.router`` 和 ``config.graph.router``。"""

    if not isinstance(config, Mapping):
        return {}

    result: Dict[str, Any] = {}
    direct = config.get("router")
    graph = config.get("graph")

    if isinstance(direct, Mapping):
        result.update(copy.deepcopy(dict(direct)))
    if isinstance(graph, Mapping):
        nested = graph.get("router")
        if isinstance(nested, Mapping):
            result.update(copy.deepcopy(dict(nested)))

    return result


def _route_with_llm(
    *,
    query: str,
    runtime: Optional["RuntimeContext"],
    router_config: Mapping[str, Any],
) -> Tuple[RouteType, str]:
    """通过 Runtime 中的 LLMClient 执行结构化路由判断。"""

    if runtime is None:
        raise RuntimeError("RuntimeContext is required for LLM routing.")

    llm = getattr(runtime, "llm_client", None)
    if llm is None:
        raise RuntimeError("Runtime LLM client is unavailable.")

    system_prompt, user_prompt = _build_router_prompts(query, runtime)
    raw_output = _call_llm(
        llm=llm,
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=_bounded_float(
            router_config.get("temperature", 0.0),
            default=0.0,
            minimum=0.0,
            maximum=1.0,
        ),
        max_tokens=_bounded_int(
            router_config.get("max_tokens", 180),
            default=180,
            minimum=32,
            maximum=512,
        ),
        model=_optional_text(router_config.get("model")),
    )
    decision = _parse_router_decision(raw_output)

    raw_route = str(decision.get("route", "") or "").strip().lower()
    if raw_route not in LLM_ROUTER_ROUTES:
        raise ValueError("LLM router returned an unsupported route.")

    reason = _sanitize_reason(str(decision.get("reason", "") or "").strip())
    if not reason:
        reason = _default_reason(raw_route)

    return raw_route, reason  # type: ignore[return-value]


def _build_router_prompts(
    query: str,
    runtime: Optional["RuntimeContext"],
) -> Tuple[str, str]:
    """从 PromptManager 读取 Router Prompt，不可用时使用内置安全模板。"""

    system_prompt = DEFAULT_ROUTER_SYSTEM_PROMPT
    user_prompt = DEFAULT_ROUTER_USER_TEMPLATE.format(query=query)

    if runtime is None:
        return system_prompt, user_prompt

    manager = getattr(runtime, "prompt_manager", None)
    if manager is None:
        return system_prompt, user_prompt

    try:
        get_prompt = getattr(manager, "get", None)
        if not callable(get_prompt):
            get_prompt = getattr(manager, "get_prompt", None)
        if not callable(get_prompt):
            return system_prompt, user_prompt

        loaded_system = get_prompt(
            "router.system",
            default=DEFAULT_ROUTER_SYSTEM_PROMPT,
        )
        loaded_user = get_prompt(
            "router.user",
            default=DEFAULT_ROUTER_USER_TEMPLATE,
            variables={"query": query},
        )

        if str(loaded_system or "").strip():
            system_prompt = str(loaded_system).strip()
        if str(loaded_user or "").strip():
            user_prompt = str(loaded_user).strip()
    except Exception as exc:
        logger = getattr(runtime, "logger", None)
        if logger is not None:
            log = getattr(logger, "warning", None)
            if callable(log):
                log("Failed to load router prompts; built-in prompts are used: %s", exc)

    return system_prompt, user_prompt


def _call_llm(
    *,
    llm: Any,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    model: Optional[str],
) -> str:
    """兼容项目统一 LLMClient 与测试替身。"""

    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model:
        kwargs["model"] = model

    if hasattr(llm, "generate_with_metadata"):
        result = llm.generate_with_metadata(**kwargs)
        return _extract_llm_content(result)

    if hasattr(llm, "generate"):
        result = llm.generate(**kwargs)
        return _extract_llm_content(result)

    if hasattr(llm, "chat"):
        result = llm.chat(**kwargs)
        return _extract_llm_content(result)

    raise AttributeError("Runtime LLM client has no supported generation method.")


def _extract_llm_content(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, Mapping):
        for key in ("content", "answer", "text", "output"):
            value = result.get(key)
            if value is not None:
                return str(value).strip()
    content = getattr(result, "content", None)
    if content is not None:
        return str(content).strip()
    return str(result).strip()


def _parse_router_decision(text: str) -> Dict[str, Any]:
    """从模型输出中提取第一个合法 JSON 对象。"""

    normalized = str(text or "").strip()
    if not normalized:
        raise ValueError("LLM router returned an empty response.")

    normalized = re.sub(r"^```(?:json)?\s*", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s*```$", "", normalized)

    decoder = json.JSONDecoder()
    for index, character in enumerate(normalized):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(normalized[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    raise ValueError("No valid JSON object found in LLM router output.")


def _route_with_rules(query: str) -> Tuple[RouteType, str]:
    """使用确定性规则完成路由，并作为 LLM Router 的降级路径。"""

    query_lower = query.casefold()

    if len(query_lower) <= 1:
        return "clarify", "问题过短，无法判断用户想问什么。"

    compact = re.sub(r"[\s。！？!?，,；;：:]+", "", query_lower)
    if compact in _VAGUE_QUERIES:
        return "clarify", _DEFAULT_REASONS["clarify"]

    # 明确写作、翻译、代码与文本变换任务优先进入直接回答流程。
    if any(pattern.search(query_lower) for pattern in _DIRECT_PATTERNS):
        return "direct_llm", _DEFAULT_REASONS["direct_llm"]

    if any(pattern.search(query_lower) for pattern in _RELATION_PATTERNS):
        return "kg_rag", _DEFAULT_REASONS["kg_rag"]

    if any(keyword in query_lower for keyword in _STRUCTURED_FACT_KEYWORDS):
        return "kg_rag", _DEFAULT_REASONS["kg_rag"]

    if _looks_like_entity_question(query):
        return "kg_rag", _DEFAULT_REASONS["kg_rag"]

    return "direct_llm", _DEFAULT_REASONS["direct_llm"]


def _looks_like_entity_question(query: str) -> bool:
    """保守判断问题是否明显涉及具体实体事实。"""

    query_lower = query.casefold()

    factual_patterns = (
        r"^谁(?:是|和|与)",
        r".+(?:是谁|属于哪里|隶属于|创始人是谁|负责人是谁)$",
        r"(?:哪两个|哪些).*(?:有关|相连|属于)",
        r"\bwho\s+is\b",
        r"\bwhich\b.*\b(?:related|connected|belongs)\b",
    )
    if any(re.search(pattern, query_lower, re.IGNORECASE) for pattern in factual_patterns):
        return True

    # 两个或更多英文专名通常表示实体间关系问题；单个专名不直接触发，避免
    # 将一般概念解释误路由到 KG-RAG。
    capitalized = re.findall(r"\b[A-Z][A-Za-z0-9_.-]*\b", query)
    return len(capitalized) >= 2


def _normalize_route(
    route: Any,
    reason: Any,
) -> Tuple[RouteType, str, Optional[str]]:
    normalized_route = str(route or "direct_llm").strip().lower()
    normalized_reason = _sanitize_reason(str(reason or "").strip())

    if normalized_route not in VALID_ROUTES:
        return (
            "direct_llm",
            normalized_reason or _DEFAULT_REASONS["direct_llm"],
            "Invalid router result was replaced with direct_llm.",
        )

    return (
        normalized_route,  # type: ignore[return-value]
        normalized_reason or _default_reason(normalized_route),
        None,
    )


def _default_reason(route: str) -> str:
    return _DEFAULT_REASONS.get(route, _DEFAULT_REASONS["direct_llm"])


def _sanitize_reason(reason: str) -> str:
    normalized = re.sub(r"\s+", " ", str(reason or "")).strip()
    if not normalized:
        return ""

    lowered = normalized.casefold()
    if any(term in lowered for term in _FORBIDDEN_REASON_TERMS):
        return ""

    # 防止异常栈、URL、文件路径等实现细节进入后续回答。
    if "http://" in lowered or "https://" in lowered or "\\" in normalized:
        return ""

    return normalized[:240]


def _normalize_query(query: Any) -> str:
    normalized = str(query or "").strip()
    return re.sub(r"\s+", " ", normalized)


def _copy_metadata(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("state.metadata must be a mapping.")
    return copy.deepcopy(dict(value))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value or "").strip().casefold()
    if normalized in {"true", "1", "yes", "on", "enabled"}:
        return True
    if normalized in {"false", "0", "no", "off", "disabled", ""}:
        return False
    raise ValueError("router.use_llm must be a boolean value.")


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
        return default
    return min(max(number, minimum), maximum)


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
        return default
    return min(max(number, minimum), maximum)


def _optional_text(value: Any) -> Optional[str]:
    normalized = str(value or "").strip()
    return normalized or None


def _log_llm_fallback(
    runtime: Optional["RuntimeContext"],
    exc: Exception,
) -> None:
    if runtime is None:
        return
    logger = getattr(runtime, "logger", None)
    log = getattr(logger, "warning", None)
    if callable(log):
        log("LLM query routing failed; using rule fallback: %s", exc)


def _fallback_reason_code(exc: Exception) -> str:
    if isinstance(exc, RuntimeError):
        return "runtime_dependency_unavailable"
    if isinstance(exc, (ValueError, json.JSONDecodeError)):
        return "invalid_llm_output"
    return "llm_call_failed"


__all__ = [
    "VALID_ROUTES",
    "query_router_node",
    "create_query_router_node",
]
