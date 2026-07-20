# -*- coding: utf-8 -*-
"""Agent 业务服务。

正式调用方向：
    API / CLI / Evaluation -> AgentService -> KGRAGAgent -> CompiledGraph

本模块只负责外部参数校验、请求级选项白名单、Agent 调用和服务级状态；
不再加载节点算法，也不直接构建 AgentState；用户、项目和会话标识仅透传给 Agent 供 Memory 隔离。
"""

from __future__ import annotations

import logging
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from kg_rag_agent.agents import KGRAGAgent
from kg_rag_agent.agents.schemas import AgentResult, RequestOptions


_ALLOWED_REQUEST_OPTIONS = {
    "retrieval_top_k",
    "path_max_depth",
    "temperature",
    "max_tokens",
    "language",
    "include_citations",
    "allowed_tools",
}


class AgentService:
    """KG-RAG Agent 的统一业务服务入口。"""

    def __init__(
        self,
        *,
        agent: Optional[KGRAGAgent] = None,
        runtime: Optional[Any] = None,
        graph: Optional[Any] = None,
        graph_factory: Optional[Any] = None,
        config: Optional[Mapping[str, Any]] = None,
        config_path: Optional[str] = None,
        auto_build_graph: bool = True,
        validate: bool = True,
    ) -> None:
        if agent is not None and any(
            value is not None
            for value in (
                runtime,
                graph,
                graph_factory,
                config,
                config_path,
            )
        ):
            raise ValueError(
                "agent cannot be combined with runtime, graph, graph_factory, "
                "config, or config_path."
            )

        self._owns_agent = agent is None
        self.agent = agent or KGRAGAgent(
            runtime=runtime,
            graph=graph,
            graph_factory=graph_factory,
            config=config,
            config_path=config_path,
            auto_build_graph=auto_build_graph,
            validate=validate,
        )

        self.logger = self._resolve_logger()

    @property
    def runtime(self) -> Any:
        """返回 Agent 使用的 RuntimeContext。"""

        return self.agent.runtime

    @property
    def config(self) -> Dict[str, Any]:
        """返回只用于读取的配置副本，避免外部修改共享 Settings。"""

        settings = getattr(self.runtime, "settings", None)
        to_dict = getattr(settings, "to_dict", None)
        return dict(to_dict()) if callable(to_dict) else {}

    @property
    def graph(self) -> Any:
        """兼容旧代码：返回当前已注入或已构建的 Graph，不触发构建。"""

        return getattr(self.agent, "_graph", None)

    @graph.setter
    def graph(self, value: Any) -> None:
        """兼容测试和迁移阶段的 FakeGraph 注入。"""

        setattr(self.agent, "_graph", value)

    def _resolve_logger(self) -> logging.Logger:
        try:
            logger = getattr(self.runtime, "logger", None)
        except Exception:
            logger = None
        return logger or logging.getLogger("kg_rag_agent.services.agent")

    def get_graph(self) -> Any:
        """获取 Agent 当前的 CompiledGraph。"""

        return self.agent.get_graph()

    def rebuild_graph(self) -> Any:
        """重新构建 Agent 的 CompiledGraph。"""

        return self.agent.rebuild_graph()

    def _build_graph(self) -> Any:
        """旧内部接口兼容；新代码应调用 :meth:`rebuild_graph`。"""

        return self.rebuild_graph()

    def ask(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request_options: Optional[Mapping[str, Any]] = None,
        config_overrides: Optional[Mapping[str, Any]] = None,
        include_raw_state: bool = False,
    ) -> AgentResult:
        """执行一次问答。

        ``config_overrides`` 仅作为迁移期兼容入口。服务只提取文档允许的
        请求级字段，未知字段会被忽略，不能覆盖路径、凭据和系统权限。
        """

        normalized_query = str(query or "").strip()
        if not normalized_query:
            raise ValueError("query must not be empty.")

        normalized_options = self._normalize_request_options(
            request_options=request_options,
            config_overrides=config_overrides,
        )
        legacy_overrides = self._legacy_graph_overrides(normalized_options)

        return self.agent.ask(
            query=normalized_query,
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
            request_id=request_id,
            messages=messages,
            chat_history=chat_history,
            metadata=metadata,
            request_options=normalized_options,
            config_overrides=legacy_overrides or None,
            include_raw_state=include_raw_state,
        )

    def invoke(
        self,
        query_or_state: Union[str, Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """调用问题字符串，或直接执行一个完整 State。"""

        if isinstance(query_or_state, str):
            include_raw_state = bool(kwargs.pop("include_raw_state", False))
            include_identifiers = bool(kwargs.pop("include_identifiers", False))
            include_memory_status = bool(
                kwargs.pop("include_memory_status", False)
            )
            result = self.ask(
                query_or_state,
                include_raw_state=include_raw_state,
                **kwargs,
            )
            return result.to_dict(
                include_raw_state=include_raw_state,
                include_identifiers=include_identifiers,
                include_memory_status=include_memory_status,
            )

        if isinstance(query_or_state, dict):
            return self.invoke_state(
                query_or_state,
                runnable_config=kwargs.pop("runnable_config", None),
                **kwargs,
            )

        raise TypeError("query_or_state must be str or dict.")

    def invoke_state(
        self,
        state: Dict[str, Any],
        *,
        runnable_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """直接执行完整 State，主要用于测试和高级调用。"""

        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported invoke_state arguments: {unknown}")

        if runnable_config is None:
            return dict(self.agent.invoke(state))
        return dict(
            self.agent.invoke(
                state,
                runnable_config=runnable_config,
            )
        )

    def batch_ask(
        self,
        queries: Iterable[str],
        *,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_options: Optional[Mapping[str, Any]] = None,
        config_overrides: Optional[Mapping[str, Any]] = None,
        include_raw_state: bool = False,
    ) -> List[AgentResult]:
        """串行执行多个问题。"""

        return [
            self.ask(
                query,
                user_id=user_id,
                project_id=project_id,
                session_id=session_id,
                request_options=request_options,
                config_overrides=config_overrides,
                include_raw_state=include_raw_state,
            )
            for query in queries
        ]

    def stream(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request_options: Optional[Mapping[str, Any]] = None,
        config_overrides: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """返回 LangGraph 原始事件流。"""

        normalized_options = self._normalize_request_options(
            request_options=request_options,
            config_overrides=config_overrides,
        )
        legacy_overrides = self._legacy_graph_overrides(normalized_options)

        return self.agent.stream(
            query=query,
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
            request_id=request_id,
            messages=messages,
            chat_history=chat_history,
            metadata=metadata,
            request_options=normalized_options,
            config_overrides=legacy_overrides or None,
        )

    def _build_result(
        self,
        *,
        final_state: Mapping[str, Any],
        request_id: str,
        include_raw_state: bool,
    ) -> AgentResult:
        """旧内部接口兼容：从最终 State 构造 AgentResult。"""

        return AgentResult.from_state(
            final_state,
            request_id=request_id,
            include_raw_state=include_raw_state,
        )

    def _normalize_request_options(
        self,
        *,
        request_options: Optional[Mapping[str, Any]],
        config_overrides: Optional[Mapping[str, Any]],
    ) -> RequestOptions:
        raw: Dict[str, Any] = {}

        if config_overrides:
            extracted, ignored = _extract_legacy_request_options(
                config_overrides
            )
            raw.update(extracted)
            if ignored:
                self.logger.warning(
                    "Ignored non-whitelisted request overrides: %s",
                    sorted(ignored),
                )

        if request_options:
            unknown = set(request_options) - _ALLOWED_REQUEST_OPTIONS
            if unknown:
                raise ValueError(
                    "Unsupported request options: "
                    + ", ".join(sorted(unknown))
                )
            raw.update(dict(request_options))

        return _validate_request_options(raw)

    @staticmethod
    def _legacy_graph_overrides(
        options: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """把请求选项映射到旧 Graph 当前仍读取的安全配置字段。"""

        overrides: Dict[str, Any] = {}

        if "retrieval_top_k" in options:
            value = int(options["retrieval_top_k"])
            overrides["retrieval"] = {
                "top_k": value,
                "entity_top_k": value,
                "doc_top_k": value,
            }
            overrides.setdefault("graph", {})["entity_linking"] = {
                "top_k": value
            }

        if "path_max_depth" in options:
            overrides["kg_retrieval"] = {
                "max_path_length": int(options["path_max_depth"])
            }

        generation: Dict[str, Any] = {}
        if "temperature" in options:
            generation["temperature"] = float(options["temperature"])
        if "max_tokens" in options:
            generation["max_tokens"] = int(options["max_tokens"])

        if generation:
            overrides["generation"] = deepcopy(generation)
            overrides.setdefault("graph", {})["generation"] = deepcopy(
                generation
            )
            overrides["model"] = deepcopy(generation)

        return overrides

    def health_check(self) -> Dict[str, Any]:
        """服务健康检查。"""

        health = dict(self.agent.health_check())
        return {
            "ok": bool(health.get("ok", False)),
            "service": "AgentService",
            "agent": health,
            "time": _utc_now(),
        }

    def info(self) -> Dict[str, Any]:
        """返回不包含凭据的服务摘要。"""

        agent_info = dict(self.agent.info())
        return {
            "service": "AgentService",
            "agent": agent_info,
            "request_option_whitelist": sorted(_ALLOWED_REQUEST_OPTIONS),
            "time": _utc_now(),
        }

    def close(self) -> None:
        """释放由本服务创建的 Agent 和 Runtime。"""

        if self._owns_agent:
            self.agent.close()

    def __enter__(self) -> "AgentService":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @staticmethod
    def _make_request_id() -> str:
        return "req_" + uuid.uuid4().hex[:16]

    @staticmethod
    def _make_session_id() -> str:
        return "sess_" + uuid.uuid4().hex[:16]

    @staticmethod
    def _truncate(text: Any, max_length: int = 160) -> str:
        value = str(text or "").replace("\n", " ").strip()
        return value if len(value) <= max_length else value[:max_length] + "..."

    @staticmethod
    def _safe_float(value: Any, *, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


_GLOBAL_AGENT_SERVICE: Optional[AgentService] = None


def get_default_agent_service(
    *,
    refresh: bool = False,
    **kwargs: Any,
) -> AgentService:
    """获取进程内默认 AgentService。"""

    global _GLOBAL_AGENT_SERVICE

    if refresh and _GLOBAL_AGENT_SERVICE is not None:
        _GLOBAL_AGENT_SERVICE.close()
        _GLOBAL_AGENT_SERVICE = None

    if _GLOBAL_AGENT_SERVICE is None:
        _GLOBAL_AGENT_SERVICE = AgentService(**kwargs)
    elif kwargs:
        raise ValueError(
            "Default AgentService already exists; use refresh=True to rebuild it."
        )

    return _GLOBAL_AGENT_SERVICE


def ask(query: str, **kwargs: Any) -> AgentResult:
    """函数式问答入口。"""

    return get_default_agent_service().ask(query, **kwargs)


def invoke(
    query_or_state: Union[str, Dict[str, Any]],
    **kwargs: Any,
) -> Dict[str, Any]:
    """函数式 invoke 入口。"""

    return get_default_agent_service().invoke(query_or_state, **kwargs)


def _extract_legacy_request_options(
    overrides: Mapping[str, Any],
) -> tuple[Dict[str, Any], set[str]]:
    """从旧嵌套配置中仅提取允许的请求级字段。"""

    data = dict(overrides or {})
    result: Dict[str, Any] = {}
    consumed: set[str] = set()

    for key in _ALLOWED_REQUEST_OPTIONS:
        if key in data:
            result[key] = data[key]
            consumed.add(key)

    retrieval = data.get("retrieval")
    if isinstance(retrieval, Mapping) and "top_k" in retrieval:
        result["retrieval_top_k"] = retrieval["top_k"]
        consumed.add("retrieval.top_k")

    graph = data.get("graph")
    if isinstance(graph, Mapping):
        entity_linking = graph.get("entity_linking")
        if isinstance(entity_linking, Mapping) and "top_k" in entity_linking:
            result.setdefault("retrieval_top_k", entity_linking["top_k"])
            consumed.add("graph.entity_linking.top_k")

        generation = graph.get("generation")
        if isinstance(generation, Mapping):
            for key in ("temperature", "max_tokens"):
                if key in generation:
                    result[key] = generation[key]
                    consumed.add(f"graph.generation.{key}")

        kg_retrieval = graph.get("kg_retrieval")
        if isinstance(kg_retrieval, Mapping):
            for key in ("max_path_length", "path_max_depth"):
                if key in kg_retrieval:
                    result["path_max_depth"] = kg_retrieval[key]
                    consumed.add(f"graph.kg_retrieval.{key}")

    generation = data.get("generation")
    if isinstance(generation, Mapping):
        for key in ("temperature", "max_tokens"):
            if key in generation:
                result[key] = generation[key]
                consumed.add(f"generation.{key}")

    model = data.get("model")
    if isinstance(model, Mapping):
        for key in ("temperature", "max_tokens"):
            if key in model:
                result.setdefault(key, model[key])
                consumed.add(f"model.{key}")

    kg_retrieval = data.get("kg_retrieval")
    if isinstance(kg_retrieval, Mapping):
        for key in ("max_path_length", "path_max_depth"):
            if key in kg_retrieval:
                result["path_max_depth"] = kg_retrieval[key]
                consumed.add(f"kg_retrieval.{key}")

    ignored = _leaf_paths(data) - consumed
    return result, ignored


def _leaf_paths(
    value: Mapping[str, Any],
    prefix: str = "",
) -> set[str]:
    paths: set[str] = set()
    for key, item in value.items():
        current = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping) and item:
            paths.update(_leaf_paths(item, current))
        else:
            paths.add(current)
    return paths


def _validate_request_options(raw: Mapping[str, Any]) -> RequestOptions:
    result: RequestOptions = {}

    if "retrieval_top_k" in raw:
        value = _bounded_int(raw["retrieval_top_k"], "retrieval_top_k", 1, 100)
        result["retrieval_top_k"] = value

    if "path_max_depth" in raw:
        value = _bounded_int(raw["path_max_depth"], "path_max_depth", 1, 6)
        result["path_max_depth"] = value

    if "temperature" in raw:
        value = _bounded_float(raw["temperature"], "temperature", 0.0, 2.0)
        result["temperature"] = value

    if "max_tokens" in raw:
        value = _bounded_int(raw["max_tokens"], "max_tokens", 1, 8192)
        result["max_tokens"] = value

    if "language" in raw:
        language = str(raw["language"] or "").strip()
        if not language or len(language) > 32:
            raise ValueError("language must contain 1 to 32 characters.")
        result["language"] = language

    if "include_citations" in raw:
        value = raw["include_citations"]
        if not isinstance(value, bool):
            raise TypeError("include_citations must be bool.")
        result["include_citations"] = value

    if "allowed_tools" in raw:
        value = raw["allowed_tools"]
        if not isinstance(value, (list, tuple)):
            raise TypeError("allowed_tools must be a list of strings.")
        tools: List[str] = []
        for item in value:
            name = str(item or "").strip()
            if not name:
                raise ValueError("allowed_tools cannot contain empty names.")
            if name not in tools:
                tools.append(name)
        if len(tools) > 64:
            raise ValueError("allowed_tools cannot contain more than 64 names.")
        result["allowed_tools"] = tools

    return result


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be int.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be int.") from exc
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return number


def _bounded_float(
    value: Any,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be float.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be float.") from exc
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return number


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "AgentResult",
    "RequestOptions",
    "AgentService",
    "get_default_agent_service",
    "ask",
    "invoke",
]
