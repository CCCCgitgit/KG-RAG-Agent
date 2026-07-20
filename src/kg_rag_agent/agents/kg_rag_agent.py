# -*- coding: utf-8 -*-
"""KG-RAG Agent 门面实现。

本模块只负责：

* 标准化 Agent 请求；
* 构造可序列化的初始 AgentState；
* 调用已编译的 LangGraph；
* 将最终状态转换为稳定的 AgentResult。

模型、图谱、向量库、MemoryManager、ToolRegistry 和 MCP 连接等共享对象均由
RuntimeContext 管理，不写入 AgentState。
"""

from __future__ import annotations

import copy
import inspect
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union

from .base_agent import BaseAgent
from .schemas import AgentRequest, AgentResult, MemoryStatus, RequestOptions

GraphFactory = Callable[..., Any]


class KGRAGAgent(BaseAgent):
    """KG-RAG Agent 的统一门面。

    正式依赖方向：

    ``AgentService -> KGRAGAgent -> CompiledGraph``

    ``service=`` 仅用于旧代码迁移。兼容模式不会在本模块中导入
    :class:`AgentService`，也不会成为正式调用路径。
    """

    def __init__(
        self,
        *,
        runtime: Optional[Any] = None,
        graph: Optional[Any] = None,
        graph_factory: Optional[GraphFactory] = None,
        service: Optional[Any] = None,
        config: Optional[Mapping[str, Any]] = None,
        config_path: Optional[str] = None,
        auto_build_graph: bool = True,
        validate: bool = True,
    ) -> None:
        if service is not None and any(
            value is not None
            for value in (runtime, graph, graph_factory, config, config_path)
        ):
            raise ValueError(
                "service compatibility mode cannot be combined with runtime, "
                "graph, graph_factory, config, or config_path."
            )

        self._legacy_service = service
        self._runtime = runtime
        self._graph = graph
        self._graph_factory = graph_factory
        self._owns_runtime = False
        self._validate = bool(validate)

        if self._legacy_service is None and self._runtime is None:
            self._runtime = self._create_runtime(
                config=config,
                config_path=config_path,
                validate=validate,
            )
            self._owns_runtime = True

        if (
            self._legacy_service is None
            and auto_build_graph
            and self._graph is None
        ):
            self._graph = self._build_graph()

    @property
    def runtime(self) -> Any:
        """返回共享 RuntimeContext。"""

        if self._runtime is None:
            raise RuntimeError("RuntimeContext is unavailable in service mode.")
        ensure_open = getattr(self._runtime, "ensure_open", None)
        if callable(ensure_open):
            ensure_open()
        return self._runtime

    @property
    def service(self) -> Any:
        """旧版兼容属性；新代码不应依赖该属性。"""

        if self._legacy_service is None:
            raise AttributeError(
                "KGRAGAgent no longer owns AgentService in direct graph mode."
            )
        return self._legacy_service

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
        request_options: Optional[RequestOptions] = None,
        config_overrides: Optional[Mapping[str, Any]] = None,
        include_raw_state: bool = False,
        **kwargs: Any,
    ) -> AgentResult:
        """执行一次问答并返回标准 :class:`AgentResult`。

        ``user_id``、``project_id`` 和 ``session_id`` 会进入 AgentState，供
        MemoryManager 做隔离；它们不会被拼入 Prompt 或系统配置。
        """

        if self._legacy_service is not None:
            legacy_result = _call_with_supported_kwargs(
                self._legacy_service.ask,
                {
                    "query": query,
                    "user_id": user_id,
                    "project_id": project_id,
                    "session_id": session_id,
                    "request_id": request_id,
                    "messages": messages,
                    "chat_history": chat_history,
                    "metadata": metadata,
                    "request_options": request_options,
                    "config_overrides": config_overrides,
                    "include_raw_state": include_raw_state,
                    **kwargs,
                },
            )
            return _coerce_agent_result(
                legacy_result,
                include_raw_state=include_raw_state,
            )

        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported ask keyword arguments: {unknown}")

        request = AgentRequest(
            query=query,
            request_id=request_id or _make_id("req"),
            session_id=session_id or _make_id("sess"),
            user_id=user_id or "",
            project_id=project_id or "",
            messages=messages or [],
            chat_history=chat_history or [],
            metadata=metadata or {},
            request_options=request_options or {},
        )

        logger = getattr(self.runtime, "logger", None)
        if logger is not None:
            logger.info(
                "Agent request started | request_id=%s | session_id=%s | "
                "user_id=%s | project_id=%s | query=%s",
                request.request_id,
                request.session_id,
                request.user_id,
                request.project_id,
                _truncate(request.query),
            )

        try:
            initial_state = self._build_initial_state(
                request,
                config_overrides=config_overrides,
            )
            final_state = self.get_graph().invoke(
                initial_state,
                config=self._runnable_config(request),
            )
            result = AgentResult.from_state(
                final_state,
                request_id=request.request_id,
                include_raw_state=include_raw_state,
            )
        except Exception as exc:
            if logger is not None:
                logger.error(
                    "Agent request failed | request_id=%s | error=%s",
                    request.request_id,
                    exc,
                    exc_info=True,
                )
            result = AgentResult.error(
                request_id=request.request_id,
                message=str(exc),
                session_id=request.session_id,
                user_id=request.user_id,
                project_id=request.project_id,
            )

        return result

    def invoke(
        self,
        query_or_state: Union[str, Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """调用问题字符串，或直接执行一个完整 AgentState。"""

        if self._legacy_service is not None:
            return dict(self._legacy_service.invoke(query_or_state, **kwargs))

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
            runnable_config = kwargs.pop("runnable_config", None)
            if kwargs:
                unknown = ", ".join(sorted(kwargs))
                raise TypeError(
                    "Unsupported keyword arguments for state invocation: "
                    f"{unknown}"
                )
            if runnable_config is None:
                runnable_config = self._state_runnable_config(query_or_state)
            return dict(
                self.get_graph().invoke(
                    query_or_state,
                    config=runnable_config,
                )
            )

        raise TypeError("query_or_state must be str or dict.")

    def batch_ask(
        self,
        queries: Iterable[str],
        **kwargs: Any,
    ) -> List[AgentResult]:
        """串行执行多个问题。"""

        if self._legacy_service is not None:
            results = _call_with_supported_kwargs(
                self._legacy_service.batch_ask,
                {"queries": list(queries), **kwargs},
            )
            return [_coerce_agent_result(item) for item in results]
        return super().batch_ask(queries, **kwargs)

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
        request_options: Optional[RequestOptions] = None,
        config_overrides: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """返回 LangGraph 原始事件流。"""

        if self._legacy_service is not None:
            return _call_with_supported_kwargs(
                self._legacy_service.stream,
                {
                    "query": query,
                    "user_id": user_id,
                    "project_id": project_id,
                    "session_id": session_id,
                    "request_id": request_id,
                    "messages": messages,
                    "chat_history": chat_history,
                    "metadata": metadata,
                    "request_options": request_options,
                    "config_overrides": config_overrides,
                    **kwargs,
                },
            )

        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported stream keyword arguments: {unknown}")

        request = AgentRequest(
            query=query,
            request_id=request_id or _make_id("req"),
            session_id=session_id or _make_id("sess"),
            user_id=user_id or "",
            project_id=project_id or "",
            messages=messages or [],
            chat_history=chat_history or [],
            metadata=metadata or {},
            request_options=request_options or {},
        )
        initial_state = self._build_initial_state(
            request,
            config_overrides=config_overrides,
        )
        return self.get_graph().stream(
            initial_state,
            config=self._runnable_config(request),
        )

    def get_graph(self) -> Any:
        """获取已编译 Graph，必要时延迟构建。"""

        if self._legacy_service is not None:
            return self._legacy_service.get_graph()
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def rebuild_graph(self) -> Any:
        """重新构建 CompiledGraph。"""

        if self._legacy_service is not None:
            return self._legacy_service.rebuild_graph()
        self._graph = self._build_graph()
        return self._graph

    def health_check(self) -> Dict[str, Any]:
        """检查 Agent、Runtime、Graph 和 Memory 依赖是否可用。"""

        if self._legacy_service is not None:
            health = dict(self._legacy_service.health_check())
            health["agent"] = self.__class__.__name__
            health["compatibility_mode"] = True
            return health

        try:
            graph = self.get_graph()
            runtime = self.runtime
            return {
                "ok": graph is not None,
                "agent": self.__class__.__name__,
                "graph_built": graph is not None,
                "runtime_closed": bool(getattr(runtime, "is_closed", False)),
                "memory_available": getattr(runtime, "memory_manager", None)
                is not None,
                "time": _utc_now(),
            }
        except Exception as exc:
            return {
                "ok": False,
                "agent": self.__class__.__name__,
                "graph_built": False,
                "error": str(exc),
                "time": _utc_now(),
            }

    def info(self) -> Dict[str, Any]:
        """返回不包含凭据、Memory 正文和大型对象的 Agent 摘要。"""

        if self._legacy_service is not None:
            info = dict(self._legacy_service.info())
            info["agent"] = self.__class__.__name__
            info["compatibility_mode"] = True
            return info

        runtime_summary = (
            self.runtime.summary()
            if callable(getattr(self.runtime, "summary", None))
            else {}
        )
        return {
            "agent": self.__class__.__name__,
            "graph_built": self._graph is not None,
            "compatibility_mode": False,
            "runtime": runtime_summary,
            "time": _utc_now(),
        }

    def close(self) -> None:
        """释放由本 Agent 自行创建的 RuntimeContext。"""

        if not self._owns_runtime or self._runtime is None:
            return
        close_method = getattr(self._runtime, "close", None)
        if callable(close_method):
            close_method()

    def __enter__(self) -> "KGRAGAgent":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _create_runtime(
        self,
        *,
        config: Optional[Mapping[str, Any]],
        config_path: Optional[str],
        validate: bool,
    ) -> Any:
        from kg_rag_agent.runtime import RuntimeSettings, create_runtime

        if config_path is not None and config is not None:
            settings = RuntimeSettings.load(
                config_path=config_path,
                validate=validate,
            ).with_overrides(
                config,
                validate=validate,
            )
            return create_runtime(settings=settings)

        return create_runtime(
            config=config,
            config_path=config_path,
            validate=validate,
        )

    def _build_graph(self) -> Any:
        factory = self._graph_factory
        if factory is None:
            from kg_rag_agent.graph.builder import build_graph

            factory = build_graph

        graph_config = dict(getattr(self.runtime.settings, "graph", {}) or {})
        available_kwargs = {
            "runtime": self.runtime,
            "checkpointer": graph_config.get("checkpointer"),
            "interrupt_before": graph_config.get("interrupt_before"),
            "interrupt_after": graph_config.get("interrupt_after"),
        }
        return _call_factory(factory, available_kwargs)

    def _build_initial_state(
        self,
        request: AgentRequest,
        *,
        config_overrides: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        from kg_rag_agent.graph.state import build_initial_state

        settings = self.runtime.settings
        if config_overrides:
            runtime_config = settings.with_overrides(
                config_overrides,
                validate=False,
            ).to_dict()
        else:
            runtime_config = settings.to_dict()

        # ``config`` 仅用于仍处在迁移期的旧 Node。正式请求参数通过
        # ``request_options`` 传递，Memory 隔离标识通过独立字段传递。
        return dict(
            build_initial_state(
                query=request.query,
                request_id=request.request_id,
                session_id=request.session_id,
                user_id=request.user_id,
                project_id=request.project_id,
                messages=request.messages,
                chat_history=request.chat_history,
                metadata=request.state_metadata(),
                request_options=request.request_options,
                config=runtime_config,
            )
        )

    @staticmethod
    def _runnable_config(request: AgentRequest) -> Dict[str, Any]:
        """构造 LangGraph 运行配置。

        Checkpointer 仍以 ``session_id`` 作为线程标识；用户和项目隔离由
        AgentState 与 MemoryManager 显式处理。
        """

        return {
            "configurable": {
                "thread_id": request.session_id,
                "request_id": request.request_id,
                "user_id": request.user_id,
                "project_id": request.project_id,
            }
        }

    @staticmethod
    def _state_runnable_config(state: Mapping[str, Any]) -> Dict[str, Any]:
        session_id = str(state.get("session_id", "") or "").strip()
        if not session_id:
            return {}
        return {
            "configurable": {
                "thread_id": session_id,
                "request_id": str(state.get("request_id", "") or ""),
                "user_id": str(state.get("user_id", "") or ""),
                "project_id": str(state.get("project_id", "") or ""),
            }
        }


# 旧代码兼容别名；新代码统一使用 KGRAGAgent。
KGRAgent = KGRAGAgent


def create_agent(**kwargs: Any) -> KGRAGAgent:
    """创建 :class:`KGRAGAgent`。"""

    return KGRAGAgent(**kwargs)


def _call_factory(factory: GraphFactory, values: Mapping[str, Any]) -> Any:
    """仅向 Graph Factory 传入其声明支持的参数。"""

    signature = inspect.signature(factory)
    parameters = signature.parameters
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    kwargs: Dict[str, Any] = {}
    for name, value in values.items():
        if accepts_var_kwargs or name in parameters:
            kwargs[name] = value

    return factory(**kwargs)


def _call_with_supported_kwargs(
    callable_obj: Callable[..., Any],
    values: Mapping[str, Any],
) -> Any:
    """兼容调用旧 Service，只传递其签名声明支持的关键字参数。"""

    signature = inspect.signature(callable_obj)
    parameters = signature.parameters
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    kwargs: Dict[str, Any] = {}
    for name, value in values.items():
        if accepts_var_kwargs or name in parameters:
            kwargs[name] = value
    return callable_obj(**kwargs)


def _coerce_agent_result(
    value: Any,
    *,
    include_raw_state: bool = False,
) -> AgentResult:
    """把旧 Service 结果或 Mapping 转换为新的 AgentResult。"""

    if isinstance(value, AgentResult):
        return value

    if isinstance(value, Mapping):
        data = dict(value)
    elif callable(getattr(value, "to_dict", None)):
        to_dict = getattr(value, "to_dict")
        try:
            data = dict(
                to_dict(
                    include_raw_state=include_raw_state,
                    include_identifiers=True,
                    include_memory_status=True,
                )
            )
        except TypeError:
            data = dict(to_dict(include_raw_state=include_raw_state))
        if include_raw_state and getattr(value, "raw_state", None) is not None:
            data["raw_state"] = getattr(value, "raw_state")
    else:
        data = {
            key: getattr(value, key)
            for key in (
                "answer",
                "request_id",
                "route",
                "answerability",
                "semantic_score",
                "citations",
                "traces",
                "warnings",
                "has_error",
                "error_message",
                "session_id",
                "user_id",
                "project_id",
                "memory_status",
                "raw_state",
            )
            if hasattr(value, key)
        }

    raw_memory_status = data.get("memory_status")
    memory_status: MemoryStatus = {}
    if isinstance(raw_memory_status, Mapping):
        memory_status = copy.deepcopy(dict(raw_memory_status))  # type: ignore[assignment]

    raw_state = data.get("raw_state")
    normalized_raw_state = None
    if include_raw_state and isinstance(raw_state, Mapping):
        normalized_raw_state = copy.deepcopy(dict(raw_state))

    return AgentResult(
        answer=str(data.get("answer", "") or ""),
        request_id=str(data.get("request_id", "") or ""),
        route=str(data.get("route", "") or ""),
        answerability=str(data.get("answerability", "") or ""),
        semantic_score=_safe_float(data.get("semantic_score", 0.0)),
        citations=_copy_mapping_list(data.get("citations")),
        traces=_copy_mapping_list(data.get("traces")),
        warnings=[str(item) for item in (data.get("warnings") or [])],
        has_error=bool(data.get("has_error", False)),
        error_message=str(data.get("error_message", "") or ""),
        session_id=str(data.get("session_id", "") or ""),
        user_id=str(data.get("user_id", "") or ""),
        project_id=str(data.get("project_id", "") or ""),
        memory_status=memory_status,
        raw_state=normalized_raw_state,
    )


def _copy_mapping_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        copy.deepcopy(dict(item))
        for item in value
        if isinstance(item, Mapping)
    ]


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truncate(value: Any, max_length: int = 160) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= max_length else text[:max_length] + "..."


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "KGRAgent",
    "KGRAGAgent",
    "create_agent",
]
