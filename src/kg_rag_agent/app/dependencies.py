# -*- coding: utf-8 -*-
"""FastAPI 应用依赖注入与生命周期边界。

本模块统一管理：

* :class:`RuntimeContext`；
* :class:`KGRAGAgent`；
* :class:`AgentService`；
* FastAPI ``app.state`` 中的应用级依赖。

API 路由只通过 ``Depends`` 获取 ``AgentService``，不会直接创建 Runtime、
Agent、Graph、LLM、Memory 或 ToolRegistry。重型组件由应用启动阶段创建一次，
并在应用关闭时按所有权释放。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from fastapi import FastAPI, Request

from .errors import APIError
from .settings import APISettings


@dataclass(slots=True)
class AppDependencies:
    """FastAPI 应用级依赖集合。

    ``owns_*`` 表示依赖是否由当前容器创建。外部注入的测试对象或共享对象不会
    被本容器擅自关闭。
    """

    runtime: Any = None
    agent: Any = None
    agent_service: Any = None
    owns_runtime: bool = False
    owns_agent_service: bool = False
    startup_error: Optional[BaseException] = None
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def is_ready(self) -> bool:
        return (
            not self._closed
            and self.startup_error is None
            and self.agent_service is not None
        )

    @property
    def is_closed(self) -> bool:
        return self._closed

    def summary(self) -> dict[str, Any]:
        """返回不暴露密钥、Memory 正文和运行时对象内容的诊断摘要。"""

        return {
            "ready": self.is_ready,
            "closed": self._closed,
            "startup_error": (
                type(self.startup_error).__name__
                if self.startup_error is not None
                else None
            ),
            "runtime_available": self.runtime is not None,
            "agent_available": self.agent is not None,
            "agent_service_available": self.agent_service is not None,
            "owns_runtime": self.owns_runtime,
            "owns_agent_service": self.owns_agent_service,
        }

    def close(self) -> None:
        """按所有权释放 Service 和 Runtime。"""

        if self._closed:
            return

        if self.owns_agent_service and self.agent_service is not None:
            close = getattr(self.agent_service, "close", None)
            if callable(close):
                close()

        if self.owns_runtime and self.runtime is not None:
            close = getattr(self.runtime, "close", None)
            if callable(close):
                close()

        self._closed = True

    def __enter__(self) -> "AppDependencies":
        if self._closed:
            raise RuntimeError("AppDependencies has already been closed.")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def build_app_dependencies(
    *,
    agent_service: Any = None,
    agent: Any = None,
    runtime: Any = None,
    config: Optional[Mapping[str, Any]] = None,
    config_path: Optional[str] = None,
    config_dir: Optional[str] = None,
    project_root: Optional[str] = None,
    runtime_options: Any = None,
    auto_build_graph: bool = True,
    validate: bool = True,
) -> AppDependencies:
    """创建一组相互一致的 Runtime、Agent 和 Service。"""

    if agent_service is not None:
        if any(
            value is not None
            for value in (
                agent,
                runtime,
                config,
                config_path,
                config_dir,
                project_root,
                runtime_options,
            )
        ):
            raise ValueError(
                "agent_service cannot be combined with agent, runtime, "
                "configuration sources, or runtime_options."
            )
        resolved_agent = getattr(agent_service, "agent", None)
        resolved_runtime = _safe_runtime_from_service(agent_service)
        return AppDependencies(
            runtime=resolved_runtime,
            agent=resolved_agent,
            agent_service=agent_service,
        )

    if agent is not None:
        if any(
            value is not None
            for value in (
                runtime,
                config,
                config_path,
                config_dir,
                project_root,
                runtime_options,
            )
        ):
            raise ValueError(
                "agent cannot be combined with runtime, configuration sources, "
                "or runtime_options."
            )

        from kg_rag_agent.services import AgentService

        service = AgentService(agent=agent, validate=validate)
        return AppDependencies(
            runtime=_safe_runtime_from_agent(agent),
            agent=agent,
            agent_service=service,
            owns_agent_service=True,
        )

    created_runtime = False
    resolved_runtime = runtime

    try:
        if resolved_runtime is None:
            from kg_rag_agent.runtime import create_runtime

            resolved_runtime = create_runtime(
                config=config,
                config_path=config_path,
                config_dir=config_dir,
                project_root=project_root,
                options=runtime_options,
                validate=validate,
            )
            created_runtime = True
        elif any(
            value is not None
            for value in (
                config,
                config_path,
                config_dir,
                project_root,
                runtime_options,
            )
        ):
            raise ValueError(
                "runtime cannot be combined with configuration sources or "
                "runtime_options."
            )

        from kg_rag_agent.services import AgentService

        service = AgentService(
            runtime=resolved_runtime,
            auto_build_graph=auto_build_graph,
            validate=validate,
        )
        return AppDependencies(
            runtime=resolved_runtime,
            agent=getattr(service, "agent", None),
            agent_service=service,
            owns_runtime=created_runtime,
            owns_agent_service=True,
        )
    except Exception:
        if created_runtime and resolved_runtime is not None:
            close = getattr(resolved_runtime, "close", None)
            if callable(close):
                close()
        raise


def install_app_dependencies(
    app: FastAPI,
    dependencies: AppDependencies,
    *,
    api_settings: Optional[APISettings] = None,
) -> None:
    """将依赖安装到 ``app.state``。"""

    if api_settings is not None:
        app.state.api_settings = api_settings

    app.state.dependencies = dependencies
    app.state.runtime = dependencies.runtime
    app.state.kg_rag_agent = dependencies.agent
    app.state.agent_service = dependencies.agent_service
    app.state.startup_error = dependencies.startup_error


def get_app_dependencies(request: Request) -> AppDependencies:
    """获取应用级依赖容器。"""

    dependencies = getattr(request.app.state, "dependencies", None)
    if isinstance(dependencies, AppDependencies):
        return dependencies

    service = getattr(request.app.state, "agent_service", None)
    if service is not None:
        return AppDependencies(
            runtime=getattr(request.app.state, "runtime", None)
            or _safe_runtime_from_service(service),
            agent=getattr(request.app.state, "kg_rag_agent", None)
            or getattr(service, "agent", None),
            agent_service=service,
            startup_error=getattr(request.app.state, "startup_error", None),
        )

    startup_error = getattr(request.app.state, "startup_error", None)
    raise _unavailable_error(
        error_code="app_dependencies_unavailable",
        message="Application dependencies are unavailable.",
        startup_error=startup_error,
    )


def get_api_settings(request: Request) -> APISettings:
    """获取已经校验的 APISettings。"""

    settings = getattr(request.app.state, "api_settings", None)
    if not isinstance(settings, APISettings):
        raise APIError(
            status_code=503,
            error_code="api_not_initialized",
            message="API settings are unavailable.",
        )
    return settings


def get_runtime_context(request: Request) -> Any:
    """获取共享 RuntimeContext。"""

    dependencies = get_app_dependencies(request)
    runtime = dependencies.runtime
    if runtime is None:
        raise _unavailable_error(
            error_code="runtime_unavailable",
            message="Runtime context is unavailable.",
            startup_error=dependencies.startup_error,
        )

    ensure_open = getattr(runtime, "ensure_open", None)
    if callable(ensure_open):
        try:
            ensure_open()
        except Exception as exc:
            raise _unavailable_error(
                error_code="runtime_closed",
                message="Runtime context is unavailable.",
                startup_error=exc,
            ) from exc

    return runtime


def get_kgrag_agent(request: Request) -> Any:
    """获取应用级 KGRAGAgent。"""

    dependencies = get_app_dependencies(request)
    agent = dependencies.agent
    if agent is None:
        raise _unavailable_error(
            error_code="agent_unavailable",
            message="KG-RAG Agent is unavailable.",
            startup_error=dependencies.startup_error,
        )
    return agent


def get_agent_service(request: Request) -> Any:
    """获取 API 路由唯一允许直接调用的 AgentService。"""

    dependencies = get_app_dependencies(request)
    service = dependencies.agent_service
    if service is None:
        raise _unavailable_error(
            error_code="agent_service_unavailable",
            message="Agent service is unavailable.",
            startup_error=dependencies.startup_error,
        )
    return service


def _safe_runtime_from_service(service: Any) -> Any:
    try:
        return getattr(service, "runtime", None)
    except Exception:
        return None


def _safe_runtime_from_agent(agent: Any) -> Any:
    try:
        return getattr(agent, "runtime", None)
    except Exception:
        return None


def _unavailable_error(
    *,
    error_code: str,
    message: str,
    startup_error: Optional[BaseException],
) -> APIError:
    details: dict[str, Any] = {}
    if startup_error is not None:
        details["startup_error"] = type(startup_error).__name__

    return APIError(
        status_code=503,
        error_code=error_code,
        message=message,
        details=details,
    )


__all__ = [
    "AppDependencies",
    "build_app_dependencies",
    "install_app_dependencies",
    "get_app_dependencies",
    "get_api_settings",
    "get_runtime_context",
    "get_kgrag_agent",
    "get_agent_service",
]
