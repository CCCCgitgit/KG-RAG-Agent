# -*- coding: utf-8 -*-
"""FastAPI 应用工厂与应用级生命周期管理。

本模块只负责 Web 应用装配：

* 读取并校验 :class:`APISettings`；
* 在 FastAPI 生命周期中创建和释放 Runtime、Agent 与 Service；
* 将共享依赖安装到 ``app.state``；
* 注册 API 路由、CORS 和统一异常处理；
* 提供 Uvicorn 可直接加载的模块级 ``app``。

KG-RAG、Memory、检索、工具和回答算法不得在本模块实现。
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import (
    AppDependencies,
    build_app_dependencies,
    install_app_dependencies,
)
from .errors import register_exception_handlers
from .settings import APISettings

APP_TITLE = "KG-RAG Agent API"
APP_DESCRIPTION = "LangGraph-based KG-RAG Agent service."
APP_VERSION = "0.1.0"

ServiceFactory = Callable[[], Any | Awaitable[Any]]

logger = logging.getLogger("kg_rag_agent.app.server")


def create_app(
    *,
    settings: Optional[APISettings] = None,
    agent_service: Any = None,
    agent: Any = None,
    runtime: Any = None,
    service_factory: Optional[ServiceFactory] = None,
    config: Optional[Mapping[str, Any]] = None,
    config_path: Optional[str] = None,
    config_dir: Optional[str] = None,
    project_root: Optional[str] = None,
    runtime_options: Any = None,
    initialize_service: Optional[bool] = None,
    auto_build_graph: bool = True,
    validate: bool = True,
) -> FastAPI:
    """创建 FastAPI 应用。

    依赖来源只能选择一种：

    * 直接注入 ``agent_service``；
    * 直接注入 ``agent``；
    * 注入 ``runtime``；
    * 通过配置创建 Runtime；
    * 使用 ``service_factory`` 自定义创建 Service。

    未显式注入依赖时，默认在 FastAPI startup 阶段创建，避免模块导入时
    初始化 LLM、图谱、向量库和 Memory Store。
    """

    api_settings = settings or APISettings.from_env()
    api_settings.validate()

    should_initialize = (
        api_settings.initialize_service_on_startup
        if initialize_service is None
        else bool(initialize_service)
    )

    _validate_dependency_sources(
        agent_service=agent_service,
        agent=agent,
        runtime=runtime,
        service_factory=service_factory,
        config=config,
        config_path=config_path,
        config_dir=config_dir,
        project_root=project_root,
        runtime_options=runtime_options,
    )

    initial_dependencies = _build_initial_dependencies(
        agent_service=agent_service,
        agent=agent,
        runtime=runtime,
        auto_build_graph=auto_build_graph,
        validate=validate,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        dependencies = getattr(
            app.state,
            "dependencies",
            initial_dependencies,
        )

        if dependencies.agent_service is None and should_initialize:
            try:
                dependencies = await _create_startup_dependencies(
                    service_factory=service_factory,
                    config=config,
                    config_path=config_path,
                    config_dir=config_dir,
                    project_root=project_root,
                    runtime_options=runtime_options,
                    auto_build_graph=auto_build_graph,
                    validate=validate,
                )
            except Exception as exc:
                logger.exception("Application dependency startup failed")
                dependencies = AppDependencies(startup_error=exc)

            install_app_dependencies(
                app,
                dependencies,
                api_settings=api_settings,
            )

        try:
            yield
        finally:
            current = getattr(app.state, "dependencies", dependencies)
            try:
                current.close()
            except Exception:
                logger.exception("Application dependency shutdown failed")

    app = FastAPI(
        title=api_settings.title,
        description=api_settings.description,
        version=api_settings.version,
        lifespan=lifespan,
        docs_url="/docs" if api_settings.docs_enabled else None,
        redoc_url="/redoc" if api_settings.docs_enabled else None,
        openapi_url=(
            "/openapi.json"
            if api_settings.docs_enabled
            else None
        ),
    )

    install_app_dependencies(
        app,
        initial_dependencies,
        api_settings=api_settings,
    )

    _configure_cors(app, api_settings)
    register_exception_handlers(app)
    register_routers(app, prefix=api_settings.api_prefix)

    return app


def register_routers(
    app: FastAPI,
    *,
    prefix: str = "/api",
) -> None:
    """注册健康检查和聊天 API。"""

    normalized_prefix = _normalize_prefix(prefix)

    from .api import chat_router, health_router

    app.include_router(
        health_router,
        prefix=normalized_prefix,
        tags=["health"],
    )
    app.include_router(
        chat_router,
        prefix=normalized_prefix,
        tags=["chat"],
    )


def _build_initial_dependencies(
    *,
    agent_service: Any,
    agent: Any,
    runtime: Any,
    auto_build_graph: bool,
    validate: bool,
) -> AppDependencies:
    """只处理显式注入对象；配置驱动创建延迟到 startup。"""

    if agent_service is not None:
        return build_app_dependencies(agent_service=agent_service)

    if agent is not None:
        return build_app_dependencies(
            agent=agent,
            validate=validate,
        )

    if runtime is not None:
        return build_app_dependencies(
            runtime=runtime,
            auto_build_graph=auto_build_graph,
            validate=validate,
        )

    return AppDependencies()


async def _create_startup_dependencies(
    *,
    service_factory: Optional[ServiceFactory],
    config: Optional[Mapping[str, Any]],
    config_path: Optional[str],
    config_dir: Optional[str],
    project_root: Optional[str],
    runtime_options: Any,
    auto_build_graph: bool,
    validate: bool,
) -> AppDependencies:
    if service_factory is not None:
        produced = service_factory()
        if inspect.isawaitable(produced):
            produced = await produced

        if isinstance(produced, AppDependencies):
            return produced

        if produced is None:
            raise RuntimeError(
                "service_factory returned None."
            )

        dependencies = build_app_dependencies(
            agent_service=produced,
        )
        # Factory 在 startup 中创建的 Service 由应用生命周期负责释放。
        dependencies.owns_agent_service = True
        return dependencies

    return build_app_dependencies(
        config=config,
        config_path=config_path,
        config_dir=config_dir,
        project_root=project_root,
        runtime_options=runtime_options,
        auto_build_graph=auto_build_graph,
        validate=validate,
    )


def _configure_cors(
    app: FastAPI,
    settings: APISettings,
) -> None:
    if not settings.cors_origins:
        return

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=list(settings.cors_allow_methods),
        allow_headers=list(settings.cors_allow_headers),
    )


def _validate_dependency_sources(
    *,
    agent_service: Any,
    agent: Any,
    runtime: Any,
    service_factory: Optional[ServiceFactory],
    config: Optional[Mapping[str, Any]],
    config_path: Optional[str],
    config_dir: Optional[str],
    project_root: Optional[str],
    runtime_options: Any,
) -> None:
    injected_count = sum(
        value is not None
        for value in (
            agent_service,
            agent,
            runtime,
            service_factory,
        )
    )
    if injected_count > 1:
        raise ValueError(
            "Only one of agent_service, agent, runtime, or "
            "service_factory may be provided."
        )

    has_config_source = any(
        value is not None
        for value in (
            config,
            config_path,
            config_dir,
            project_root,
            runtime_options,
        )
    )
    if injected_count and has_config_source:
        raise ValueError(
            "Injected dependencies cannot be combined with "
            "configuration sources or runtime_options."
        )


def _normalize_prefix(prefix: str) -> str:
    value = str(prefix or "/api").strip()
    if not value.startswith("/"):
        value = "/" + value
    value = value.rstrip("/")
    return value or "/api"


app = create_app()


__all__ = [
    "APP_TITLE",
    "APP_DESCRIPTION",
    "APP_VERSION",
    "ServiceFactory",
    "create_app",
    "register_routers",
    "app",
]
