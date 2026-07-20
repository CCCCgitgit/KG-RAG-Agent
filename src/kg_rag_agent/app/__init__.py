# -*- coding: utf-8 -*-
"""KG-RAG Agent Web API 公共入口。

该包仅暴露应用装配与依赖注入接口，不在导入阶段主动创建 Runtime、
LangGraph、LLM、知识图谱、向量库或 Memory Store。模块级 ``app`` 只完成
FastAPI 路由装配，重型组件在应用生命周期启动阶段统一初始化。
"""

from .dependencies import (
    AppDependencies,
    build_app_dependencies,
    get_agent_service,
    get_api_settings,
    get_app_dependencies,
    get_kgrag_agent,
    get_runtime_context,
    install_app_dependencies,
)
from .server import (
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    ServiceFactory,
    app,
    create_app,
    register_routers,
)
from .settings import APISettings

__all__ = [
    "APISettings",
    "AppDependencies",
    "ServiceFactory",
    "APP_TITLE",
    "APP_DESCRIPTION",
    "APP_VERSION",
    "build_app_dependencies",
    "install_app_dependencies",
    "get_app_dependencies",
    "get_api_settings",
    "get_runtime_context",
    "get_kgrag_agent",
    "get_agent_service",
    "create_app",
    "register_routers",
    "app",
]
