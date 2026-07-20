# -*- coding: utf-8 -*-
"""FastAPI 路由包的统一导出入口。"""

from .chat import (
    batch_chat,
    chat,
    chat_info,
    router as chat_router,
    stream_chat,
)
from .health import (
    health_check,
    readiness_check,
    router as health_router,
    status_check,
)

__all__ = [
    "chat_router",
    "health_router",
    "chat",
    "batch_chat",
    "stream_chat",
    "chat_info",
    "health_check",
    "readiness_check",
    "status_check",
]
