# -*- coding: utf-8 -*-
"""FastAPI 请求与响应 Schema 的统一导出入口。

外部模块可以继续从 ``kg_rag_agent.app.schemas`` 导入常用请求、响应和
Memory 状态模型；具体实现仍分别位于 ``request.py`` 与 ``response.py``。
"""

from .request import (
    APIRequestModel,
    BatchChatRequest,
    ChatMessage,
    ChatOptions,
    ChatRequest,
    StreamChatRequest,
)
from .response import (
    APIResponseModel,
    BatchChatResponse,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    MemoryStatusResponse,
    ReadinessResponse,
    StatusResponse,
)

__all__ = [
    "APIRequestModel",
    "ChatMessage",
    "ChatOptions",
    "ChatRequest",
    "BatchChatRequest",
    "StreamChatRequest",
    "APIResponseModel",
    "MemoryStatusResponse",
    "ChatResponse",
    "BatchChatResponse",
    "HealthResponse",
    "ReadinessResponse",
    "StatusResponse",
    "ErrorResponse",
]
