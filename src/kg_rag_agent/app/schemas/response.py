# -*- coding: utf-8 -*-
"""FastAPI 对外响应 Schema。

本模块只定义可公开、可序列化的 API 响应结构。它不会暴露 Runtime、
Memory 正文、Store 路径、模型客户端或其他内部运行时对象。
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class APIResponseModel(BaseModel):
    """所有 API 响应模型的公共基类。"""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


class MemoryStatusResponse(APIResponseModel):
    """本次请求的 Memory 执行摘要。

    这里只返回状态和计数，不返回会话摘要、长期 Memory 正文、命名空间、
    持久化路径或底层 Store 信息。
    """

    loaded: bool = False
    written: bool = False
    recent_message_count: int = Field(default=0, ge=0)
    retrieved_memory_count: int = Field(default=0, ge=0)
    estimated_tokens: int = Field(default=0, ge=0)
    summary_used: bool = False
    written_count: int = Field(default=0, ge=0)
    skipped_count: int = Field(default=0, ge=0)

    @classmethod
    def from_value(cls, value: Any) -> "MemoryStatusResponse":
        """从 Mapping 或兼容对象构造受控 Memory 状态。"""

        if isinstance(value, cls):
            return value.model_copy(deep=True)
        if isinstance(value, Mapping):
            return cls.model_validate(dict(value))
        if hasattr(value, "model_dump"):
            return cls.model_validate(value.model_dump())
        if hasattr(value, "to_dict"):
            return cls.model_validate(value.to_dict())
        raise TypeError("memory_status must be a mapping-like object.")


class ChatResponse(APIResponseModel):
    """单次聊天请求的稳定响应。"""

    answer: str
    request_id: str

    # 请求隔离标识。API 层可按需要决定是否输出。
    session_id: str = ""
    user_id: str = ""
    project_id: str = ""

    route: str = ""
    answerability: str = ""
    semantic_score: float = 0.0
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    traces: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    has_error: bool = False
    error_message: str = ""

    # 仅包含 Memory 状态摘要，不包含 Memory 正文。
    memory_status: Optional[MemoryStatusResponse] = None

    # 仅在服务端显式允许且请求明确要求时返回。
    raw_state: Optional[Dict[str, Any]] = None

    @classmethod
    def from_agent_result(
        cls,
        result: Any,
        *,
        include_raw_state: bool,
        include_identifiers: bool = True,
        include_memory_status: bool = False,
    ) -> "ChatResponse":
        """将 ``AgentResult`` 或兼容 Mapping 转换为 API 响应。

        参数开关在转换阶段执行，避免调用方通过伪造结果对象绕过响应边界。
        """

        data = _agent_result_to_dict(
            result,
            include_raw_state=include_raw_state,
            include_identifiers=include_identifiers,
            include_memory_status=include_memory_status,
        )

        if not include_raw_state:
            data.pop("raw_state", None)
        if not include_identifiers:
            data.pop("session_id", None)
            data.pop("user_id", None)
            data.pop("project_id", None)
        if not include_memory_status:
            data.pop("memory_status", None)

        # 旧版结果可能没有这些字段；使用稳定默认值补齐。
        if include_identifiers:
            data.setdefault("session_id", "")
            data.setdefault("user_id", "")
            data.setdefault("project_id", "")

        if include_memory_status:
            status = data.get("memory_status")
            if status is None:
                data["memory_status"] = MemoryStatusResponse()
            elif not isinstance(status, MemoryStatusResponse):
                data["memory_status"] = MemoryStatusResponse.from_value(status)

        return cls.model_validate(data)


class BatchChatResponse(APIResponseModel):
    """批量聊天响应。"""

    results: List[ChatResponse] = Field(default_factory=list)
    count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_count(self) -> "BatchChatResponse":
        if self.count != len(self.results):
            raise ValueError("count must equal the number of results.")
        return self


class HealthResponse(APIResponseModel):
    status: str = "ok"
    service: str = "kg-rag-agent"
    version: str = "0.1.0"
    timestamp: str


class ReadinessResponse(APIResponseModel):
    ready: bool
    service: str = "kg-rag-agent"
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str


class StatusResponse(APIResponseModel):
    running: bool = True
    message: str = "KG-RAG Agent API is running."


class ErrorResponse(APIResponseModel):
    error_code: str
    message: str
    request_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


def _agent_result_to_dict(
    result: Any,
    *,
    include_raw_state: bool,
    include_identifiers: bool,
    include_memory_status: bool,
) -> Dict[str, Any]:
    """兼容新旧 ``AgentResult.to_dict`` 签名并返回深拷贝。"""

    if isinstance(result, Mapping):
        return copy.deepcopy(dict(result))

    to_dict = getattr(result, "to_dict", None)
    if not callable(to_dict):
        if hasattr(result, "model_dump"):
            dumped = result.model_dump()
            if not isinstance(dumped, Mapping):
                raise TypeError("Agent result model_dump() must return a mapping.")
            return copy.deepcopy(dict(dumped))
        raise TypeError("Agent result must provide to_dict(), model_dump(), or be a mapping.")

    kwargs: Dict[str, Any] = {}
    try:
        parameters = inspect.signature(to_dict).parameters
    except (TypeError, ValueError):
        parameters = {}

    if "include_raw_state" in parameters:
        kwargs["include_raw_state"] = include_raw_state
    if "include_identifiers" in parameters:
        kwargs["include_identifiers"] = include_identifiers
    if "include_memory_status" in parameters:
        kwargs["include_memory_status"] = include_memory_status

    dumped = to_dict(**kwargs)
    if not isinstance(dumped, Mapping):
        raise TypeError("Agent result to_dict() must return a mapping.")
    return copy.deepcopy(dict(dumped))


__all__ = [
    "APIResponseModel",
    "MemoryStatusResponse",
    "ChatResponse",
    "BatchChatResponse",
    "HealthResponse",
    "ReadinessResponse",
    "StatusResponse",
    "ErrorResponse",
]
