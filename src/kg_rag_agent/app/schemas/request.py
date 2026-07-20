# -*- coding: utf-8 -*-
"""FastAPI 请求 Schema。

本模块只接收可公开、可序列化的请求数据：

- ``user_id``、``project_id``、``session_id`` 用于会话与 Memory 隔离；
- ``ChatOptions`` 只允许覆盖经过审核的请求级参数；
- Memory 正文、Memory Namespace、Store 路径和系统权限不得由外部请求传入。
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


_IDENTIFIER_MAX_LENGTH = 128
_QUERY_MAX_LENGTH = 20_000
_MESSAGE_MAX_LENGTH = 100_000
_METADATA_MAX_KEYS = 64
_METADATA_MAX_BYTES = 32_768
_MAX_MESSAGES = 128
_MAX_BATCH_SIZE = 100
_MAX_ALLOWED_TOOLS = 64


class APIRequestModel(BaseModel):
    """API 请求模型基类。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class ChatMessage(APIRequestModel):
    """外部传入的一条轻量对话消息。"""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1, max_length=_MESSAGE_MAX_LENGTH)


class ChatOptions(APIRequestModel):
    """单次请求允许覆盖的白名单参数。

    这里必须与 ``agents.schemas.RequestOptions`` 和
    ``services.agent_service`` 的白名单保持一致。路径、密钥、Provider 地址、
    Prompt 路径、Memory Namespace、Tool/MCP 系统权限等稳定配置不得出现在这里。
    """

    retrieval_top_k: Optional[int] = Field(default=None, ge=1, le=100)
    path_max_depth: Optional[int] = Field(default=None, ge=1, le=6)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    language: Optional[str] = Field(default=None, min_length=1, max_length=32)
    include_citations: Optional[bool] = None
    allowed_tools: Optional[
        List[Annotated[str, Field(min_length=1, max_length=128)]]
    ] = Field(default=None, max_length=_MAX_ALLOWED_TOOLS)

    @field_validator("allowed_tools")
    @classmethod
    def deduplicate_tools(
        cls,
        value: Optional[List[str]],
    ) -> Optional[List[str]]:
        if value is None:
            return None

        result: List[str] = []
        seen: set[str] = set()
        for item in value:
            name = item.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            result.append(name)
        return result

    def to_service_options(self) -> Dict[str, Any]:
        """转换为 AgentService 接受的请求级参数。"""

        return self.model_dump(exclude_none=True)


Identifier = Annotated[
    str,
    Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH),
]
MessageList = Annotated[List[ChatMessage], Field(max_length=_MAX_MESSAGES)]
QueryText = Annotated[str, Field(min_length=1, max_length=_QUERY_MAX_LENGTH)]


class ChatRequest(APIRequestModel):
    """单轮聊天请求。"""

    query: QueryText
    user_id: Optional[Identifier] = None
    project_id: Optional[Identifier] = None
    session_id: Optional[Identifier] = None
    request_id: Optional[Identifier] = None

    messages: Optional[MessageList] = None
    chat_history: Optional[MessageList] = None
    metadata: Optional[Dict[str, Any]] = None
    options: Optional[ChatOptions] = None

    # ``raw_state`` 与 Memory 状态均属于受控诊断输出，不暴露 Memory 正文。
    include_raw_state: bool = False
    include_memory_status: bool = False

    @field_validator("metadata")
    @classmethod
    def validate_metadata(
        cls,
        value: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return _validate_metadata(value)


class BatchChatRequest(APIRequestModel):
    """批量聊天请求。

    同一个批次共享用户、项目和会话标识。需要为每条问题使用独立会话时，
    应由调用方拆分为多个单轮请求，避免 Memory 会话边界不清晰。
    """

    queries: Annotated[
        List[QueryText],
        Field(min_length=1, max_length=_MAX_BATCH_SIZE),
    ]
    user_id: Optional[Identifier] = None
    project_id: Optional[Identifier] = None
    session_id: Optional[Identifier] = None
    metadata: Optional[Dict[str, Any]] = None
    options: Optional[ChatOptions] = None
    include_raw_state: bool = False
    include_memory_status: bool = False

    @field_validator("metadata")
    @classmethod
    def validate_metadata(
        cls,
        value: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return _validate_metadata(value)


class StreamChatRequest(APIRequestModel):
    """流式聊天请求。"""

    query: QueryText
    user_id: Optional[Identifier] = None
    project_id: Optional[Identifier] = None
    session_id: Optional[Identifier] = None
    request_id: Optional[Identifier] = None

    messages: Optional[MessageList] = None
    chat_history: Optional[MessageList] = None
    metadata: Optional[Dict[str, Any]] = None
    options: Optional[ChatOptions] = None

    @field_validator("metadata")
    @classmethod
    def validate_metadata(
        cls,
        value: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return _validate_metadata(value)


def _validate_metadata(
    value: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """限制 Metadata 的数量和序列化大小。"""

    if value is None:
        return None
    if len(value) > _METADATA_MAX_KEYS:
        raise ValueError(
            f"metadata cannot contain more than {_METADATA_MAX_KEYS} keys."
        )

    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be JSON serializable.") from exc

    if len(serialized.encode("utf-8")) > _METADATA_MAX_BYTES:
        raise ValueError(
            f"metadata cannot exceed {_METADATA_MAX_BYTES // 1024} KiB."
        )
    return value


__all__ = [
    "ChatMessage",
    "ChatOptions",
    "ChatRequest",
    "BatchChatRequest",
    "StreamChatRequest",
]
