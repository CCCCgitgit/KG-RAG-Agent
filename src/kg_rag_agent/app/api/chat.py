# -*- coding: utf-8 -*-
"""聊天 API。

本模块只负责：

- 校验并转换 FastAPI 请求；
- 调用 :class:`AgentService`；
- 控制 ``raw_state``、Memory 状态和请求标识的对外暴露；
- 将服务异常转换为稳定的 API 错误；
- 对流式事件中的 Runtime、配置和 Memory 正文做脱敏。

KG-RAG、Memory、检索和回答算法不得在 API 层实现。
"""

from __future__ import annotations

import inspect
import json
import logging
import uuid
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from kg_rag_agent.app.dependencies import get_agent_service, get_api_settings
from kg_rag_agent.app.errors import APIError
from kg_rag_agent.app.schemas.request import (
    BatchChatRequest,
    ChatMessage,
    ChatRequest,
    StreamChatRequest,
)
from kg_rag_agent.app.schemas.response import BatchChatResponse, ChatResponse
from kg_rag_agent.app.settings import APISettings

router = APIRouter(prefix="/chat")
logger = logging.getLogger("kg_rag_agent.app.chat")

# 原始 Graph Stream 中不得直接向客户端暴露这些字段。Memory 的可公开摘要
# 由 ChatResponse.memory_status 提供，而不是返回 Memory 正文或 Store 细节。
_STREAM_REDACTED_KEYS = frozenset(
    {
        "config",
        "raw_state",
        "runtime",
        "runtime_context",
        "llm_client",
        "embedding_client",
        "graph_store",
        "vector_store",
        "prompt_manager",
        "memory_manager",
        "tool_registry",
        "mcp_client_manager",
        "memory_context",
        "memory_text",
        "memory_candidates",
        "memory_write_result",
    }
)


def _message_to_dict(message: Any) -> Dict[str, Any]:
    """将 Pydantic 消息或 Mapping 转换为普通字典。"""

    if isinstance(message, Mapping):
        return dict(message)
    if hasattr(message, "model_dump"):
        return dict(message.model_dump())
    if hasattr(message, "dict"):
        return dict(message.dict())
    return {
        "role": getattr(message, "role", ""),
        "content": getattr(message, "content", ""),
    }


def _messages_to_dicts(
    messages: List[ChatMessage] | None,
) -> List[Dict[str, Any]]:
    if not messages:
        return []
    return [_message_to_dict(item) for item in messages]


def _request_options(request: Any) -> Dict[str, Any]:
    """提取请求级白名单参数。"""

    options = getattr(request, "options", None)
    if options is None:
        return {}
    return dict(options.to_service_options())


def _allow_raw_state(
    requested: bool,
    settings: APISettings,
    request_id: str | None,
) -> bool:
    if requested and not settings.allow_raw_state:
        raise APIError(
            status_code=403,
            error_code="raw_state_disabled",
            message="Raw AgentState output is disabled.",
            request_id=request_id,
        )
    return bool(requested)


def _response_from_result(
    result: Any,
    *,
    include_raw_state: bool,
    include_memory_status: bool,
) -> ChatResponse:
    """将 AgentResult 转换为 API Response。

    ``ChatResponse`` 在 Memory 接入期间可能处于新旧 Schema 过渡状态。
    这里根据工厂方法的实际签名传参，保证本文件可以与当前版本及下一步
    更新后的 ``response.py`` 同时工作。
    """

    factory = ChatResponse.from_agent_result
    parameters = inspect.signature(factory).parameters
    kwargs: Dict[str, Any] = {
        "include_raw_state": include_raw_state,
    }
    if "include_identifiers" in parameters:
        kwargs["include_identifiers"] = True
    if "include_memory_status" in parameters:
        kwargs["include_memory_status"] = include_memory_status
    return factory(result, **kwargs)


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    service: Any = Depends(get_agent_service),
    settings: APISettings = Depends(get_api_settings),
) -> ChatResponse:
    """执行一次标准聊天请求。"""

    request_id = request.request_id or _make_request_id()
    include_raw_state = _allow_raw_state(
        request.include_raw_state,
        settings,
        request_id,
    )

    try:
        result = service.ask(
            query=request.query,
            user_id=request.user_id,
            project_id=request.project_id,
            session_id=request.session_id,
            request_id=request_id,
            messages=_messages_to_dicts(request.messages),
            chat_history=_messages_to_dicts(request.chat_history),
            metadata=request.metadata,
            request_options=_request_options(request),
            include_raw_state=include_raw_state,
        )
        return _response_from_result(
            result,
            include_raw_state=include_raw_state,
            include_memory_status=request.include_memory_status,
        )
    except APIError:
        raise
    except (TypeError, ValueError) as exc:
        raise APIError(
            status_code=400,
            error_code="invalid_agent_request",
            message=str(exc),
            request_id=request_id,
        ) from exc
    except Exception as exc:
        logger.exception("Chat request failed | request_id=%s", request_id)
        raise APIError(
            status_code=500,
            error_code="chat_execution_failed",
            message="The chat request could not be completed.",
            request_id=request_id,
        ) from exc


@router.post("/batch", response_model=BatchChatResponse)
def batch_chat(
    request: BatchChatRequest,
    service: Any = Depends(get_agent_service),
    settings: APISettings = Depends(get_api_settings),
) -> BatchChatResponse:
    """在同一用户、项目和会话边界内串行执行批量问题。"""

    if len(request.queries) > settings.max_batch_size:
        raise APIError(
            status_code=400,
            error_code="batch_size_exceeded",
            message=f"Batch size cannot exceed {settings.max_batch_size}.",
        )

    include_raw_state = _allow_raw_state(
        request.include_raw_state,
        settings,
        None,
    )
    options = _request_options(request)
    responses: List[ChatResponse] = []

    try:
        for query in request.queries:
            result = service.ask(
                query=query,
                user_id=request.user_id,
                project_id=request.project_id,
                session_id=request.session_id,
                metadata=request.metadata,
                request_options=options,
                include_raw_state=include_raw_state,
            )
            responses.append(
                _response_from_result(
                    result,
                    include_raw_state=include_raw_state,
                    include_memory_status=request.include_memory_status,
                )
            )
        return BatchChatResponse(results=responses, count=len(responses))
    except APIError:
        raise
    except (TypeError, ValueError) as exc:
        raise APIError(
            status_code=400,
            error_code="invalid_batch_request",
            message=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Batch chat request failed")
        raise APIError(
            status_code=500,
            error_code="batch_execution_failed",
            message="The batch request could not be completed.",
        ) from exc


@router.post("/stream")
def stream_chat(
    request: StreamChatRequest,
    service: Any = Depends(get_agent_service),
) -> StreamingResponse:
    """以 NDJSON 返回 LangGraph 事件流。"""

    request_id = request.request_id or _make_request_id()

    try:
        events = service.stream(
            query=request.query,
            user_id=request.user_id,
            project_id=request.project_id,
            session_id=request.session_id,
            request_id=request_id,
            messages=_messages_to_dicts(request.messages),
            chat_history=_messages_to_dicts(request.chat_history),
            metadata=request.metadata,
            request_options=_request_options(request),
        )
    except (TypeError, ValueError) as exc:
        raise APIError(
            status_code=400,
            error_code="invalid_stream_request",
            message=str(exc),
            request_id=request_id,
        ) from exc
    except Exception as exc:
        logger.exception("Stream creation failed | request_id=%s", request_id)
        raise APIError(
            status_code=500,
            error_code="stream_creation_failed",
            message="The stream could not be created.",
            request_id=request_id,
        ) from exc

    return StreamingResponse(
        _serialize_stream(events, request_id=request_id),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )


@router.get("/info")
def chat_info(
    service: Any = Depends(get_agent_service),
) -> Dict[str, Any]:
    """返回不包含密钥和连接对象的 AgentService 摘要。"""

    try:
        return dict(service.info())
    except Exception as exc:
        logger.exception("Agent service info failed")
        raise APIError(
            status_code=500,
            error_code="service_info_failed",
            message="Agent service information is unavailable.",
        ) from exc


def _serialize_stream(
    events: Iterable[Any],
    *,
    request_id: str,
) -> Iterable[str]:
    """序列化并脱敏流式事件。"""

    try:
        for event in events:
            yield json.dumps(
                {
                    "request_id": request_id,
                    "event": _sanitize_stream_payload(event),
                },
                ensure_ascii=False,
                allow_nan=False,
                default=str,
            ) + "\n"
    except Exception:
        logger.exception("Stream iteration failed | request_id=%s", request_id)
        yield json.dumps(
            {
                "request_id": request_id,
                "error": {
                    "error_code": "stream_execution_failed",
                    "message": "The stream terminated unexpectedly.",
                },
            },
            ensure_ascii=False,
        ) + "\n"


def _sanitize_stream_payload(
    value: Any,
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> Any:
    """递归移除事件中的 Runtime 与 Memory 正文。

    LangGraph 事件通常由普通 Mapping/List 组成。遇到不可安全展开的对象时，
    保留字符串表示；最大递归深度和循环引用检查防止异常对象拖垮流接口。
    """

    if _depth > 12:
        return "[truncated]"

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if _seen is None:
        _seen = set()

    value_id = id(value)
    if value_id in _seen:
        return "[circular]"

    if isinstance(value, Mapping):
        _seen.add(value_id)
        try:
            output: Dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)
                if key_text.lower() in _STREAM_REDACTED_KEYS:
                    output[key_text] = "[redacted]"
                    continue
                output[key_text] = _sanitize_stream_payload(
                    item,
                    _depth=_depth + 1,
                    _seen=_seen,
                )
            return output
        finally:
            _seen.discard(value_id)

    if isinstance(value, (list, tuple, set, frozenset)):
        _seen.add(value_id)
        try:
            return [
                _sanitize_stream_payload(
                    item,
                    _depth=_depth + 1,
                    _seen=_seen,
                )
                for item in value
            ]
        finally:
            _seen.discard(value_id)

    if hasattr(value, "model_dump"):
        try:
            return _sanitize_stream_payload(
                value.model_dump(),
                _depth=_depth + 1,
                _seen=_seen,
            )
        except Exception:
            return str(value)

    if hasattr(value, "to_dict"):
        try:
            return _sanitize_stream_payload(
                value.to_dict(),
                _depth=_depth + 1,
                _seen=_seen,
            )
        except Exception:
            return str(value)

    return str(value)


def _make_request_id() -> str:
    return "req_" + uuid.uuid4().hex[:16]


__all__ = [
    "router",
    "chat",
    "batch_chat",
    "stream_chat",
    "chat_info",
    "_message_to_dict",
    "_messages_to_dicts",
    "_request_options",
    "_response_from_result",
    "_sanitize_stream_payload",
]
