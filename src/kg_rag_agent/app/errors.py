# -*- coding: utf-8 -*-
"""API 层异常及统一异常响应。"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .schemas.response import ErrorResponse


class APIError(Exception):
    """可安全返回给客户端的 API 异常。"""

    def __init__(
        self,
        *,
        status_code: int,
        error_code: str,
        message: str,
        request_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.error_code = str(error_code)
        self.message = str(message)
        self.request_id = request_id
        self.details = dict(details or {})


def register_exception_handlers(app: FastAPI) -> None:
    """注册 API 统一异常处理器。"""

    @app.exception_handler(APIError)
    async def handle_api_error(
        request: Request,
        exc: APIError,
    ) -> JSONResponse:
        request_id = exc.request_id or request.headers.get("X-Request-ID")
        payload = ErrorResponse(
            error_code=exc.error_code,
            message=exc.message,
            request_id=request_id,
            details=exc.details,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger = logging.getLogger("kg_rag_agent.app")
        logger.exception("Unhandled API error", exc_info=exc)
        payload = ErrorResponse(
            error_code="internal_server_error",
            message="The request could not be completed.",
            request_id=request.headers.get("X-Request-ID"),
        )
        return JSONResponse(
            status_code=500,
            content=payload.model_dump(mode="json"),
        )


__all__ = ["APIError", "register_exception_handlers"]
