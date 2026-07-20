# -*- coding: utf-8 -*-
"""存活、就绪和状态接口。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends

from kg_rag_agent.app.dependencies import get_agent_service, get_api_settings
from kg_rag_agent.app.errors import APIError
from kg_rag_agent.app.schemas.response import (
    HealthResponse,
    ReadinessResponse,
    StatusResponse,
)
from kg_rag_agent.app.settings import APISettings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(
    settings: APISettings = Depends(get_api_settings),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="kg-rag-agent",
        version=settings.version,
        timestamp=_utc_now(),
    )


@router.get("/ready", response_model=ReadinessResponse)
def readiness_check(
    service: Any = Depends(get_agent_service),
) -> ReadinessResponse:
    try:
        health = dict(service.health_check())
    except Exception as exc:
        raise APIError(
            status_code=503,
            error_code="readiness_check_failed",
            message="Agent service is not ready.",
        ) from exc

    ready = bool(health.get("ok", False))
    if not ready:
        raise APIError(
            status_code=503,
            error_code="agent_service_not_ready",
            message="Agent service is not ready.",
            details={"health": health},
        )

    return ReadinessResponse(
        ready=True,
        service="kg-rag-agent",
        details=health,
        timestamp=_utc_now(),
    )


@router.get("/status", response_model=StatusResponse)
def status_check() -> StatusResponse:
    return StatusResponse()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "router",
    "health_check",
    "readiness_check",
    "status_check",
]
