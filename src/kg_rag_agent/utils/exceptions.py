# -*- coding: utf-8 -*-
"""Project-wide, infrastructure-level exceptions."""

from __future__ import annotations

from typing import Any, Mapping, Optional


class KGRAGError(Exception):
    """Base exception for project infrastructure errors."""

    code = "kg_rag_error"

    def __init__(
        self,
        message: str,
        *,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(str(message))
        self.message = str(message)
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }


class ConfigurationError(KGRAGError):
    code = "configuration_error"


class ConfigurationValidationError(ConfigurationError):
    code = "configuration_validation_error"


class DependencyUnavailableError(KGRAGError):
    code = "dependency_unavailable"


class ResourceNotFoundError(KGRAGError):
    code = "resource_not_found"


class PermissionDeniedError(KGRAGError):
    code = "permission_denied"


class OperationTimeoutError(KGRAGError):
    code = "operation_timeout"
