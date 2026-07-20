# -*- coding: utf-8 -*-
"""Tools 层统一异常。"""

from __future__ import annotations

from typing import Any, Mapping, Optional


class ToolError(RuntimeError):
    """Tools 层基础异常。"""

    code = "tool_error"

    def __init__(
        self,
        message: str,
        *,
        tool_name: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(str(message))
        self.tool_name = tool_name
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "tool_name": self.tool_name,
            "details": dict(self.details),
        }


class ToolValidationError(ToolError):
    """工具名、参数或 Schema 不合法。"""

    code = "tool_validation_error"


class ToolPermissionError(ToolError):
    """工具或权限未授权。"""

    code = "tool_permission_error"


class ToolNotFoundError(ToolError):
    """工具未注册。"""

    code = "tool_not_found"


class ToolExecutionError(ToolError):
    """工具执行失败。"""

    code = "tool_execution_error"


class ToolTimeoutError(ToolExecutionError):
    """工具执行超时。"""

    code = "tool_timeout"


class ToolResultTooLargeError(ToolExecutionError):
    """工具结果超过限制。"""

    code = "tool_result_too_large"


class ToolPathError(ToolPermissionError):
    """文件路径越过允许的沙箱边界。"""

    code = "tool_path_error"


__all__ = [
    "ToolError",
    "ToolValidationError",
    "ToolPermissionError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "ToolResultTooLargeError",
    "ToolPathError",
]
