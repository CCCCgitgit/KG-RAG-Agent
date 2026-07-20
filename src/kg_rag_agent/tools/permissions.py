# -*- coding: utf-8 -*-
"""Tools 层权限策略。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .errors import ToolPermissionError
from .schemas import (
    ToolCallContext,
    ToolPermission,
    ToolSpec,
    normalize_permissions,
)


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """应用级默认工具权限策略。"""

    allowed_permissions: frozenset[ToolPermission] = field(
        default_factory=lambda: frozenset(
            {ToolPermission.READ, ToolPermission.EXECUTE}
        )
    )
    allowed_tools: Optional[frozenset[str]] = None
    denied_tools: frozenset[str] = field(default_factory=frozenset)
    max_calls_per_request: int = 20
    allow_destructive: bool = False

    @classmethod
    def create(
        cls,
        *,
        allowed_permissions: Optional[Sequence[str | ToolPermission]] = None,
        allowed_tools: Optional[Sequence[str]] = None,
        denied_tools: Optional[Sequence[str]] = None,
        max_calls_per_request: int = 20,
        allow_destructive: bool = False,
    ) -> "ToolPolicy":
        permissions = (
            normalize_permissions(allowed_permissions)
            if allowed_permissions is not None
            else frozenset({ToolPermission.READ, ToolPermission.EXECUTE})
        )
        allow_set = None
        if allowed_tools is not None:
            allow_set = frozenset(
                str(item).strip()
                for item in allowed_tools
                if str(item).strip()
            )
        deny_set = frozenset(
            str(item).strip()
            for item in (denied_tools or ())
            if str(item).strip()
        )
        if max_calls_per_request <= 0:
            raise ValueError("max_calls_per_request must be > 0")
        return cls(
            allowed_permissions=permissions,
            allowed_tools=allow_set,
            denied_tools=deny_set,
            max_calls_per_request=int(max_calls_per_request),
            allow_destructive=bool(allow_destructive),
        )

    def authorize(
        self,
        spec: ToolSpec,
        context: Optional[ToolCallContext] = None,
    ) -> None:
        """校验工具、权限和调用预算。"""

        name = spec.name
        if not spec.enabled:
            raise ToolPermissionError(
                "Tool is disabled.",
                tool_name=name,
            )
        if name in self.denied_tools:
            raise ToolPermissionError(
                "Tool is denied by policy.",
                tool_name=name,
            )
        if self.allowed_tools is not None and name not in self.allowed_tools:
            raise ToolPermissionError(
                "Tool is not in the policy allowlist.",
                tool_name=name,
            )
        if spec.destructive and not self.allow_destructive:
            raise ToolPermissionError(
                "Destructive tool calls are disabled.",
                tool_name=name,
            )
        missing = spec.permissions.difference(self.allowed_permissions)
        if missing:
            raise ToolPermissionError(
                "Tool requires permissions not granted by policy.",
                tool_name=name,
                details={"missing_permissions": sorted(x.value for x in missing)},
            )

        if context is None:
            return
        if context.allowed_tools is not None and name not in context.allowed_tools:
            raise ToolPermissionError(
                "Tool is not allowed for this request.",
                tool_name=name,
            )
        if context.allowed_permissions is not None:
            missing = spec.permissions.difference(context.allowed_permissions)
            if missing:
                raise ToolPermissionError(
                    "Tool permission is not allowed for this request.",
                    tool_name=name,
                    details={
                        "missing_permissions": sorted(x.value for x in missing)
                    },
                )
        effective_max = min(context.max_calls, self.max_calls_per_request)
        if context.call_count >= effective_max:
            raise ToolPermissionError(
                "Tool call budget exceeded.",
                tool_name=name,
                details={"max_calls": effective_max},
            )


DEFAULT_TOOL_POLICY = ToolPolicy()


__all__ = ["ToolPolicy", "DEFAULT_TOOL_POLICY"]
