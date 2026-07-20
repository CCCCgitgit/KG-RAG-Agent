# -*- coding: utf-8 -*-
"""Tools 层统一 Schema。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence


class ToolPermission(str, Enum):
    """工具权限分类。"""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    NETWORK = "network"


def normalize_permissions(
    values: Optional[Sequence[str | ToolPermission]],
) -> frozenset[ToolPermission]:
    result: set[ToolPermission] = set()
    for value in values or ():
        if isinstance(value, ToolPermission):
            result.add(value)
            continue
        normalized = str(value or "").strip().lower()
        if not normalized:
            continue
        try:
            result.add(ToolPermission(normalized))
        except ValueError as exc:
            raise ValueError(f"Unsupported tool permission: {value}") from exc
    return frozenset(result)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """一个可注册工具的公开描述。"""

    name: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    permissions: frozenset[ToolPermission] = field(default_factory=frozenset)
    timeout_seconds: float = 30.0
    max_result_chars: int = 100_000
    enabled: bool = True
    destructive: bool = False
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized_name = str(self.name or "").strip()
        if not normalized_name:
            raise ValueError("tool name must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.max_result_chars <= 0:
            raise ValueError("max_result_chars must be > 0")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(
            self,
            "permissions",
            normalize_permissions(tuple(self.permissions)),
        )
        object.__setattr__(self, "tags", tuple(str(item) for item in self.tags))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "permissions": sorted(item.value for item in self.permissions),
            "timeout_seconds": self.timeout_seconds,
            "max_result_chars": self.max_result_chars,
            "enabled": self.enabled,
            "destructive": self.destructive,
            "tags": list(self.tags),
        }


@dataclass(slots=True)
class ToolCallContext:
    """单次请求中的工具调用上下文。"""

    request_id: str = ""
    session_id: str = ""
    user_id: str = ""
    allowed_tools: Optional[frozenset[str]] = None
    allowed_permissions: Optional[frozenset[ToolPermission]] = None
    max_calls: int = 20
    call_count: int = 0
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        request_id: str = "",
        session_id: str = "",
        user_id: str = "",
        allowed_tools: Optional[Sequence[str]] = None,
        allowed_permissions: Optional[Sequence[str | ToolPermission]] = None,
        max_calls: int = 20,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "ToolCallContext":
        tools = None
        if allowed_tools is not None:
            tools = frozenset(
                str(item).strip()
                for item in allowed_tools
                if str(item).strip()
            )
        permissions = None
        if allowed_permissions is not None:
            permissions = normalize_permissions(allowed_permissions)
        if max_calls <= 0:
            raise ValueError("max_calls must be > 0")
        return cls(
            request_id=str(request_id or ""),
            session_id=str(session_id or ""),
            user_id=str(user_id or ""),
            allowed_tools=tools,
            allowed_permissions=permissions,
            max_calls=int(max_calls),
            metadata=dict(metadata or {}),
        )


@dataclass(slots=True)
class ToolResult:
    """工具执行的标准结果。"""

    ok: bool
    tool_name: str
    data: Any = None
    error: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tool_name": self.tool_name,
            "data": self.data,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ToolRegistration:
    """Registry 内部保存的工具实现。"""

    spec: ToolSpec
    handler: Callable[..., Any]


__all__ = [
    "ToolPermission",
    "ToolSpec",
    "ToolCallContext",
    "ToolResult",
    "ToolRegistration",
    "normalize_permissions",
]
