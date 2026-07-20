# -*- coding: utf-8 -*-
"""内部工具注册、权限校验和统一执行入口。"""

from __future__ import annotations

import inspect
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from threading import RLock
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Optional, Sequence, TYPE_CHECKING

from .errors import (
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolTimeoutError,
    ToolValidationError,
)
from .file_tools import FileTools
from .kg_tools import KGTools
from .permissions import DEFAULT_TOOL_POLICY, ToolPolicy
from .schemas import (
    ToolCallContext,
    ToolPermission,
    ToolRegistration,
    ToolResult,
    ToolSpec,
)
from .vector_tools import VectorTools

if TYPE_CHECKING:
    from kg_rag_agent.runtime.context import RuntimeContext


class ToolRegistry:
    """项目内部唯一的工具注册与调用入口。"""

    def __init__(
        self,
        *,
        policy: Optional[ToolPolicy] = None,
        max_workers: int = 4,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be > 0")
        self.policy = policy or DEFAULT_TOOL_POLICY
        self._registrations: dict[str, ToolRegistration] = {}
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="kg-rag-tool",
        )
        self._closed = False

    @property
    def is_closed(self) -> bool:
        return self._closed

    def register(
        self,
        spec: ToolSpec,
        handler: Callable[..., Any],
        *,
        overwrite: bool = False,
    ) -> None:
        if self._closed:
            raise RuntimeError("ToolRegistry has already been closed.")
        if not callable(handler):
            raise TypeError("tool handler must be callable")
        with self._lock:
            if spec.name in self._registrations and not overwrite:
                raise KeyError(f"Tool already registered: {spec.name}")
            self._registrations[spec.name] = ToolRegistration(
                spec=spec,
                handler=handler,
            )

    def register_function(
        self,
        name: str,
        handler: Callable[..., Any],
        *,
        description: str,
        input_schema: Optional[Mapping[str, Any]] = None,
        permissions: Sequence[ToolPermission | str] = (),
        timeout_seconds: float = 30.0,
        max_result_chars: int = 100_000,
        destructive: bool = False,
        tags: Sequence[str] = (),
        overwrite: bool = False,
    ) -> None:
        self.register(
            ToolSpec(
                name=name,
                description=description,
                input_schema=dict(input_schema or {}),
                permissions=frozenset(permissions),
                timeout_seconds=timeout_seconds,
                max_result_chars=max_result_chars,
                destructive=destructive,
                tags=tuple(tags),
            ),
            handler,
            overwrite=overwrite,
        )

    def unregister(self, name: str) -> ToolRegistration:
        normalized = _normalize_tool_name(name)
        with self._lock:
            try:
                return self._registrations.pop(normalized)
            except KeyError as exc:
                raise ToolNotFoundError(
                    "Tool is not registered.",
                    tool_name=normalized,
                ) from exc

    def has(self, name: str) -> bool:
        normalized = _normalize_tool_name(name)
        with self._lock:
            return normalized in self._registrations

    def get(self, name: str) -> ToolRegistration:
        normalized = _normalize_tool_name(name)
        with self._lock:
            registration = self._registrations.get(normalized)
        if registration is None:
            raise ToolNotFoundError(
                "Tool is not registered.",
                tool_name=normalized,
            )
        return registration

    def get_spec(self, name: str) -> ToolSpec:
        return self.get(name).spec

    def list_specs(
        self,
        *,
        include_disabled: bool = False,
        tags: Optional[Sequence[str]] = None,
    ) -> list[dict[str, Any]]:
        required_tags = {str(item) for item in (tags or ())}
        with self._lock:
            registrations = list(self._registrations.values())
        specs: list[dict[str, Any]] = []
        for registration in registrations:
            spec = registration.spec
            if not include_disabled and not spec.enabled:
                continue
            if required_tags and not required_tags.intersection(spec.tags):
                continue
            specs.append(spec.to_dict())
        return sorted(specs, key=lambda item: item["name"])

    def invoke(
        self,
        name: str,
        arguments: Optional[Mapping[str, Any]] = None,
        *,
        context: Optional[ToolCallContext] = None,
        raise_errors: bool = False,
    ) -> ToolResult:
        """执行工具并返回统一结果。"""

        started = perf_counter()
        normalized = _normalize_tool_name(name)
        try:
            registration = self.get(normalized)
            spec = registration.spec
            self.policy.authorize(spec, context)
            payload = _validate_arguments(spec, arguments or {})
            if context is not None:
                context.call_count += 1

            future = self._executor.submit(
                _invoke_handler,
                registration.handler,
                payload,
                context,
            )
            try:
                data = future.result(timeout=spec.timeout_seconds)
            except FutureTimeoutError as exc:
                future.cancel()
                raise ToolTimeoutError(
                    "Tool execution timed out.",
                    tool_name=normalized,
                    details={"timeout_seconds": spec.timeout_seconds},
                ) from exc

            data, truncated = _limit_result(data, spec.max_result_chars)
            return ToolResult(
                ok=True,
                tool_name=normalized,
                data=data,
                metadata={
                    "duration_ms": round((perf_counter() - started) * 1000, 3),
                    "truncated": truncated,
                    "permissions": sorted(x.value for x in spec.permissions),
                },
            )
        except ToolError as exc:
            if raise_errors:
                raise
            return ToolResult(
                ok=False,
                tool_name=normalized,
                error=exc.to_dict(),
                metadata={
                    "duration_ms": round((perf_counter() - started) * 1000, 3),
                },
            )
        except Exception as exc:
            wrapped = ToolExecutionError(
                str(exc) or exc.__class__.__name__,
                tool_name=normalized,
                details={"exception_type": exc.__class__.__name__},
            )
            if raise_errors:
                raise wrapped from exc
            return ToolResult(
                ok=False,
                tool_name=normalized,
                error=wrapped.to_dict(),
                metadata={
                    "duration_ms": round((perf_counter() - started) * 1000, 3),
                },
            )

    def invoke_data(
        self,
        name: str,
        arguments: Optional[Mapping[str, Any]] = None,
        *,
        context: Optional[ToolCallContext] = None,
    ) -> Any:
        """执行工具；失败时抛异常，成功时只返回 data。"""

        result = self.invoke(
            name,
            arguments,
            context=context,
            raise_errors=True,
        )
        return result.data

    def close(self) -> None:
        if self._closed:
            return
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._closed = True

    def __enter__(self) -> "ToolRegistry":
        if self._closed:
            raise RuntimeError("ToolRegistry has already been closed.")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def build_default_tool_registry(
    *,
    runtime: Optional["RuntimeContext"] = None,
    file_base_dir: Optional[str] = None,
    policy: Optional[ToolPolicy] = None,
    max_workers: int = 4,
) -> ToolRegistry:
    """注册项目内置 KG、Retrieval 和文件工具。"""

    registry = ToolRegistry(policy=policy, max_workers=max_workers)
    if file_base_dir is None:
        if runtime is not None:
            file_base_dir = str(runtime.settings.project_root / "data")
        else:
            file_base_dir = str(Path.cwd())
    file_tools = FileTools(
        base_dir=file_base_dir,
        enforce_base_dir=True,
        allow_delete=True,
    )
    kg_tools = KGTools(runtime=runtime)
    vector_tools = VectorTools(runtime=runtime)

    _register_file_tools(registry, file_tools)
    _register_kg_tools(registry, kg_tools)
    _register_vector_tools(registry, vector_tools)

    if runtime is not None:
        runtime.register("tool_registry", registry, overwrite=True)
    return registry


def _register_file_tools(registry: ToolRegistry, tools: FileTools) -> None:
    obj = _object_schema
    registry.register_function(
        "file.read_text", tools.read_text,
        description="读取沙箱内文本文件。",
        input_schema=obj({"file_path": _string(), "max_chars": _integer(0)}, ["file_path"]),
        permissions=(ToolPermission.READ,), tags=("file", "read"),
    )
    registry.register_function(
        "file.read_json", tools.read_json,
        description="读取沙箱内 JSON 文件。",
        input_schema=obj({"file_path": _string()}, ["file_path"]),
        permissions=(ToolPermission.READ,), tags=("file", "read"),
    )
    registry.register_function(
        "file.file_info", tools.file_info,
        description="获取沙箱内文件或目录信息。",
        input_schema=obj({"file_path": _string(), "include_hash": _boolean()}, ["file_path"]),
        permissions=(ToolPermission.READ,), tags=("file", "read"),
    )
    registry.register_function(
        "file.list_files", tools.list_files,
        description="列出沙箱目录中的文件。",
        input_schema=obj({
            "directory": _string(), "pattern": _string(), "recursive": _boolean(),
            "include_dirs": _boolean(), "limit": _integer(1),
        }, ["directory"]),
        permissions=(ToolPermission.READ,), tags=("file", "read"),
    )
    registry.register_function(
        "file.load_texts", tools.load_texts,
        description="批量读取沙箱目录中的文本文件。",
        input_schema=obj({
            "directory": _string(), "pattern": _string(), "recursive": _boolean(),
            "limit": _integer(1), "max_chars_per_file": _integer(0),
        }, ["directory"]),
        permissions=(ToolPermission.READ,), tags=("file", "read"),
    )
    registry.register_function(
        "file.ensure_dir", tools.ensure_dir,
        description="在沙箱内创建目录。",
        input_schema=obj({"directory": _string()}, ["directory"]),
        permissions=(ToolPermission.WRITE,), tags=("file", "write"),
    )
    registry.register_function(
        "file.write_text", tools.write_text,
        description="向沙箱内写入文本文件。",
        input_schema=obj({
            "file_path": _string(), "content": _string(), "append": _boolean(),
            "create_parent": _boolean(),
        }, ["file_path", "content"]),
        permissions=(ToolPermission.WRITE,), tags=("file", "write"),
    )
    registry.register_function(
        "file.write_json", tools.write_json,
        description="向沙箱内写入 JSON 文件。",
        input_schema=obj({
            "file_path": _string(), "data": {}, "ensure_ascii": _boolean(),
            "indent": _integer(0), "create_parent": _boolean(),
        }, ["file_path", "data"]),
        permissions=(ToolPermission.WRITE,), tags=("file", "write"),
    )
    registry.register_function(
        "file.copy", tools.copy_file,
        description="在沙箱内复制文件。",
        input_schema=obj({
            "source": _string(), "target": _string(), "overwrite": _boolean(),
            "create_parent": _boolean(),
        }, ["source", "target"]),
        permissions=(ToolPermission.READ, ToolPermission.WRITE), tags=("file", "write"),
    )
    registry.register_function(
        "file.move", tools.move_file,
        description="在沙箱内移动文件。",
        input_schema=obj({
            "source": _string(), "target": _string(), "overwrite": _boolean(),
            "create_parent": _boolean(),
        }, ["source", "target"]),
        permissions=(ToolPermission.READ, ToolPermission.WRITE, ToolPermission.DELETE),
        destructive=True, tags=("file", "write", "destructive"),
    )
    registry.register_function(
        "file.delete", tools.delete_file,
        description="删除沙箱内文件。",
        input_schema=obj({"file_path": _string(), "missing_ok": _boolean()}, ["file_path"]),
        permissions=(ToolPermission.DELETE,), destructive=True,
        tags=("file", "destructive"),
    )


def _register_kg_tools(registry: ToolRegistry, tools: KGTools) -> None:
    common = (ToolPermission.READ, ToolPermission.EXECUTE)
    registry.register_function(
        "kg.graph_info", tools.graph_info,
        description="获取知识图谱统计信息。",
        input_schema=_object_schema({}, []), permissions=common, tags=("kg", "read"),
    )
    registry.register_function(
        "kg.relation", tools.relation,
        description="查询两个实体之间的直接关系。",
        input_schema=_object_schema({
            "source": _string(), "target": _string(), "include_reverse": _boolean(),
            "max_results": _integer(1), "include_evidence": _boolean(),
        }, ["source", "target"]), permissions=common, tags=("kg", "read"),
    )
    registry.register_function(
        "kg.path", tools.path,
        description="查询两个实体之间的多跳路径。",
        input_schema=_object_schema({
            "source": _string(), "target": _string(), "max_paths": _integer(1),
            "max_path_length": _integer(1), "include_evidence": _boolean(),
        }, ["source", "target"]), permissions=common, tags=("kg", "read"),
    )
    registry.register_function(
        "kg.neighbors", tools.neighbors,
        description="查询实体的一跳邻居。",
        input_schema=_object_schema({
            "entity": _string(), "max_neighbors": _integer(1),
            "direction": {"type": "string", "enum": ["in", "out", "both"]},
            "include_evidence": _boolean(),
        }, ["entity"]), permissions=common, tags=("kg", "read"),
    )
    registry.register_function(
        "kg.subgraph", tools.subgraph,
        description="围绕实体抽取局部子图。",
        input_schema=_object_schema({
            "entities": _array(), "max_depth": _integer(1), "max_nodes": _integer(1),
            "max_edges": _integer(1), "direction": {"type": "string", "enum": ["in", "out", "both"]},
            "include_evidence": _boolean(),
        }, ["entities"]), permissions=common, tags=("kg", "read"),
    )
    registry.register_function(
        "kg.retrieve", tools.retrieve,
        description="围绕实体集合执行综合知识图谱检索。",
        input_schema=_object_schema({
            "entities": _array(), "max_neighbors": _integer(1), "max_paths": _integer(1),
            "max_path_length": _integer(1), "include_subgraph": _boolean(),
            "subgraph_depth": _integer(1), "max_evidence": _integer(1),
        }, ["entities"]), permissions=common, tags=("kg", "read"),
    )


def _register_vector_tools(registry: ToolRegistry, tools: VectorTools) -> None:
    common = (ToolPermission.READ, ToolPermission.EXECUTE)
    registry.register_function(
        "vector.embed_text", tools.embed_text,
        description="将文本编码为向量。",
        input_schema=_object_schema({"text": _string()}, ["text"]),
        permissions=(ToolPermission.EXECUTE,), max_result_chars=300_000,
        tags=("retrieval", "embedding"),
    )
    registry.register_function(
        "vector.search_entities", tools.search_entities,
        description="从实体向量库召回实体候选。",
        input_schema=_object_schema({
            "query": _string(), "top_k": _integer(1), "where": {"type": "object"},
        }, ["query"]), permissions=common, tags=("retrieval", "read"),
    )
    registry.register_function(
        "vector.search", tools.search_vectors,
        description="从通用向量库召回文档。",
        input_schema=_object_schema({
            "query": _string(), "top_k": _integer(1), "where": {"type": "object"},
            "min_score": _number(0),
        }, ["query"]), permissions=common, tags=("retrieval", "read"),
    )
    registry.register_function(
        "vector.hybrid_search", tools.hybrid_search,
        description="执行向量、实体和关键词混合召回。",
        input_schema=_object_schema({
            "query": _string(), "top_k": _integer(1), "vector_top_k": _integer(1),
            "entity_top_k": _integer(1), "keyword_top_k": _integer(1),
            "vector_where": {"type": "object"}, "entity_where": {"type": "object"},
            "keyword_corpus": _array(), "min_score": _number(0),
        }, ["query"]), permissions=common, tags=("retrieval", "read"),
    )
    registry.register_function(
        "vector.rerank", tools.rerank,
        description="对召回候选结果执行重排序。",
        input_schema=_object_schema({
            "query": _string(), "candidates": _array(), "top_k": _integer(1),
            "entities": _array(), "min_score": _number(0),
        }, ["query", "candidates"]), permissions=common, tags=("retrieval", "read"),
    )
    registry.register_function(
        "vector.retrieve_and_rerank", tools.retrieve_and_rerank,
        description="先执行混合召回，再执行重排序。",
        input_schema=_object_schema({
            "query": _string(), "top_k": _integer(1), "final_top_k": _integer(1),
            "entities": _array(), "keyword_corpus": _array(), "min_score": _number(0),
        }, ["query"]), permissions=common, tags=("retrieval", "read"),
    )
    registry.register_function(
        "vector.add_entities", tools.add_entities,
        description="向实体向量库写入实体，仅用于离线构建或受控维护。",
        input_schema=_object_schema({
            "entities": _array(), "batch_size": _integer(1),
        }, ["entities"]), permissions=(ToolPermission.WRITE, ToolPermission.EXECUTE),
        tags=("retrieval", "write"),
    )


def _invoke_handler(
    handler: Callable[..., Any],
    arguments: Mapping[str, Any],
    context: Optional[ToolCallContext],
) -> Any:
    signature = inspect.signature(handler)
    if "tool_context" in signature.parameters:
        return handler(**dict(arguments), tool_context=context)
    return handler(**dict(arguments))


def _validate_arguments(
    spec: ToolSpec,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(arguments, Mapping):
        raise ToolValidationError(
            "Tool arguments must be an object.",
            tool_name=spec.name,
        )
    payload = dict(arguments)
    schema = dict(spec.input_schema or {})
    if not schema:
        return payload
    if schema.get("type") not in (None, "object"):
        raise ToolValidationError(
            "Only object input schemas are supported by the registry.",
            tool_name=spec.name,
        )
    required = set(schema.get("required", []) or [])
    missing = sorted(key for key in required if key not in payload)
    if missing:
        raise ToolValidationError(
            "Required tool arguments are missing.",
            tool_name=spec.name,
            details={"missing": missing},
        )
    properties = schema.get("properties", {}) or {}
    if schema.get("additionalProperties") is False:
        unknown = sorted(key for key in payload if key not in properties)
        if unknown:
            raise ToolValidationError(
                "Unknown tool arguments are not allowed.",
                tool_name=spec.name,
                details={"unknown": unknown},
            )
    for key, value in payload.items():
        field_schema = properties.get(key)
        if field_schema:
            _validate_value(spec.name, key, value, field_schema)
    return payload


def _validate_value(
    tool_name: str,
    field: str,
    value: Any,
    schema: Mapping[str, Any],
) -> None:
    if value is None:
        return
    expected = schema.get("type")
    valid = True
    if expected == "string":
        valid = isinstance(value, str)
    elif expected == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif expected == "number":
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif expected == "boolean":
        valid = isinstance(value, bool)
    elif expected == "array":
        valid = isinstance(value, (list, tuple))
    elif expected == "object":
        valid = isinstance(value, Mapping)
    if not valid:
        raise ToolValidationError(
            f"Invalid type for argument: {field}",
            tool_name=tool_name,
            details={"field": field, "expected": expected},
        )
    if "enum" in schema and value not in schema["enum"]:
        raise ToolValidationError(
            f"Unsupported value for argument: {field}",
            tool_name=tool_name,
            details={"field": field, "allowed": list(schema["enum"])},
        )
    if value is not None and "minimum" in schema and value < schema["minimum"]:
        raise ToolValidationError(
            f"Argument is below minimum: {field}",
            tool_name=tool_name,
            details={"field": field, "minimum": schema["minimum"]},
        )


def _limit_result(data: Any, max_chars: int) -> tuple[Any, bool]:
    try:
        serialized = json.dumps(data, ensure_ascii=False, default=str)
    except Exception:
        serialized = repr(data)
    if len(serialized) <= max_chars:
        return data, False
    return {
        "truncated": True,
        "original_chars": len(serialized),
        "preview": serialized[:max_chars],
    }, True


def _normalize_tool_name(name: str) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        raise ToolValidationError("tool name must not be empty")
    return normalized


def _object_schema(
    properties: Mapping[str, Any],
    required: Sequence[str],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def _string() -> dict[str, Any]:
    return {"type": "string"}


def _integer(minimum: Optional[int] = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "integer"}
    if minimum is not None:
        schema["minimum"] = minimum
    return schema


def _number(minimum: Optional[float] = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "number"}
    if minimum is not None:
        schema["minimum"] = minimum
    return schema


def _boolean() -> dict[str, Any]:
    return {"type": "boolean"}


def _array() -> dict[str, Any]:
    return {"type": "array"}


__all__ = ["ToolRegistry", "build_default_tool_registry"]
