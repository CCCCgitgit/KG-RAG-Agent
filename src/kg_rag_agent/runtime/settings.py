# -*- coding: utf-8 -*-
"""
runtime/settings.py

统一运行时配置对象。

职责：
    1. 复用现有 utils.config_loader 加载 YAML 与环境变量。
    2. 为 RuntimeContext 提供只读、可校验的配置视图。
    3. 兼容当前 graph.yaml 中的嵌套节点配置与旧的顶层节点配置。
    4. 统一项目路径解析，避免各模块重复判断当前工作目录。

本模块不负责创建 LLM、图谱、向量库或 LangGraph。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from kg_rag_agent.utils.config_loader import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONFIG_FILES,
    apply_env_overrides,
    deep_merge,
    fill_default_config,
    get_project_root,
    load_all_configs,
    load_yaml_file,
    resolve_project_path,
    validate_config,
)


# 当前 graph.yaml 中仍位于 graph.* 下的节点级配置。
# RuntimeSettings.section() 会把旧顶层默认值与 graph.yaml 中的值合并，
# 从而为后续 Node 重构提供一个稳定入口。
GRAPH_NODE_SECTIONS: Tuple[str, ...] = (
    "router",
    "mention_extraction",
    "entity_linking",
    "entity_grounding",
    "semantic_scoring",
    "reasoning",
    "generation",
)

SENSITIVE_KEYWORDS: Tuple[str, ...] = (
    "api_key",
    "token",
    "password",
    "secret",
    "private_key",
    "credential",
)


class RuntimeSettingsError(ValueError):
    """运行时配置无效。"""


def _freeze(value: Any) -> Any:
    """递归转换为只读结构。"""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )

    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)

    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)

    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)

    return value


def _thaw(value: Any) -> Any:
    """将只读结构递归转换为普通 Python 容器。"""

    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}

    if isinstance(value, tuple):
        return [_thaw(item) for item in value]

    if isinstance(value, frozenset):
        return {_thaw(item) for item in value}

    return copy.deepcopy(value)


def _lookup(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """按点号路径读取 Mapping。"""

    if not path:
        return mapping

    current: Any = mapping

    for part in str(path).split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]

    return current


def _redact(value: Any, key_path: str = "") -> Any:
    """生成可记录日志的脱敏配置副本。"""

    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            current_path = f"{key_path}.{key_text}" if key_path else key_text

            if any(keyword in lowered for keyword in SENSITIVE_KEYWORDS):
                result[key_text] = "***REDACTED***"
            else:
                result[key_text] = _redact(item, current_path)
        return result

    if isinstance(value, (tuple, list)):
        return [_redact(item, key_path) for item in value]

    return copy.deepcopy(value)


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """
    统一运行时配置。

    `data` 在对象内部为递归只读结构。需要传给旧模块时使用
    :meth:`to_dict` 获取普通字典副本，避免共享配置被意外修改。
    """

    data: Mapping[str, Any]
    project_root: Path
    config_dir: Path
    source_files: Tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        root = Path(self.project_root).expanduser().resolve()
        config_dir = Path(self.config_dir).expanduser()
        if not config_dir.is_absolute():
            config_dir = root / config_dir

        object.__setattr__(self, "project_root", root)
        object.__setattr__(self, "config_dir", config_dir.resolve())
        object.__setattr__(self, "data", _freeze(dict(self.data or {})))
        object.__setattr__(
            self,
            "source_files",
            tuple(Path(path).expanduser().resolve() for path in self.source_files),
        )

    @classmethod
    def load(
        cls,
        config_path: Optional[str | Path] = None,
        *,
        config_dir: Optional[str | Path] = None,
        apply_env: bool = True,
        validate: bool = True,
        ignore_missing: bool = True,
        project_root: Optional[str | Path] = None,
    ) -> "RuntimeSettings":
        """从项目配置文件创建 RuntimeSettings。"""

        root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else get_project_root().resolve()
        )

        resolved_config_dir = cls._resolve_config_dir(
            root=root,
            config_dir=config_dir,
        )

        source_files: Tuple[Path, ...]

        if config_path is not None:
            resolved_path = cls._resolve_input_path(root, config_path)
            section_name = resolved_path.stem
            file_data = load_yaml_file(resolved_path)
            config = fill_default_config({section_name: file_data})
            if apply_env:
                config = apply_env_overrides(config)
            source_files = (resolved_path,)
        else:
            config = load_all_configs(
                config_dir=resolved_config_dir,
                apply_env=apply_env,
                ignore_missing=ignore_missing,
            )
            source_files = tuple(
                (resolved_config_dir / filename).resolve()
                for filename in DEFAULT_CONFIG_FILES
                if (resolved_config_dir / filename).exists()
            )

        instance = cls(
            data=config,
            project_root=root,
            config_dir=resolved_config_dir,
            source_files=source_files,
        )

        if validate:
            instance.validate()

        return instance

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
        *,
        project_root: Optional[str | Path] = None,
        config_dir: Optional[str | Path] = None,
        apply_env: bool = True,
        validate: bool = True,
    ) -> "RuntimeSettings":
        """从已有配置字典创建 RuntimeSettings。"""

        root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else get_project_root().resolve()
        )
        resolved_config_dir = cls._resolve_config_dir(root, config_dir)

        normalized = fill_default_config(copy.deepcopy(dict(config or {})))
        if apply_env:
            normalized = apply_env_overrides(normalized)

        instance = cls(
            data=normalized,
            project_root=root,
            config_dir=resolved_config_dir,
            source_files=(),
        )

        if validate:
            instance.validate()

        return instance

    @staticmethod
    def _resolve_config_dir(
        root: Path,
        config_dir: Optional[str | Path],
    ) -> Path:
        candidate = Path(config_dir or DEFAULT_CONFIG_DIR).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate.resolve()

    @staticmethod
    def _resolve_input_path(root: Path, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()

        cwd_candidate = Path.cwd() / candidate
        if cwd_candidate.exists():
            return cwd_candidate.resolve()

        return (root / candidate).resolve()

    def validate(self) -> None:
        """执行现有项目的基础配置校验。"""

        ok, errors = validate_config(self.to_dict())
        if not ok:
            message = "; ".join(errors) if errors else "unknown config error"
            raise RuntimeSettingsError(message)

    def get(self, path: str, default: Any = None) -> Any:
        """按点号路径读取配置，返回只读值。"""

        return _lookup(self.data, path, default)

    def require(self, path: str) -> Any:
        """读取必填配置；不存在时抛出 RuntimeSettingsError。"""

        marker = object()
        value = self.get(path, marker)
        if value is marker:
            raise RuntimeSettingsError(f"missing required config: {path}")
        return value

    def section(self, name: str) -> Dict[str, Any]:
        """
        获取配置段的普通字典副本。

        对 Node 级配置同时兼容：
            config["generation"]
            config["graph"]["generation"]

        合并时 graph.yaml 中的嵌套值优先。
        """

        name = str(name or "").strip()
        if not name:
            return {}

        direct = self.get(name, {})
        direct_dict = _thaw(direct) if isinstance(direct, Mapping) else {}

        if name not in GRAPH_NODE_SECTIONS:
            return direct_dict

        nested = self.get(f"graph.{name}", {})
        nested_dict = _thaw(nested) if isinstance(nested, Mapping) else {}

        return deep_merge(direct_dict, nested_dict)

    def to_dict(self) -> Dict[str, Any]:
        """返回可修改的深拷贝，供旧模块兼容使用。"""

        value = _thaw(self.data)
        return value if isinstance(value, dict) else {}

    def safe_dict(self) -> Dict[str, Any]:
        """返回适合日志和诊断输出的脱敏配置。"""

        value = _redact(self.data)
        return value if isinstance(value, dict) else {}

    def resolve_path(
        self,
        path: str | Path,
        *,
        must_exist: bool = False,
    ) -> Path:
        """按项目根目录解析路径。"""

        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_root / candidate

        resolved = candidate.resolve()

        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Project path not found: {resolved}")

        return resolved

    def with_overrides(
        self,
        overrides: Mapping[str, Any],
        *,
        apply_env: bool = False,
        validate: bool = True,
    ) -> "RuntimeSettings":
        """
        创建合并后的新配置对象。

        该方法面向系统启动或测试，不应直接接收未经白名单过滤的用户请求。
        """

        merged = deep_merge(self.to_dict(), dict(overrides or {}))
        if apply_env:
            merged = apply_env_overrides(merged)

        return RuntimeSettings.from_mapping(
            merged,
            project_root=self.project_root,
            config_dir=self.config_dir,
            apply_env=False,
            validate=validate,
        )

    @property
    def model(self) -> Dict[str, Any]:
        return self.section("model")

    @property
    def graph(self) -> Dict[str, Any]:
        return self.section("graph")

    @property
    def kg(self) -> Dict[str, Any]:
        return self.section("kg")

    @property
    def retrieval(self) -> Dict[str, Any]:
        return self.section("retrieval")

    @property
    def prompt(self) -> Dict[str, Any]:
        return self.section("prompt")

    @property
    def tools(self) -> Dict[str, Any]:
        return self.section("tools")

    @property
    def memory(self) -> Dict[str, Any]:
        return self.section("memory")

    @property
    def mcp(self) -> Dict[str, Any]:
        return self.section("mcp")

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self.section("evaluation")

    @property
    def logging(self) -> Dict[str, Any]:
        return self.section("logging")
