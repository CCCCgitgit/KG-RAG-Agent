# -*- coding: utf-8 -*-
"""Prompt 加载、版本信息、缓存与变量渲染。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from typing import Any, Dict, Mapping, Optional

from .errors import (
    PromptConfigurationError,
    PromptNotFoundError,
    PromptRenderError,
)

DEFAULT_CONFIG_PATH = "configs/prompt.yaml"
PACKAGE_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
DEFAULT_SOURCE_PROMPT_DIR = "src/kg_rag_agent/prompts"
LEGACY_PROMPT_DIR = "prompts"
_PROMPT_SUFFIXES = {".txt", ".md", ".prompt"}


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """Prompt 正文及可追踪元数据。"""

    name: str
    content: str
    version: str = "legacy"
    language: str = "zh"
    source: str = "inline"
    source_path: str = ""
    required_variables: tuple[str, ...] = ()
    output_schema: str = ""
    default_parameters: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True
    checksum: str = ""

    def to_dict(self, *, include_content: bool = False) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "language": self.language,
            "source": self.source,
            "source_path": self.source_path,
            "required_variables": list(self.required_variables),
            "output_schema": self.output_schema,
            "default_parameters": dict(self.default_parameters),
            "enabled": self.enabled,
            "checksum": self.checksum,
        }
        if include_content:
            result["content"] = self.content
        return result


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _safe_load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise PromptConfigurationError("PyYAML is required to load prompt config") from exc

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PromptConfigurationError(f"failed to load prompt config: {path}") from exc

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PromptConfigurationError("prompt config root must be a mapping")
    return value


def _checksum(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class PromptManager:
    """统一管理旧内联 Prompt 与目标目录化 Prompt。"""

    def __init__(
        self,
        *,
        project_root: str | Path | None = None,
        config_path: str | Path | None = None,
        prompt_dir: str | Path | None = None,
        auto_load: bool = True,
    ) -> None:
        self.project_root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else get_project_root()
        )
        self.config_path = self._resolve_path(config_path or DEFAULT_CONFIG_PATH)
        self.prompt_dir = self._resolve_prompt_dir(prompt_dir)

        self.config: Dict[str, Any] = {}
        self.prompts: Dict[str, str] = {}
        self.records: Dict[str, PromptRecord] = {}
        self._aliases: Dict[str, str] = {}

        if auto_load:
            self.reload()

    def reload(self) -> None:
        self.config = _safe_load_yaml(self.config_path)
        self.prompts = {}
        self.records = {}
        self._aliases = {}

        self._load_inline_prompts()
        self._load_structured_prompts()
        self._load_prompt_files()

    def _resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return candidate.resolve()

    def _resolve_prompt_dir(self, prompt_dir: str | Path | None) -> Path:
        if prompt_dir is not None:
            return self._resolve_path(prompt_dir)

        # 优先使用当前已安装包内的 Prompt 资源。该路径在 editable install、
        # Wheel、Docker 镜像和源码运行模式下都有效。
        package_dir = PACKAGE_PROMPT_DIR.expanduser().resolve()
        source_dir = self._resolve_path(DEFAULT_SOURCE_PROMPT_DIR)
        legacy_dir = self._resolve_path(LEGACY_PROMPT_DIR)

        for candidate in (package_dir, source_dir, legacy_dir):
            if candidate.exists():
                return candidate
        return package_dir

    def _load_inline_prompts(self) -> None:
        language = str(self.config.get("language", "zh") or "zh")
        version = str(self.config.get("version", "legacy") or "legacy")

        for key, value in self.config.items():
            if not isinstance(value, str):
                continue
            if not self._is_inline_prompt_key(key):
                continue

            content = value.strip()
            if not content:
                continue
            self._register(
                PromptRecord(
                    name=key,
                    content=content,
                    version=version,
                    language=language,
                    source="inline",
                    checksum=_checksum(content),
                ),
                overwrite=False,
            )

    def _load_structured_prompts(self) -> None:
        section = self.config.get("prompts", {})
        if section is None:
            return
        if not isinstance(section, Mapping):
            raise PromptConfigurationError("prompt.yaml field 'prompts' must be a mapping")

        default_language = str(self.config.get("language", "zh") or "zh")

        for raw_name, raw_spec in section.items():
            name = str(raw_name or "").strip()
            if not name:
                raise PromptConfigurationError("prompt name must not be empty")
            if isinstance(raw_spec, str):
                spec: Mapping[str, Any] = {"content": raw_spec}
            elif isinstance(raw_spec, Mapping):
                spec = raw_spec
            else:
                raise PromptConfigurationError(f"invalid prompt spec: {name}")

            enabled = bool(spec.get("enabled", True))
            if not enabled:
                continue

            content, source, source_path = self._read_structured_content(name, spec)
            required = _string_tuple(
                spec.get("required_variables", spec.get("variables", ()))
            )
            default_parameters = spec.get("default_parameters", spec.get("parameters", {}))
            if default_parameters is None:
                default_parameters = {}
            if not isinstance(default_parameters, Mapping):
                raise PromptConfigurationError(
                    f"default_parameters must be a mapping: {name}"
                )

            record = PromptRecord(
                name=name,
                content=content,
                version=str(spec.get("version", "v1") or "v1"),
                language=str(spec.get("language", default_language) or default_language),
                source=source,
                source_path=source_path,
                required_variables=required,
                output_schema=str(spec.get("output_schema", "") or ""),
                default_parameters=dict(default_parameters),
                enabled=True,
                checksum=_checksum(content),
            )
            self._register(record, overwrite=True)

    def _read_structured_content(
        self,
        name: str,
        spec: Mapping[str, Any],
    ) -> tuple[str, str, str]:
        inline = spec.get("content")
        path_value = spec.get("path")

        if inline is not None and path_value is not None:
            raise PromptConfigurationError(
                f"prompt cannot define both content and path: {name}"
            )

        if inline is not None:
            content = str(inline).strip()
            if not content:
                raise PromptConfigurationError(f"prompt content is empty: {name}")
            return content, "structured_inline", ""

        if path_value is None:
            raise PromptConfigurationError(f"prompt requires content or path: {name}")

        path = self._resolve_path(str(path_value))
        self._ensure_allowed_prompt_path(path)
        if not path.is_file():
            raise PromptConfigurationError(f"prompt file not found: {path}")

        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise PromptConfigurationError(f"prompt file is empty: {path}")
        return content, "file", str(path)

    def _load_prompt_files(self) -> None:
        if not self.prompt_dir.exists():
            return
        if not self.prompt_dir.is_dir():
            raise PromptConfigurationError(
                f"prompt_dir is not a directory: {self.prompt_dir}"
            )

        for path in sorted(self.prompt_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _PROMPT_SUFFIXES:
                continue
            self._ensure_allowed_prompt_path(path.resolve())
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                continue

            relative = path.relative_to(self.prompt_dir).with_suffix("")
            name = ".".join(relative.parts)
            record = PromptRecord(
                name=name,
                content=content,
                version="file",
                language=str(self.config.get("language", "zh") or "zh"),
                source="file",
                source_path=str(path.resolve()),
                checksum=_checksum(content),
            )
            # 结构化配置优先于目录自动发现。
            self._register(record, overwrite=False)

    def _ensure_allowed_prompt_path(self, path: Path) -> None:
        allowed = self.prompt_dir.resolve()
        try:
            path.resolve().relative_to(allowed)
        except ValueError as exc:
            raise PromptConfigurationError(
                f"prompt path is outside prompt directory: {path}"
            ) from exc

    def _register(self, record: PromptRecord, *, overwrite: bool) -> None:
        name = record.name.strip()
        if not name:
            raise PromptConfigurationError("prompt name must not be empty")
        if not overwrite and name in self.records:
            return

        self.records[name] = record
        self.prompts[name] = record.content
        self._register_aliases(name)

    def _register_aliases(self, name: str) -> None:
        for alias in self._candidate_names(name):
            self._aliases.setdefault(alias, name)

    def resolve_name(self, name: str) -> Optional[str]:
        for candidate in self._candidate_names(name):
            if candidate in self.records:
                return candidate
            target = self._aliases.get(candidate)
            if target in self.records:
                return target
        return None

    def get_record(self, name: str, *, required: bool = False) -> Optional[PromptRecord]:
        resolved = self.resolve_name(name)
        if resolved is None:
            if required:
                raise PromptNotFoundError(str(name))
            return None
        return self.records[resolved]

    def get(
        self,
        name: str,
        default: str = "",
        variables: Optional[Mapping[str, Any]] = None,
        *,
        required: bool = False,
        strict_variables: bool = False,
    ) -> str:
        record = self.get_record(name, required=required)
        if record is None:
            return self._render_text(
                default,
                variables,
                required_variables=(),
                strict_variables=strict_variables,
                prompt_name=str(name),
            )
        return self._render_text(
            record.content,
            variables,
            required_variables=record.required_variables,
            strict_variables=strict_variables,
            prompt_name=record.name,
        )

    def get_prompt(
        self,
        name: str,
        default: str = "",
        variables: Optional[Mapping[str, Any]] = None,
        *,
        required: bool = False,
        strict_variables: bool = False,
    ) -> str:
        return self.get(
            name=name,
            default=default,
            variables=variables,
            required=required,
            strict_variables=strict_variables,
        )

    def format_prompt(
        self,
        name: str,
        variables: Optional[Mapping[str, Any]] = None,
        default: str = "",
        *,
        required: bool = False,
        strict_variables: bool = False,
    ) -> str:
        return self.get(
            name=name,
            default=default,
            variables=variables,
            required=required,
            strict_variables=strict_variables,
        )

    def render(
        self,
        name: str,
        variables: Mapping[str, Any],
        *,
        strict_variables: bool = True,
    ) -> str:
        """正式 Prompt 渲染入口；默认要求 Prompt 存在且变量完整。"""

        return self.get(
            name=name,
            variables=variables,
            required=True,
            strict_variables=strict_variables,
        )

    def has_prompt(self, name: str) -> bool:
        return self.resolve_name(name) is not None

    def list_prompts(self) -> list[str]:
        return sorted(self.records)

    def as_dict(self) -> Dict[str, str]:
        return {name: record.content for name, record in self.records.items()}

    def manifest(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: record.to_dict(include_content=False)
            for name, record in sorted(self.records.items())
        }

    def _render_text(
        self,
        template: str,
        variables: Optional[Mapping[str, Any]],
        *,
        required_variables: tuple[str, ...],
        strict_variables: bool,
        prompt_name: str,
    ) -> str:
        if not variables:
            if strict_variables and required_variables:
                raise PromptRenderError(
                    f"missing prompt variables for {prompt_name}: "
                    + ", ".join(required_variables)
                )
            return template

        values = dict(variables)
        placeholders = _placeholder_names(template)
        required = set(required_variables) or placeholders
        missing = sorted(name for name in required if name not in values)
        if strict_variables and missing:
            raise PromptRenderError(
                f"missing prompt variables for {prompt_name}: " + ", ".join(missing)
            )

        try:
            if strict_variables:
                return template.format_map(values)
            return template.format_map(_SafeFormatDict(values))
        except Exception as exc:
            raise PromptRenderError(f"failed to render prompt: {prompt_name}") from exc

    @staticmethod
    def _is_inline_prompt_key(key: str) -> bool:
        return key.endswith("_prompt") or key in {
            "system_prompt",
            "direct_llm_system_prompt",
            "generation_system_prompt",
        }

    @staticmethod
    def _candidate_names(name: str) -> list[str]:
        raw = str(name or "").strip().replace("/", ".")
        if not raw:
            return []

        candidates = [raw]
        if raw.endswith("_prompt"):
            base = raw[: -len("_prompt")]
            candidates.extend([base, f"{base}.system"])
        else:
            candidates.extend([f"{raw}_prompt", f"{raw}.system"])

        if raw.endswith("_system_prompt"):
            base = raw[: -len("_system_prompt")]
            candidates.extend([base, f"{base}.system", f"{base}_prompt"])
        if raw.endswith(".system"):
            base = raw[: -len(".system")]
            candidates.extend([base, f"{base}_prompt", f"{base}_system_prompt"])

        result: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        return result


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        raise PromptConfigurationError("required_variables must be a string or list")

    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        normalized = str(item or "").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)


def _placeholder_names(template: str) -> set[str]:
    names: set[str] = set()
    try:
        for _, field_name, _, _ in Formatter().parse(template):
            if not field_name:
                continue
            root = field_name.split(".", 1)[0].split("[", 1)[0]
            if root:
                names.add(root)
    except ValueError:
        # 模板包含普通 JSON 花括号时，交给实际 format 阶段处理。
        return set()
    return names


_DEFAULT_PROMPT_MANAGER: Optional[PromptManager] = None


def get_default_prompt_manager() -> PromptManager:
    global _DEFAULT_PROMPT_MANAGER
    if _DEFAULT_PROMPT_MANAGER is None:
        _DEFAULT_PROMPT_MANAGER = PromptManager()
    return _DEFAULT_PROMPT_MANAGER


def reset_default_prompt_manager() -> None:
    global _DEFAULT_PROMPT_MANAGER
    _DEFAULT_PROMPT_MANAGER = None


def get_prompt(
    name: str,
    default: str = "",
    variables: Optional[Mapping[str, Any]] = None,
) -> str:
    return get_default_prompt_manager().get(
        name=name,
        default=default,
        variables=variables,
    )


def format_prompt(
    name: str,
    variables: Optional[Mapping[str, Any]] = None,
    default: str = "",
) -> str:
    return get_default_prompt_manager().format_prompt(
        name=name,
        variables=variables,
        default=default,
    )


def render_prompt(name: str, variables: Mapping[str, Any]) -> str:
    return get_default_prompt_manager().render(name=name, variables=variables)


def list_prompts() -> list[str]:
    return get_default_prompt_manager().list_prompts()


__all__ = [
    "PromptRecord",
    "PromptManager",
    "PACKAGE_PROMPT_DIR",
    "get_default_prompt_manager",
    "reset_default_prompt_manager",
    "get_prompt",
    "format_prompt",
    "render_prompt",
    "list_prompts",
]
