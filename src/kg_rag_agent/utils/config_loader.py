# -*- coding: utf-8 -*-
"""Configuration loading and mapping utilities.

The loader keeps compatibility with the original public API while providing a
single source of truth for project paths, defaults, environment overrides, and
validation.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from .exceptions import ConfigurationError, ConfigurationValidationError
from .paths import get_project_root, resolve_project_path

DEFAULT_CONFIG_DIR = "configs"

# Missing optional files are ignored by default. Keeping all target files here
# lets new modules become active without changing the loader again.
DEFAULT_CONFIG_FILES = [
    "model.yaml",
    "graph.yaml",
    "retrieval.yaml",
    "kg.yaml",
    "prompt.yaml",
    "tools.yaml",
    "memory.yaml",
    "evaluation.yaml",
    "mcp.yaml",
    "logging.yaml",
]


def get_config_dir(config_dir: Optional[str | Path] = None) -> Path:
    """Return the resolved configuration directory."""

    return resolve_project_path(config_dir or DEFAULT_CONFIG_DIR)


def _require_yaml() -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ConfigurationError(
            "PyYAML is not installed. Please install pyyaml first."
        ) from exc
    return yaml


def load_yaml_file(path: str | Path) -> Dict[str, Any]:
    """Load one YAML mapping."""

    resolved_path = resolve_project_path(path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    yaml = _require_yaml()
    try:
        with resolved_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except OSError as exc:
        raise ConfigurationError(f"Failed to read config: {resolved_path}") from exc
    except Exception as exc:
        raise ConfigurationError(f"Invalid YAML config: {resolved_path}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Config file must contain a YAML mapping object: {resolved_path}"
        )
    return data


def save_yaml_file(
    data: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    """Save one YAML mapping."""

    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping")

    resolved_path = resolve_project_path(path)
    if resolved_path.exists() and not overwrite:
        raise FileExistsError(f"Config file already exists: {resolved_path}")

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    yaml = _require_yaml()
    try:
        with resolved_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                dict(data),
                file,
                allow_unicode=True,
                sort_keys=False,
                indent=2,
            )
    except OSError as exc:
        raise ConfigurationError(f"Failed to write config: {resolved_path}") from exc

    return resolved_path


def load_config(
    config_path: Optional[str | Path] = None,
    *,
    config_dir: Optional[str | Path] = None,
    apply_env: bool = True,
) -> Dict[str, Any]:
    """Load one config file or all default config sections."""

    if config_path is not None:
        config = load_yaml_file(config_path)
    else:
        config = load_all_configs(config_dir=config_dir, apply_env=False)

    return apply_env_overrides(config) if apply_env else config


def load_all_configs(
    *,
    config_dir: Optional[str | Path] = None,
    config_files: Optional[Iterable[str]] = None,
    apply_env: bool = True,
    ignore_missing: bool = True,
) -> Dict[str, Any]:
    """Load multiple YAML files under one top-level section per file."""

    config_root = get_config_dir(config_dir)
    filenames = list(config_files or DEFAULT_CONFIG_FILES)
    merged: Dict[str, Any] = {}

    for filename in filenames:
        path = config_root / str(filename)
        if not path.is_file():
            if ignore_missing:
                continue
            raise FileNotFoundError(f"Config file not found: {path}")

        merged[path.stem] = load_yaml_file(path)

    merged = fill_default_config(merged)
    return apply_env_overrides(merged) if apply_env else merged


def get_default_config() -> Dict[str, Any]:
    """Return safe defaults matching the existing project behavior."""

    return {
        "model": {
            "provider": "deepseek",
            "api_key_env": "DEEPSEEK_API_KEY",
            "base_url": "https://api.deepseek.com",
            "model_name": "deepseek-chat",
            "temperature": 0.2,
            "max_tokens": 1024,
            "timeout": 60,
            "max_retries": 2,
        },
        "graph": {
            "graph_path": "data/demo/kg/graph.pkl",
            "checkpointer": None,
            "interrupt_before": None,
            "interrupt_after": None,
        },
        # Legacy top-level node defaults are retained. RuntimeSettings merges
        # them with the corresponding graph.<section> values.
        "router": {"use_llm": False},
        "mention_extraction": {
            "use_llm": False,
            "min_confidence": 0.3,
            "max_mentions": 8,
        },
        "entity_linking": {
            "top_k": 5,
            "min_score": 0.0,
            "keep_empty_candidates": True,
            "chroma_dir": "data/demo/vector_store/chroma_entity_db",
            "collection_name": "kg_entities",
            "model_name": (
                "sentence-transformers/"
                "paraphrase-multilingual-MiniLM-L12-v2"
            ),
            "alias_path": "data/demo/processed/alias_map.json",
            "auto_select_threshold": 0.72,
            "margin_threshold": 0.05,
            "cache_key": "default",
        },
        "entity_grounding": {
            "min_score": 0.0,
            "max_entities": 16,
            "one_entity_per_mention": True,
            "require_in_graph": True,
            "allow_linear_scan": True,
        },
        "kg": {
            "graph_path": "data/demo/kg/graph.pkl",
            "enable_relation_search": True,
            "enable_path_search": True,
            "enable_neighbor_search": True,
            "enable_subgraph_search": False,
            "max_evidence": 30,
            "max_neighbors_per_entity": 10,
            "max_paths_per_pair": 5,
            "max_path_length": 4,
        },
        "retrieval": {
            "embedding_model": (
                "sentence-transformers/"
                "paraphrase-multilingual-MiniLM-L12-v2"
            ),
            "entity_chroma_dir": "data/demo/vector_store/chroma_entity_db",
            "doc_chroma_dir": "data/demo/vector_store/chroma_doc_db",
            "entity_collection_name": "kg_entities",
            "doc_collection_name": "kg_documents",
            "top_k": 5,
            "local_files_only": True,
            "allow_hash_fallback": False,
        },
        "semantic_scoring": {
            "use_llm": False,
            "max_selected_evidence": 8,
            "min_evidence_score": 0.15,
            "answerable_threshold": 0.55,
            "uncertain_threshold": 0.25,
        },
        "reasoning": {"use_llm": False, "max_reasoning_steps": 5},
        "generation": {
            "use_llm": True,
            "temperature": 0.2,
            "max_tokens": 1200,
        },
        "prompt": {
            "language": "zh",
            "style": "natural",
            "expose_internal_process": False,
        },
        "tools": {
            "enabled": True,
            "default_timeout_seconds": 30,
            "max_calls_per_request": 20,
        },
        "memory": {"enabled": False},
        "evaluation": {"output_dir": "outputs/evaluation"},
        "mcp": {"enabled": False},
        "logging": {
            "logger_name": "kg_rag_agent",
            "level": "INFO",
            "log_dir": "outputs/logs",
            "log_file": "kg_rag_agent.log",
            "enable_console": True,
            "enable_file": True,
            "propagate": False,
        },
        "prompts": {
            "direct_llm_system_prompt": (
                "你是一个可靠、自然、专业的中文智能助手。"
                "请根据用户问题直接回答，表达清晰，避免编造。"
                "如果信息不足，请自然地说明需要补充哪些信息。"
                "不要向用户暴露内部路由、检索、图结构、节点流程等"
                "系统实现细节。"
            )
        },
    }


def fill_default_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep-copied config with defaults filled."""

    return deep_merge(get_default_config(), dict(config or {}))


def apply_env_overrides(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply approved environment-variable overrides."""

    result = copy.deepcopy(dict(config or {}))
    env_mapping = {
        "LLM_PROVIDER": ("model.provider", str),
        "LLM_MODEL": ("model.model_name", str),
        "LLM_BASE_URL": ("model.base_url", str),
        "LLM_TIMEOUT": ("model.timeout", float),
        "LLM_MAX_RETRIES": ("model.max_retries", int),
        "DEEPSEEK_BASE_URL": ("model.base_url", str),
        "DEEPSEEK_MODEL": ("model.model_name", str),
        "KG_GRAPH_PATH": ("kg.graph_path", str),
        "ROUTER_USE_LLM": ("router.use_llm", parse_bool),
        "GENERATION_USE_LLM": ("generation.use_llm", parse_bool),
        "SEMANTIC_SCORING_USE_LLM": (
            "semantic_scoring.use_llm",
            parse_bool,
        ),
        "REASONING_USE_LLM": ("reasoning.use_llm", parse_bool),
        "TOOLS_ENABLED": ("tools.enabled", parse_bool),
        "MEMORY_ENABLED": ("memory.enabled", parse_bool),
        "MCP_ENABLED": ("mcp.enabled", parse_bool),
        "LOG_LEVEL": ("logging.level", str),
    }

    for env_key, (config_path, caster) in env_mapping.items():
        raw_value = os.getenv(env_key)
        if raw_value is None:
            continue
        try:
            value = caster(raw_value)
        except (TypeError, ValueError):
            continue
        set_config_value(result, config_path, value)

    return result


def parse_bool(value: Any) -> bool:
    """Parse a common boolean representation."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0

    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "disable", "disabled", ""}:
        return False
    return False


def get_config_value(
    config: Mapping[str, Any],
    path: str,
    default: Any = None,
) -> Any:
    """Read a value using a dot-separated path."""

    if not path:
        return default

    current: Any = config
    for part in str(path).split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def set_config_value(
    config: MutableMapping[str, Any],
    path: str,
    value: Any,
) -> None:
    """Write a value using a dot-separated path."""

    if not path:
        return
    if not isinstance(config, MutableMapping):
        raise TypeError("config must be a mutable mapping")

    parts = str(path).split(".")
    current: MutableMapping[str, Any] = config
    for part in parts[:-1]:
        existing = current.get(part)
        if not isinstance(existing, MutableMapping):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def has_config_value(config: Mapping[str, Any], path: str) -> bool:
    """Return whether a dot-separated path exists."""

    sentinel = object()
    return get_config_value(config, path, sentinel) is not sentinel


def deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> Dict[str, Any]:
    """Recursively merge mappings without mutating inputs."""

    result: Dict[str, Any] = copy.deepcopy(dict(base or {}))
    for key, value in dict(override or {}).items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def merge_configs(*configs: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge multiple mappings from left to right."""

    result: Dict[str, Any] = {}
    for config in configs:
        result = deep_merge(result, config or {})
    return result


def validate_config(config: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    """Perform non-I/O configuration validation."""

    errors: List[str] = []
    if not isinstance(config, Mapping):
        return False, ["config must be a mapping"]

    for section in ("model", "kg", "retrieval", "generation"):
        if not isinstance(config.get(section), Mapping):
            errors.append(f"missing or invalid config section: {section}")

    if not str(get_config_value(config, "model.provider", "") or "").strip():
        errors.append("model.provider is empty")

    graph_path = (
        get_config_value(config, "kg.graph_path", "")
        or get_config_value(config, "graph.graph_path", "")
    )
    if not str(graph_path or "").strip():
        errors.append("kg.graph_path or graph.graph_path is empty")

    temperature = get_config_value(config, "model.temperature", 0.2)
    try:
        if not 0.0 <= float(temperature) <= 2.0:
            errors.append("model.temperature must be between 0 and 2")
    except (TypeError, ValueError):
        errors.append("model.temperature must be numeric")

    max_tokens = get_config_value(config, "model.max_tokens", 1024)
    try:
        if int(max_tokens) <= 0:
            errors.append("model.max_tokens must be greater than 0")
    except (TypeError, ValueError):
        errors.append("model.max_tokens must be an integer")

    return not errors, errors


def require_valid_config(config: Mapping[str, Any]) -> None:
    """Raise a structured error when config validation fails."""

    ok, errors = validate_config(config)
    if not ok:
        raise ConfigurationValidationError(
            "Invalid project configuration",
            details={"errors": errors},
        )


class ConfigLoader:
    """Stateful configuration loader compatible with the original API."""

    def __init__(
        self,
        *,
        config_dir: Optional[str | Path] = None,
        config_files: Optional[Iterable[str]] = None,
        apply_env: bool = True,
        ignore_missing: bool = True,
    ) -> None:
        self.config_dir = config_dir or DEFAULT_CONFIG_DIR
        self.config_files = list(config_files or DEFAULT_CONFIG_FILES)
        self.apply_env = bool(apply_env)
        self.ignore_missing = bool(ignore_missing)
        self.config: Dict[str, Any] = {}

    def load(self, *, force: bool = True) -> Dict[str, Any]:
        if self.config and not force:
            return copy.deepcopy(self.config)
        self.config = load_all_configs(
            config_dir=self.config_dir,
            config_files=self.config_files,
            apply_env=self.apply_env,
            ignore_missing=self.ignore_missing,
        )
        return copy.deepcopy(self.config)

    def reload(self) -> Dict[str, Any]:
        return self.load(force=True)

    def get(self, path: str, default: Any = None) -> Any:
        if not self.config:
            self.load()
        return get_config_value(self.config, path, default)

    def set(self, path: str, value: Any) -> None:
        if not self.config:
            self.load()
        set_config_value(self.config, path, value)

    def validate(self) -> Tuple[bool, List[str]]:
        if not self.config:
            self.load()
        return validate_config(self.config)

    def to_dict(self) -> Dict[str, Any]:
        if not self.config:
            self.load()
        return copy.deepcopy(self.config)


def dump_config_json(config: Mapping[str, Any], *, indent: int = 2) -> str:
    """Serialize config for diagnostics."""

    return json.dumps(dict(config), ensure_ascii=False, indent=indent, default=str)
