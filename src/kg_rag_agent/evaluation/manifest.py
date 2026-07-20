# -*- coding: utf-8 -*-
"""评估 Run Manifest 构建与哈希工具。"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    run_id: str,
    dataset_name: str,
    dataset_version: str,
    started_at: str,
    finished_at: str = "",
    cases: Optional[Sequence[Any]] = None,
    request_options: Optional[Mapping[str, Any]] = None,
    agent_service: Any = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    settings = _extract_settings(agent_service)
    model = settings.get("model", {}) if isinstance(settings, Mapping) else {}
    graph = settings.get("graph", {}) if isinstance(settings, Mapping) else {}
    retrieval = settings.get("retrieval", {}) if isinstance(settings, Mapping) else {}
    prompt = settings.get("prompt", {}) if isinstance(settings, Mapping) else {}
    project_root = _extract_project_root(agent_service)
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "dataset_size": len(cases or []),
        "dataset_hash": stable_hash([
            item.to_dict() if hasattr(item, "to_dict") else item
            for item in (cases or [])
        ]),
        "started_at": started_at,
        "finished_at": finished_at,
        "code_version": _git_revision(project_root),
        "config_hash": stable_hash(settings),
        "request_options": dict(request_options or {}),
        "model": _safe_model_info(model),
        "prompt_version": _find_version(prompt),
        "graph_checksum": stable_hash(graph),
        "vector_store_version": _find_version(retrieval),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
            "process_id": os.getpid(),
        },
    }
    if extra:
        manifest["extra"] = dict(extra)
    return manifest


def _extract_settings(agent_service: Any) -> Dict[str, Any]:
    if agent_service is None:
        return {}
    try:
        config = getattr(agent_service, "config", {})
        if isinstance(config, Mapping):
            return dict(config)
    except Exception:
        pass
    try:
        settings = agent_service.runtime.settings
        to_dict = getattr(settings, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
            return dict(value) if isinstance(value, Mapping) else {}
    except Exception:
        pass
    return {}


def _extract_project_root(agent_service: Any) -> Path:
    try:
        root = agent_service.runtime.settings.project_root
        return Path(root).resolve()
    except Exception:
        return Path.cwd().resolve()


def _git_revision(project_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def _safe_model_info(model: Any) -> Dict[str, Any]:
    if not isinstance(model, Mapping):
        return {}
    result: Dict[str, Any] = {}
    for key in ("provider", "model", "model_name", "name"):
        if key in model and model[key] not in (None, ""):
            result[key] = model[key]
    return result


def _find_version(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("version", "prompt_version", "build_id", "collection_version"):
            item = value.get(key)
            if item not in (None, ""):
                return str(item)
        for nested in value.values():
            found = _find_version(nested)
            if found:
                return found
    return ""


__all__ = [
    "utc_now_iso",
    "stable_hash",
    "file_sha256",
    "build_manifest",
]
