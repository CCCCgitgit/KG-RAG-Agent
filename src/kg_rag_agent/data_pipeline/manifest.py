# -*- coding: utf-8 -*-
"""构建产物 Manifest 与一致性校验。"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .paths import PipelinePaths, resolve_project_path

MANIFEST_FILENAME = "build_manifest.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    resolved = resolve_project_path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: str | Path, *, required: bool = True) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    exists = resolved.exists() and resolved.is_file()
    record: dict[str, Any] = {
        "path": resolved.as_posix(),
        "required": bool(required),
        "exists": exists,
        "size_bytes": resolved.stat().st_size if exists else 0,
        "sha256": sha256_file(resolved) if exists else None,
    }
    return record


def write_json_atomic(data: Mapping[str, Any], path: str | Path) -> Path:
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(resolved)
    return resolved


def build_manifest(
    *,
    build_id: str,
    paths: PipelinePaths,
    options: Mapping[str, Any],
    stages: Mapping[str, Any],
    include_vector_store: bool,
) -> dict[str, Any]:
    artifacts = {
        "entities": artifact_record(paths.processed_dir / "entities.json"),
        "relations": artifact_record(paths.processed_dir / "relations.json"),
        "triples": artifact_record(paths.processed_dir / "triples.csv"),
        "entity_index": artifact_record(paths.processed_dir / "entity_index.json"),
        "relation_index": artifact_record(paths.processed_dir / "relation_index.json"),
        "alias_map": artifact_record(paths.processed_dir / "alias_map.json"),
        "graph": artifact_record(paths.kg_dir / "graph.pkl"),
        "graph_stats": artifact_record(paths.kg_dir / "graph_stats.json"),
    }
    if include_vector_store:
        artifacts["vector_stats"] = artifact_record(
            paths.vector_store_dir / "vector_store_stats.json"
        )

    required_ok = all(
        item["exists"]
        for item in artifacts.values()
        if item.get("required")
    )

    return {
        "schema_version": "1.0",
        "build_id": str(build_id),
        "created_at": utc_now_iso(),
        "profile": paths.profile,
        "ok": bool(required_ok),
        "paths": paths.as_dict(),
        "options": dict(options),
        "stages": dict(stages),
        "artifacts": artifacts,
    }


def write_manifest_copies(
    manifest: Mapping[str, Any],
    *,
    paths: PipelinePaths,
    include_vector_store: bool,
) -> list[str]:
    targets = [
        paths.manifest_path,
        paths.processed_dir / MANIFEST_FILENAME,
        paths.kg_dir / MANIFEST_FILENAME,
    ]
    if include_vector_store:
        targets.append(paths.vector_store_dir / MANIFEST_FILENAME)

    written: list[str] = []
    seen: set[str] = set()
    for target in targets:
        key = target.resolve(strict=False).as_posix()
        if key in seen:
            continue
        seen.add(key)
        written.append(write_json_atomic(manifest, target).as_posix())
    return written


def load_manifest(path: str | Path) -> dict[str, Any]:
    resolved = resolve_project_path(path, must_exist=True)
    with resolved.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a JSON object: {resolved}")
    return data


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    verify_checksums: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    build_id = str(manifest.get("build_id") or "").strip()
    if not build_id:
        errors.append("missing_build_id")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        errors.append("missing_artifacts")
        artifacts = {}

    checked = 0
    for name, raw in artifacts.items():
        if not isinstance(raw, Mapping):
            errors.append(f"invalid_artifact:{name}")
            continue
        path = raw.get("path")
        required = bool(raw.get("required", True))
        if not path:
            if required:
                errors.append(f"missing_path:{name}")
            continue
        resolved = resolve_project_path(str(path))
        if not resolved.is_file():
            if required:
                errors.append(f"missing_file:{name}")
            continue
        checked += 1
        if verify_checksums and raw.get("sha256"):
            if sha256_file(resolved) != raw.get("sha256"):
                errors.append(f"checksum_mismatch:{name}")

    return {
        "ok": not errors,
        "build_id": build_id or None,
        "checked_artifacts": checked,
        "errors": errors,
    }


__all__ = [
    "MANIFEST_FILENAME",
    "artifact_record",
    "build_manifest",
    "load_manifest",
    "sha256_file",
    "utc_now_iso",
    "validate_manifest",
    "write_json_atomic",
    "write_manifest_copies",
]
