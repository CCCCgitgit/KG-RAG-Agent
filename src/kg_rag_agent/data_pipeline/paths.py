# -*- coding: utf-8 -*-
"""离线构建路径解析与 Profile 隔离。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

LEGACY_RAW_DIR = "data/raw"
LEGACY_PROCESSED_DIR = "data/processed"
LEGACY_KG_DIR = "data/kg"
LEGACY_VECTOR_STORE_DIR = "data/vector_store"

SUPPORTED_PROFILES = frozenset({"demo", "production"})


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path, *, must_exist: bool = False) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = get_project_root() / value
    value = value.resolve(strict=False)
    if must_exist and not value.exists():
        raise FileNotFoundError(f"Path not found: {value}")
    return value


def normalize_profile(profile: Optional[str]) -> Optional[str]:
    if profile is None:
        return None
    value = str(profile).strip().lower()
    if not value:
        return None
    if value not in SUPPORTED_PROFILES:
        raise ValueError(
            f"Unsupported data profile: {profile!r}. "
            f"Expected one of: {sorted(SUPPORTED_PROFILES)}"
        )
    return value


@dataclass(frozen=True)
class PipelinePaths:
    """一次离线构建使用的全部路径。"""

    profile: Optional[str]
    raw_dir: Path
    processed_dir: Path
    kg_dir: Path
    vector_store_dir: Path
    manifest_path: Path

    @classmethod
    def build(
        cls,
        *,
        profile: Optional[str] = None,
        raw_dir: str | Path | None = None,
        processed_dir: str | Path | None = None,
        kg_dir: str | Path | None = None,
        vector_store_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
    ) -> "PipelinePaths":
        normalized_profile = normalize_profile(profile)

        if normalized_profile:
            profile_root = get_project_root() / "data" / normalized_profile
            raw = resolve_project_path(raw_dir or profile_root / "raw")
            processed = resolve_project_path(processed_dir or profile_root / "processed")
            kg = resolve_project_path(kg_dir or profile_root / "kg")
            vector = resolve_project_path(vector_store_dir or profile_root / "vector_store")
            manifest = resolve_project_path(manifest_path or profile_root / "build_manifest.json")
        else:
            raw = resolve_project_path(raw_dir or LEGACY_RAW_DIR)
            processed = resolve_project_path(processed_dir or LEGACY_PROCESSED_DIR)
            kg = resolve_project_path(kg_dir or LEGACY_KG_DIR)
            vector = resolve_project_path(vector_store_dir or LEGACY_VECTOR_STORE_DIR)
            manifest = resolve_project_path(manifest_path or processed / "build_manifest.json")

        return cls(
            profile=normalized_profile,
            raw_dir=raw,
            processed_dir=processed,
            kg_dir=kg,
            vector_store_dir=vector,
            manifest_path=manifest,
        )

    def ensure_output_dirs(self) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.kg_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, str | None]:
        return {
            "profile": self.profile,
            "raw_dir": self.raw_dir.as_posix(),
            "processed_dir": self.processed_dir.as_posix(),
            "kg_dir": self.kg_dir.as_posix(),
            "vector_store_dir": self.vector_store_dir.as_posix(),
            "manifest_path": self.manifest_path.as_posix(),
        }


__all__ = [
    "LEGACY_RAW_DIR",
    "LEGACY_PROCESSED_DIR",
    "LEGACY_KG_DIR",
    "LEGACY_VECTOR_STORE_DIR",
    "SUPPORTED_PROFILES",
    "PipelinePaths",
    "get_project_root",
    "resolve_project_path",
    "normalize_profile",
]
