# -*- coding: utf-8 -*-
"""Project path utilities.

This module contains only generic path handling. Domain modules should not
implement their own project-root discovery or path-sandbox logic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT_ENV = "KG_RAG_AGENT_ROOT"
_PROJECT_MARKERS = ("pyproject.toml", ".git", "requirements.txt")


class PathBoundaryError(ValueError):
    """Raised when a resolved path escapes an allowed directory."""


def _candidate_roots(start: Path) -> Iterable[Path]:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    yield current
    yield from current.parents


def get_project_root(start: Optional[str | Path] = None) -> Path:
    """Return the project root.

    Resolution order:
        1. ``KG_RAG_AGENT_ROOT`` environment variable.
        2. Walk upward from ``start`` when supplied.
        3. Walk upward from this source file.
        4. Fall back to the historical ``src/...`` layout.
    """

    env_root = str(os.getenv(PROJECT_ROOT_ENV, "") or "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()

    search_starts = []
    if start is not None:
        search_starts.append(Path(start).expanduser())
    search_starts.append(Path(__file__).resolve())

    for search_start in search_starts:
        for candidate in _candidate_roots(search_start):
            if (candidate / "pyproject.toml").is_file():
                return candidate
            if (candidate / "src" / "kg_rag_agent").is_dir():
                return candidate

    # Compatible fallback for src/kg_rag_agent/utils/paths.py.
    return Path(__file__).resolve().parents[3]


def resolve_project_path(
    path: str | Path,
    *,
    project_root: Optional[str | Path] = None,
    prefer_cwd_if_exists: bool = True,
    must_exist: bool = False,
) -> Path:
    """Resolve an absolute or project-relative path.

    The legacy behavior is preserved: an existing relative path under the
    current working directory takes precedence; otherwise it is resolved
    against the project root.
    """

    input_path = Path(path).expanduser()

    if input_path.is_absolute():
        resolved = input_path.resolve()
    else:
        cwd_candidate = (Path.cwd() / input_path).resolve()
        if prefer_cwd_if_exists and cwd_candidate.exists():
            resolved = cwd_candidate
        else:
            root = (
                Path(project_root).expanduser().resolve()
                if project_root is not None
                else get_project_root()
            )
            resolved = (root / input_path).resolve()

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")

    return resolved


def ensure_directory(
    path: str | Path,
    *,
    project_root: Optional[str | Path] = None,
) -> Path:
    """Resolve and create a directory."""

    resolved = resolve_project_path(path, project_root=project_root)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_within_directory(
    path: str | Path,
    base_dir: str | Path,
    *,
    project_root: Optional[str | Path] = None,
    allow_base: bool = True,
) -> Path:
    """Resolve ``path`` and ensure it remains inside ``base_dir``."""

    base = resolve_project_path(base_dir, project_root=project_root).resolve()
    candidate_path = Path(path).expanduser()
    candidate = (
        candidate_path.resolve()
        if candidate_path.is_absolute()
        else (base / candidate_path).resolve()
    )

    if candidate == base:
        if allow_base:
            return candidate
        raise PathBoundaryError(f"Path must be below base directory: {base}")

    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise PathBoundaryError(
            f"Path escapes allowed directory: {candidate} (base: {base})"
        ) from exc

    return candidate


def relative_to_project(
    path: str | Path,
    *,
    project_root: Optional[str | Path] = None,
) -> Path:
    """Return a project-relative path when possible."""

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else get_project_root()
    )
    resolved = resolve_project_path(path, project_root=root)
    try:
        return resolved.relative_to(root)
    except ValueError:
        return resolved
