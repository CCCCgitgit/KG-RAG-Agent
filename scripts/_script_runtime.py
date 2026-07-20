# -*- coding: utf-8 -*-
"""scripts 层共享的项目路径与输出辅助。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def bootstrap_src() -> Path:
    src = project_root() / "src"
    text = str(src)
    if text not in sys.path:
        sys.path.insert(0, text)
    old = [item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item]
    if text not in old:
        os.environ["PYTHONPATH"] = os.pathsep.join([text, *old])
    return src


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root() / candidate
    return candidate.resolve(strict=False)


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("value must be greater than 0")
    return parsed


bootstrap_src()

__all__ = [
    "bootstrap_src",
    "positive_int",
    "print_json",
    "project_root",
    "resolve_project_path",
]
