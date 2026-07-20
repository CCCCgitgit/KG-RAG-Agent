# -*- coding: utf-8 -*-
"""将原始 KG 文件标准化为 processed 数据。"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

try:
    from _script_runtime import positive_int, print_json
except ImportError:  # pragma: no cover
    from scripts._script_runtime import positive_int, print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build processed KG-RAG data.")
    parser.add_argument("--profile", choices=["demo", "production"], default="demo")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--max-rows", type=positive_int, default=None)
    parser.add_argument("--check-only", action="store_true")
    return parser


def run(
    *,
    profile: Optional[str] = "demo",
    raw_dir: Optional[str] = None,
    processed_dir: Optional[str] = None,
    max_rows: Optional[int] = None,
    check_only: bool = False,
) -> dict[str, Any]:
    from kg_rag_agent.data_pipeline import PipelinePaths, process_raw_data

    paths = PipelinePaths.build(
        profile=profile,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )
    required = [paths.raw_dir / "ent2ids", paths.raw_dir / "relation2ids", paths.raw_dir / "path_graph"]
    missing = [path.as_posix() for path in required if not path.is_file()]
    check = {"ok": not missing, "raw_dir": paths.raw_dir.as_posix(), "missing": missing}
    if check_only or missing:
        return {"ok": not missing, "stage": "check_raw_inputs", "check": check}
    stats = process_raw_data(
        raw_dir=paths.raw_dir,
        processed_dir=paths.processed_dir,
        max_rows=max_rows,
    )
    return {"ok": True, "stage": "build_data", "profile": paths.profile, "stats": stats}


def main() -> None:
    args = build_parser().parse_args()
    result = run(
        profile=args.profile,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        max_rows=args.max_rows,
        check_only=args.check_only,
    )
    print_json(result)
    if not result.get("ok", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
