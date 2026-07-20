# -*- coding: utf-8 -*-
"""根据 processed triples 构建知识图谱。"""
from __future__ import annotations

import argparse
from typing import Any, Optional

try:
    from _script_runtime import print_json
except ImportError:  # pragma: no cover
    from scripts._script_runtime import print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the KG-RAG knowledge graph.")
    parser.add_argument("--profile", choices=["demo", "production"], default="demo")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--kg-dir", default=None)
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser


def run(
    *,
    profile: Optional[str] = "demo",
    processed_dir: Optional[str] = None,
    kg_dir: Optional[str] = None,
    overwrite: bool = True,
    check_only: bool = False,
) -> dict[str, Any]:
    from kg_rag_agent.data_pipeline import PipelinePaths, build_graph

    paths = PipelinePaths.build(profile=profile, processed_dir=processed_dir, kg_dir=kg_dir)
    triples = paths.processed_dir / "triples.csv"
    check = {"ok": triples.is_file(), "triples": triples.as_posix()}
    if check_only or not check["ok"]:
        return {"ok": bool(check["ok"]), "stage": "check_processed_inputs", "check": check}
    stats = build_graph(
        processed_dir=paths.processed_dir,
        kg_dir=paths.kg_dir,
        overwrite=overwrite,
    )
    return {"ok": True, "stage": "build_kg", "profile": paths.profile, "stats": stats}


def main() -> None:
    args = build_parser().parse_args()
    result = run(
        profile=args.profile,
        processed_dir=args.processed_dir,
        kg_dir=args.kg_dir,
        overwrite=not args.no_overwrite,
        check_only=args.check_only,
    )
    print_json(result)
    if not result.get("ok", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
