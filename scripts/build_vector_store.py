# -*- coding: utf-8 -*-
"""根据标准实体或 graph.pkl 构建实体向量库。"""
from __future__ import annotations

import argparse
from typing import Any, Optional

try:
    from _script_runtime import positive_int, print_json
except ImportError:  # pragma: no cover
    from scripts._script_runtime import positive_int, print_json

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the KG-RAG entity vector store.")
    parser.add_argument("--profile", choices=["demo", "production"], default="demo")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--kg-dir", default=None)
    parser.add_argument("--vector-store-dir", default=None)
    parser.add_argument("--collection-name", default="kg_entities")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-hash-fallback", action="store_true")
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser


def run(
    *,
    profile: Optional[str] = "demo",
    processed_dir: Optional[str] = None,
    kg_dir: Optional[str] = None,
    vector_store_dir: Optional[str] = None,
    collection_name: str = "kg_entities",
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 128,
    local_files_only: bool = True,
    allow_hash_fallback: bool = False,
    reset: bool = True,
    check_only: bool = False,
) -> dict[str, Any]:
    from kg_rag_agent.data_pipeline import PipelinePaths, build_vector_store

    paths = PipelinePaths.build(
        profile=profile,
        processed_dir=processed_dir,
        kg_dir=kg_dir,
        vector_store_dir=vector_store_dir,
    )
    candidates = [paths.processed_dir / "entities.json", paths.kg_dir / "graph.pkl"]
    check = {
        "ok": any(path.is_file() for path in candidates),
        "candidates": [path.as_posix() for path in candidates],
    }
    if check_only or not check["ok"]:
        return {"ok": bool(check["ok"]), "stage": "check_vector_inputs", "check": check}
    stats = build_vector_store(
        processed_dir=paths.processed_dir,
        kg_dir=paths.kg_dir,
        vector_store_dir=paths.vector_store_dir,
        collection_name=collection_name,
        model_name=model_name,
        local_files_only=local_files_only,
        allow_hash_fallback=allow_hash_fallback,
        batch_size=batch_size,
        reset=reset,
    )
    return {"ok": True, "stage": "build_vector_store", "profile": paths.profile, "stats": stats}


def main() -> None:
    args = build_parser().parse_args()
    result = run(
        profile=args.profile,
        processed_dir=args.processed_dir,
        kg_dir=args.kg_dir,
        vector_store_dir=args.vector_store_dir,
        collection_name=args.collection_name,
        model_name=args.model_name,
        batch_size=args.batch_size,
        local_files_only=not args.allow_download,
        allow_hash_fallback=args.allow_hash_fallback,
        reset=not args.no_reset,
        check_only=args.check_only,
    )
    print_json(result)
    if not result.get("ok", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
