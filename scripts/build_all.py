# -*- coding: utf-8 -*-
"""一键构建 processed、KG、向量库及构建清单。"""
from __future__ import annotations

import argparse
from typing import Any, Optional

try:
    from _script_runtime import positive_int, print_json
except ImportError:  # pragma: no cover
    from scripts._script_runtime import positive_int, print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build all KG-RAG offline artifacts.")
    parser.add_argument("--profile", choices=["demo", "production"], default="demo")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--kg-dir", default=None)
    parser.add_argument("--vector-store-dir", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--build-id", default=None)
    parser.add_argument("--max-rows", type=positive_int, default=None)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--collection-name", default="kg_entities")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-hash-fallback", action="store_true")
    parser.add_argument("--no-overwrite-graph", action="store_true")
    parser.add_argument("--no-reset-vector-store", action="store_true")
    parser.add_argument("--skip-vector-store", action="store_true")
    return parser


def run(
    *,
    profile: Optional[str] = "demo",
    raw_dir: Optional[str] = None,
    processed_dir: Optional[str] = None,
    kg_dir: Optional[str] = None,
    vector_store_dir: Optional[str] = None,
    manifest_path: Optional[str] = None,
    build_id: Optional[str] = None,
    max_rows: Optional[int] = None,
    batch_size: int = 128,
    collection_name: str = "kg_entities",
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    local_files_only: bool = True,
    allow_hash_fallback: bool = False,
    overwrite_graph: bool = True,
    reset_vector_store: bool = True,
    skip_vector_store: bool = False,
) -> dict[str, Any]:
    from kg_rag_agent.data_pipeline import build_all

    return build_all(
        profile=profile,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        kg_dir=kg_dir,
        vector_store_dir=vector_store_dir,
        manifest_path=manifest_path,
        build_id=build_id,
        max_rows=max_rows,
        batch_size=batch_size,
        collection_name=collection_name,
        model_name=model_name,
        local_files_only=local_files_only,
        allow_hash_fallback=allow_hash_fallback,
        overwrite_graph=overwrite_graph,
        reset_vector_store=reset_vector_store,
        skip_vector_store=skip_vector_store,
    )


def main() -> None:
    args = build_parser().parse_args()
    result = run(
        profile=args.profile,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        kg_dir=args.kg_dir,
        vector_store_dir=args.vector_store_dir,
        manifest_path=args.manifest_path,
        build_id=args.build_id,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        collection_name=args.collection_name,
        model_name=args.model_name,
        local_files_only=not args.allow_download,
        allow_hash_fallback=args.allow_hash_fallback,
        overwrite_graph=not args.no_overwrite_graph,
        reset_vector_store=not args.no_reset_vector_store,
        skip_vector_store=args.skip_vector_store,
    )
    print_json(result)
    if not result.get("ok", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
