# -*- coding: utf-8 -*-
"""KG-RAG Agent 离线构建总编排。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from .data_loader import process_raw_data
from .graph_builder import build_graph
from .manifest import build_manifest, write_manifest_copies
from .schemas import BuildOptions, StageResult
from .vector_builder import build_vector_store


class DataPipeline:
    """只负责离线数据产物构建，不参与在线问答。"""

    def __init__(self, options: BuildOptions) -> None:
        self.options = options

    def run(self) -> dict[str, Any]:
        options = self.options
        options.paths.ensure_output_dirs()
        stage_results: dict[str, StageResult] = {}

        try:
            stats = process_raw_data(
                raw_dir=options.paths.raw_dir,
                processed_dir=options.paths.processed_dir,
                max_rows=options.max_rows,
            )
            stats["build_id"] = options.build_id
            stats["profile"] = options.paths.profile
            stage_results["data"] = StageResult("data", True, stats)
        except Exception as exc:
            stage_results["data"] = StageResult("data", False, error=str(exc))
            return self._finalize(stage_results, include_vector_store=False)

        try:
            stats = build_graph(
                processed_dir=options.paths.processed_dir,
                kg_dir=options.paths.kg_dir,
                overwrite=options.overwrite_graph,
            )
            stats["build_id"] = options.build_id
            stats["profile"] = options.paths.profile
            stage_results["graph"] = StageResult("graph", True, stats)
        except Exception as exc:
            stage_results["graph"] = StageResult("graph", False, error=str(exc))
            return self._finalize(stage_results, include_vector_store=False)

        if options.skip_vector_store:
            stage_results["vector_store"] = StageResult(
                "vector_store",
                True,
                {"skipped": True, "reason": "skip_vector_store"},
            )
            return self._finalize(stage_results, include_vector_store=False)

        try:
            stats = build_vector_store(
                processed_dir=options.paths.processed_dir,
                kg_dir=options.paths.kg_dir,
                vector_store_dir=options.paths.vector_store_dir,
                collection_name=options.collection_name,
                model_name=options.model_name,
                local_files_only=options.local_files_only,
                allow_hash_fallback=options.allow_hash_fallback,
                batch_size=options.batch_size,
                reset=options.reset_vector_store,
            )
            stats["build_id"] = options.build_id
            stats["profile"] = options.paths.profile
            stage_results["vector_store"] = StageResult("vector_store", True, stats)
        except Exception as exc:
            stage_results["vector_store"] = StageResult(
                "vector_store", False, error=str(exc)
            )

        return self._finalize(stage_results, include_vector_store=True)

    def _finalize(
        self,
        stage_results: dict[str, StageResult],
        *,
        include_vector_store: bool,
    ) -> dict[str, Any]:
        stage_dict = {name: item.as_dict() for name, item in stage_results.items()}
        manifest = build_manifest(
            build_id=self.options.build_id,
            paths=self.options.paths,
            options=self.options.as_dict(),
            stages=stage_dict,
            include_vector_store=include_vector_store,
        )
        manifest_paths = write_manifest_copies(
            manifest,
            paths=self.options.paths,
            include_vector_store=include_vector_store,
        )
        stages_ok = all(item.ok for item in stage_results.values())
        return {
            "ok": bool(stages_ok and manifest.get("ok")),
            "build_id": self.options.build_id,
            "profile": self.options.paths.profile,
            "paths": self.options.paths.as_dict(),
            "stages": stage_dict,
            "manifest": manifest,
            "manifest_paths": manifest_paths,
        }


def build_all(
    *,
    options: Optional[BuildOptions] = None,
    profile: Optional[str] = None,
    raw_dir: str | None = None,
    processed_dir: str | None = None,
    kg_dir: str | None = None,
    vector_store_dir: str | None = None,
    manifest_path: str | None = None,
    build_id: Optional[str] = None,
    max_rows: Optional[int] = None,
    overwrite_graph: bool = True,
    local_files_only: bool = True,
    allow_hash_fallback: bool = False,
    batch_size: int = 128,
    reset_vector_store: bool = True,
    skip_vector_store: bool = False,
    collection_name: str = "kg_entities",
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> dict[str, Any]:
    """兼容旧参数的一键离线构建入口。"""

    resolved_options = options or BuildOptions.build(
        profile=profile,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        kg_dir=kg_dir,
        vector_store_dir=vector_store_dir,
        manifest_path=manifest_path,
        build_id=build_id,
        max_rows=max_rows,
        overwrite_graph=overwrite_graph,
        reset_vector_store=reset_vector_store,
        skip_vector_store=skip_vector_store,
        collection_name=collection_name,
        model_name=model_name,
        local_files_only=local_files_only,
        allow_hash_fallback=allow_hash_fallback,
        batch_size=batch_size,
    )
    return DataPipeline(resolved_options).run()


__all__ = ["DataPipeline", "build_all"]
