# -*- coding: utf-8 -*-
"""离线构建统一参数与结果协议。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from .paths import PipelinePaths


def new_build_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"


@dataclass(frozen=True)
class BuildOptions:
    """完整离线构建的统一参数。"""

    paths: PipelinePaths
    build_id: str = field(default_factory=new_build_id)
    max_rows: Optional[int] = None
    overwrite_graph: bool = True
    reset_vector_store: bool = True
    skip_vector_store: bool = False
    collection_name: str = "kg_entities"
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    local_files_only: bool = True
    allow_hash_fallback: bool = False
    batch_size: int = 128

    def __post_init__(self) -> None:
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows must be greater than 0 when provided.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        if not str(self.build_id).strip():
            raise ValueError("build_id must not be empty.")
        if not str(self.collection_name).strip():
            raise ValueError("collection_name must not be empty.")
        if not str(self.model_name).strip():
            raise ValueError("model_name must not be empty.")

    @classmethod
    def build(
        cls,
        *,
        profile: Optional[str] = None,
        raw_dir: str | None = None,
        processed_dir: str | None = None,
        kg_dir: str | None = None,
        vector_store_dir: str | None = None,
        manifest_path: str | None = None,
        build_id: Optional[str] = None,
        max_rows: Optional[int] = None,
        overwrite_graph: bool = True,
        reset_vector_store: bool = True,
        skip_vector_store: bool = False,
        collection_name: str = "kg_entities",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        local_files_only: bool = True,
        allow_hash_fallback: bool = False,
        batch_size: int = 128,
    ) -> "BuildOptions":
        paths = PipelinePaths.build(
            profile=profile,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            kg_dir=kg_dir,
            vector_store_dir=vector_store_dir,
            manifest_path=manifest_path,
        )
        return cls(
            paths=paths,
            build_id=build_id or new_build_id(),
            max_rows=max_rows,
            overwrite_graph=bool(overwrite_graph),
            reset_vector_store=bool(reset_vector_store),
            skip_vector_store=bool(skip_vector_store),
            collection_name=str(collection_name),
            model_name=str(model_name),
            local_files_only=bool(local_files_only),
            allow_hash_fallback=bool(allow_hash_fallback),
            batch_size=int(batch_size),
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["paths"] = self.paths.as_dict()
        return result


@dataclass
class StageResult:
    name: str
    ok: bool
    stats: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["BuildOptions", "StageResult", "new_build_id"]
