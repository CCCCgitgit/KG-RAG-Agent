# -*- coding: utf-8 -*-
"""将旧版顶层 data 目录迁移到 demo / production Profile。

本脚本只处理用户已经存在的本地数据，不会下载或重建模型。默认复制文件；
使用 ``--cleanup`` 时，在确认目标文件存在后删除旧目录。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
PRODUCTION_ROOT = DATA_ROOT / "production"

PRODUCTION_PROCESSED_FILES = (
    "data_loader_stats.json",
    "entity_index.json",
    "relation_index.json",
    "relations.json",
    "triples.csv",
)


def _copy_path(source: Path, target: Path, *, overwrite: bool) -> bool:
    if not source.exists():
        return False
    if target.exists():
        if not overwrite:
            return False
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    return True


def migrate(*, overwrite: bool = False, cleanup: bool = False) -> dict[str, object]:
    copied: list[str] = []
    skipped: list[str] = []

    raw_source = DATA_ROOT / "raw"
    raw_target = PRODUCTION_ROOT / "raw"
    if _copy_path(raw_source, raw_target, overwrite=overwrite):
        copied.append(raw_target.relative_to(PROJECT_ROOT).as_posix())
    elif raw_source.exists():
        skipped.append(raw_target.relative_to(PROJECT_ROOT).as_posix())

    legacy_processed = DATA_ROOT / "processed"
    production_processed = PRODUCTION_ROOT / "processed"
    for filename in PRODUCTION_PROCESSED_FILES:
        source = legacy_processed / filename
        target = production_processed / filename
        if _copy_path(source, target, overwrite=overwrite):
            copied.append(target.relative_to(PROJECT_ROOT).as_posix())
        elif source.exists():
            skipped.append(target.relative_to(PROJECT_ROOT).as_posix())

    if cleanup:
        cleanup_targets = (
            DATA_ROOT / "raw",
            DATA_ROOT / "processed",
            DATA_ROOT / "kg",
            DATA_ROOT / "vector_store",
            DATA_ROOT / "examples",
        )
        for path in cleanup_targets:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()

    return {
        "ok": True,
        "copied": copied,
        "skipped_existing": skipped,
        "cleanup": cleanup,
        "next_command": "python scripts/build_all.py --profile production",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    import json

    print(
        json.dumps(
            migrate(overwrite=args.overwrite, cleanup=args.cleanup),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
