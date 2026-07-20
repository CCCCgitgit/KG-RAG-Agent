# -*- coding: utf-8 -*-
"""基于 JSON 文件的轻量持久化 Memory Store。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Optional

from ..models import MemoryQuery, MemoryRecord, MemoryStatus
from .in_memory import InMemoryMemoryStore


class JSONMemoryStore(InMemoryMemoryStore):
    """适合单进程开发和小规模部署的原子 JSON Store。

    多 Worker 或多实例生产环境应替换为数据库 Store；此类不会伪装成分布式安全存储。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file_lock = RLock()
        super().__init__()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to load memory store: {self.path}") from exc
        if not isinstance(raw, list):
            raise ValueError("Memory store JSON root must be a list.")
        for item in raw:
            if not isinstance(item, dict):
                continue
            record = MemoryRecord.from_dict(item)
            self._records[record.memory_id] = record

    def _flush(self) -> None:
        payload = [record.to_dict() for record in self._records.values()]
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)

    def upsert(self, record: MemoryRecord) -> MemoryRecord:
        with self._file_lock:
            result = super().upsert(record)
            self._flush()
            return result

    def set_status(
        self,
        memory_id: str,
        status: MemoryStatus,
    ) -> Optional[MemoryRecord]:
        with self._file_lock:
            result = super().set_status(memory_id, status)
            if result is not None:
                self._flush()
            return result

    def delete(self, memory_id: str, *, hard: bool = False) -> bool:
        with self._file_lock:
            deleted = super().delete(memory_id, hard=hard)
            if deleted:
                self._flush()
            return deleted


__all__ = ["JSONMemoryStore"]
