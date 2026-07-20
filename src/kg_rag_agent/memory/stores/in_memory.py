# -*- coding: utf-8 -*-
"""线程安全的内存 Memory Store。"""

from __future__ import annotations

import copy
from threading import RLock
from typing import Optional

from ..models import MemoryQuery, MemoryRecord, MemoryStatus, utc_now
from .base import MemoryStore


class InMemoryMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._lock = RLock()

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        with self._lock:
            record = self._records.get(str(memory_id))
            return copy.deepcopy(record) if record is not None else None

    def upsert(self, record: MemoryRecord) -> MemoryRecord:
        if not isinstance(record, MemoryRecord):
            raise TypeError("record must be MemoryRecord")
        with self._lock:
            self._records[record.memory_id] = copy.deepcopy(record)
            return copy.deepcopy(record)

    def list(self, query: MemoryQuery) -> list[MemoryRecord]:
        query = query.normalized()
        with self._lock:
            records = [copy.deepcopy(item) for item in self._records.values()]
        result: list[MemoryRecord] = []
        for record in records:
            if record.namespace != query.namespace:
                continue
            if query.user_id and record.user_id != query.user_id:
                continue
            if query.project_id and record.project_id != query.project_id:
                continue
            if query.memory_types and record.memory_type not in query.memory_types:
                continue
            if record.session_id:
                if not query.include_session_memories:
                    continue
                if query.session_id and record.session_id != query.session_id:
                    continue
            if not record.is_active:
                continue
            result.append(record)
        result.sort(key=lambda item: item.updated_at, reverse=True)
        return result

    def set_status(
        self,
        memory_id: str,
        status: MemoryStatus,
    ) -> Optional[MemoryRecord]:
        status = status if isinstance(status, MemoryStatus) else MemoryStatus(str(status))
        with self._lock:
            record = self._records.get(str(memory_id))
            if record is None:
                return None
            record.status = status
            record.updated_at = utc_now()
            return copy.deepcopy(record)

    def delete(self, memory_id: str, *, hard: bool = False) -> bool:
        with self._lock:
            if str(memory_id) not in self._records:
                return False
            if hard:
                self._records.pop(str(memory_id), None)
            else:
                record = self._records[str(memory_id)]
                record.status = MemoryStatus.DELETED
                record.updated_at = utc_now()
            return True

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def count(self) -> int:
        with self._lock:
            return len(self._records)


__all__ = ["InMemoryMemoryStore"]
