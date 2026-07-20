# -*- coding: utf-8 -*-
"""Memory Store 抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

from ..models import MemoryQuery, MemoryRecord, MemoryStatus


class MemoryStore(ABC):
    """长期 Memory 存储适配器接口。"""

    @abstractmethod
    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, record: MemoryRecord) -> MemoryRecord:
        raise NotImplementedError

    def upsert_many(self, records: Iterable[MemoryRecord]) -> list[MemoryRecord]:
        return [self.upsert(record) for record in records]

    @abstractmethod
    def list(self, query: MemoryQuery) -> list[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def set_status(
        self,
        memory_id: str,
        status: MemoryStatus,
    ) -> Optional[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, memory_id: str, *, hard: bool = False) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        """默认 Store 无需释放资源。"""


__all__ = ["MemoryStore"]
