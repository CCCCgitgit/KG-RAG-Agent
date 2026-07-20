# -*- coding: utf-8 -*-
"""Agent 门面层抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Union

from .schemas import AgentResult


class BaseAgent(ABC):
    """所有具体 Agent 应遵循的稳定接口。"""

    @abstractmethod
    def ask(self, query: str, **kwargs: Any) -> AgentResult:
        """执行一次标准问答。"""

        raise NotImplementedError

    @abstractmethod
    def invoke(
        self,
        query_or_state: Union[str, Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """使用问题字符串或完整状态调用 Agent。"""

        raise NotImplementedError

    def batch_ask(
        self,
        queries: Iterable[str],
        **kwargs: Any,
    ) -> List[AgentResult]:
        """默认串行执行多个问题。"""

        return [self.ask(query, **kwargs) for query in queries]

    def stream(self, query: str, **kwargs: Any) -> Any:
        """流式接口由具体 Agent 实现。"""

        raise NotImplementedError("stream() is not implemented by this agent.")

    def health_check(self) -> Dict[str, Any]:
        """返回最小健康状态。"""

        return {"ok": True, "agent": self.__class__.__name__}

    def info(self) -> Dict[str, Any]:
        """返回最小 Agent 信息。"""

        return {"agent": self.__class__.__name__}
