from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class FakeGraph:
    """不依赖真实 LangGraph 的最小 CompiledGraph 替身。"""

    def __init__(self, *, answer: str = "测试回答") -> None:
        self.answer = answer
        self.invocations: list[dict[str, Any]] = []
        self.stream_invocations: list[dict[str, Any]] = []

    def invoke(self, state: dict[str, Any], config: dict[str, Any] | None = None):
        self.invocations.append({"state": dict(state), "config": config})
        result = dict(state)
        result.update(
            {
                "final_answer": self.answer,
                "route": "kg_rag",
                "answerability": "answerable",
                "semantic_score": 0.91,
                "citations": [{"evidence_id": "ev_1"}],
                "traces": [{"node": "fake"}],
                "warnings": [],
                "has_error": False,
                "error_message": "",
            }
        )
        return result

    def stream(self, state: dict[str, Any], config: dict[str, Any] | None = None):
        self.stream_invocations.append({"state": dict(state), "config": config})
        yield {"event": "started"}
        yield {"event": "completed", "answer": self.answer}


class FakeAgent:
    def __init__(self, result_factory=None) -> None:
        self.runtime = None
        self.calls: list[dict[str, Any]] = []
        self._result_factory = result_factory
        self._graph = FakeGraph()

    def ask(self, query: str, **kwargs: Any):
        from kg_rag_agent.agents.schemas import AgentResult

        self.calls.append({"query": query, **kwargs})
        if self._result_factory is not None:
            return self._result_factory(query, kwargs)
        return AgentResult(
            answer=f"回答：{query}",
            request_id=kwargs.get("request_id") or "req_test",
            route="kg_rag",
            answerability="answerable",
            semantic_score=0.8,
        )

    def invoke(self, state, **kwargs):
        return dict(state)

    def stream(self, query: str, **kwargs: Any):
        self.calls.append({"query": query, **kwargs})
        yield {"event": "completed", "query": query}

    def get_graph(self):
        return self._graph

    def rebuild_graph(self):
        self._graph = FakeGraph(answer="重建后回答")
        return self._graph

    def health_check(self):
        return {"ok": True, "agent": "FakeAgent"}

    def info(self):
        return {"agent": "FakeAgent"}

    def close(self):
        return None


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("kg_rag_agent.tests")


@pytest.fixture
def fake_graph() -> FakeGraph:
    return FakeGraph()


@pytest.fixture
def fake_agent() -> FakeAgent:
    return FakeAgent()
