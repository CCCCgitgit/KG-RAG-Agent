from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kg_rag_agent.agents.schemas import AgentResult
from kg_rag_agent.app.server import create_app
from kg_rag_agent.app.settings import APISettings


class FakeService:
    def ask(self, query: str, **kwargs):
        return AgentResult(
            answer=f"API:{query}",
            request_id=kwargs.get("request_id") or "req_api",
            route="kg_rag",
            answerability="answerable",
        )

    def stream(self, query: str, **kwargs):
        yield {"event": "done", "query": query}

    def health_check(self):
        return {"ok": True}

    def info(self):
        return {"service": "fake"}


def _client(*, allow_raw_state: bool = False):
    settings = APISettings(
        initialize_service_on_startup=False,
        allow_raw_state=allow_raw_state,
        cors_origins=(),
    )
    return TestClient(create_app(settings=settings, agent_service=FakeService()))


def test_chat_endpoint_and_validation():
    with _client() as client:
        response = client.post("/api/chat", json={"query": "hello"})
        assert response.status_code == 200
        assert response.json()["answer"] == "API:hello"
        invalid = client.post("/api/chat", json={"query": ""})
        assert invalid.status_code == 422


def test_raw_state_permission_boundary():
    with _client(allow_raw_state=False) as client:
        response = client.post(
            "/api/chat",
            json={"query": "hello", "include_raw_state": True},
        )
        assert response.status_code == 403
        assert response.json()["error_code"] == "raw_state_disabled"


def test_health_and_stream_endpoints():
    with _client() as client:
        assert client.get("/api/health").status_code == 200
        response = client.post("/api/chat/stream", json={"query": "hello"})
        assert response.status_code == 200
        assert '"event": {"event": "done"' in response.text
