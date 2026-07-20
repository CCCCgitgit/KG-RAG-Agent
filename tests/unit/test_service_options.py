from __future__ import annotations

import pytest

from kg_rag_agent.services.agent_service import AgentService


def test_agent_service_forwards_only_whitelisted_options(fake_agent):
    service = AgentService(agent=fake_agent)
    result = service.ask(
        "test",
        request_options={
            "retrieval_top_k": 8,
            "temperature": 0.2,
            "include_citations": True,
        },
        config_overrides={
            "graph_path": "/unsafe/path",
            "model": {"api_key": "secret"},
        },
    )
    assert result.answer == "回答：test"
    options = fake_agent.calls[-1]["request_options"]
    assert options == {
        "retrieval_top_k": 8,
        "temperature": 0.2,
        "include_citations": True,
    }
    assert "graph_path" not in options


def test_agent_service_rejects_unknown_request_option(fake_agent):
    service = AgentService(agent=fake_agent)
    with pytest.raises(ValueError):
        service.ask("test", request_options={"graph_path": "x"})


@pytest.mark.parametrize(
    ("options", "error_type"),
    [
        ({"retrieval_top_k": 0}, ValueError),
        ({"path_max_depth": 7}, ValueError),
        ({"temperature": 3}, ValueError),
        ({"max_tokens": 0}, ValueError),
        ({"include_citations": "yes"}, TypeError),
        ({"allowed_tools": "file.read"}, TypeError),
    ],
)
def test_agent_service_validates_option_boundaries(fake_agent, options, error_type):
    service = AgentService(agent=fake_agent)
    with pytest.raises(error_type):
        service.ask("test", request_options=options)


def test_agent_service_batch_and_health(fake_agent):
    service = AgentService(agent=fake_agent)
    results = service.batch_ask(["a", "b"])
    assert [item.answer for item in results] == ["回答：a", "回答：b"]
    assert service.health_check()["ok"] is True
    assert service.info()["request_option_whitelist"]
