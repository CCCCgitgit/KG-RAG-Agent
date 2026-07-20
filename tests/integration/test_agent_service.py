from __future__ import annotations

from kg_rag_agent.services import AgentService


def test_service_agent_boundary_with_fake_agent(fake_agent):
    service = AgentService(agent=fake_agent)
    result = service.ask(
        "A 与 B 有什么关系？",
        request_id="req_fixed",
        session_id="sess_fixed",
        request_options={"retrieval_top_k": 10},
    )
    assert result.request_id == "req_fixed"
    assert fake_agent.calls[-1]["session_id"] == "sess_fixed"
    assert fake_agent.calls[-1]["request_options"] == {"retrieval_top_k": 10}


def test_service_stream_reuses_same_agent(fake_agent):
    service = AgentService(agent=fake_agent)
    events = list(service.stream("stream query", request_id="req_stream"))
    assert events[-1]["event"] == "completed"
    assert fake_agent.calls[-1]["request_id"] == "req_stream"
