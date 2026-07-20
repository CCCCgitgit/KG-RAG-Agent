from __future__ import annotations

from kg_rag_agent.agents.schemas import AgentRequest, AgentResult


def test_agent_request_normalizes_and_copies_input():
    messages = [{"role": "user", "content": "hello"}]
    request = AgentRequest(
        query="  question  ",
        request_id=" req_1 ",
        session_id=" sess_1 ",
        messages=messages,
        metadata={"source": "test"},
        request_options={"retrieval_top_k": 5},
    )
    messages[0]["content"] = "changed"
    assert request.query == "question"
    assert request.request_id == "req_1"
    assert request.messages[0]["content"] == "hello"
    assert request.state_metadata()["request_options"]["retrieval_top_k"] == 5


def test_agent_result_from_state_and_raw_state_boundary():
    state = {
        "request_id": "req_2",
        "final_answer": "answer",
        "route": "kg_rag",
        "answerability": "answerable",
        "semantic_score": "0.75",
        "citations": [{"id": "c1"}],
        "traces": [{"node": "generation"}],
    }
    result = AgentResult.from_state(state, request_id="fallback", include_raw_state=True)
    assert result.answer == "answer"
    assert result.semantic_score == 0.75
    assert result.raw_state == state
    assert "raw_state" not in result.to_dict()
    assert result.to_dict(include_raw_state=True)["raw_state"] == state


def test_agent_result_error_has_stable_shape():
    result = AgentResult.error(request_id="req_err", message="boom")
    assert result.has_error is True
    assert result.route == "error"
    assert result.answerability == "unanswerable"
    assert result.error_message == "boom"
