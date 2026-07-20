# -*- coding: utf-8 -*-
"""Memory 读写、隔离与安全策略的集成测试。"""

from __future__ import annotations

import json
from typing import Any

import pytest

from kg_rag_agent.graph.nodes.memory_load_node import memory_load_node
from kg_rag_agent.graph.nodes.memory_write_node import memory_write_node
from kg_rag_agent.memory import (
    InMemoryMemoryStore,
    MemoryManager,
    MemoryPolicy,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.memory,
]


def _create_manager() -> MemoryManager:
    policy = MemoryPolicy(
        enabled=True,
        write_enabled=True,
        long_term_enabled=True,
        namespace_prefix="test_memory",
        max_messages=10,
        max_summary_tokens=256,
        max_retrieved_items=8,
        max_context_tokens=2000,
        min_relevance_score=0.0,
        max_write_candidates=8,
        require_user_id_for_long_term=True,
        reject_sensitive_content=True,
    )
    return MemoryManager(
        policy=policy,
        store=InMemoryMemoryStore(),
    )


def _state(
    *,
    query: str,
    answer: str = "",
    user_id: str = "user_a",
    project_id: str = "project_a",
    session_id: str = "session_a",
    fail_open: bool = False,
) -> dict[str, Any]:
    return {
        "request_id": "req_memory_test",
        "query": query,
        "normalized_query": query,
        "final_answer": answer,
        "user_id": user_id,
        "project_id": project_id,
        "session_id": session_id,
        "memory_candidates": [],
        "memory_loaded": False,
        "memory_written": False,
        "memory_context": {},
        "memory_text": "",
        "memory_write_result": {},
        "warnings": [],
        "traces": [],
        "config": {
            "memory": {
                "fail_open": fail_open,
                "track_conversation": True,
                "summary_enabled": False,
                "capture_explicit_memory": True,
            }
        },
    }


def test_explicit_memory_is_written_then_loaded_with_recent_turn() -> None:
    manager = _create_manager()

    first_turn = _state(
        query="请记住我偏好使用中文回答",
        answer="好的，后续我会优先使用中文回答。",
    )
    write_update = memory_write_node(
        first_turn,
        memory_manager=manager,
    )

    assert write_update["memory_written"] is True
    assert write_update["memory_write_result"]["written_count"] == 1
    assert manager.store.count() == 1

    recent = manager.get_recent_messages(
        "session_a",
        user_id="user_a",
        project_id="project_a",
    )
    assert [item["role"] for item in recent] == ["user", "assistant"]
    assert "偏好使用中文回答" in recent[0]["content"]

    second_turn = _state(
        query="请继续用中文回答这个问题",
        answer="",
    )
    load_update = memory_load_node(
        second_turn,
        memory_manager=manager,
    )

    assert load_update["memory_loaded"] is True
    assert "偏好使用中文回答" in load_update["memory_text"]
    assert len(load_update["memory_context"]["recent_messages"]) == 2
    assert len(load_update["memory_context"]["memories"]) == 1
    assert load_update["memory_context"]["estimated_tokens"] > 0

    # AgentState 中只能出现可序列化快照，不能写入 Manager 或 Store 对象。
    json.dumps(load_update, ensure_ascii=False, default=str)
    assert all(
        value is not manager and value is not manager.store
        for value in load_update.values()
    )


def test_long_term_memory_isolated_by_user_project_and_session() -> None:
    manager = _create_manager()

    result = manager.write_candidates(
        [
            {
                "content": "项目 A 的回答必须使用中文。",
                "memory_type": "project",
                "source": "integration_test",
                "confidence": 1.0,
            }
        ],
        user_id="user_a",
        project_id="project_a",
        session_id="",
    )
    assert result.written_count == 1

    manager.add_messages(
        session_id="session_a",
        user_id="user_a",
        project_id="project_a",
        messages=[
            {"role": "user", "content": "这是 session_a 的消息。"},
            {"role": "assistant", "content": "这是 session_a 的回答。"},
        ],
    )

    same_scope = manager.load_context(
        query="项目 A 中文回答要求",
        user_id="user_a",
        project_id="project_a",
        session_id="session_a",
    )
    assert len(same_scope.memories) == 1
    assert len(same_scope.recent_messages) == 2

    other_project = manager.load_context(
        query="项目 A 中文回答要求",
        user_id="user_a",
        project_id="project_b",
        session_id="session_a",
    )
    assert other_project.memories == []
    assert other_project.recent_messages == []

    other_user = manager.load_context(
        query="项目 A 中文回答要求",
        user_id="user_b",
        project_id="project_a",
        session_id="session_a",
    )
    assert other_user.memories == []
    assert other_user.recent_messages == []

    other_session = manager.load_context(
        query="项目 A 中文回答要求",
        user_id="user_a",
        project_id="project_a",
        session_id="session_b",
    )
    assert len(other_session.memories) == 1
    assert other_session.recent_messages == []


def test_short_term_memory_uses_full_tenant_scope() -> None:
    manager = _create_manager()

    manager.add_messages(
        session_id="shared_session",
        user_id="user_a",
        project_id="project_a",
        messages=[{"role": "user", "content": "仅属于 A 的消息。"}],
    )

    same_scope = manager.get_recent_messages(
        "shared_session",
        user_id="user_a",
        project_id="project_a",
    )
    other_user = manager.get_recent_messages(
        "shared_session",
        user_id="user_b",
        project_id="project_a",
    )
    other_project = manager.get_recent_messages(
        "shared_session",
        user_id="user_a",
        project_id="project_b",
    )

    assert len(same_scope) == 1
    assert other_user == []
    assert other_project == []


def test_sensitive_explicit_memory_is_rejected_but_turn_is_recorded() -> None:
    manager = _create_manager()

    state = _state(
        query="请记住我的 API key=sk-abcdefghijklmnop",
        answer="我不会将敏感凭据写入长期 Memory。",
    )
    update = memory_write_node(
        state,
        memory_manager=manager,
    )

    result = update["memory_write_result"]
    assert update["memory_written"] is True
    assert result["written_count"] == 0
    assert manager.store.count() == 0
    assert any(
        item.get("reason") == "sensitive_content"
        for item in result["skipped"]
    )

    # 对话上下文可以保留在受限的会话缓冲区，但不得进入长期 Store。
    assert len(
        manager.get_recent_messages(
            "session_a",
            user_id="user_a",
            project_id="project_a",
        )
    ) == 2


def test_memory_load_failure_can_fail_open_without_blocking_answer_flow() -> None:
    class BrokenManager:
        enabled = True

        def load_context(self, **_: Any) -> Any:
            raise RuntimeError("temporary memory backend failure")

    update = memory_load_node(
        _state(
            query="继续回答",
            fail_open=True,
        ),
        memory_manager=BrokenManager(),
    )

    assert update["memory_loaded"] is False
    assert update["memory_text"] == ""
    assert update["memory_context"]["memories"] == []
    assert update["warnings"]
    assert "continued without Memory enhancement" in update["warnings"][0]
    assert "has_error" not in update


def test_duplicate_memory_write_is_idempotently_skipped() -> None:
    manager = _create_manager()
    state = _state(
        query="请记住我喜欢简洁回答",
        answer="好的。",
    )

    first = memory_write_node(state, memory_manager=manager)
    assert first["memory_write_result"]["written_count"] == 1

    duplicate_state = {
        **state,
        **first,
        "memory_written": True,
    }
    second = memory_write_node(
        duplicate_state,
        memory_manager=manager,
    )

    assert second["memory_written"] is True
    assert second["traces"][0]["payload"]["skipped"] is True
    assert manager.store.count() == 1
