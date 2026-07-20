from __future__ import annotations


def test_core_layer_imports():
    from kg_rag_agent.agents.schemas import AgentResult
    from kg_rag_agent.app.settings import APISettings
    from kg_rag_agent.evaluation.schemas import EvaluationCase
    from kg_rag_agent.runtime import RuntimeContext, RuntimeSettings
    from kg_rag_agent.services import AgentService
    from kg_rag_agent.tools import ToolRegistry

    assert AgentResult
    assert APISettings
    assert EvaluationCase
    assert RuntimeContext
    assert RuntimeSettings
    assert AgentService
    assert ToolRegistry
