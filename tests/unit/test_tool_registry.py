from __future__ import annotations

from kg_rag_agent.tools.permissions import ToolPolicy
from kg_rag_agent.tools.registry import ToolRegistry
from kg_rag_agent.tools.schemas import ToolCallContext, ToolPermission


def _schema():
    return {
        "type": "object",
        "properties": {"value": {"type": "integer", "minimum": 1}},
        "required": ["value"],
        "additionalProperties": False,
    }


def test_tool_registry_validates_executes_and_normalizes_result():
    registry = ToolRegistry(
        policy=ToolPolicy.create(allowed_permissions=["read"], max_calls_per_request=2)
    )
    registry.register_function(
        "math.double",
        lambda value: value * 2,
        description="double",
        input_schema=_schema(),
        permissions=[ToolPermission.READ],
    )
    context = ToolCallContext.create(max_calls=2)
    result = registry.invoke("math.double", {"value": 4}, context=context)
    assert result.ok is True
    assert result.data == 8
    assert context.call_count == 1
    registry.close()


def test_tool_registry_rejects_invalid_arguments_and_budget():
    registry = ToolRegistry(
        policy=ToolPolicy.create(allowed_permissions=["read"], max_calls_per_request=1)
    )
    registry.register_function(
        "math.double",
        lambda value: value * 2,
        description="double",
        input_schema=_schema(),
        permissions=[ToolPermission.READ],
    )
    invalid = registry.invoke("math.double", {"value": 0})
    assert invalid.ok is False
    assert invalid.error["code"] == "tool_validation_error"

    context = ToolCallContext.create(max_calls=1)
    assert registry.invoke("math.double", {"value": 1}, context=context).ok
    denied = registry.invoke("math.double", {"value": 2}, context=context)
    assert denied.ok is False
    assert denied.error["code"] == "tool_permission_error"
    registry.close()
