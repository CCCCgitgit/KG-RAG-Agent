from __future__ import annotations

import logging

import pytest

from kg_rag_agent.runtime import RuntimeContext, RuntimeDependencyError, RuntimeSettings


def _settings(tmp_path):
    return RuntimeSettings.from_mapping(
        {}, project_root=tmp_path, config_dir=tmp_path / "configs", validate=False
    )


def test_runtime_settings_isolation_and_overrides(tmp_path):
    settings = RuntimeSettings.from_mapping(
        {"model": {"name": "a"}, "retrieval": {"top_k": 5}},
        project_root=tmp_path,
        config_dir=tmp_path / "configs",
        validate=False,
    )
    copy_data = settings.to_dict()
    copy_data["model"]["name"] = "changed"
    assert settings.get("model.name") == "a"

    updated = settings.with_overrides(
        {"retrieval": {"top_k": 9}}, validate=False
    )
    assert updated.get("retrieval.top_k") == 9
    assert settings.get("retrieval.top_k") == 5


def test_runtime_context_register_require_and_close(tmp_path):
    runtime = RuntimeContext(
        settings=_settings(tmp_path), logger=logging.getLogger("runtime-test")
    )
    runtime.register("custom_dependency", {"ok": True})
    assert runtime.require("custom_dependency") == {"ok": True}
    assert runtime.unregister("custom_dependency") == {"ok": True}
    with pytest.raises(RuntimeDependencyError):
        runtime.require("custom_dependency")

    runtime.close()
    assert runtime.is_closed is True
    with pytest.raises(RuntimeError):
        runtime.get("llm_client")


def test_runtime_context_rejects_duplicate_dependency(tmp_path):
    runtime = RuntimeContext(
        settings=_settings(tmp_path), logger=logging.getLogger("runtime-test")
    )
    runtime.register("sample", 1)
    with pytest.raises(KeyError):
        runtime.register("sample", 2)
    runtime.register("sample", 2, overwrite=True)
    assert runtime.require("sample") == 2
