from __future__ import annotations

from pathlib import Path

from kg_rag_agent.llm.prompt_manager import PACKAGE_PROMPT_DIR, PromptManager


def test_prompt_manager_prefers_packaged_prompt_templates() -> None:
    project_root = Path(__file__).resolve().parents[2]
    manager = PromptManager(
        project_root=project_root,
        config_path=project_root / "configs" / "prompt.yaml",
        prompt_dir=None,
        auto_load=True,
    )

    assert manager.prompt_dir == PACKAGE_PROMPT_DIR.resolve()
    assert manager.get_record("generation.system", required=True).source == "file"
    assert manager.get_record("direct_llm.system", required=True).source == "file"
    assert "generation.system" in manager.list_prompts()
    assert "router.user" in manager.list_prompts()
