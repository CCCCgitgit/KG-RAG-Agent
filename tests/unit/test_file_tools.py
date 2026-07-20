from __future__ import annotations

import pytest

from kg_rag_agent.tools.errors import ToolPathError, ToolPermissionError
from kg_rag_agent.tools.file_tools import FileTools


def test_file_tools_text_and_json_roundtrip(tmp_path):
    tools = FileTools(base_dir=tmp_path, enforce_base_dir=True)
    tools.write_text("notes/a.txt", "hello")
    assert tools.read_text("notes/a.txt") == "hello"

    tools.write_json("data/item.json", {"value": 3})
    assert tools.read_json("data/item.json") == {"value": 3}


def test_file_tools_blocks_path_escape(tmp_path):
    tools = FileTools(base_dir=tmp_path, enforce_base_dir=True)
    with pytest.raises(ToolPathError):
        tools.read_text("../outside.txt")
    with pytest.raises(ToolPathError):
        tools.write_text(tmp_path.parent / "outside.txt", "x")


def test_file_tools_delete_permission(tmp_path):
    tools = FileTools(base_dir=tmp_path, enforce_base_dir=True, allow_delete=False)
    tools.write_text("a.txt", "hello")
    with pytest.raises(ToolPermissionError):
        tools.delete_file("a.txt")
