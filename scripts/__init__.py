# -*- coding: utf-8 -*-
"""KG-RAG Agent 轻量脚本包。

该目录只保存开发、构建、评估和兼容启动脚本。导入 ``scripts`` 时不会
自动导入任何具体脚本，也不会初始化 Runtime、AgentService、LangGraph、
LLM、知识图谱、向量库或 Memory Store。
"""

__all__ = [
    "build_all",
    "build_data",
    "build_demo_graph",
    "build_entity_vector_store",
    "build_kg",
    "build_vector_store",
    "evaluate",
    "migrate_legacy_data",
    "run_cli",
    "run_api",
    "run_server",
]
