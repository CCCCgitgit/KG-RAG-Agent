# -*- coding: utf-8 -*-
"""兼容脚本形式的 KG-RAG Agent FastAPI 启动入口。

正式的 Web 启动、Runtime 创建和生命周期管理位于
``kg_rag_agent.main`` 与 ``kg_rag_agent.app``。本脚本只负责补充 ``src``
路径并将参数转发给统一入口，不维护第二套 Uvicorn 或 Service 初始化逻辑。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

try:
    from _script_runtime import bootstrap_src
except ImportError:  # pragma: no cover
    from scripts._script_runtime import bootstrap_src

bootstrap_src()

from kg_rag_agent.main import main as _main  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    """以 ``--serve`` 模式调用统一命令行入口。"""

    forwarded = list(argv) if argv is not None else None

    if forwarded is None:
        import sys

        forwarded = sys.argv[1:]

    # 允许旧调用方显式传入 --serve，但内部只保留一次。
    arguments = [item for item in forwarded if item != "--serve"]
    return _main(["--serve", *arguments])


def run_api(argv: Optional[Sequence[str]] = None) -> int:
    """语义明确的 API 启动别名。"""

    return main(argv)


def run_server(argv: Optional[Sequence[str]] = None) -> int:
    """保留旧脚本可能使用的函数名称。"""

    return main(argv)


__all__ = [
    "main",
    "run_api",
    "run_server",
]


if __name__ == "__main__":
    raise SystemExit(main())
