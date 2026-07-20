# -*- coding: utf-8 -*-
"""兼容旧调用方式的 FastAPI 服务启动脚本。

正式的参数解析、Uvicorn 启动和应用工厂位于 ``kg_rag_agent.main``。
本脚本只保留旧的 ``run_server`` 调用接口，并把参数转发到统一入口。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

try:
    from _script_runtime import bootstrap_src
except ImportError:  # pragma: no cover
    from scripts._script_runtime import bootstrap_src

bootstrap_src()

from kg_rag_agent.main import (  # noqa: E402
    build_parser,
    main as _main,
)

DEFAULT_APP = "kg_rag_agent.main:create_server_app"
LEGACY_DEFAULT_APP = "kg_rag_agent.app.server:app"


def run_server(
    *,
    app: str = DEFAULT_APP,
    host: str = "127.0.0.1",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
    log_level: str = "info",
    factory: bool = True,
    config: Optional[str] = None,
    api_prefix: Optional[str] = None,
    docs_enabled: bool = True,
    allow_raw_state: bool = False,
    initialize_service: bool = True,
) -> int:
    """兼容旧函数接口并转发到统一 ``--serve`` 入口。

    ``app`` 与 ``factory`` 参数仅为兼容历史调用保留。系统固定使用
    ``kg_rag_agent.main:create_server_app`` 工厂，以确保 Runtime、
    AgentService 和 Memory 都由 FastAPI 生命周期统一管理。
    """

    if app not in {DEFAULT_APP, LEGACY_DEFAULT_APP}:
        raise ValueError(
            "Custom ASGI app targets are no longer supported by this "
            "compatibility script. Use uvicorn directly for a custom app."
        )

    arguments = [
        "--serve",
        "--host",
        str(host),
        "--port",
        str(port),
        "--workers",
        str(workers),
        "--log-level",
        str(log_level),
    ]

    if reload:
        arguments.append("--reload")
    if config:
        arguments.extend(["--config", str(config)])
    if api_prefix:
        arguments.extend(["--api-prefix", str(api_prefix)])
    if not docs_enabled:
        arguments.append("--no-docs")
    if allow_raw_state:
        arguments.append("--allow-api-raw-state")
    if not initialize_service:
        arguments.append("--no-service-init")

    # ``factory`` is intentionally ignored. The unified entry always uses
    # Uvicorn factory mode, including when legacy callers pass factory=False.
    _ = factory

    return _main(arguments)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """运行统一 CLI；未显式提供时自动加入 ``--serve``。"""

    arguments = list(argv) if argv is not None else None
    if arguments is None:
        import sys

        arguments = sys.argv[1:]

    forwarded = [item for item in arguments if item != "--serve"]
    return _main(["--serve", *forwarded])


__all__ = [
    "DEFAULT_APP",
    "LEGACY_DEFAULT_APP",
    "build_parser",
    "run_server",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
