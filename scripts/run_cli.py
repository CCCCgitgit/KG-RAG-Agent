# -*- coding: utf-8 -*-
"""兼容旧调用方式的 KG-RAG Agent CLI 脚本。

正式 CLI 实现位于 ``kg_rag_agent.main``。本文件只负责：

* 在未执行 editable install 时补充 ``src`` 路径；
* 转发到统一的 CLI 实现；
* 保留旧脚本中常用函数名称和参数默认值。

业务逻辑、Runtime、AgentService 和 Memory 生命周期不得在此重复实现。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

try:
    from _script_runtime import bootstrap_src
except ImportError:  # pragma: no cover
    from scripts._script_runtime import bootstrap_src

bootstrap_src()

from kg_rag_agent.main import (  # noqa: E402
    build_parser,
    create_cli_service,
    main as _main,
    render_result as _render_result,
    request_options_from_args,
    run_interactive as _run_interactive,
    run_once as _run_once,
)


def create_service(
    *,
    config_path: Optional[str],
    validate: bool,
) -> Any:
    """兼容旧脚本名称，实际创建逻辑由统一 CLI 入口负责。"""

    return create_cli_service(
        config_path=config_path,
        validate=validate,
    )


def render_result(
    result: Any,
    *,
    as_json: bool,
    include_raw_state: bool,
    include_identifiers: bool = False,
    include_memory_status: bool = False,
) -> None:
    """兼容旧输出接口，并支持新的标识符和 Memory 状态控制。"""

    _render_result(
        result,
        as_json=as_json,
        include_raw_state=include_raw_state,
        include_identifiers=include_identifiers,
        include_memory_status=include_memory_status,
    )


def run_once(
    service: Any,
    query: str,
    *,
    options: Mapping[str, Any],
    as_json: bool,
    raw: bool,
    user_id: Optional[str] = None,
    project_id: Optional[str] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    include_identifiers: bool = False,
    include_memory_status: bool = False,
) -> Any:
    """兼容旧单轮调用，同时透传 Memory 隔离标识。"""

    return _run_once(
        service,
        query,
        options=options,
        user_id=user_id,
        project_id=project_id,
        session_id=session_id,
        request_id=request_id,
        as_json=as_json,
        raw=raw,
        include_identifiers=include_identifiers,
        include_memory_status=include_memory_status,
    )


def run_interactive(
    service: Any,
    *,
    options: Mapping[str, Any],
    as_json: bool,
    raw: bool,
    user_id: Optional[str] = None,
    project_id: Optional[str] = None,
    session_id: Optional[str] = None,
    include_identifiers: bool = False,
    include_memory_status: bool = False,
) -> None:
    """兼容旧交互调用，并在统一 session_id 下维持 Memory 连续性。"""

    _run_interactive(
        service,
        options=options,
        user_id=user_id,
        project_id=project_id,
        session_id=session_id,
        as_json=as_json,
        raw=raw,
        include_identifiers=include_identifiers,
        include_memory_status=include_memory_status,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """转发到 ``kg_rag_agent.main.main``。"""

    return _main(argv)


__all__ = [
    "build_parser",
    "request_options_from_args",
    "create_service",
    "render_result",
    "run_once",
    "run_interactive",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
