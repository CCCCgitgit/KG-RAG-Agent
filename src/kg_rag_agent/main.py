# -*- coding: utf-8 -*-
"""KG-RAG Agent 安装后的统一命令行入口。

默认运行单次或交互式 CLI；使用 ``--serve`` 启动 FastAPI。两种模式均复用
AgentService、RuntimeContext 和 MemoryManager，不在模块导入阶段初始化重型依赖。
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Optional, Sequence

_PACKAGE_VERSION = "0.1.0"
_SERVER_FACTORY = "kg_rag_agent.main:create_server_app"
_SERVER_CONFIG_ENV = "KG_RAG_MAIN_CONFIG_PATH"
_SERVER_VALIDATE_ENV = "KG_RAG_MAIN_VALIDATE"

_LOG_LEVELS = (
    "critical",
    "error",
    "warning",
    "info",
    "debug",
    "trace",
)


def build_parser() -> argparse.ArgumentParser:
    """构造兼容旧 CLI 的统一参数解析器。"""

    parser = argparse.ArgumentParser(
        prog="kg-rag-agent",
        description=(
            "Run KG-RAG Agent in CLI mode, or start the FastAPI service "
            "with --serve."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_PACKAGE_VERSION}",
    )

    mode_group = parser.add_argument_group("execution mode")
    mode_group.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI service instead of the CLI.",
    )
    mode_group.add_argument(
        "--interactive",
        action="store_true",
        help="Keep one CLI session open for multiple questions.",
    )

    request_group = parser.add_argument_group("request")
    request_group.add_argument("-q", "--query", default="")
    request_group.add_argument("-c", "--config", default=None)
    request_group.add_argument("--user-id", default=None)
    request_group.add_argument("--project-id", default=None)
    request_group.add_argument("--session-id", default=None)
    request_group.add_argument("--request-id", default=None)
    request_group.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print the complete serializable result as JSON.",
    )
    request_group.add_argument(
        "--raw",
        action="store_true",
        help="Include the final raw AgentState in local CLI JSON output.",
    )
    request_group.add_argument(
        "--memory-status",
        action="store_true",
        help="Include the non-sensitive Memory execution summary.",
    )
    request_group.add_argument(
        "--include-identifiers",
        action="store_true",
        help="Include user/project/session identifiers in JSON output.",
    )
    request_group.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip startup dependency validation.",
    )

    option_group = parser.add_argument_group("request options")
    option_group.add_argument("--retrieval-top-k", type=int, default=None)
    option_group.add_argument("--path-max-depth", type=int, default=None)
    option_group.add_argument("--temperature", type=float, default=None)
    option_group.add_argument("--max-tokens", type=int, default=None)
    option_group.add_argument("--language", default=None)
    option_group.add_argument(
        "--allowed-tool",
        action="append",
        default=None,
        dest="allowed_tools",
        help="Allow one tool for this request; repeat for multiple tools.",
    )
    option_group.add_argument("--no-citations", action="store_true")

    server_group = parser.add_argument_group("FastAPI server")
    server_group.add_argument("--host", default="127.0.0.1")
    server_group.add_argument("--port", type=int, default=8000)
    server_group.add_argument("--workers", type=int, default=1)
    server_group.add_argument("--reload", action="store_true")
    server_group.add_argument(
        "--log-level",
        choices=_LOG_LEVELS,
        default="info",
    )
    server_group.add_argument("--api-prefix", default=None)
    server_group.add_argument(
        "--no-docs",
        action="store_true",
        help="Disable OpenAPI, Swagger UI and ReDoc.",
    )
    server_group.add_argument(
        "--allow-api-raw-state",
        action="store_true",
        help="Allow authenticated/trusted API callers to request raw_state.",
    )
    server_group.add_argument(
        "--no-service-init",
        action="store_true",
        help="Start HTTP routes without initializing AgentService.",
    )

    return parser


def request_options_from_args(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """提取 AgentService 允许的请求级白名单参数。"""

    values = {
        "retrieval_top_k": args.retrieval_top_k,
        "path_max_depth": args.path_max_depth,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "language": args.language,
        "include_citations": not args.no_citations,
        "allowed_tools": args.allowed_tools,
    }
    return {
        key: value
        for key, value in values.items()
        if value is not None
    }


def create_cli_service(
    *,
    config_path: Optional[str],
    validate: bool,
) -> Any:
    """延迟创建 CLI 使用的 AgentService。"""

    from kg_rag_agent.services import AgentService

    return AgentService(
        config_path=config_path,
        validate=validate,
    )


def create_server_app() -> Any:
    """供 Uvicorn ``--factory`` 模式调用的 FastAPI 工厂。

    ``run_server`` 使用临时环境变量把配置文件和校验开关传给 Uvicorn 的
    worker/reload 子进程，避免在模块导入阶段创建 Runtime。
    """

    from kg_rag_agent.app import APISettings, create_app

    config_path = _clean_optional(
        os.getenv(_SERVER_CONFIG_ENV)
    )
    validate = _parse_env_bool(
        os.getenv(_SERVER_VALIDATE_ENV),
        default=True,
    )
    settings = APISettings.from_env()

    return create_app(
        settings=settings,
        config_path=config_path,
        validate=validate,
    )


def render_result(
    result: Any,
    *,
    as_json: bool,
    include_raw_state: bool,
    include_identifiers: bool,
    include_memory_status: bool,
) -> None:
    """输出标准 AgentResult，并避免默认暴露原始状态和 Memory 正文。"""

    if as_json:
        payload = _result_to_dict(
            result,
            include_raw_state=include_raw_state,
            include_identifiers=include_identifiers,
            include_memory_status=include_memory_status,
        )
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return

    answer = getattr(result, "answer", None)
    if answer is None and isinstance(result, Mapping):
        answer = result.get("answer") or result.get("final_answer")
    print(str(answer or "未生成有效回答。"))

    if include_memory_status:
        payload = _result_to_dict(
            result,
            include_raw_state=False,
            include_identifiers=False,
            include_memory_status=True,
        )
        status = payload.get("memory_status")
        if isinstance(status, Mapping):
            print(
                "[Memory] "
                f"loaded={bool(status.get('loaded', False))}, "
                f"written={bool(status.get('written', False))}, "
                f"recent={_safe_int(status.get('recent_message_count'))}, "
                f"retrieved={_safe_int(status.get('retrieved_memory_count'))}"
            )


def run_once(
    service: Any,
    query: str,
    *,
    options: Mapping[str, Any],
    user_id: Optional[str],
    project_id: Optional[str],
    session_id: Optional[str],
    request_id: Optional[str],
    as_json: bool,
    raw: bool,
    include_identifiers: bool,
    include_memory_status: bool,
) -> Any:
    """执行一次 CLI 问答。"""

    normalized_query = str(query or "").strip()
    if not normalized_query:
        raise ValueError("query must not be empty.")

    result = service.ask(
        normalized_query,
        user_id=_clean_optional(user_id),
        project_id=_clean_optional(project_id),
        session_id=_clean_optional(session_id),
        request_id=_clean_optional(request_id),
        request_options=dict(options),
        include_raw_state=raw,
    )
    render_result(
        result,
        as_json=as_json,
        include_raw_state=raw,
        include_identifiers=include_identifiers,
        include_memory_status=include_memory_status,
    )
    return result


def run_interactive(
    service: Any,
    *,
    options: Mapping[str, Any],
    user_id: Optional[str],
    project_id: Optional[str],
    session_id: Optional[str],
    as_json: bool,
    raw: bool,
    include_identifiers: bool,
    include_memory_status: bool,
) -> None:
    """在固定 session_id 下运行多轮 CLI，保证会话 Memory 连续。"""

    resolved_session_id = (
        _clean_optional(session_id)
        or "sess_cli_" + uuid.uuid4().hex[:16]
    )

    print(
        "KG-RAG Agent CLI；输入 exit、quit 或 q 退出。"
        f"\n当前 session_id: {resolved_session_id}"
    )

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if query.lower() in {"exit", "quit", "q"}:
            return
        if not query:
            continue

        run_once(
            service,
            query,
            options=options,
            user_id=user_id,
            project_id=project_id,
            session_id=resolved_session_id,
            request_id=None,
            as_json=as_json,
            raw=raw,
            include_identifiers=include_identifiers,
            include_memory_status=include_memory_status,
        )


def run_cli(args: argparse.Namespace) -> int:
    """运行单次或交互式 CLI，并统一释放 Service/Runtime。"""

    if args.request_id and args.interactive:
        raise ValueError(
            "--request-id cannot be reused in interactive mode."
        )

    service = create_cli_service(
        config_path=_clean_optional(args.config),
        validate=not args.no_validate,
    )

    try:
        options = request_options_from_args(args)
        include_identifiers = bool(
            args.include_identifiers
            or args.user_id
            or args.project_id
            or args.session_id
        )

        if args.interactive:
            run_interactive(
                service,
                options=options,
                user_id=args.user_id,
                project_id=args.project_id,
                session_id=args.session_id,
                as_json=args.as_json,
                raw=args.raw,
                include_identifiers=include_identifiers,
                include_memory_status=args.memory_status,
            )
            return 0

        query = str(args.query or "").strip()
        if not query:
            query = input("请输入问题：").strip()

        run_once(
            service,
            query,
            options=options,
            user_id=args.user_id,
            project_id=args.project_id,
            session_id=args.session_id,
            request_id=args.request_id,
            as_json=args.as_json,
            raw=args.raw,
            include_identifiers=include_identifiers,
            include_memory_status=args.memory_status,
        )
        return 0
    finally:
        close = getattr(service, "close", None)
        if callable(close):
            close()


def run_server(args: argparse.Namespace) -> int:
    """通过 Uvicorn 启动 FastAPI，支持 reload 和多 worker。"""

    if not 1 <= args.port <= 65_535:
        raise ValueError("--port must be between 1 and 65535.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.reload and args.workers != 1:
        raise ValueError("--reload cannot be combined with --workers > 1.")

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            '缺少 Uvicorn，请安装 API 依赖：pip install -e ".[api]"'
        ) from exc

    environment = {
        _SERVER_CONFIG_ENV: _clean_optional(args.config),
        _SERVER_VALIDATE_ENV: _format_env_bool(
            not args.no_validate
        ),
        "KG_RAG_API_DOCS_ENABLED": _format_env_bool(
            not args.no_docs
        ),
        "KG_RAG_API_ALLOW_RAW_STATE": _format_env_bool(
            args.allow_api_raw_state
        ),
        "KG_RAG_API_INITIALIZE_SERVICE": _format_env_bool(
            not args.no_service_init
        ),
    }
    if args.api_prefix is not None:
        environment["KG_RAG_API_PREFIX"] = _normalize_api_prefix(
            args.api_prefix
        )

    with _temporary_environ(environment):
        uvicorn.run(
            _SERVER_FACTORY,
            factory=True,
            host=str(args.host),
            port=int(args.port),
            workers=int(args.workers),
            reload=bool(args.reload),
            log_level=str(args.log_level),
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """统一入口；返回进程退出码，便于脚本和测试复用。"""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.serve:
            return run_server(args)
        return run_cli(args)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover


def _result_to_dict(
    result: Any,
    *,
    include_raw_state: bool,
    include_identifiers: bool,
    include_memory_status: bool,
) -> dict[str, Any]:
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        try:
            return dict(
                to_dict(
                    include_raw_state=include_raw_state,
                    include_identifiers=include_identifiers,
                    include_memory_status=include_memory_status,
                )
            )
        except TypeError:
            # 兼容尚未升级到新 AgentResult 参数的旧实现。
            payload = dict(
                to_dict(include_raw_state=include_raw_state)
            )
    elif isinstance(result, Mapping):
        payload = dict(result)
    else:
        return {"answer": str(result)}

    if not include_raw_state:
        payload.pop("raw_state", None)
    if not include_identifiers:
        payload.pop("user_id", None)
        payload.pop("project_id", None)
        payload.pop("session_id", None)
    if not include_memory_status:
        payload.pop("memory_status", None)
    return payload


@contextmanager
def _temporary_environ(
    values: Mapping[str, Optional[str]],
) -> Iterator[None]:
    previous: dict[str, Optional[str]] = {}
    try:
        for key, value in values.items():
            previous[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _normalize_api_prefix(value: Any) -> str:
    prefix = str(value or "/api").strip()
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    return prefix.rstrip("/") or "/api"


def _clean_optional(value: Any) -> Optional[str]:
    normalized = str(value or "").strip()
    return normalized or None


def _safe_int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _format_env_bool(value: bool) -> str:
    return "true" if value else "false"


def _parse_env_bool(
    value: Optional[str],
    *,
    default: bool,
) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean environment value: {value!r}."
    )


__all__ = [
    "build_parser",
    "request_options_from_args",
    "create_cli_service",
    "create_server_app",
    "render_result",
    "run_once",
    "run_interactive",
    "run_cli",
    "run_server",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
