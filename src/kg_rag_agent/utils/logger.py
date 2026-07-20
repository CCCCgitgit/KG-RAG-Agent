# -*- coding: utf-8 -*-
"""Central logging utilities."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Mapping, Optional

from .paths import get_project_root, resolve_project_path

DEFAULT_LOGGER_NAME = "kg_rag_agent"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_DIR = "outputs/logs"
DEFAULT_LOG_FILE = "kg_rag_agent.log"
DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOGGER_LOCK = RLock()
_GLOBAL_LOGGER: Optional[logging.Logger] = None


def parse_log_level(level: Any) -> int:
    """Convert a level string or integer into a logging level."""

    if isinstance(level, int):
        return level
    normalized = str(level or DEFAULT_LOG_LEVEL).strip().upper()
    return {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.CRITICAL,
    }.get(normalized, logging.INFO)


def build_console_handler(
    *,
    level: Any,
    formatter: logging.Formatter,
) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(parse_log_level(level))
    handler.setFormatter(formatter)
    setattr(handler, "_kg_rag_kind", "console")
    return handler


def build_file_handler(
    *,
    log_dir: str | Path,
    log_file: str,
    level: Any,
    formatter: logging.Formatter,
    max_bytes: int = 0,
    backup_count: int = 0,
) -> logging.Handler:
    directory = resolve_project_path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = (directory / str(log_file)).resolve()

    if int(max_bytes or 0) > 0:
        handler: logging.Handler = RotatingFileHandler(
            filename=str(log_path),
            mode="a",
            maxBytes=int(max_bytes),
            backupCount=max(0, int(backup_count or 0)),
            encoding="utf-8",
        )
    else:
        handler = logging.FileHandler(
            filename=str(log_path),
            mode="a",
            encoding="utf-8",
        )

    handler.setLevel(parse_log_level(level))
    handler.setFormatter(formatter)
    setattr(handler, "_kg_rag_kind", "file")
    setattr(handler, "_kg_rag_path", str(log_path))
    return handler


def clear_handlers(logger: logging.Logger) -> None:
    """Remove and close all handlers from a logger."""

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
    setattr(logger, "_kg_rag_initialized", False)


def get_logger(
    name: str = DEFAULT_LOGGER_NAME,
    *,
    level: Any = DEFAULT_LOG_LEVEL,
    log_dir: str | Path = DEFAULT_LOG_DIR,
    log_file: str = DEFAULT_LOG_FILE,
    enable_console: bool = True,
    enable_file: bool = True,
    reset_handlers: bool = False,
    propagate: bool = False,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    max_bytes: int = 0,
    backup_count: int = 0,
) -> logging.Logger:
    """Create or reuse a consistently configured logger."""

    logger_name = str(name or DEFAULT_LOGGER_NAME).strip() or DEFAULT_LOGGER_NAME

    with _LOGGER_LOCK:
        logger = logging.getLogger(logger_name)
        logger.setLevel(parse_log_level(level))
        logger.propagate = bool(propagate)

        if reset_handlers:
            clear_handlers(logger)

        if getattr(logger, "_kg_rag_initialized", False):
            return logger

        formatter = logging.Formatter(
            fmt=str(log_format or DEFAULT_LOG_FORMAT),
            datefmt=str(date_format or DEFAULT_DATE_FORMAT),
        )

        if enable_console:
            logger.addHandler(
                build_console_handler(level=level, formatter=formatter)
            )
        if enable_file:
            logger.addHandler(
                build_file_handler(
                    log_dir=log_dir,
                    log_file=log_file,
                    level=level,
                    formatter=formatter,
                    max_bytes=max_bytes,
                    backup_count=backup_count,
                )
            )

        setattr(logger, "_kg_rag_initialized", True)
        return logger


def setup_logger(
    name: str = DEFAULT_LOGGER_NAME,
    *,
    config: Optional[Mapping[str, Any]] = None,
    reset_handlers: bool = False,
) -> logging.Logger:
    """Initialize a logger from either full config or logging subsection."""

    raw_config: Mapping[str, Any] = config or {}
    logging_config = raw_config.get("logging", raw_config)
    if not isinstance(logging_config, Mapping):
        logging_config = {}

    configured_name = logging_config.get("logger_name")
    effective_name = (
        str(configured_name)
        if configured_name and name == DEFAULT_LOGGER_NAME
        else name
    )

    return get_logger(
        name=effective_name,
        level=logging_config.get("level", DEFAULT_LOG_LEVEL),
        log_dir=logging_config.get("log_dir", DEFAULT_LOG_DIR),
        log_file=logging_config.get("log_file", DEFAULT_LOG_FILE),
        enable_console=bool(logging_config.get("enable_console", True)),
        enable_file=bool(logging_config.get("enable_file", True)),
        reset_handlers=reset_handlers,
        propagate=bool(logging_config.get("propagate", False)),
        log_format=str(logging_config.get("format", DEFAULT_LOG_FORMAT)),
        date_format=str(
            logging_config.get("date_format", DEFAULT_DATE_FORMAT)
        ),
        max_bytes=_safe_int(logging_config.get("max_bytes"), 0),
        backup_count=_safe_int(logging_config.get("backup_count"), 0),
    )


def get_default_logger() -> logging.Logger:
    global _GLOBAL_LOGGER
    with _LOGGER_LOCK:
        if _GLOBAL_LOGGER is None:
            _GLOBAL_LOGGER = get_logger()
        return _GLOBAL_LOGGER


def reset_default_logger(
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> logging.Logger:
    global _GLOBAL_LOGGER
    with _LOGGER_LOCK:
        _GLOBAL_LOGGER = setup_logger(config=config, reset_handlers=True)
        return _GLOBAL_LOGGER


def log_exception(
    logger: logging.Logger,
    message: str,
    exc: Exception,
) -> None:
    logger.error(
        "%s | %s: %s",
        message,
        type(exc).__name__,
        str(exc),
        exc_info=(type(exc), exc, exc.__traceback__),
    )


def log_node_start(
    logger: logging.Logger,
    node_name: str,
    *,
    request_id: str = "",
    query: str = "",
) -> None:
    logger.info(
        "[NODE START] %s | request_id=%s | query=%s",
        node_name,
        request_id,
        truncate_text(query, 120),
    )


def log_node_end(
    logger: logging.Logger,
    node_name: str,
    *,
    request_id: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    logger.info(
        "[NODE END] %s | request_id=%s | extra=%s",
        node_name,
        request_id,
        extra or {},
    )


def log_node_error(
    logger: logging.Logger,
    node_name: str,
    exc: Exception,
    *,
    request_id: str = "",
) -> None:
    logger.error(
        "[NODE ERROR] %s | request_id=%s | %s: %s",
        node_name,
        request_id,
        type(exc).__name__,
        str(exc),
        exc_info=(type(exc), exc, exc.__traceback__),
    )


def truncate_text(text: Any, max_length: int = 200) -> str:
    """Flatten and safely truncate text for logs."""

    limit = max(0, int(max_length))
    value = " ".join(str(text or "").split())
    if limit == 0:
        return ""
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3] + "..."


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
