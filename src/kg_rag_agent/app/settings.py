# -*- coding: utf-8 -*-
"""Web API 层配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Sequence


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean.")


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"Environment variable {name} must be between {minimum} and {maximum}."
        )
    return value


def _split_csv(value: str | None, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items


@dataclass(frozen=True, slots=True)
class APISettings:
    """FastAPI 运行参数，不包含业务算法配置。"""

    title: str = "KG-RAG Agent API"
    description: str = "LangGraph-based KG-RAG Agent service."
    version: str = "0.1.0"
    api_prefix: str = "/api"

    docs_enabled: bool = True
    initialize_service_on_startup: bool = True
    allow_raw_state: bool = False
    max_batch_size: int = 20

    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: (
            "http://localhost",
            "http://127.0.0.1",
        )
    )
    cors_allow_credentials: bool = False
    cors_allow_methods: tuple[str, ...] = ("GET", "POST", "OPTIONS")
    cors_allow_headers: tuple[str, ...] = (
        "Accept",
        "Authorization",
        "Content-Type",
        "X-Request-ID",
    )

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "APISettings":
        """从环境变量构造 APISettings。"""

        source = os.environ if environ is None else environ
        defaults = cls()

        def get(name: str) -> str | None:
            return source.get(name)

        prefix = (get("KG_RAG_API_PREFIX") or "/api").strip()
        if not prefix.startswith("/"):
            prefix = "/" + prefix
        prefix = prefix.rstrip("/") or "/api"

        settings = cls(
            title=(get("KG_RAG_API_TITLE") or defaults.title).strip(),
            description=(
                get("KG_RAG_API_DESCRIPTION") or defaults.description
            ).strip(),
            version=(get("KG_RAG_API_VERSION") or defaults.version).strip(),
            api_prefix=prefix,
            docs_enabled=_mapping_bool(
                source,
                "KG_RAG_API_DOCS_ENABLED",
                True,
            ),
            initialize_service_on_startup=_mapping_bool(
                source,
                "KG_RAG_API_INITIALIZE_SERVICE",
                True,
            ),
            allow_raw_state=_mapping_bool(
                source,
                "KG_RAG_API_ALLOW_RAW_STATE",
                False,
            ),
            max_batch_size=_mapping_int(
                source,
                "KG_RAG_API_MAX_BATCH_SIZE",
                20,
                1,
                100,
            ),
            cors_origins=_split_csv(
                get("KG_RAG_API_CORS_ORIGINS"),
                ("http://localhost", "http://127.0.0.1"),
            ),
            cors_allow_credentials=_mapping_bool(
                source,
                "KG_RAG_API_CORS_ALLOW_CREDENTIALS",
                False,
            ),
            cors_allow_methods=_split_csv(
                get("KG_RAG_API_CORS_ALLOW_METHODS"),
                ("GET", "POST", "OPTIONS"),
            ),
            cors_allow_headers=_split_csv(
                get("KG_RAG_API_CORS_ALLOW_HEADERS"),
                (
                    "Accept",
                    "Authorization",
                    "Content-Type",
                    "X-Request-ID",
                ),
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.title:
            raise ValueError("API title must not be empty.")
        if not self.version:
            raise ValueError("API version must not be empty.")
        if not self.api_prefix.startswith("/"):
            raise ValueError("api_prefix must start with '/'.")
        if not 1 <= self.max_batch_size <= 100:
            raise ValueError("max_batch_size must be between 1 and 100.")
        if self.cors_allow_credentials and "*" in self.cors_origins:
            raise ValueError(
                "Wildcard CORS origin cannot be combined with credentials."
            )


def _mapping_bool(
    source: Mapping[str, str],
    name: str,
    default: bool,
) -> bool:
    value = source.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean.")


def _mapping_int(
    source: Mapping[str, str],
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = source.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"Environment variable {name} must be between {minimum} and {maximum}."
        )
    return value


__all__ = ["APISettings"]
