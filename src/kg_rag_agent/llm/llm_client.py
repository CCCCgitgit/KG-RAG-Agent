# -*- coding: utf-8 -*-
"""
OpenAI-compatible LLM 统一客户端。

本模块只负责模型请求、消息规范化、响应解析和调用级重试；不负责
Graph 编排、Prompt 业务选择、KG 查询或 Evidence 构建。
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .errors import LLMConfigurationError, LLMRequestError

DEFAULT_PROVIDER = "deepseek"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2

_ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
_RESERVED_REQUEST_KEYS = {"model", "messages", "temperature", "max_tokens"}


@dataclass(slots=True)
class LLMResponse:
    """标准化模型响应。"""

    content: str
    model: str = ""
    provider: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    raw: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.raw is not None:
            data["raw_type"] = type(self.raw).__name__
            data.pop("raw", None)
        return data


def get_project_root() -> Path:
    """返回项目根目录。"""

    return Path(__file__).resolve().parents[3]


def load_env_file() -> None:
    """加载项目根目录下的 ``.env``；缺少依赖时静默跳过。"""

    env_path = get_project_root() / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except Exception:
        return


class LLMClient:
    """OpenAI-compatible 模型客户端。"""

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        default_temperature: float = DEFAULT_TEMPERATURE,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        lazy_init: bool = True,
    ) -> None:
        load_env_file()

        self.provider = str(
            provider if provider is not None else os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
        ).strip().lower()
        if not self.provider:
            raise LLMConfigurationError("provider must not be empty")

        self.api_key = api_key if api_key is not None else self._load_api_key()
        self.base_url = base_url if base_url is not None else self._load_base_url()
        self.model = model if model is not None else self._load_model()

        timeout_value = os.getenv("LLM_TIMEOUT", str(DEFAULT_TIMEOUT)) if timeout is None else timeout
        retry_value = (
            os.getenv("LLM_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))
            if max_retries is None
            else max_retries
        )

        self.timeout = _validate_positive_float(timeout_value, "timeout")
        self.max_retries = _validate_non_negative_int(retry_value, "max_retries")
        self.default_temperature = _validate_temperature(default_temperature)
        self.default_max_tokens = _validate_max_tokens(default_max_tokens)

        self.client: Optional[Any] = None
        if not lazy_init:
            self._ensure_client()

    def _load_api_key(self) -> str:
        if self.provider == "deepseek":
            return os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY") or ""
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or ""
        return os.getenv("LLM_API_KEY") or ""

    def _load_base_url(self) -> str:
        if self.provider == "deepseek":
            return (
                os.getenv("DEEPSEEK_BASE_URL")
                or os.getenv("LLM_BASE_URL")
                or DEFAULT_DEEPSEEK_BASE_URL
            )
        if self.provider == "openai":
            return os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or ""
        return os.getenv("LLM_BASE_URL") or ""

    def _load_model(self) -> str:
        if self.provider == "deepseek":
            return (
                os.getenv("DEEPSEEK_MODEL")
                or os.getenv("LLM_MODEL")
                or DEFAULT_DEEPSEEK_MODEL
            )
        if self.provider == "openai":
            return os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        return os.getenv("LLM_MODEL") or DEFAULT_DEEPSEEK_MODEL

    def _ensure_client(self) -> None:
        if self.client is not None:
            return
        if not str(self.api_key or "").strip():
            raise ValueError(
                "LLM API key is missing. Please set DEEPSEEK_API_KEY, "
                "OPENAI_API_KEY or LLM_API_KEY."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package is not installed") from exc

        kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            # 本类统一承担重试，避免 SDK 与业务层双重重试。
            "max_retries": 0,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url

        self.client = OpenAI(**kwargs)

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        result = self.generate_with_metadata(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            **kwargs,
        )
        return str(result.get("content", "") or "")

    def chat(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        *,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        return self.chat_with_metadata(
            messages=messages,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            **kwargs,
        ).content

    def chat_with_metadata(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        *,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        final_messages = build_messages(
            messages=messages,
            prompt=prompt,
            system_prompt=system_prompt,
        )
        return self._chat_completion(
            messages=final_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            **kwargs,
        )

    def generate_with_metadata(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        response = self.chat_with_metadata(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            **kwargs,
        )
        return response.to_dict()

    def _chat_completion(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        model: Optional[str],
        **kwargs: Any,
    ) -> LLMResponse:
        self._ensure_client()
        assert self.client is not None

        final_model = str(model if model is not None else self.model).strip()
        if not final_model:
            raise LLMConfigurationError("model must not be empty")

        final_temperature = (
            self.default_temperature
            if temperature is None
            else _validate_temperature(temperature)
        )
        final_max_tokens = (
            self.default_max_tokens
            if max_tokens is None
            else _validate_max_tokens(max_tokens)
        )

        collisions = _RESERVED_REQUEST_KEYS.intersection(kwargs)
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(f"reserved request parameters must use explicit arguments: {names}")

        request_kwargs: Dict[str, Any] = {
            "model": final_model,
            "messages": sanitize_messages(messages),
            "temperature": final_temperature,
            "max_tokens": final_max_tokens,
        }
        request_kwargs.update(kwargs)

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                raw_response = self.client.chat.completions.create(**request_kwargs)
                latency = time.perf_counter() - start_time
                return parse_openai_compatible_response(
                    raw_response,
                    provider=self.provider,
                    model=final_model,
                    latency_seconds=latency,
                )
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(0.5 * (2**attempt), 4.0))

        raise LLMRequestError(
            f"LLM request failed after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error

    def health_check(self) -> Dict[str, Any]:
        try:
            content = self.generate("请只回复 OK", temperature=0.0, max_tokens=8)
            return {
                "ok": True,
                "provider": self.provider,
                "model": self.model,
                "content": content,
            }
        except Exception as exc:
            return {
                "ok": False,
                "provider": self.provider,
                "model": self.model,
                "error": str(exc),
            }

    def info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_temperature": self.default_temperature,
            "default_max_tokens": self.default_max_tokens,
            "client_initialized": self.client is not None,
            "has_api_key": bool(self.api_key),
        }

    def close(self) -> None:
        """释放底层 SDK 资源；重复调用安全。"""

        client = self.client
        self.client = None
        if client is None:
            return
        close_method = getattr(client, "close", None)
        if callable(close_method):
            close_method()


def build_messages(
    *,
    messages: Optional[List[Dict[str, str]]] = None,
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """构造标准 OpenAI-compatible 消息列表。"""

    if messages:
        explicit = sanitize_messages(messages)
        if explicit:
            return explicit

    result: List[Dict[str, str]] = []
    if system_prompt:
        result.append({"role": "system", "content": str(system_prompt)})
    result.append({"role": "user", "content": str(prompt or "")})
    return sanitize_messages(result)


def sanitize_messages(
    messages: Sequence[Mapping[str, Any]],
) -> List[Dict[str, str]]:
    """过滤无效消息，并统一为 ``role/content`` 两个字段。"""

    cleaned: List[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue

        role = str(message.get("role", "user") or "user").strip().lower()
        content = str(message.get("content", "") or "")
        if role not in _ALLOWED_ROLES:
            role = "user"
        if not content.strip():
            continue
        cleaned.append({"role": role, "content": content})

    return cleaned or [{"role": "user", "content": ""}]


def parse_openai_compatible_response(
    response: Any,
    *,
    provider: str,
    model: str,
    latency_seconds: float,
) -> LLMResponse:
    """将 OpenAI-compatible SDK 响应转换为 ``LLMResponse``。"""

    content = ""
    finish_reason = ""

    try:
        choices = _read_value(response, "choices", []) or []
        choice = choices[0]
        finish_reason = str(_read_value(choice, "finish_reason", "") or "")
        message = _read_value(choice, "message", None)
        if message is not None:
            content = str(_read_value(message, "content", "") or "")
        else:
            content = str(_read_value(choice, "text", "") or "")
    except Exception:
        content = str(response or "")

    usage = _read_value(response, "usage", None)
    prompt_tokens = _safe_int(_read_value(usage, "prompt_tokens", 0))
    completion_tokens = _safe_int(_read_value(usage, "completion_tokens", 0))
    total_tokens = _safe_int(_read_value(usage, "total_tokens", 0))
    response_model = str(_read_value(response, "model", "") or model)

    return LLMResponse(
        content=content.strip(),
        model=response_model,
        provider=str(provider or ""),
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_seconds=float(latency_seconds),
        raw=response,
    )


def _read_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _validate_positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LLMConfigurationError(f"{name} must be a number") from exc
    if parsed <= 0:
        raise LLMConfigurationError(f"{name} must be greater than 0")
    return parsed


def _validate_non_negative_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LLMConfigurationError(f"{name} must be an integer") from exc
    if parsed < 0:
        raise LLMConfigurationError(f"{name} must be greater than or equal to 0")
    return parsed


def _validate_temperature(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LLMConfigurationError("temperature must be a number") from exc
    if not 0.0 <= parsed <= 2.0:
        raise LLMConfigurationError("temperature must be between 0 and 2")
    return parsed


def _validate_max_tokens(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LLMConfigurationError("max_tokens must be an integer") from exc
    if parsed <= 0:
        raise LLMConfigurationError("max_tokens must be greater than 0")
    return parsed


_GLOBAL_LLM_CLIENT: Optional[LLMClient] = None


def get_default_llm_client() -> LLMClient:
    """兼容旧调用的进程级默认客户端；正式运行优先使用 RuntimeContext。"""

    global _GLOBAL_LLM_CLIENT
    if _GLOBAL_LLM_CLIENT is None:
        _GLOBAL_LLM_CLIENT = LLMClient()
    return _GLOBAL_LLM_CLIENT


def reset_default_llm_client() -> None:
    """关闭并清除默认客户端，供测试和进程重载使用。"""

    global _GLOBAL_LLM_CLIENT
    if _GLOBAL_LLM_CLIENT is not None:
        _GLOBAL_LLM_CLIENT.close()
    _GLOBAL_LLM_CLIENT = None


def generate(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    return get_default_llm_client().generate(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def chat(
    messages: List[Dict[str, str]],
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    return get_default_llm_client().chat(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


__all__ = [
    "DEFAULT_PROVIDER",
    "DEFAULT_DEEPSEEK_BASE_URL",
    "DEFAULT_DEEPSEEK_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    "LLMResponse",
    "LLMClient",
    "get_project_root",
    "load_env_file",
    "build_messages",
    "sanitize_messages",
    "parse_openai_compatible_response",
    "get_default_llm_client",
    "reset_default_llm_client",
    "generate",
    "chat",
]
