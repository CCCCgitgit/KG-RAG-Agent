# -*- coding: utf-8 -*-
"""DeepSeek Provider 适配器。"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .llm_client import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    LLMClient,
)


class DeepSeekClient(LLMClient):
    """固定 ``provider=deepseek`` 的 OpenAI-compatible 客户端。"""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        default_temperature: float = DEFAULT_TEMPERATURE,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        lazy_init: bool = True,
    ) -> None:
        resolved_api_key = (
            api_key
            if api_key is not None
            else os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
        )
        resolved_base_url = (
            base_url
            if base_url is not None
            else os.getenv("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL
        )
        resolved_model = (
            model
            if model is not None
            else os.getenv("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL
        )
        resolved_timeout = (
            timeout
            if timeout is not None
            else float(os.getenv("LLM_TIMEOUT", str(DEFAULT_TIMEOUT)))
        )
        resolved_retries = (
            max_retries
            if max_retries is not None
            else int(os.getenv("LLM_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))
        )

        super().__init__(
            provider="deepseek",
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            model=resolved_model,
            timeout=resolved_timeout,
            max_retries=resolved_retries,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
            lazy_init=lazy_init,
        )

    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt
            or (
                "你是一个严格的 JSON 生成助手。"
                "请只输出合法 JSON，不要输出 Markdown，不要输出解释性文字。"
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_short_answer(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt
            or (
                "你是一个可靠、简洁、专业的中文助手。"
                "请直接回答问题，不要输出无关内容。"
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_long_answer(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt
            or (
                "你是一个可靠、专业、表达清晰的中文助手。"
                "请在不编造信息的前提下，给出结构清楚的回答。"
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )


_GLOBAL_DEEPSEEK_CLIENT: Optional[DeepSeekClient] = None


def get_default_deepseek_client() -> DeepSeekClient:
    """兼容旧调用的默认 DeepSeek 客户端。"""

    global _GLOBAL_DEEPSEEK_CLIENT
    if _GLOBAL_DEEPSEEK_CLIENT is None:
        _GLOBAL_DEEPSEEK_CLIENT = DeepSeekClient()
    return _GLOBAL_DEEPSEEK_CLIENT


def reset_default_deepseek_client() -> None:
    """关闭并清除默认 DeepSeek 客户端。"""

    global _GLOBAL_DEEPSEEK_CLIENT
    if _GLOBAL_DEEPSEEK_CLIENT is not None:
        _GLOBAL_DEEPSEEK_CLIENT.close()
    _GLOBAL_DEEPSEEK_CLIENT = None


def deepseek_generate(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    return get_default_deepseek_client().generate(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def deepseek_chat(
    messages: List[Dict[str, str]],
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    return get_default_deepseek_client().chat(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def deepseek_generate_with_metadata(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    return get_default_deepseek_client().generate_with_metadata(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


__all__ = [
    "DeepSeekClient",
    "get_default_deepseek_client",
    "reset_default_deepseek_client",
    "deepseek_generate",
    "deepseek_chat",
    "deepseek_generate_with_metadata",
]
