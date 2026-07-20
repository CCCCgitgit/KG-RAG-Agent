# -*- coding: utf-8 -*-
"""KG-RAG Agent 的模型调用与 Prompt 管理层。"""

from __future__ import annotations

from .deepseek_client import (
    DeepSeekClient,
    deepseek_chat,
    deepseek_generate,
    deepseek_generate_with_metadata,
    get_default_deepseek_client,
    reset_default_deepseek_client,
)
from .errors import (
    LLMConfigurationError,
    LLMError,
    LLMRequestError,
    LLMResponseError,
    PromptConfigurationError,
    PromptError,
    PromptNotFoundError,
    PromptRenderError,
)
from .llm_client import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROVIDER,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    LLMClient,
    LLMResponse,
    build_messages,
    chat,
    generate,
    get_default_llm_client,
    parse_openai_compatible_response,
    reset_default_llm_client,
    sanitize_messages,
)
from .prompt_manager import (
    PromptManager,
    PromptRecord,
    format_prompt,
    get_default_prompt_manager,
    get_prompt,
    list_prompts,
    render_prompt,
    reset_default_prompt_manager,
)


def load_prompt(name: str, default: str = "") -> str:
    """兼容旧接口；新代码使用 ``get_prompt``。"""

    return get_prompt(name=name, default=default)


__all__ = [
    "DEFAULT_PROVIDER",
    "DEFAULT_DEEPSEEK_BASE_URL",
    "DEFAULT_DEEPSEEK_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    "LLMError",
    "LLMConfigurationError",
    "LLMRequestError",
    "LLMResponseError",
    "LLMClient",
    "LLMResponse",
    "get_default_llm_client",
    "reset_default_llm_client",
    "generate",
    "chat",
    "build_messages",
    "sanitize_messages",
    "parse_openai_compatible_response",
    "DeepSeekClient",
    "get_default_deepseek_client",
    "reset_default_deepseek_client",
    "deepseek_generate",
    "deepseek_chat",
    "deepseek_generate_with_metadata",
    "PromptError",
    "PromptNotFoundError",
    "PromptConfigurationError",
    "PromptRenderError",
    "PromptRecord",
    "PromptManager",
    "get_default_prompt_manager",
    "reset_default_prompt_manager",
    "get_prompt",
    "format_prompt",
    "render_prompt",
    "list_prompts",
    "load_prompt",
]
