# -*- coding: utf-8 -*-
"""LLM 与 Prompt 层统一异常。"""

from __future__ import annotations


class LLMError(RuntimeError):
    """LLM 层基础异常。"""


class LLMConfigurationError(LLMError, ValueError):
    """模型客户端配置无效。"""


class LLMRequestError(LLMError):
    """模型请求失败。"""


class LLMResponseError(LLMError):
    """模型响应无法解析。"""


class PromptError(RuntimeError):
    """Prompt 管理基础异常。"""


class PromptNotFoundError(PromptError, KeyError):
    """Prompt 名称不存在。"""


class PromptConfigurationError(PromptError, ValueError):
    """Prompt 配置无效。"""


class PromptRenderError(PromptError, ValueError):
    """Prompt 变量渲染失败。"""


__all__ = [
    "LLMError",
    "LLMConfigurationError",
    "LLMRequestError",
    "LLMResponseError",
    "PromptError",
    "PromptNotFoundError",
    "PromptConfigurationError",
    "PromptRenderError",
]
