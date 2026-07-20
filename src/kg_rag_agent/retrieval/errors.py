
# -*- coding: utf-8 -*-
"""Retrieval 层统一异常。"""
from __future__ import annotations

class RetrievalError(RuntimeError):
    """Retrieval 层基础异常。"""

class RetrievalValidationError(RetrievalError, ValueError):
    """输入参数或数据格式不合法。"""

class RetrievalConfigurationError(RetrievalError):
    """Retrieval 配置不完整或互相冲突。"""

class RetrievalDependencyError(RetrievalError, ImportError):
    """外部依赖缺失。"""

class RetrievalBackendError(RetrievalError):
    """Embedding、Chroma 或重排后端执行失败。"""
