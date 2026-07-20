
# -*- coding: utf-8 -*-
"""Retrieval 内部参数标准化工具。"""
from __future__ import annotations
import math
from typing import Any, Dict, Mapping, Optional
from .errors import RetrievalValidationError

def normalize_text(value: Any, *, field: str = "text") -> str:
    if value is None: return ""
    try: return str(value).strip()
    except Exception as exc: raise RetrievalValidationError(f"{field} must be text-like") from exc

def require_non_empty_text(value: Any, *, field: str) -> str:
    text=normalize_text(value,field=field)
    if not text: raise RetrievalValidationError(f"{field} must not be empty")
    return text

def positive_int(value: Any, *, field: str, default: Optional[int]=None) -> int:
    raw=default if value is None else value
    if raw is None: raise RetrievalValidationError(f"{field} is required")
    try: result=int(raw)
    except (TypeError,ValueError) as exc: raise RetrievalValidationError(f"{field} must be an integer") from exc
    if result<=0: raise RetrievalValidationError(f"{field} must be greater than 0")
    return result

def non_negative_float(value: Any, *, field: str, default: Optional[float]=None) -> float:
    raw=default if value is None else value
    if raw is None: raise RetrievalValidationError(f"{field} is required")
    try: result=float(raw)
    except (TypeError,ValueError) as exc: raise RetrievalValidationError(f"{field} must be numeric") from exc
    if not math.isfinite(result) or result<0: raise RetrievalValidationError(f"{field} must be a finite non-negative number")
    return result

def optional_mapping(value: Any, *, field: str) -> Optional[Dict[str, Any]]:
    if value is None: return None
    if not isinstance(value,Mapping): raise RetrievalValidationError(f"{field} must be a mapping")
    return dict(value)

def validate_weights(**weights: Any) -> Dict[str,float]:
    normalized={k:non_negative_float(v,field=k) for k,v in weights.items()}
    if normalized and not any(v>0 for v in normalized.values()):
        raise RetrievalValidationError("at least one weight must be greater than 0")
    return normalized
