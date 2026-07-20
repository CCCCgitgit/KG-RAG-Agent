# -*- coding: utf-8 -*-
"""旧入口兼容：转发到 build_vector_store.py。"""
from __future__ import annotations

try:
    from build_vector_store import build_parser, main, run
except ImportError:  # pragma: no cover
    from scripts.build_vector_store import build_parser, main, run

__all__ = ["build_parser", "run", "main"]

if __name__ == "__main__":
    main()
