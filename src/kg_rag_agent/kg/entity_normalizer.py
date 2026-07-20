# -*- coding: utf-8 -*-
"""Backward-compatible entity-normalization imports.

The canonical implementation now lives in ``kg_rag_agent.entity_resolution``.
This module remains so existing Graph nodes, tests, and integrations keep the
same import path during the staged migration.
"""

from __future__ import annotations

from kg_rag_agent.entity_resolution import normalizer as _impl

# Re-export every non-private compatibility symbol from the canonical module.
for _name in dir(_impl):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_impl, _name)

__all__ = sorted(name for name in globals() if not name.startswith("_"))
