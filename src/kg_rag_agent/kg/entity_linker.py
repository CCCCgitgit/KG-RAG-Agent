# -*- coding: utf-8 -*-
"""Backward-compatible entity-linking imports.

The canonical implementation now lives in ``kg_rag_agent.entity_resolution``.
This module remains so existing RuntimeFactory and Graph nodes keep working
until their imports are migrated.
"""

from __future__ import annotations

from kg_rag_agent.entity_resolution import linker as _impl

for _name in dir(_impl):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_impl, _name)

__all__ = sorted(name for name in globals() if not name.startswith("_"))
