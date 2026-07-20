# -*- coding: utf-8 -*-
"""Evidence 与 Citation 的稳定对齐。"""

from __future__ import annotations

from typing import List

from .schemas import Citation, EvidenceItem


class CitationBuilder:
    def build(self, evidence: List[EvidenceItem]) -> List[Citation]:
        citations: List[Citation] = []
        for idx, item in enumerate(evidence or [], start=1):
            evidence_id = str(item.get("evidence_id", "") or "").strip() or f"E{idx}"
            citations.append(
                Citation(
                    citation_id=f"E{idx}",
                    evidence_id=evidence_id,
                    type=str(item.get("evidence_type", "") or ""),
                    source_entity=str(item.get("source_entity", "") or ""),
                    relation=str(item.get("relation", "") or ""),
                    target_entity=str(item.get("target_entity", "") or ""),
                    text=str(item.get("text", "") or ""),
                    score=float(item.get("score", 0.0) or 0.0),
                )
            )
        return citations


def build_citations(evidence: List[EvidenceItem]) -> List[Citation]:
    return CitationBuilder().build(evidence)
