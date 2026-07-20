# -*- coding: utf-8 -*-
"""Answering 领域能力的统一顺序调用入口。"""

from __future__ import annotations

from typing import Any, List

from .composer import AnswerComposer
from .evidence_selector import EvidenceSelector
from .reasoner import AnswerReasoner
from .schemas import (
    AnsweringOptions,
    AnsweringPipelineResult,
    EvidenceItem,
)


class AnsweringPipeline:
    """执行 Evidence 选择、结构化推理和最终答案组合。"""

    def __init__(self, llm_client: Any | None = None) -> None:
        self.selector = EvidenceSelector(llm_client=llm_client)
        self.reasoner = AnswerReasoner(llm_client=llm_client)
        self.composer = AnswerComposer(llm_client=llm_client)

    def run(
        self,
        *,
        query: str,
        evidence: List[EvidenceItem],
        options: AnsweringOptions | None = None,
    ) -> AnsweringPipelineResult:
        opts = options or AnsweringOptions()
        selection = self.selector.select(
            query=query,
            evidence=evidence,
            options=opts.evidence_selection,
        )
        reasoning = self.reasoner.reason(
            query=query,
            evidence=selection.evidence,
            evidence_text=selection.evidence_text,
            answerability=selection.result.get("answerability", "uncertain"),
            semantic_score=float(selection.result.get("score", 0.0) or 0.0),
            options=opts.reasoning,
        )
        answer = self.composer.compose(
            query=query,
            evidence=selection.evidence,
            reasoning=reasoning.result,
            evidence_text=selection.evidence_text,
            reasoning_text=reasoning.reasoning_text,
            answerability=selection.result.get("answerability", "uncertain"),
            semantic_score=float(selection.result.get("score", 0.0) or 0.0),
            scoring_reason=str(selection.result.get("reason", "") or ""),
            options=opts.generation,
        )
        return AnsweringPipelineResult(
            selection=selection,
            reasoning=reasoning,
            answer=answer,
        )
