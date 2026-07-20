# -*- coding: utf-8 -*-
"""回答领域的统一数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, TypedDict

AnswerabilityType = Literal["answerable", "uncertain", "unanswerable"]


class EvidenceItem(TypedDict, total=False):
    evidence_id: str
    evidence_type: str
    source_entity: str
    target_entity: str
    relation: str
    path: List[str]
    triples: List[Dict[str, Any]]
    text: str
    score: float
    metadata: Dict[str, Any]


class SemanticScoringResult(TypedDict, total=False):
    score: float
    answerability: AnswerabilityType
    reason: str
    selected_evidence_ids: List[str]
    rejected_evidence_ids: List[str]


class ReasoningResult(TypedDict, total=False):
    reasoning_chain: List[str]
    conclusion: str
    used_evidence_ids: List[str]
    confidence: float
    metadata: Dict[str, Any]


class Citation(TypedDict, total=False):
    citation_id: str
    evidence_id: str
    type: str
    source_entity: str
    relation: str
    target_entity: str
    text: str
    score: float


@dataclass(frozen=True)
class EvidenceSelectionOptions:
    use_llm: bool = False
    max_selected_evidence: int = 8
    min_evidence_score: float = 0.15
    answerable_threshold: float = 0.55
    uncertain_threshold: float = 0.25

    def normalized(self) -> "EvidenceSelectionOptions":
        max_selected = max(int(self.max_selected_evidence), 1)
        min_score = min(max(float(self.min_evidence_score), 0.0), 1.0)
        uncertain = min(max(float(self.uncertain_threshold), 0.0), 1.0)
        answerable = min(max(float(self.answerable_threshold), uncertain), 1.0)
        return EvidenceSelectionOptions(
            use_llm=bool(self.use_llm),
            max_selected_evidence=max_selected,
            min_evidence_score=min_score,
            answerable_threshold=answerable,
            uncertain_threshold=uncertain,
        )


@dataclass(frozen=True)
class ReasoningOptions:
    use_llm: bool = False
    max_reasoning_steps: int = 5

    def normalized(self) -> "ReasoningOptions":
        return ReasoningOptions(
            use_llm=bool(self.use_llm),
            max_reasoning_steps=max(1, min(int(self.max_reasoning_steps), 20)),
        )


@dataclass(frozen=True)
class GenerationOptions:
    use_llm: bool = True
    temperature: float = 0.2
    max_tokens: int = 1200
    include_citations: bool = True

    def normalized(self) -> "GenerationOptions":
        return GenerationOptions(
            use_llm=bool(self.use_llm),
            temperature=min(max(float(self.temperature), 0.0), 2.0),
            max_tokens=max(1, min(int(self.max_tokens), 32768)),
            include_citations=bool(self.include_citations),
        )


@dataclass
class EvidenceSelection:
    result: SemanticScoringResult
    evidence: List[EvidenceItem]
    evidence_text: str
    scoring_type: str


@dataclass
class ReasoningOutput:
    result: ReasoningResult
    reasoning_text: str
    reasoning_type: str


@dataclass
class AnswerResult:
    answer: str
    citations: List[Citation] = field(default_factory=list)
    generation_type: str = "rule"
    answerability: AnswerabilityType = "uncertain"
    semantic_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnsweringOptions:
    evidence_selection: EvidenceSelectionOptions = field(default_factory=EvidenceSelectionOptions)
    reasoning: ReasoningOptions = field(default_factory=ReasoningOptions)
    generation: GenerationOptions = field(default_factory=GenerationOptions)


@dataclass
class AnsweringPipelineResult:
    selection: EvidenceSelection
    reasoning: ReasoningOutput
    answer: AnswerResult
