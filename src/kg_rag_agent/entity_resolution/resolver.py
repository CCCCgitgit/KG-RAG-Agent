# -*- coding: utf-8 -*-
"""End-to-end entity-resolution façade."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .grounder import EntityGrounder
from .linker import EntityLinker
from .mention_extractor import MentionExtractor
from .schemas import (
    EntityCandidate,
    EntityResolutionResult,
    GroundingOptions,
    LinkingOptions,
    MentionExtractionOptions,
)


class EntityResolutionPipeline:
    def __init__(
        self,
        *,
        mention_extractor: MentionExtractor,
        entity_linker: EntityLinker,
        entity_grounder: Optional[EntityGrounder] = None,
    ) -> None:
        self.mention_extractor = mention_extractor
        self.entity_linker = entity_linker
        self.entity_grounder = entity_grounder

    def resolve(
        self,
        query: str,
        *,
        extraction_options: Optional[MentionExtractionOptions | Mapping[str, Any]] = None,
        linking_options: Optional[LinkingOptions | Mapping[str, Any]] = None,
        grounding_options: Optional[GroundingOptions | Mapping[str, Any]] = None,
        extraction_config: Optional[Mapping[str, Any]] = None,
    ) -> EntityResolutionResult:
        extraction = self.mention_extractor.extract(
            query,
            options=extraction_options,
            config=extraction_config,
        )
        link_options = (
            linking_options
            if isinstance(linking_options, LinkingOptions)
            else LinkingOptions.from_mapping(linking_options)
        )
        candidates_by_mention: Dict[str, List[EntityCandidate]] = {}
        unlinked: List[str] = []
        warnings = list(extraction.warnings)
        raw_linking: Dict[str, Dict[str, Any]] = {}

        for mention in extraction.mentions:
            mention_text = str(mention.get("text", "") or "").strip()
            if not mention_text:
                continue
            raw = self.entity_linker.link(mention_text, top_k=link_options.top_k)
            raw_linking[mention_text] = raw
            candidates = _normalize_candidates(mention_text, raw)
            candidates_by_mention[mention_text] = candidates
            if not candidates:
                unlinked.append(mention_text)

        grounded_entities = []
        ungrounded: List[str] = []
        if self.entity_grounder is not None:
            grounded = self.entity_grounder.ground(
                candidates_by_mention,
                options=grounding_options,
            )
            grounded_entities = grounded.grounded_entities
            ungrounded = grounded.ungrounded_mentions
            warnings.extend(grounded.warnings)

        return EntityResolutionResult(
            query=str(query or ""),
            mentions=extraction.mentions,
            candidates_by_mention=candidates_by_mention,
            grounded_entities=grounded_entities,
            unlinked_mentions=unlinked,
            ungrounded_mentions=ungrounded,
            warnings=warnings,
            metadata={
                "extractor_type": extraction.extractor_type,
                "raw_extraction_output": extraction.raw_output,
                "raw_linking_results": raw_linking,
            },
        )


def _normalize_candidates(mention: str, raw_result: Mapping[str, Any]) -> List[EntityCandidate]:
    raw_candidates = raw_result.get("candidates", []) if isinstance(raw_result, Mapping) else []
    if not isinstance(raw_candidates, list):
        return []
    result: List[EntityCandidate] = []
    seen = set()
    selected_name = str(raw_result.get("selected_entity", "") or "")
    selected_id = str(raw_result.get("selected_entity_id", "") or "")

    for rank, item in enumerate(raw_candidates, start=1):
        if not isinstance(item, Mapping):
            continue
        metadata = dict(item.get("metadata", {}) or {})
        entity_name = str(
            item.get("entity_name")
            or item.get("entity")
            or item.get("name")
            or metadata.get("entity_name")
            or ""
        ).strip()
        entity_id = str(
            item.get("entity_id")
            or item.get("id")
            or metadata.get("entity_id")
            or ""
        ).strip()
        if not entity_name and not entity_id:
            continue
        dedup_key = (entity_id.casefold(), entity_name.casefold())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        aliases = item.get("aliases") or metadata.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        if not isinstance(aliases, list):
            aliases = []
        metadata.setdefault("rank", int(item.get("rank", rank) or rank))
        metadata.setdefault(
            "selected",
            bool(
                (selected_id and entity_id == selected_id)
                or (selected_name and entity_name == selected_name)
                or metadata.get("selected", False)
            ),
        )
        result.append(
            EntityCandidate(
                mention=mention,
                entity_id=entity_id,
                entity_name=entity_name,
                score=_clip_score(item.get("score", 0.0)),
                source=str(item.get("source", raw_result.get("linking_method", "")) or ""),
                aliases=[str(alias) for alias in aliases if str(alias).strip()],
                metadata=metadata,
            )
        )
    return result


def _clip_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))
