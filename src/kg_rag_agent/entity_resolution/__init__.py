# -*- coding: utf-8 -*-
"""Canonical entity-resolution public API."""

from .grounder import EntityGrounder
from .linker import EntityLinker, batch_link_entities, link_entity
from .mention_extractor import MentionExtractor, extract_mentions
from .normalizer import EntityNormalizer
from .resolver import EntityResolutionPipeline
from .schemas import (
    EntityCandidate,
    EntityResolutionResult,
    GroundedEntity,
    GroundingOptions,
    GroundingResult,
    LinkingOptions,
    LinkingResult,
    Mention,
    MentionExtractionOptions,
    MentionExtractionResult,
)

__all__ = [
    "Mention",
    "EntityCandidate",
    "GroundedEntity",
    "MentionExtractionOptions",
    "LinkingOptions",
    "GroundingOptions",
    "MentionExtractionResult",
    "LinkingResult",
    "GroundingResult",
    "EntityResolutionResult",
    "MentionExtractor",
    "extract_mentions",
    "EntityNormalizer",
    "EntityLinker",
    "link_entity",
    "batch_link_entities",
    "EntityGrounder",
    "EntityResolutionPipeline",
]
