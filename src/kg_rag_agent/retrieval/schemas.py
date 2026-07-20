
# -*- coding: utf-8 -*-
"""Retrieval 层共享类型协议。"""
from __future__ import annotations
from typing import Any, Dict, List, NotRequired, TypedDict

class RetrievalItem(TypedDict):
    id: str
    text: str
    content: str
    score: float
    source: str
    source_type: str
    metadata: Dict[str,Any]
    rank: NotRequired[int]
    distance: NotRequired[float|None]
    raw_score: NotRequired[float]
    retrieval_sources: NotRequired[List[str]]
    raw_result: NotRequired[Any]

class VectorSearchItem(RetrievalItem,total=False):
    embedding: List[float]

class EntitySearchItem(TypedDict,total=False):
    rank: int
    vector_id: str
    entity_id: str
    entity_name: str
    aliases: List[str]
    document: str
    metadata: Dict[str,Any]
    distance: float|None
    score: float

class RerankItem(RetrievalItem,total=False):
    rerank_rank: int
    rerank_score: float
    rerank_method: str
    rerank_detail: Dict[str,Any]
