# -*- coding: utf-8 -*-
"""Knowledge-graph storage, query, and evidence construction APIs."""

from .graph_loader import (
    GraphLoader,
    clear_graph_cache,
    get_cached_graph_paths,
    get_edge_attrs,
    get_edges,
    get_graph_stats,
    get_node_attrs,
    get_nodes,
    get_num_edges,
    get_num_nodes,
    has_node,
    load_graph,
    load_graph_stats,
    save_graph,
    save_graph_stats,
    validate_graph,
)
from .entity_normalizer import (
    EntityNormalizer,
    build_node_name_index,
    clean_mention_text,
    deduplicate_candidates,
    find_node_key,
    is_valid_entity_text,
    load_alias_map,
    normalize_alias_map,
    normalize_candidate,
    normalize_candidates,
    normalize_entity_name,
    normalize_mention,
    normalize_node_key,
    normalize_text,
    resolve_alias,
    save_alias_map,
)
from .entity_linker import (
    EntityLinker,
    batch_link_entities,
    get_default_linker,
    link_entity,
)
from .relation_search import find_relations, relation_search, search_relation
from .path_search import find_paths, path_search, search_path
from .neighbor_search import find_neighbors, neighbor_search, search_neighbors
from .subgraph_search import ego_graph_search, search_subgraph, subgraph_search
from .evidence_builder import (
    EvidenceBuilder,
    build_evidence_from_results,
    build_evidence_item,
    build_evidence_text,
    build_relation_text,
    deduplicate_evidence,
    filter_evidence_by_score,
    make_evidence_id,
    normalize_evidence_item,
    normalize_evidence_list,
    sort_evidence,
    triples_to_text,
)
from .retriever import KGRetriever
from .schemas import (
    EvidenceItem,
    KGRetrievalOptions,
    KGRetrievalResult,
    NeighborItem,
    PathItem,
    RelationItem,
    SubgraphEdge,
    SubgraphNode,
    Triple,
)

__all__ = [
    "GraphLoader", "load_graph", "save_graph", "validate_graph",
    "get_graph_stats", "get_num_nodes", "get_num_edges", "has_node",
    "get_nodes", "get_edges", "get_node_attrs", "get_edge_attrs",
    "clear_graph_cache", "get_cached_graph_paths", "save_graph_stats",
    "load_graph_stats",
    "EntityNormalizer", "normalize_text", "normalize_mention",
    "normalize_entity_name", "normalize_node_key", "clean_mention_text",
    "is_valid_entity_text", "load_alias_map", "save_alias_map",
    "normalize_alias_map", "resolve_alias", "normalize_candidate",
    "normalize_candidates", "deduplicate_candidates",
    "build_node_name_index", "find_node_key",
    "EntityLinker", "get_default_linker", "link_entity",
    "batch_link_entities",
    "relation_search", "search_relation", "find_relations",
    "path_search", "search_path", "find_paths",
    "neighbor_search", "search_neighbors", "find_neighbors",
    "subgraph_search", "search_subgraph", "ego_graph_search",
    "EvidenceBuilder", "build_evidence_item", "build_evidence_from_results",
    "normalize_evidence_item", "normalize_evidence_list",
    "deduplicate_evidence", "sort_evidence", "filter_evidence_by_score",
    "build_evidence_text", "build_relation_text", "triples_to_text",
    "make_evidence_id",
    "Triple", "RelationItem", "PathItem", "NeighborItem",
    "SubgraphNode", "SubgraphEdge", "EvidenceItem",
    "KGRetrievalOptions", "KGRetrievalResult", "KGRetriever",
]
