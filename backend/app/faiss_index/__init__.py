"""
FAISS Similarity Search Module
===============================
Production-grade FAISS-based similarity search for NBA players.

Sub-modules:
- index_builder: Build FAISS indices from pipeline feature matrices
- service: module-level state + functions with blue-green index swapping
- ranking: Hybrid scoring, feature attribution, block-weighted similarity
- evaluation: Recall@k evaluation and regression testing

Usage:
    from app.faiss_index import service

    service.initialize(data_dir="data/nba-aba-baa-stats/versions/56")
    results = service.search_player("jamesle01", k=10)
"""

from .index_builder import (
    build_player_faiss_index,
    load_faiss_index,
    load_metadata,
    l2_normalize,
)
from .service import EntityType
from .ranking import (
    ScoringContext,
    build_scoring_context,
    compute_hybrid_score,
    compute_feature_attributions,
    compute_block_level_attribution,
    build_block_index_map,
    PLAYER_BLOCK_PATTERNS,
)
from .evaluation import (
    evaluate_recall_at_k,
    build_labeled_pairs_from_clusters,
    capture_regression_baseline,
    compare_to_regression_baseline,
    save_regression_baseline,
    load_regression_baseline,
    CANONICAL_PLAYERS,
)

__all__ = [
    "build_player_faiss_index",
    "load_faiss_index",
    "load_metadata",
    "l2_normalize",
    "EntityType",
    "ScoringContext",
    "build_scoring_context",
    "compute_hybrid_score",
    "compute_feature_attributions",
    "compute_block_level_attribution",
    "build_block_index_map",
    "PLAYER_BLOCK_PATTERNS",
    "evaluate_recall_at_k",
    "build_labeled_pairs_from_clusters",
    "capture_regression_baseline",
    "compare_to_regression_baseline",
    "save_regression_baseline",
    "load_regression_baseline",
    "CANONICAL_PLAYERS",
]
