"""
Ranking & Re-ranking for FAISS Similarity Search
=================================================
Hybrid scoring: blends raw cosine similarity with block-weighted similarity
and role bonuses. Also provides feature-level attribution for explainability.

Design (from faiss_similarity_search_design.md §6):
- hybrid_score = α·cos_sim + β·block_sim + γ·role_bonus
- Default weights: α=0.6, β=0.35, γ=0.05
- Feature attribution decomposes cosine sim into per-feature contributions
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# FEATURE BLOCK DEFINITIONS (aligned with constants.py)
# ═══════════════════════════════════════════════════════════════

# Player feature block → canonical feature substrings
PLAYER_BLOCK_PATTERNS = {
    "scoring": [
        "pts_per_game", "usg_percent", "ts_percent", "fg_percent",
        "fg_per_game", "fga_per_game", "x3p_per_game", "x3pa_per_game",
        "ft_per_game", "fta_per_game", "x2p_per_game", "x2pa_per_game",
        "scoring_score",
    ],
    "playmaking": [
        "ast_per_game", "ast_percent", "tov_per_game", "tov_percent",
        "points_generated_by_assists", "playmaking_score",
    ],
    "rebounding": [
        "orb_percent", "drb_percent", "trb_percent",
        "orb_per_game", "drb_per_game", "trb_per_game",
        "rebounding_score",
    ],
    "defense": [
        "stl_percent", "blk_percent", "dbpm", "dws",
        "stl_per_game", "blk_per_game", "defense_score",
    ],
    "shooting": [
        "avg_dist_fga", "x3p_ar", "f_tr", "e_fg_percent",
        "x3p_percent", "ft_percent",
        "percent_fga_from", "fg_percent_from", "spacing_score",
    ],
    "positional": [
        "pg_percent", "sg_percent", "sf_percent", "pf_percent", "c_percent",
        "versatility_score",
    ],
    "advanced": [
        "per", "bpm", "obpm", "vorp", "ws", "ows", "ws_48",
    ],
}


# ═══════════════════════════════════════════════════════════════
# BLOCK INDEX MAP BUILDING
# ═══════════════════════════════════════════════════════════════

def build_block_index_map(
    feature_names: list[str],
    entity_type: str = "player",
) -> dict[str, np.ndarray]:
    """
    Map each semantic block to the indices of its features in the vector.

    Returns dict: block_name → np.array of column indices.

    Feature names may have '_era_adj' suffix from the pipeline — we strip it
    before matching against block patterns.
    """
    block_patterns = PLAYER_BLOCK_PATTERNS
    block_map = {}

    for block_name, patterns in block_patterns.items():
        indices = []
        for i, fname in enumerate(feature_names):
            # Strip era_adj suffix for matching
            base = fname.replace("_era_adj", "")
            for pat in patterns:
                if pat.lower() in base.lower():
                    indices.append(i)
                    break
        if indices:
            block_map[block_name] = np.array(indices, dtype=np.int64)

    return block_map


# ═══════════════════════════════════════════════════════════════
# HYBRID SCORING
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ScoringContext:
    """Immutable, precomputed inputs for hybrid similarity scoring.

    Built once per index install via ``build_scoring_context`` — the block
    index map is the expensive part (substring matching over all features).
    """

    feature_names: tuple[str, ...]
    entity_type: str
    alpha: float
    beta: float
    gamma: float
    block_map: dict[str, np.ndarray] = field(compare=False)


def build_scoring_context(
    feature_names: list[str],
    entity_type: str = "player",
    alpha: float = 0.60,
    beta: float = 0.35,
    gamma: float = 0.05,
) -> ScoringContext:
    """Precompute the hybrid scoring context for an index's feature layout."""
    return ScoringContext(
        feature_names=tuple(feature_names),
        entity_type=entity_type,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        block_map=build_block_index_map(feature_names, entity_type),
    )


def _compute_role_bonus(
    entity_type: str, query_meta: dict, candidate_meta: dict
) -> float:
    """Compute role bonus for matching primary position (players)."""
    score = 0.0

    if entity_type == "player":
        q_pos = query_meta.get("primary_position", "")
        c_pos = candidate_meta.get("primary_position", "")
        if q_pos and c_pos and q_pos == c_pos:
            score += 1.0  # same primary position

    return score


def compute_hybrid_score(
    scorer: ScoringContext,
    query_vec: np.ndarray,       # L2-normalized, shape (d,)
    candidate_vec: np.ndarray,   # L2-normalized, shape (d,)
    query_meta: dict,
    candidate_meta: dict,
    block_weights: Optional[dict[str, float]] = None,
) -> dict:
    """
    Compute hybrid score and return breakdown.

    Returns dict with:
        - hybrid_score: float
        - cos_sim: float (raw cosine from FAISS)
        - block_sim: float (weighted block cosine)
        - role_bonus: float (position/era match bonus)
        - block_contributions: dict[block_name → contribution]
    """
    if block_weights is None:
        # Default: all blocks weight 1.0
        block_weights = {name: 1.0 for name in scorer.block_map}

    # 1. Full cosine similarity (already have this from FAISS, but recompute for safety)
    cos_full = float(np.dot(query_vec, candidate_vec))

    # 2. Block-weighted similarity
    block_scores = {}
    total_weight = 0.0
    for block_name, feat_indices in scorer.block_map.items():
        w = block_weights.get(block_name, 1.0)
        if len(feat_indices) == 0:
            continue

        q_block = query_vec[feat_indices]
        c_block = candidate_vec[feat_indices]

        # Re-normalize sub-vectors for block-specific cosine
        q_norm = q_block / (np.linalg.norm(q_block) + 1e-8)
        c_norm = c_block / (np.linalg.norm(c_block) + 1e-8)
        block_cos = float(np.dot(q_norm, c_norm))

        block_scores[block_name] = w * block_cos
        total_weight += w

    block_sim = sum(block_scores.values()) / total_weight if total_weight > 0 else 0.0

    # 3. Role bonus
    role_bonus = _compute_role_bonus(scorer.entity_type, query_meta, candidate_meta)

    # 4. Hybrid score
    hybrid = scorer.alpha * cos_full + scorer.beta * block_sim + scorer.gamma * role_bonus

    return {
        "hybrid_score": hybrid,
        "cos_sim": cos_full,
        "block_sim": block_sim,
        "role_bonus": role_bonus,
        "block_contributions": block_scores,
    }


# ═══════════════════════════════════════════════════════════════
# FEATURE ATTRIBUTION (EXPLAINABILITY)
# ═══════════════════════════════════════════════════════════════

def compute_feature_attributions(
    query_vec: np.ndarray,       # L2-normalized, shape (d,)
    candidate_vec: np.ndarray,   # L2-normalized, shape (d,)
    feature_names: list[str],
    top_n: int = 5,
) -> list[dict]:
    """
    Compute per-feature contribution to cosine similarity.

    cos_sim = Σ(query_i * candidate_i)
    Each feature contributes: query_i * candidate_i / cos_sim
    """
    cos_sim = float(np.dot(query_vec, candidate_vec))
    if cos_sim <= 0 or not np.isfinite(cos_sim):
        return []

    raw_contributions = query_vec * candidate_vec

    # Handle the case where cos_sim is very small
    if abs(cos_sim) < 1e-8:
        return []

    normalized = raw_contributions / cos_sim

    # Get top-N by absolute contribution
    top_indices = np.argsort(-np.abs(normalized))[:top_n]

    results = []
    for i in top_indices:
        fname = feature_names[i].replace("_era_adj", "")
        results.append({
            "feature": fname,
            "contribution": float(normalized[i]),
            "abs_contribution": float(abs(normalized[i])),
            "query_value": float(query_vec[i]),
            "candidate_value": float(candidate_vec[i]),
        })

    # Sort by absolute contribution descending
    results.sort(key=lambda x: x["abs_contribution"], reverse=True)
    return results


def compute_block_level_attribution(
    query_vec: np.ndarray,
    candidate_vec: np.ndarray,
    feature_names: list[str],
    entity_type: str = "player",
) -> dict[str, float]:
    """
    Compute block-level contribution summary.
    
    Returns dict: block_name → fractional contribution to cosine similarity.
    Values sum to ~1.0.
    """
    block_map = build_block_index_map(feature_names, entity_type)
    cos_sim = float(np.dot(query_vec, candidate_vec))
    if cos_sim <= 0 or not np.isfinite(cos_sim):
        return {}

    raw_contributions = query_vec * candidate_vec
    block_attrib = {}

    for block_name, feat_indices in block_map.items():
        if len(feat_indices) == 0:
            continue
        block_contrib = float(np.sum(raw_contributions[feat_indices]))
        block_attrib[block_name] = block_contrib / cos_sim

    return block_attrib


# ═══════════════════════════════════════════════════════════════
# BATCH SCORING (vectorized for performance)
# ═══════════════════════════════════════════════════════════════

def compute_hybrid_scores_batch(
    scorer: ScoringContext,
    query_vec: np.ndarray,           # (d,) L2-normalized
    candidate_matrix: np.ndarray,   # (n, d) L2-normalized
    query_meta: dict,
    candidate_metas: list[dict],
    block_weights: Optional[dict[str, float]] = None,
) -> list[dict]:
    """
    Vectorized batch hybrid scoring.

    Computes hybrid scores for all candidates in a single matrix operation
    instead of looping per-candidate.

    Returns list of dicts (same shape as compute_hybrid_score) — one per candidate.
    """
    n_candidates = candidate_matrix.shape[0]
    if n_candidates == 0:
        return []

    if block_weights is None:
        block_weights = {name: 1.0 for name in scorer.block_map}

    # 1. Full cosine similarities: (n,) = candidate_matrix @ query_vec
    cos_sims = candidate_matrix @ query_vec  # (n,)

    # 2. Block-weighted similarities (vectorized)
    block_scores_list = []
    total_weight = 0.0

    for block_name, feat_indices in scorer.block_map.items():
        if len(feat_indices) == 0:
            continue
        w = block_weights.get(block_name, 1.0)

        q_block = query_vec[feat_indices]                    # (db,)
        c_blocks = candidate_matrix[:, feat_indices]          # (n, db)

        # Re-normalize sub-vectors
        q_norm = q_block / (np.linalg.norm(q_block) + 1e-8)
        c_norms = np.linalg.norm(c_blocks, axis=1, keepdims=True)  # (n, 1)
        c_normed = c_blocks / (c_norms + 1e-8)               # (n, db)

        block_cos = c_normed @ q_norm  # (n,)

        block_scores_list.append((block_name, w, block_cos))
        total_weight += w

    if total_weight > 0:
        block_sim = np.zeros(n_candidates)
        for _, w, bcos in block_scores_list:
            block_sim += w * bcos
        block_sim /= total_weight
    else:
        block_sim = np.zeros(n_candidates)

    # 3. Role bonuses (still per-candidate, but cheap)
    role_bonuses = np.zeros(n_candidates)
    if scorer.entity_type == "player":
        q_pos = query_meta.get("primary_position", "")
        if q_pos:
            for i, cmeta in enumerate(candidate_metas):
                c_pos = cmeta.get("primary_position", "")
                if c_pos and c_pos == q_pos:
                    role_bonuses[i] = 1.0

    # 4. Hybrid scores
    hybrid_scores = (
        scorer.alpha * cos_sims
        + scorer.beta * block_sim
        + scorer.gamma * role_bonuses
    )

    # Build per-candidate result dicts
    results = []
    for i in range(n_candidates):
        block_contributions = {}
        for name, w, bcos in block_scores_list:
            block_contributions[name] = w * bcos[i]
        results.append({
            "hybrid_score": float(hybrid_scores[i]),
            "cos_sim": float(cos_sims[i]),
            "block_sim": float(block_sim[i]),
            "role_bonus": float(role_bonuses[i]),
            "block_contributions": block_contributions,
        })
    return results


def compute_feature_attributions_batch(
    query_vec: np.ndarray,           # (d,) L2-normalized
    candidate_matrix: np.ndarray,    # (n, d) L2-normalized
    feature_names: list[str],
    top_n: int = 5,
) -> list[list[dict]]:
    """
    Vectorized batch feature attribution.

    Returns list (one per candidate) of top-N feature attribution dicts.
    """
    n_candidates = candidate_matrix.shape[0]
    if n_candidates == 0:
        return []

    # Raw contributions: (n, d) = candidate_matrix * query_vec (broadcast)
    raw_contributions = candidate_matrix * query_vec[np.newaxis, :]

    # Cosine sims: (n,)
    cos_sims = candidate_matrix @ query_vec

    results = []
    for i in range(n_candidates):
        cos_sim = float(cos_sims[i])
        if cos_sim <= 0 or not np.isfinite(cos_sim) or abs(cos_sim) < 1e-8:
            results.append([])
            continue

        normalized = raw_contributions[i] / cos_sim
        top_indices = np.argsort(-np.abs(normalized))[:top_n]

        feat_list = []
        for idx in top_indices:
            fname = feature_names[idx].replace("_era_adj", "")
            feat_list.append({
                "feature": fname,
                "contribution": float(normalized[idx]),
                "abs_contribution": float(abs(normalized[idx])),
                "query_value": float(query_vec[idx]),
                "candidate_value": float(candidate_matrix[i, idx]),
            })
        feat_list.sort(key=lambda x: x["abs_contribution"], reverse=True)
        results.append(feat_list)

    return results
