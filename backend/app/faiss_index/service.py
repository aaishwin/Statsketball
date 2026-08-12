"""
FAISS Similarity Search Service (functional)
=============================================
Module-level state managing the player FAISS index with blue-green
swapping for zero-downtime index updates.

Design (from faiss_similarity_search_design.md §3, §5):
- Two index slots: "active" (serves queries) and "standby" (built into)
- Queries read the active slot without locking; rebuilds write standby then atomically swap
- In-memory TTLCache for query results (invalidated on rebuild)

Thread safety:
- Python GIL makes dict pointer assignment atomic for single-threaded reads.
- ``_swap_lock`` only protects the swap operation itself (held for microseconds).
- ``_rebuild_lock`` serializes full rebuilds (acquired non-blocking → 409 upstream).

Public API (all module functions — no classes, no singleton):
    configure(data_dir, output_dir)
    initialize(data_dir, output_dir)          — build the player index from CSVs
    load_prebuilt_indices(output_dir)          — load index/metadata from disk
    search_player(player_id, k, filters, ...)
    search_by_vector(vector, entity_type, ...)
    rebuild_player_index()
    get_index_info(entity_type)
    resolve_player(query) / suggest_players(query, limit)
    active_indices() / cache_size()
    acquire_rebuild_lock() / release_rebuild_lock()
"""

import bisect
import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Final, Optional

import numpy as np
import pandas as pd
from cachetools import TTLCache

from .index_builder import (
    l2_normalize,
    build_player_faiss_index,
    load_faiss_index,
    load_metadata,
)
from .ranking import (
    ScoringContext,
    build_scoring_context,
    compute_hybrid_scores_batch,
    compute_feature_attributions_batch,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# ENTITY TYPE CONSTANTS
# ═══════════════════════════════════════════════════════════════
# Plain string constants — the API layer's ``app.api.models.EntityType``
# (a str Enum) carries the same values; this module works with plain strings
# to avoid an api→faiss_index dependency inversion.

PLAYER: Final[str] = "player"
ENTITY_TYPES: Final[tuple[str]] = (PLAYER,)


class EntityType:
    """Backwards-compatible namespace for entity type constants.

    Deprecated: prefer the module-level ``PLAYER`` constant.
    Retained so existing ``from .service import EntityType`` call sites work.
    """

    PLAYER: Final[str] = PLAYER


# ═══════════════════════════════════════════════════════════════
# REBUILD CONCURRENCY GUARD
# ═══════════════════════════════════════════════════════════════
# A single non-blocking lock serializes full index rebuilds. The blue-green
# _swap_lock only protects the pointer swap (microseconds); it does NOT
# prevent two concurrent rebuilds from running full feature-engineering +
# FAISS pipelines simultaneously and racing writes to the same
# faiss_output/ files (index/metadata pairs can end up mismatched).
# This lock is acquired non-blocking so a second concurrent rebuild request
# returns 409 REBUILD_IN_PROGRESS instead of blocking or corrupting artifacts.
_rebuild_lock = threading.Lock()


def acquire_rebuild_lock() -> bool:
    """Try to acquire the global rebuild lock without blocking.

    Returns True if acquired (caller MUST release it), False if a rebuild
    is already in progress.
    """
    return _rebuild_lock.acquire(blocking=False)


def release_rebuild_lock() -> None:
    """Release the global rebuild lock if held. Safe to call when not held."""
    try:
        _rebuild_lock.release()
    except RuntimeError:
        # Lock was not held — nothing to do.
        pass


# ═══════════════════════════════════════════════════════════════
# MODULE STATE
# ═══════════════════════════════════════════════════════════════

_swap_lock = threading.Lock()

# entity_type → {"active": slot | None, "standby": slot | None}
# A slot is a dict: index, version, built_at, id_to_idx, dimension,
# vector_count, feature_names.
_indices: dict[str, dict[str, Optional[dict[str, Any]]]] = {
    PLAYER: {"active": None, "standby": None},
}

# entity_type → {entity_id → metadata dict}
_metadata: dict[str, dict[str, dict[str, Any]]] = {PLAYER: {}}

# entity_type → {entity_id → row index in FAISS index}
_id_to_idx: dict[str, dict[str, int]] = {PLAYER: {}}

# entity_type → {row index → metadata dict}. Built ONCE per index swap so
# searches never rebuild it per request (previously an O(n) dict build on
# every uncached query).
_idx_to_meta: dict[str, dict[int, dict[str, Any]]] = {PLAYER: {}}

# entity_type → precomputed hybrid scoring context (weights + block map)
_scorers: dict[str, Optional[ScoringContext]] = {PLAYER: None}

# Query result cache. Keys are (entity_type, md5) tuples so per-entity-type
# invalidation on rebuild actually works (a plain md5 string key made the
# entity-type filter unmatchable — silently clearing nothing).
_cache: TTLCache = TTLCache(maxsize=10_000, ttl=3600)  # 1 hour TTL

# Paths stored for rebuild operations
_data_dir: Optional[str] = None
_output_dir: Optional[str] = None

# Name lookup: lowercase display name → entity_id (players only)
_name_to_entity_id: dict[str, str] = {}
# Reverse: entity_id → display name
_entity_id_to_name: dict[str, str] = {}
# Sorted list of (lowercase_name, entity_id) for prefix matching
_sorted_player_names: list[tuple[str, str]] = []


def configure(data_dir: Optional[str] = None, output_dir: Optional[str] = None) -> None:
    """Set the data/output directories used by rebuild operations."""
    global _data_dir, _output_dir
    if data_dir is not None:
        _data_dir = data_dir
    if output_dir is not None:
        _output_dir = output_dir


def reset_state() -> None:
    """Reset all module state (test isolation helper)."""
    global _data_dir, _output_dir
    with _swap_lock:
        for et in ENTITY_TYPES:
            _indices[et]["active"] = None
            _indices[et]["standby"] = None
            _metadata[et] = {}
            _id_to_idx[et] = {}
            _idx_to_meta[et] = {}
            _scorers[et] = None
        _cache.clear()
        _name_to_entity_id.clear()
        _entity_id_to_name.clear()
        _sorted_player_names.clear()
        _data_dir = None
        _output_dir = None


# ═══════════════════════════════════════════════════════════════
# NAME LOOKUP / RESOLUTION
# ═══════════════════════════════════════════════════════════════

def _build_name_lookup(entity_type: str = PLAYER) -> None:
    """Build name→entity_id lookup tables from loaded metadata.

    Called after indices are loaded/built for the player entity type.
    """
    global _name_to_entity_id, _entity_id_to_name, _sorted_player_names
    if entity_type != PLAYER:
        return

    name_to_id: dict[str, str] = {}
    id_to_name: dict[str, str] = {}
    sorted_names: list[tuple[str, str]] = []

    for eid, meta in _metadata[PLAYER].items():
        display_name = meta.get("entity_name", eid)
        lower = display_name.lower().strip()
        name_to_id[lower] = eid
        id_to_name[eid] = display_name
        sorted_names.append((lower, eid))

    # Sort for binary-search prefix matching
    sorted_names.sort(key=lambda x: x[0])

    _name_to_entity_id = name_to_id
    _entity_id_to_name = id_to_name
    _sorted_player_names = sorted_names
    logger.info(
        f"Built player name lookup: {len(name_to_id)} names, "
        f"{len(sorted_names)} sorted entries"
    )


def resolve_player(query: str) -> Optional[str]:
    """Resolve a player query (name or entity_id) to an entity_id.

    Tries: exact entity_id → exact name → prefix name match.
    Returns None if no resolution found.
    """
    query = query.strip()
    if not query:
        return None

    # 1. Exact entity_id match
    if query in _id_to_idx.get(PLAYER, {}):
        return query

    # 2. Exact name match (case-insensitive)
    lower = query.lower()
    if lower in _name_to_entity_id:
        return _name_to_entity_id[lower]

    # 3. Prefix match on sorted names
    sorted_names = _sorted_player_names
    if not sorted_names:
        return None

    idx = bisect.bisect_left(sorted_names, (lower, ""))
    if idx < len(sorted_names) and sorted_names[idx][0].startswith(lower):
        return sorted_names[idx][1]

    return None


def suggest_players(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Return player suggestions matching the query (prefix, then substring).

    Returns list of dicts: {entity_id, entity_name, metadata}.
    """
    query = query.strip().lower()
    if not query:
        return []

    sorted_names = _sorted_player_names
    if not sorted_names:
        return []

    # Binary search for prefix start
    start = bisect.bisect_left(sorted_names, (query, ""))
    results: list[dict[str, Any]] = []

    for i in range(start, min(start + limit * 3, len(sorted_names))):
        name, eid = sorted_names[i]
        if not name.startswith(query):
            break

        meta = _metadata[PLAYER].get(eid, {})
        results.append({
            "entity_id": eid,
            "entity_name": meta.get("entity_name", eid),
            "metadata": _sanitize_metadata(meta),
        })

        if len(results) >= limit:
            break

    # If prefix match found nothing, try substring fallback
    if not results:
        for name, eid in sorted_names:
            if query in name:
                meta = _metadata[PLAYER].get(eid, {})
                results.append({
                    "entity_id": eid,
                    "entity_name": meta.get("entity_name", eid),
                    "metadata": _sanitize_metadata(meta),
                })
                if len(results) >= limit:
                    break

    return results


# ═══════════════════════════════════════════════════════════════
# INITIALIZATION / INDEX INSTALLATION
# ═══════════════════════════════════════════════════════════════

def initialize(
    data_dir: str,
    output_dir: str = "./faiss_output",
    min_player_seasons: int = 5,
) -> dict[str, Any]:
    """Run the full pipeline and build the initial FAISS player index.
    Called once at startup or on season rollover.

    Returns dict with build info for the player index.
    """
    from ..clustering.player_feature_engineering import run_player_feature_engineering

    configure(data_dir=data_dir, output_dir=output_dir)
    results: dict[str, Any] = {}

    # ── Build player index ──
    logger.info("Building player FAISS index...")
    t0 = time.time()
    player_fe = run_player_feature_engineering(data_dir, min_seasons=min_player_seasons)
    player_result = build_player_index_from_features(
        player_fe["X_scaled"],
        player_fe["feature_names"],
        player_fe["metadata_df"],
        output_dir,
    )
    results["player"] = {
        **player_result,
        "build_time_ms": (time.time() - t0) * 1000,
    }
    logger.info(f"Player index built in {results['player']['build_time_ms']:.0f}ms")

    return results


def _install_slot(
    entity_type: str,
    slot: dict[str, Any],
    new_metadata: dict[str, dict[str, Any]],
    id_to_idx: dict[str, int],
    scorer: ScoringContext,
) -> None:
    """Atomically promote a built slot to active and refresh lookup tables.

    Builds the row_index→metadata map ONCE here so the search path never
    reconstructs it per request.
    """
    idx_to_meta: dict[int, dict[str, Any]] = {}
    for meta in new_metadata.values():
        ri = meta.get("row_index")
        if ri is not None:
            idx_to_meta[int(ri)] = meta

    with _swap_lock:
        _indices[entity_type]["standby"] = slot
        _indices[entity_type]["active"] = slot
        _metadata[entity_type] = new_metadata
        _id_to_idx[entity_type] = id_to_idx
        _idx_to_meta[entity_type] = idx_to_meta
        _scorers[entity_type] = scorer

    if entity_type == PLAYER:
        _build_name_lookup(PLAYER)

    _clear_cache(entity_type)


def _metadata_maps_from_list(
    metadata_list: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Build (entity_id→meta, entity_id→row_index) maps from a metadata list."""
    new_metadata: dict[str, dict[str, Any]] = {}
    id_to_idx: dict[str, int] = {}
    for meta in metadata_list:
        eid = meta["entity_id"]
        new_metadata[eid] = meta
        id_to_idx[eid] = meta["row_index"]
    return new_metadata, id_to_idx


def build_player_index_from_features(
    X_scaled: np.ndarray,
    feature_names: list[str],
    metadata_df: pd.DataFrame,
    output_dir: str,
) -> dict[str, Any]:
    """Build a player FAISS index from a pre-computed feature matrix
    and promote it to active (blue-green swap).
    """
    result = build_player_faiss_index(X_scaled, feature_names, metadata_df, output_dir)

    metadata_list = load_metadata(result["metadata_path"])["metadata"]
    new_metadata, id_to_idx = _metadata_maps_from_list(metadata_list)

    scorer = build_scoring_context(feature_names, entity_type=PLAYER)
    index = load_faiss_index(result["index_path"])

    slot: dict[str, Any] = {
        "index": index,
        "version": result["index_version"],
        "built_at": datetime.now(timezone.utc).isoformat(),
        "id_to_idx": id_to_idx,
        "dimension": result["dimension"],
        "vector_count": result["n_vectors"],
        "feature_names": feature_names,
    }

    _install_slot(PLAYER, slot, new_metadata, id_to_idx, scorer)

    return {
        "index_version": result["index_version"],
        "vector_count": result["n_vectors"],
        "dimension": result["dimension"],
    }


def load_prebuilt_indices(output_dir: str) -> None:
    """Load pre-built FAISS index and metadata from disk into module state.

    Expects ``faiss_player.index`` / ``faiss_player_metadata.json`` inside
    ``output_dir``.
    """
    entity_type = PLAYER
    prefix = "faiss_player"
    index_path = os.path.join(output_dir, f"{prefix}.index")
    metadata_path = os.path.join(output_dir, f"{prefix}_metadata.json")

    index = load_faiss_index(index_path)
    meta_pkg = load_metadata(metadata_path)

    new_metadata, id_to_idx = _metadata_maps_from_list(meta_pkg["metadata"])
    scorer = build_scoring_context(meta_pkg["feature_names"], entity_type=entity_type)

    slot: dict[str, Any] = {
        "index": index,
        "version": meta_pkg["version"],
        "built_at": datetime.now(timezone.utc).isoformat(),
        "id_to_idx": id_to_idx,
        "dimension": meta_pkg["dimension"],
        "vector_count": meta_pkg["n_vectors"],
        "feature_names": meta_pkg["feature_names"],
    }

    _install_slot(entity_type, slot, new_metadata, id_to_idx, scorer)

    logger.info(
        f"Loaded {entity_type} index: {meta_pkg['n_vectors']} vectors × "
        f"{meta_pkg['dimension']} dims, version={meta_pkg['version']}"
    )


# ═══════════════════════════════════════════════════════════════
# SEARCH
# ═══════════════════════════════════════════════════════════════

def search_player(
    player_id: str,
    k: int = 10,
    filters: Optional[dict[str, Any]] = None,
    block_weights: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Find top-k most stylistically similar players.

    Parameters
    ----------
    player_id : entity ID (e.g., "jamesle01")
    k : number of results
    filters : optional dict with 'position' key
    block_weights : optional dict mapping block name → weight

    Returns dict with results list, query info, timing.
    """
    return _search(PLAYER, player_id, k, filters, block_weights)


def _search(
    entity_type: str,
    entity_id: str,
    k: int = 10,
    filters: Optional[dict[str, Any]] = None,
    block_weights: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Core search logic shared by players and teams."""
    t0 = time.time()

    # Check cache
    cache_key = _make_cache_key(entity_type, entity_id, k, filters, block_weights)
    cached = _cache.get(cache_key)
    if cached is not None:
        cached["timing_ms"] = (time.time() - t0) * 1000
        cached["cache_hit"] = True
        return cached

    # Validate index exists
    active = _indices[entity_type]["active"]
    if active is None:
        raise RuntimeError(f"No active {entity_type} index — run initialize() first")

    # Validate entity exists
    if entity_id not in _id_to_idx[entity_type]:
        raise KeyError(f"Entity '{entity_id}' not found in {entity_type} index")

    idx_map = _id_to_idx[entity_type]
    metadata = _metadata[entity_type]
    idx_to_meta = _idx_to_meta[entity_type]  # prebuilt at index install
    scorer = _scorers[entity_type]
    index = active["index"]
    feature_names = active["feature_names"]

    query_idx = idx_map[entity_id]
    query_meta = metadata[entity_id]

    # Retrieve query vector
    query_vec = index.reconstruct(query_idx)  # already L2-normalized

    # Overfetch for post-filtering — single pass with generous fetch
    has_filters = filters and (filters.get("position")
                               or filters.get("min_season") or filters.get("max_season"))
    fetch_k = k * 5 if has_filters else k + 1
    fetch_k = min(fetch_k, active["vector_count"])

    # FAISS search — single pass
    scores, indices = index.search(query_vec.reshape(1, -1), fetch_k + 1)

    # Collect valid candidate indices (exclude -1 and self)
    valid_mask = (indices[0] != -1) & (indices[0] != query_idx)
    cand_indices = indices[0][valid_mask]

    # Apply filters and collect valid candidates
    valid_cand_indices: list[int] = []
    valid_cand_metas: list[dict[str, Any]] = []
    filters_applied: dict[str, Any] = {}

    for ci in cand_indices:
        cand_meta = idx_to_meta.get(int(ci))
        if cand_meta is None:
            continue
        if has_filters and filters is not None:
            if not _passes_filters(cand_meta, filters):
                continue
            filters_applied = {fk: fv for fk, fv in filters.items() if fv is not None}
        valid_cand_indices.append(int(ci))
        valid_cand_metas.append(cand_meta)
        if len(valid_cand_indices) >= k * 2:
            break

    # Batch reconstruct all candidate vectors at once
    results: list[dict[str, Any]] = []
    if valid_cand_indices and scorer is not None:
        cand_indices_arr = np.array(valid_cand_indices, dtype=np.int64)
        # Batch reconstruct — single C++ call instead of N calls
        candidate_matrix = index.reconstruct_batch(cand_indices_arr)  # (n, d)

        # Vectorized hybrid scoring
        hybrid_results = compute_hybrid_scores_batch(
            scorer, query_vec, candidate_matrix,
            query_meta, valid_cand_metas,
            block_weights=block_weights,
        )

        # Vectorized feature attributions
        attribution_results = compute_feature_attributions_batch(
            query_vec, candidate_matrix, feature_names, top_n=5
        )

        # Sort by hybrid score descending, take top-k
        scored = sorted(
            zip(hybrid_results, attribution_results, valid_cand_metas),
            key=lambda x: x[0]["hybrid_score"],
            reverse=True,
        )[:k]

        for rank, (hybrid, attributions, cand_meta) in enumerate(scored, 1):
            results.append({
                "entity_id": cand_meta["entity_id"],
                "entity_name": cand_meta.get("entity_name", "Unknown"),
                "score": hybrid["hybrid_score"],
                "cosine_similarity": hybrid["cos_sim"],
                "rank": rank,
                "metadata": _sanitize_metadata(cand_meta),
                "top_contributing_features": attributions,
            })

    timing_ms = (time.time() - t0) * 1000

    response: dict[str, Any] = {
        "query_id": hashlib.md5(f"{entity_type}:{entity_id}:{time.time()}".encode(), usedforsecurity=False).hexdigest()[:12],
        "query_entity": {
            "entity_id": query_meta["entity_id"],
            "entity_name": query_meta.get("entity_name", "Unknown"),
            "entity_type": entity_type,
        },
        "results": results[:k],
        "total_candidates_searched": fetch_k,
        "filters_applied": filters_applied,
        "timing_ms": timing_ms,
        "index_version": active["version"],
        "cache_hit": False,
    }

    # Cache the result
    _cache[cache_key] = response

    return response


def search_by_vector(
    vector: list[float],
    entity_type: str,
    k: int = 10,
    normalize: bool = True,
) -> dict[str, Any]:
    """Search using a raw embedding vector (programmatic API)."""
    t0 = time.time()
    active = _indices[entity_type]["active"]
    if active is None:
        raise RuntimeError(f"No active {entity_type} index")

    vec = np.array(vector, dtype=np.float32)
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)

    # Validate dimension
    if vec.shape[1] != active["dimension"]:
        raise ValueError(
            f"Vector dimension mismatch: got {vec.shape[1]}, "
            f"expected {active['dimension']}"
        )

    if normalize:
        vec = l2_normalize(vec)

    scores, indices = active["index"].search(vec, k)
    feature_names = active["feature_names"]

    # Collect valid indices
    valid_mask = indices[0] != -1
    cand_indices = indices[0][valid_mask]

    results: list[dict[str, Any]] = []
    if len(cand_indices) > 0:
        cand_indices_arr = np.array(cand_indices, dtype=np.int64)
        # Batch reconstruct
        candidate_matrix = active["index"].reconstruct_batch(cand_indices_arr)

        idx_to_meta = _idx_to_meta[entity_type]  # prebuilt at index install

        # Batch feature attributions
        attribution_results = compute_feature_attributions_batch(
            vec[0], candidate_matrix, feature_names, top_n=5
        )

        for i, faiss_idx in enumerate(cand_indices):
            cand_meta = idx_to_meta.get(int(faiss_idx))
            if cand_meta is None:
                continue
            results.append({
                "entity_id": cand_meta["entity_id"],
                "entity_name": cand_meta.get("entity_name", "Unknown"),
                "score": float(scores[0][valid_mask][i]),
                "cosine_similarity": float(scores[0][valid_mask][i]),
                "rank": len(results) + 1,
                "metadata": _sanitize_metadata(cand_meta),
                "top_contributing_features": attribution_results[i],
            })

    timing_ms = (time.time() - t0) * 1000

    return {
        "query_id": hashlib.md5(f"raw:{time.time()}".encode(), usedforsecurity=False).hexdigest()[:12],
        "query_entity": {"entity_id": "raw_vector", "entity_name": "Raw Vector Query", "entity_type": entity_type},
        "results": results,
        "total_candidates_searched": k,
        "filters_applied": {},
        "timing_ms": timing_ms,
        "index_version": active["version"],
        "cache_hit": False,
    }


# ═══════════════════════════════════════════════════════════════
# INDEX MANAGEMENT
# ═══════════════════════════════════════════════════════════════

def rebuild_player_index(force: bool = False) -> dict[str, Any]:
    """Rebuild the player index. Delegates to full re-initialization."""
    t0 = time.time()
    from ..clustering.player_feature_engineering import run_player_feature_engineering

    if _data_dir is None:
        raise RuntimeError("data_dir not set — call configure() or initialize() first")

    fe_result = run_player_feature_engineering(_data_dir)
    result = build_player_index_from_features(
        fe_result["X_scaled"],
        fe_result["feature_names"],
        fe_result["metadata_df"],
        _output_dir or "./faiss_output",
    )

    return {
        "status": "completed",
        "index_version": result["index_version"],
        "build_time_ms": (time.time() - t0) * 1000,
        "vector_count": result["vector_count"],
        "dimension": result["dimension"],
    }


def get_index_info(entity_type: str) -> Optional[dict[str, Any]]:
    """Return metadata about the currently active index."""
    active = _indices[entity_type]["active"]
    if active is None:
        return None

    return {
        "entity_type": entity_type,
        "index_version": active["version"],
        "index_type": type(active["index"]).__name__,
        "vector_count": active["vector_count"],
        "dimension": active["dimension"],
        "built_at": active["built_at"],
        "memory_bytes": _estimate_index_memory(active["vector_count"], active["dimension"]),
    }


def active_indices() -> dict[str, bool]:
    """Return dict of entity_type → bool (whether active index exists)."""
    return {et: _indices[et]["active"] is not None for et in ENTITY_TYPES}


def cache_size() -> int:
    """Number of entries currently in the query result cache."""
    return len(_cache)


def is_initialized(entity_type: str) -> bool:
    """Whether an active index exists for the given entity type."""
    return _indices[entity_type]["active"] is not None


# ═══════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════

def _passes_filters(meta: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Check if a candidate passes all metadata filters."""
    if filters.get("position") is not None:
        pos = meta.get("primary_position", "")
        if pos != filters["position"]:
            return False

    if filters.get("min_season") is not None:
        season = meta.get("season", 0)
        if season < filters["min_season"]:
            return False

    if filters.get("max_season") is not None:
        season = meta.get("season", 0)
        if season > filters["max_season"]:
            return False

    return True


def _make_cache_key(
    entity_type: str,
    entity_id: str,
    k: int,
    filters: Optional[dict[str, Any]],
    block_weights: Optional[dict[str, float]],
) -> tuple[str, str]:
    """Create a deterministic cache key: (entity_type, md5-of-params).

    The entity type rides alongside the hash so per-entity-type cache
    invalidation on rebuild can match keys structurally. (Previously the key
    was a bare md5 hexdigest and the invalidation substring-matched JSON that
    no longer existed — silently clearing nothing.)
    """
    active = _indices[entity_type]["active"]
    version = active["version"] if active else "none"

    normalized = {
        "et": entity_type,
        "eid": entity_id,
        "k": k,
        "f": filters or {},
        "bw": block_weights or {},
        "v": version,
    }
    key_str = json.dumps(normalized, sort_keys=True)
    digest = hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()
    return (entity_type, digest)


def _clear_cache(entity_type: Optional[str] = None) -> None:
    """Clear the query cache (called after index rebuild)."""
    if entity_type is not None:
        # Keys are (entity_type, digest) tuples — match structurally.
        to_remove = [key for key in _cache if key[0] == entity_type]
        for key in to_remove:
            _cache.pop(key, None)
    else:
        _cache.clear()


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Return a clean copy of metadata with only user-facing fields."""
    keep = {
        "entity_type", "primary_position", "hof",
        "debut_season", "final_season", "total_seasons",
    }
    return {k: v for k, v in meta.items() if k in keep}


def _estimate_index_memory(n_vectors: int, dim: int) -> int:
    """Estimate FAISS index memory in bytes.

    HNSWFlat stores the vector data (4 bytes/float) plus the graph
    structure (~M * 8 bytes per node for link lists).
    """
    vector_bytes = n_vectors * dim * 4
    graph_bytes = n_vectors * 32 * 8  # M=32 links × 8 bytes/link
    return vector_bytes + graph_bytes
