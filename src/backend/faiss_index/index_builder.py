"""
FAISS Index Builder for NBA Player Similarity Search
====================================================
Builds FAISS indices from the existing clustering pipeline outputs.

Design decisions (from faiss_similarity_search_design.md):
- IndexFlatIP on L2-normalized vectors (cosine similarity via inner product)
- Stores the full era-adjusted + RobustScaler-transformed vectors (no PCA)
"""

import numpy as np
import faiss
from typing import Optional
import os
import json
import logging
import pickle

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# HNSW INDEX PARAMETERS
# ═══════════════════════════════════════════════════════════════
# HNSW (Hierarchical Navigable Small World) provides approximate nearest
# neighbor search with ~10-50x speedup over IndexFlatIP at >95% recall.
# Tuned for ~22k player vectors.

HNSW_M = 32                    # Number of bi-directional links per node (higher = better recall, more memory)
HNSW_EF_CONSTRUCTION = 200     # ef during build (higher = better graph quality, slower build)
HNSW_EF_SEARCH = 64            # ef during search (higher = better recall, slower query)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize each row to unit length.

    After normalization: cos_sim(u, v) = inner_product(u_norm, v_norm)
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # guard against zero vectors
    return vectors / norms


def build_player_faiss_index(
    X_scaled: np.ndarray,
    feature_names: list[str],
    metadata_df: "pd.DataFrame",
    output_dir: str,
) -> dict:
    """
    Build a FAISS IndexFlatIP for player similarity search.

    Parameters
    ----------
    X_scaled : (n_players, n_features) RobustScaler-transformed feature matrix
    feature_names : list of feature column names
    metadata_df : DataFrame with player_id, player, primary_pos, hof
    output_dir : where to save the index and metadata

    Returns dict with:
        - index_path: path to saved FAISS index
        - metadata_path: path to saved metadata pickle
        - n_vectors, dimension, index_version (SHA of vectors)
    """
    import hashlib

    # L2-normalize for cosine similarity via inner product
    X_norm = l2_normalize(X_scaled.astype(np.float32))

    n_vectors, dim = X_norm.shape

    # Build index — HNSW for approximate nearest neighbor (fast search)
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(X_norm)

    # Compute version hash
    version = hashlib.sha256(X_norm.tobytes()).hexdigest()[:12]
    print(f"[faiss:player] Index built: {n_vectors} vectors × {dim} dims, "
          f"version={version}")

    # Build metadata lookup: row_idx → player metadata dict
    metadata = _build_player_metadata(metadata_df, feature_names, X_scaled)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "faiss_player.index")
    metadata_path = os.path.join(output_dir, "faiss_player_metadata.json")

    faiss.write_index(index, index_path)
    _save_metadata_json(metadata_path, {
        "metadata": metadata,
        "feature_names": feature_names,
        "version": version,
        "n_vectors": n_vectors,
        "dimension": dim,
    })

    print(f"[faiss:player] Saved to {index_path} ({os.path.getsize(index_path) / 1024:.1f} KB)")
    return {
        "index_path": index_path,
        "metadata_path": metadata_path,
        "n_vectors": n_vectors,
        "dimension": dim,
        "index_version": version,
    }


def load_faiss_index(index_path: str) -> faiss.Index:
    """Load a FAISS index from disk."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    return faiss.read_index(index_path)


def _save_metadata_json(path: str, data: dict) -> None:
    """Write index metadata as JSON (no arbitrary code execution on load).

    The metadata payload is plain dicts/lists/strings/numbers — fully JSON-
    serializable — so we avoid pickle's code-execution risk entirely.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _validate_metadata_package(raw: dict) -> dict:
    """Validate a deserialized metadata package has the required shape.

    A corrupted or tampered file fails loudly here instead of executing code.
    """
    required_keys = {"metadata", "feature_names", "version", "n_vectors", "dimension"}
    missing = required_keys - set(raw)
    if missing:
        raise ValueError(f"Metadata package missing required keys: {sorted(missing)}")

    if not isinstance(raw["metadata"], list):
        raise ValueError("Metadata 'metadata' field must be a list")
    if not isinstance(raw["feature_names"], list):
        raise ValueError("Metadata 'feature_names' field must be a list")
    if not all(isinstance(f, str) for f in raw["feature_names"]):
        raise ValueError("Metadata 'feature_names' must be a list of strings")
    if not isinstance(raw["version"], str):
        raise ValueError("Metadata 'version' must be a string")
    if not isinstance(raw["n_vectors"], int) or raw["n_vectors"] < 0:
        raise ValueError("Metadata 'n_vectors' must be a non-negative int")
    if not isinstance(raw["dimension"], int) or raw["dimension"] <= 0:
        raise ValueError("Metadata 'dimension' must be a positive int")
    return raw


def load_metadata(metadata_path: str) -> dict:
    """Load index metadata from disk.

    Reads the JSON format written by ``_save_metadata_json`` and validates
    its shape. Falls back to the legacy ``.pkl`` path (with a warning) only
    for one release to migrate pre-existing artifacts — pickle is then
    deprecated and will be removed.
    """
    # Primary path: JSON (safe — no code execution).
    json_path = metadata_path
    if not json_path.endswith(".json"):
        # Caller passed a legacy .pkl path; try the .json sibling first.
        json_path = metadata_path[:-4] + ".json" if metadata_path.endswith(".pkl") else metadata_path + ".json"

    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)
        return _validate_metadata_package(raw)

    # One-release fallback: legacy pickle artifact. Logs a warning so the
    # operator knows to regenerate artifacts. This path will be removed.
    pkl_path = metadata_path if metadata_path.endswith(".pkl") else (
        metadata_path[:-5] + ".pkl" if metadata_path.endswith(".json") else metadata_path + ".pkl"
    )
    if os.path.exists(pkl_path):
        logger.warning(
            "Loading legacy pickle metadata at %s — regenerate artifacts via "
            "POST /index/rebuild to migrate to the safe JSON format. Pickle "
            "support will be removed in a future release.",
            pkl_path,
        )
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    raise FileNotFoundError(f"Metadata not found: {json_path}")


# ═══════════════════════════════════════════════════════════════
# METADATA BUILDERS
# ═══════════════════════════════════════════════════════════════

def _build_player_metadata(
    metadata_df: "pd.DataFrame",
    feature_names: list[str],
    X_scaled: np.ndarray,
) -> list[dict]:
    """Build list of metadata dicts, one per player row."""
    records = []
    for idx in range(len(metadata_df)):
        row = metadata_df.iloc[idx]
        entity_id = str(row.get("player_id", f"P{idx}"))
        record = {
            "entity_id": entity_id,
            "entity_name": str(row.get("player", entity_id)),
            "entity_type": "player",
            "primary_position": str(row.get("primary_pos", "UNK")),
            "hof": bool(row.get("hof", False)),
            "debut_season": int(row.get("debut_season", 0)),
            "final_season": int(row.get("final_season", 0)),
            "total_seasons": int(row.get("total_seasons", 0)),
            "row_index": idx,
        }
        records.append(record)
    return records
