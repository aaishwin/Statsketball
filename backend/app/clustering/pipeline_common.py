"""
Shared pipeline helpers
=======================
Utilities used by the player clustering pipeline: pickle-based step caching
and per-entity z-score feature profiling.
"""

from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np


# ═══════════════════════════════════════════════════════════════
# STEP CACHING (pickle)
# ═══════════════════════════════════════════════════════════════

def cache_path(cache_dir: str, step_name: str) -> str:
    """Get the cache file path for a pipeline step."""
    return os.path.join(cache_dir, f"{step_name}.pkl")


def load_cache(cache_dir: str, step_name: str, enabled: bool) -> Any | None:
    """Load a cached step result if available and caching is enabled."""
    if not enabled:
        return None
    path = cache_path(cache_dir, step_name)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            print(f"[cache] ✓ Loaded cached {step_name}")
            return data
        except Exception as e:
            print(f"[cache] ⚠ Failed to load cache for {step_name}: {e}")
    return None


def save_cache(cache_dir: str, step_name: str, data: Any, enabled: bool) -> None:
    """Save a step result to the cache."""
    if not enabled:
        return
    path = cache_path(cache_dir, step_name)
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[cache] Saved {step_name} → {path}")
    except Exception as e:
        print(f"[cache] ⚠ Failed to save cache for {step_name}: {e}")


# ═══════════════════════════════════════════════════════════════
# Z-SCORE FEATURE PROFILING
# ═══════════════════════════════════════════════════════════════

def top_zscore_features(
    X_scaled: np.ndarray,
    feature_names: list[str],
    idx: int,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Top-N defining features for one row, as z-scores vs the global mean.

    Returns dicts with 'feature', 'z_score', 'direction' keys — identical
    structure to the former get_player_profile inline logic.
    """
    global_mean = X_scaled.mean(axis=0)
    global_std = X_scaled.std(axis=0)
    row_z = (X_scaled[idx] - global_mean) / (global_std + 1e-8)

    top_indices = np.argsort(-np.abs(row_z))[:top_n]
    top_features: list[dict[str, Any]] = []
    for i in top_indices:
        base = feature_names[i].replace("_era_adj", "")
        direction = "high" if row_z[i] > 0 else "low"
        top_features.append({
            "feature": base,
            "z_score": round(float(row_z[i]), 3),
            "direction": direction,
        })
    return top_features
