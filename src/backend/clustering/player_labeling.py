"""
Player Archetype Labeling
=========================
Auto-generates human-readable player archetype names from cluster feature profiles.
Maps 120+ (feature, direction) pairs to basketball-specific player descriptors.

Produces labels like:
  "3&D Wing • Floor General • Sharpshooter"
  "Rim Protector • Glass Cleaner • Low-Usage Big"
  "Shot-Creating Guard • High-Volume Scorer • Perimeter Stopper"
"""

import numpy as np
import pandas as pd
from typing import Optional

from ..constants import PLAYER_FEATURE_LABELS as FEATURE_LABELS, PLAYER_LABEL_PRIORITY as LABEL_PRIORITY


# ═══════════════════════════════════════════════════════════════
# FEATURE → PLAYER ARCHETYPE LABEL MAPPING
# ═══════════════════════════════════════════════════════════════
# All label mappings and priority lists imported from ..constants:
#   FEATURE_LABELS (PLAYER_FEATURE_LABELS), LABEL_PRIORITY (PLAYER_LABEL_PRIORITY)


# ═══════════════════════════════════════════════════════════════
# CLUSTER NAME GENERATION
# ═══════════════════════════════════════════════════════════════

def _get_base_feature_name(era_adj_col: str) -> str:
    return era_adj_col.replace("_era_adj", "")


def _get_direction(z_score: float) -> str:
    return "high" if z_score > 0 else "low"


def generate_player_cluster_names(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    metadata_df: pd.DataFrame,
    top_n_features: int = 4,
) -> dict[int, dict]:
    """
    Generate human-readable archetype names and profiles for each player cluster.

    Returns dict: cluster_id → {name, top_features, feature_z_scores, exemplar_players, position_breakdown}
    """
    unique_labels = sorted(set(labels) - {-1})
    global_mean = X_scaled.mean(axis=0)
    global_std = X_scaled.std(axis=0)

    cluster_profiles = {}

    for cl in unique_labels:
        mask = labels == cl
        cluster_data = X_scaled[mask]
        cluster_mean = cluster_data.mean(axis=0)

        # Z-score of cluster mean vs. global mean
        z_scores = (cluster_mean - global_mean) / (global_std + 1e-8)

        # Build (feature_base, z_score) pairs
        feature_zs = {}
        for i, fname in enumerate(feature_names):
            base = _get_base_feature_name(fname)
            feature_zs[base] = float(z_scores[i])

        # Sort by absolute Z-score, prioritizing interpretable features
        ranked = sorted(
            feature_zs.items(),
            key=lambda x: (
                -abs(x[1]),
                LABEL_PRIORITY.index(x[0]) if x[0] in LABEL_PRIORITY else 999,
            ),
        )

        # ── Build cluster name (deduplicate descriptors) ──
        descriptors = []
        seen_descriptors = set()
        detailed_features = []
        for feat, z in ranked[:top_n_features]:
            direction = _get_direction(z)
            label = FEATURE_LABELS.get((feat, direction))
            display_label = label or f"{feat}-{direction}"
            if label and label not in seen_descriptors:
                descriptors.append(label)
                seen_descriptors.add(label)
            elif label is None and display_label not in seen_descriptors:
                descriptors.append(display_label)
                seen_descriptors.add(display_label)
            detailed_features.append({
                "feature": feat,
                "z_score": round(z, 3),
                "direction": direction,
                "label": display_label,
            })

        cluster_name = " • ".join(descriptors) if descriptors else f"Archetype-{cl}"

        # ── Find exemplar players (closest to cluster centroid) ──
        centroid = cluster_mean
        distances = np.linalg.norm(cluster_data - centroid, axis=1)
        cluster_indices = np.where(mask)[0]
        top_exemplar_idx = cluster_indices[np.argsort(distances)[:8]]

        exemplars = []
        for idx in top_exemplar_idx:
            row = metadata_df.iloc[idx]
            exemplars.append(str(row["player"]))

        # ── Position breakdown ──
        cluster_meta = metadata_df[mask]
        pos_counts = cluster_meta["primary_pos"].value_counts().to_dict()

        # ── HOF rate ──
        hof_count = int(cluster_meta["hof"].sum())
        hof_rate = hof_count / len(cluster_meta) if len(cluster_meta) > 0 else 0

        # ── Avg height / weight ──
        avg_height = float(cluster_meta["ht_in_in"].mean()) if "ht_in_in" in cluster_meta.columns else 0
        avg_weight = float(cluster_meta["wt"].mean()) if "wt" in cluster_meta.columns else 0

        cluster_profiles[int(cl)] = {
            "name": cluster_name,
            "size": int(mask.sum()),
            "top_features": detailed_features,
            "all_feature_z_scores": feature_zs,
            "exemplar_players": exemplars,
            "position_breakdown": pos_counts,
            "hof_rate": round(hof_rate, 3),
            "hof_count": hof_count,
            "avg_height": round(avg_height, 1),
            "avg_weight": round(avg_weight, 1),
        }

        print(f"[label] Archetype {cl}: {cluster_name} "
              f"(n={mask.sum()}, HOF%={hof_rate:.1%}, "
              f"exemplars: {', '.join(exemplars[:4])})")

    return cluster_profiles


def compare_player_archetypes(
    cluster_profiles: dict[int, dict],
    cluster_a: int,
    cluster_b: int,
) -> dict:
    """Side-by-side comparison of two player archetypes."""
    if cluster_a not in cluster_profiles or cluster_b not in cluster_profiles:
        return {"error": "One or both clusters not found"}

    prof_a = cluster_profiles[cluster_a]
    prof_b = cluster_profiles[cluster_b]

    zs_a = prof_a["all_feature_z_scores"]
    zs_b = prof_b["all_feature_z_scores"]

    divergences = []
    for feat in zs_a:
        if feat in zs_b:
            diff = abs(zs_a[feat] - zs_b[feat])
            divergences.append((feat, diff, zs_a[feat], zs_b[feat]))

    divergences.sort(key=lambda x: -x[1])

    return {
        "archetype_a": {"id": cluster_a, "name": prof_a["name"], "size": prof_a["size"]},
        "archetype_b": {"id": cluster_b, "name": prof_b["name"], "size": prof_b["size"]},
        "key_differences": [
            {
                "feature": feat,
                "divergence": round(diff, 3),
                f"archetype_{cluster_a}_z": round(za, 3),
                f"archetype_{cluster_b}_z": round(zb, 3),
            }
            for feat, diff, za, zb in divergences[:8]
        ],
    }
