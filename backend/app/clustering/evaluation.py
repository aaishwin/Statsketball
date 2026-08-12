"""
Cluster Evaluation & Validation
===============================
Internal metrics (silhouette, Davies-Bouldin, DBCV) and
domain-specific validation (temporal stability, coach continuity proxy).

The real test: do clusters make basketball sense?
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def evaluate_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    metadata_df: pd.DataFrame,
    algorithm_name: str = "consensus",
) -> dict:
    """
    Compute comprehensive evaluation metrics for a clustering result.

    Skips noise points (label = -1) for metrics that require full assignment.

    Parameters
    ----------
    X : feature matrix (PCA space recommended)
    labels : cluster labels (-1 = noise/unassigned)
    metadata_df : DataFrame with 'season', 'team', 'w_pct'
    algorithm_name : label for reporting

    Returns dict of evaluation metrics.
    """
    results = {"algorithm": algorithm_name}

    # ── Filter to non-noise for metrics ──
    non_noise = labels != -1
    X_clean = X[non_noise]
    labels_clean = labels[non_noise]

    if len(set(labels_clean)) < 2:
        results["warning"] = "Less than 2 valid clusters — skipping metric computation"
        return results

    # ── Silhouette Score (cosine) ──
    try:
        sil = silhouette_score(X_clean, labels_clean, metric="cosine", random_state=42)
        results["silhouette_cosine"] = float(sil)
    except Exception as e:
        results["silhouette_cosine"] = f"ERROR: {e}"

    # ── Davies-Bouldin Index (lower = better) ──
    try:
        db = davies_bouldin_score(X_clean, labels_clean)
        results["davies_bouldin"] = float(db)
    except Exception as e:
        results["davies_bouldin"] = f"ERROR: {e}"

    # ── Cluster sizes ──
    unique, counts = np.unique(labels_clean, return_counts=True)
    results["n_clusters_valid"] = len(unique)
    results["cluster_sizes"] = {int(k): int(v) for k, v in zip(unique, counts)}
    results["min_cluster_size"] = int(counts.min())
    results["max_cluster_size"] = int(counts.max())
    results["size_imbalance_ratio"] = float(counts.max() / counts.min()) if counts.min() > 0 else float("inf")

    n_noise = int(np.sum(labels == -1))
    results["n_noise"] = n_noise
    results["noise_pct"] = n_noise / len(labels)

    # ── Temporal Stability ──
    results["temporal_stability"] = _compute_temporal_stability(labels, metadata_df)

    # ── Championship Style Diversity ──
    results["champion_diversity"] = _compute_champion_diversity(labels, metadata_df)

    # ── Win-pct correlation ──
    results["win_pct_by_cluster"] = _compute_win_pct_by_cluster(labels, metadata_df)

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION: {algorithm_name}")
    print(f"{'='*60}")
    print(f"  Silhouette (cosine):  {results.get('silhouette_cosine', 'N/A')}")
    print(f"  Davies-Bouldin:       {results.get('davies_bouldin', 'N/A')}")
    print(f"  Clusters:             {results['n_clusters_valid']} (+ {n_noise} noise)")
    print(f"  Size range:           {results['min_cluster_size']}–{results['max_cluster_size']}")
    print(f"  Temporal stability:   {results['temporal_stability']['pct_stable']:.1%}")
    print(f"  Champion diversity:   {results['champion_diversity']['n_clusters_with_champions']} clusters contain champs")
    print(f"{'='*60}\n")

    return results


def _compute_temporal_stability(
    labels: np.ndarray,
    metadata_df: pd.DataFrame,
) -> dict:
    """
    Measure: what % of teams stay in the same cluster from one season to the next?

    High stability (>75%) = clusters track persistent identities.
    Low stability (<50%) = clusters are noisy.
    """
    df = metadata_df.copy()
    df["label"] = labels

    # Sort by team and season
    df = df.sort_values(["team", "season"])

    transitions = 0
    stable = 0
    total_pairs = 0

    for team, group in df.groupby("team"):
        seasons = group["season"].values
        labs = group["label"].values
        for i in range(len(labs) - 1):
            if labs[i] == -1 or labs[i + 1] == -1:
                continue  # skip noise transitions
            total_pairs += 1
            if seasons[i + 1] - seasons[i] == 1:  # consecutive seasons
                transitions += 1
                if labs[i] == labs[i + 1]:
                    stable += 1

    pct_stable = stable / transitions if transitions > 0 else 0

    return {
        "consecutive_season_pairs": transitions,
        "stable_transitions": stable,
        "pct_stable": pct_stable,
    }


def _compute_champion_diversity(
    labels: np.ndarray,
    metadata_df: pd.DataFrame,
) -> dict:
    """
    Champions should NOT all cluster together.

    If all champions land in one cluster, the features are leaking "win quality"
    into "style." A healthy pipeline has champions distributed across 3+ clusters.
    """
    df = metadata_df.copy()
    df["label"] = labels

    # Identify champions: top w_pct team or playoff team with highest wins
    # For simplicity, define "champion-level" as w_pct > 0.70
    # (covers ~all actual champions + elite regular season teams)
    elite = df[df["w_pct"] > 0.70]
    elite_labels = elite["label"].values
    elite_labels = elite_labels[elite_labels != -1]

    unique_champ_clusters = len(set(elite_labels))

    return {
        "n_elite_teams": len(elite),
        "n_clusters_with_champions": unique_champ_clusters,
        "champion_cluster_distribution": (
            pd.Series(elite_labels).value_counts().to_dict()
        ),
        "healthy": unique_champ_clusters >= 3,
    }


def _compute_win_pct_by_cluster(
    labels: np.ndarray,
    metadata_df: pd.DataFrame,
) -> dict:
    """Average win percentage per cluster — should show clusters are NOT just
    sorted by quality. Good clusters have varied win rates within them."""
    df = metadata_df.copy()
    df["label"] = labels
    non_noise = df[df["label"] != -1]

    win_stats = {}
    for cluster in sorted(non_noise["label"].unique()):
        cluster_data = non_noise[non_noise["label"] == cluster]
        win_stats[int(cluster)] = {
            "mean_w_pct": float(cluster_data["w_pct"].mean()),
            "std_w_pct": float(cluster_data["w_pct"].std()),
            "min_w_pct": float(cluster_data["w_pct"].min()),
            "max_w_pct": float(cluster_data["w_pct"].max()),
        }

    return win_stats


# ═══════════════════════════════════════════════════════════════
# BOOTSTRAP STABILITY
# ═══════════════════════════════════════════════════════════════

def bootstrap_stability(
    X: np.ndarray,
    clusterer_fn,
    n_iterations: int = 50,
    sample_frac: float = 0.8,
    random_state: int = 42,
) -> dict:
    """
    Measure clustering stability under subsampling.

    Uses Adjusted Rand Index between full-fit labels and subsample-fit labels.

    Returns mean ARI and std — >0.85 is good stability.
    """
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.RandomState(random_state)
    labels_full = clusterer_fn(X)

    aris = []
    for _ in tqdm(range(n_iterations), desc="Bootstrap stability"):
        n_sample = int(len(X) * sample_frac)
        idx = rng.choice(len(X), n_sample, replace=False)
        labels_sample = clusterer_fn(X[idx])

        # Only compare on non-noise points
        mask = (labels_full[idx] != -1) & (labels_sample != -1)
        if mask.sum() > 10:
            ari = adjusted_rand_score(labels_full[idx][mask], labels_sample[mask])
            aris.append(ari)

    return {
        "mean_ari": float(np.mean(aris)) if aris else 0,
        "std_ari": float(np.std(aris)) if aris else 0,
        "n_valid_iterations": len(aris),
        "stable": bool(np.mean(aris) > 0.85) if aris else False,
    }
