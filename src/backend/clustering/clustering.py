"""
Clustering Algorithms for NBA Player Playing Styles
=================================================
Implements HDBSCAN, Agglomerative Hierarchical, and KMeans clustering,
plus an ensemble consensus approach that combines all three.

Design:
- HDBSCAN: primary (handles variable-density clusters, identifies noise)
- Agglomerative: secondary (produces dendrogram, deterministic)
- KMeans: baseline (fast, cluster centers directly interpretable)
- Ensemble: builds co-association matrix for consensus labels
"""

import numpy as np
from typing import Optional

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("[clustering] ⚠️  hdbscan not installed — HDBSCAN unavailable")


# ═══════════════════════════════════════════════════════════════
# HDBSCAN
# ═══════════════════════════════════════════════════════════════

def cluster_hdbscan(
    X_pca: np.ndarray,
    min_cluster_size: int = 20,
    min_samples: int = 3,
    metric: str = "euclidean",
) -> tuple[np.ndarray, object, dict]:
    """
    HDBSCAN clustering on PCA-reduced data.

    Parameters
    ----------
    min_cluster_size : smallest allowed cluster (20 ≈ ~1% of all player-seasons)
    min_samples : conservative core-point definition
    metric : 'euclidean' on PCA space works well

    Returns (labels, model, info_dict).
    Noise points get label = -1.
    """
    if not HAS_HDBSCAN:
        print("[hdbscan] Not available — returning all-noise labels")
        return np.full(len(X_pca), -1), None, {"n_clusters": 0, "n_noise": len(X_pca)}

    # Use at most 10 PCA dims for HDBSCAN (curse of dimensionality)
    n_dims = min(X_pca.shape[1], 10)
    X_for_hdbscan = X_pca[:, :n_dims]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=0.0,
        cluster_selection_method="leaf",  # leaf method finds more fine-grained clusters
        alpha=1.0,
        leaf_size=min_cluster_size,
    )
    labels = clusterer.fit_predict(X_for_hdbscan)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    info = {
        "algorithm": "HDBSCAN",
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": n_noise / len(labels),
        "min_cluster_size": min_cluster_size,
        "probabilities": getattr(clusterer, "probabilities_", None),
    }

    print(f"[hdbscan] {n_clusters} clusters, {n_noise} noise "
          f"({info['noise_pct']:.1%})")
    return labels, clusterer, info


# ═══════════════════════════════════════════════════════════════
# AGGLOMERATIVE HIERARCHICAL
# ═══════════════════════════════════════════════════════════════

def cluster_hierarchical(
    X_pca: np.ndarray,
    n_clusters: int = 8,
    linkage: str = "ward",
) -> tuple[np.ndarray, object, dict]:
    """
    Agglomerative hierarchical clustering.

    Ward linkage + Euclidean on PCA space = reasonable spherical clusters.
    Produces a dendrogram-friendly output for nested style relationships.
    """
    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="euclidean",
        linkage=linkage,
    )
    labels = agg.fit_predict(X_pca)

    info = {
        "algorithm": "Agglomerative",
        "n_clusters": n_clusters,
        "linkage": linkage,
        "n_noise": 0,
        "noise_pct": 0.0,
    }

    print(f"[hierarchical] {n_clusters} clusters (linkage={linkage})")
    return labels, agg, info


# ═══════════════════════════════════════════════════════════════
# KMEANS
# ═══════════════════════════════════════════════════════════════

def cluster_kmeans(
    X_pca: np.ndarray,
    n_clusters: int = 8,
    random_state: int = 42,
) -> tuple[np.ndarray, object, dict]:
    """
    KMeans clustering baseline.

    Cluster centers are directly interpretable as "average player" per style.
    """
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
        max_iter=500,
    )
    labels = km.fit_predict(X_pca)

    info = {
        "algorithm": "KMeans",
        "n_clusters": n_clusters,
        "inertia": float(km.inertia_),
        "n_noise": 0,
        "noise_pct": 0.0,
    }

    print(f"[kmeans] {n_clusters} clusters, inertia={km.inertia_:.1f}")
    return labels, km, info


# ═══════════════════════════════════════════════════════════════
# ENSEMBLE CONSENSUS
# ═══════════════════════════════════════════════════════════════

def ensemble_consensus(
    label_sets: list[tuple[np.ndarray, str]],
    X_pca: np.ndarray,
    min_consensus: float = 0.5,
    n_final_clusters: int = 8,
) -> tuple[np.ndarray, dict]:
    """
    Build a co-association matrix from multiple clustering results and
    produce consensus labels via hierarchical clustering on the matrix.

    Parameters
    ----------
    label_sets : list of (labels_array, algorithm_name) tuples
    X_pca : PCA-reduced data (used to compute silhouette)
    min_consensus : minimum fraction of algorithms agreeing to form a cluster
    n_final_clusters : target number of consensus clusters

    Returns
    -------
    consensus_labels : final cluster labels (0-indexed), -1 for low-consensus
    info : dict with per-algorithm stats and consensus quality
    """
    n_samples = len(X_pca)
    n_algorithms = len(label_sets)

    if n_algorithms == 0:
        return np.zeros(n_samples, dtype=int), {"error": "No algorithms provided"}

    # ── Build co-association matrix ──
    co_assoc = np.zeros((n_samples, n_samples))

    for labels, algo_name in tqdm(label_sets, desc="Building co-association"):
        # Remap noise (-1) to a unique negative value for this algorithm
        cleaned = labels.copy()
        noise_mask = cleaned == -1
        cleaned[noise_mask] = -1  # keep as noise

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if cleaned[i] != -1 and cleaned[j] != -1 and cleaned[i] == cleaned[j]:
                    co_assoc[i, j] += 1
                    co_assoc[j, i] += 1

    co_assoc /= n_algorithms  # normalize to [0, 1]

    # ── Cluster the co-association matrix ──
    # Use distance = 1 - co_assoc
    distance_matrix = 1.0 - co_assoc
    np.fill_diagonal(distance_matrix, 0)

    consensus_clusterer = AgglomerativeClustering(
        n_clusters=n_final_clusters,
        metric="precomputed",
        linkage="average",
    )
    consensus_labels = consensus_clusterer.fit_predict(distance_matrix)

    # ── Identify low-consensus players ──
    # A player has low consensus if its average co-association with its own
    # cluster members is below the threshold
    final_labels = consensus_labels.copy()
    for i in range(n_samples):
        cluster_mask = consensus_labels == consensus_labels[i]
        cluster_mask[i] = False
        if cluster_mask.sum() > 0:
            avg_agreement = co_assoc[i][cluster_mask].mean()
            if avg_agreement < min_consensus:
                final_labels[i] = -1  # mark as hybrid/transitional

    # ── Compute consensus quality ──
    quality_scores = {}
    for labels, algo_name in label_sets:
        # ARI between this algorithm and consensus (for non-noise points)
        mask = (labels != -1) & (final_labels != -1)
        if mask.sum() > 10:
            from sklearn.metrics import adjusted_rand_score
            ari = adjusted_rand_score(labels[mask], final_labels[mask])
            quality_scores[f"{algo_name}_ari"] = float(ari)

    n_noise_final = int(np.sum(final_labels == -1))

    info = {
        "algorithm": "Ensemble Consensus",
        "n_algorithms": n_algorithms,
        "n_final_clusters": len(set(final_labels)) - (1 if -1 in final_labels else 0),
        "n_hybrid": n_noise_final,
        "hybrid_pct": n_noise_final / n_samples,
        "algorithm_quality": quality_scores,
        "co_association_mean": float(co_assoc.mean()),
    }

    print(f"[ensemble] {info['n_final_clusters']} consensus clusters, "
          f"{n_noise_final} hybrid players ({info['hybrid_pct']:.1%})")
    print(f"[ensemble] Algorithm agreement with consensus: "
          + ", ".join(f"{k}={v:.3f}" for k, v in quality_scores.items()))

    return final_labels, info


# ═══════════════════════════════════════════════════════════════
# FULL CLUSTERING PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_clustering(
    X_pca: np.ndarray,
    n_clusters: int = 8,
    use_hdbscan: bool = True,
    min_cluster_size: int = 25,
) -> dict:
    """
    Run all clustering algorithms and produce ensemble consensus labels.

    Returns dict with labels from each algorithm + consensus labels + info.
    """
    label_sets: list[tuple[np.ndarray, str]] = []
    all_info: dict = {}

    # ── HDBSCAN ──
    if use_hdbscan:
        h_labels, h_model, h_info = cluster_hdbscan(
            X_pca, min_cluster_size=min_cluster_size
        )
        all_info["hdbscan"] = h_info
        # Only include HDBSCAN if it found meaningful clusters (>1 cluster, <50% noise)
        if h_info["n_clusters"] >= 2 and h_info["noise_pct"] < 0.50:
            label_sets.append((h_labels, "HDBSCAN"))
            print("[clustering] HDBSCAN included in ensemble "
                  f"({h_info['n_clusters']} clusters, {h_info['noise_pct']:.1%} noise)")
        else:
            print("[clustering] HDBSCAN excluded from ensemble "
                  f"({h_info['n_clusters']} clusters, {h_info['noise_pct']:.1%} noise) — "
                  "too few clusters or too much noise for reliable consensus")

    # ── Hierarchical ──
    hier_labels, hier_model, hier_info = cluster_hierarchical(
        X_pca, n_clusters=n_clusters
    )
    all_info["hierarchical"] = hier_info
    label_sets.append((hier_labels, "Hierarchical"))

    # ── KMeans ──
    km_labels, km_model, km_info = cluster_kmeans(X_pca, n_clusters=n_clusters)
    all_info["kmeans"] = km_info
    label_sets.append((km_labels, "KMeans"))

    # ── Ensemble Consensus ──
    consensus_labels, consensus_info = run_ensemble_consensus = ensemble_consensus(
        label_sets, X_pca, n_final_clusters=n_clusters
    )
    all_info["consensus"] = consensus_info

    return {
        "labels_hdbscan": h_labels if use_hdbscan else None,
        "labels_hierarchical": hier_labels,
        "labels_kmeans": km_labels,
        "labels_consensus": consensus_labels,
        "info": all_info,
        "models": {
            "hdbscan": h_model if use_hdbscan else None,
            "hierarchical": hier_model,
            "kmeans": km_model,
        },
    }
