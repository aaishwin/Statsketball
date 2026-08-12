"""
Dimensionality Reduction for Player Embeddings
==============================================
Two-stage approach: PCA (denoising) → UMAP (visualization).

PCA retains 90% variance and produces a stable embedding for similarity search.
UMAP projects to 2D for interactive visualization while preserving global structure.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Optional

# UMAP is optional — the pipeline degrades gracefully without it
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def reduce_pca(
    X: np.ndarray,
    variance_threshold: float = 0.90,
    random_state: int = 42,
) -> tuple[np.ndarray, PCA, dict]:
    """
    PCA dimensionality reduction.

    Parameters
    ----------
    X : shape (n_samples, n_features)
    variance_threshold : float, cumulative variance to retain
    random_state : int

    Returns
    -------
    X_pca : shape (n_samples, n_components)
    pca : fitted PCA model
    info : dict with explained_variance, n_components, loadings
    """
    pca = PCA(n_components=variance_threshold, random_state=random_state)
    X_pca = pca.fit_transform(X)

    info = {
        "n_components": X_pca.shape[1],
        "n_original_features": X.shape[1],
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": float(np.sum(pca.explained_variance_ratio_)),
        "components": pca.components_.tolist(),
    }

    print(f"[pca] {X.shape[1]} features → {X_pca.shape[1]} components "
          f"({info['cumulative_variance']:.1%} variance retained)")
    return X_pca, pca, info


def reduce_umap(
    X_pca: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> tuple[Optional[np.ndarray], Optional[object]]:
    """
    UMAP projection for 2D visualization.

    Falls back gracefully if umap-learn is not installed.

    Parameters
    ----------
    X_pca : PCA-reduced data
    n_components : target dimension (usually 2)
    n_neighbors : local neighborhood size (15 balances local/global)
    min_dist : minimum distance between points in embedding
    metric : distance metric ('cosine' = style direction, not magnitude)
    random_state : int

    Returns
    -------
    X_umap : (n_samples, 2) or None if UMAP unavailable
    reducer : UMAP model or None
    """
    if not HAS_UMAP:
        print("[umap] ⚠️  umap-learn not installed — using PCA first 2 components instead")
        return X_pca[:, :2], None

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=-1,
    )
    X_umap = reducer.fit_transform(X_pca)

    print(f"[umap] {X_pca.shape[1]}d PCA → {X_umap.shape[1]}d UMAP "
          f"(n_neighbors={n_neighbors}, metric={metric})")
    return X_umap, reducer


def run_dimensionality_reduction(
    X_scaled: np.ndarray,
    pca_variance: float = 0.90,
    umap_n_neighbors: int = 15,
) -> dict:
    """
    Full dimensionality reduction pipeline.

    Returns dict with X_pca, X_umap, pca_model, umap_model, and info.
    """
    # Stage 1: PCA
    X_pca, pca_model, pca_info = reduce_pca(X_scaled, variance_threshold=pca_variance)

    # Stage 2: UMAP
    X_umap, umap_model = reduce_umap(X_pca, n_neighbors=umap_n_neighbors)

    return {
        "X_pca": X_pca,
        "X_umap": X_umap,
        "pca_model": pca_model,
        "umap_model": umap_model,
        "pca_info": pca_info,
    }
