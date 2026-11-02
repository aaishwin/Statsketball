"""
Similarity Search for NBA Playing Styles
=========================================
Cosine similarity on PCA space for finding stylistically similar entities.
Emphasizes playing-style direction over quality/magnitude.

Shared by the player clustering pipeline (which adapts player metadata to the
``team``/``season``/``w_pct`` column shape this module expects).
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def compute_similarity_matrix(
    X_pca: np.ndarray,
    large_n_threshold: int = 10_000,
    sparse_n_neighbors: int = 50,
) -> np.ndarray:
    """
    Compute cosine similarity matrix on PCA space.

    For datasets with > large_n_threshold samples, automatically switches
    from full O(n²) cosine_similarity to a sparse KNN graph using
    sklearn.neighbors.NearestNeighbors to avoid memory blowup.

    Returns (n_samples, n_samples) matrix where entry [i, j] = cos_sim(i, j).
    """
    n_samples = X_pca.shape[0]

    if n_samples > large_n_threshold:
        print(f"[similarity] ⚠ {n_samples} samples > {large_n_threshold} threshold — "
              f"switching to sparse KNN ({sparse_n_neighbors} neighbors)")
        return _compute_sparse_similarity(X_pca, sparse_n_neighbors)

    sim_matrix = cosine_similarity(X_pca)
    np.fill_diagonal(sim_matrix, 0)  # exclude self
    print(f"[similarity] {sim_matrix.shape} cosine similarity matrix computed")
    return sim_matrix


def _compute_sparse_similarity(
    X_pca: np.ndarray,
    n_neighbors: int = 50,
) -> np.ndarray:
    """
    Build a sparse CSR similarity matrix using NearestNeighbors.
    Only the top n_neighbors similarities are kept per sample.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", n_jobs=-1)
    nn.fit(X_pca)
    distances, indices = nn.kneighbors(X_pca)

    # Convert cosine distance to cosine similarity: sim = 1 - distance
    similarities = 1.0 - distances

    n_samples = X_pca.shape[0]
    row_indices = np.repeat(np.arange(n_samples), n_neighbors)
    col_indices = indices.ravel()
    values = similarities.ravel()

    # Zero out self-similarity
    self_mask = row_indices == col_indices
    values[self_mask] = 0.0

    sim_sparse = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(n_samples, n_samples),
    )

    # Convert to dense for downstream compatibility (only stored values are non-zero)
    # For very large n, keep sparse; for moderate, convert to dense
    if n_samples <= 20_000:
        result = sim_sparse.toarray()
    else:
        result = sim_sparse  # type: ignore — downstream consumers may need .toarray()

    nnz = sim_sparse.nnz
    print(f"[similarity] Sparse KNN: {n_samples}×{n_samples} matrix, "
          f"{nnz} non-zero entries ({nnz / (n_samples * n_samples):.2%} density)")
    return result # type: ignore


def find_similar_teams(
    query_idx: int,
    sim_matrix: np.ndarray,
    metadata_df: pd.DataFrame,
    top_k: int = 10,
    exclude_same_franchise: bool = False,
) -> pd.DataFrame:
    """
    Find the top-k most stylistically similar entities.

    Parameters
    ----------
    query_idx : index of the query entity in the matrix
    sim_matrix : precomputed cosine similarity matrix
    metadata_df : DataFrame with 'team', 'season', 'w_pct' columns
    top_k : number of results
    exclude_same_franchise : exclude the same franchise (different seasons)

    Returns DataFrame with columns: team, season, w_pct, cosine_similarity, rank
    """
    sims = sim_matrix[query_idx].copy()

    # Build mask for exclusions
    mask = np.ones(len(sims), dtype=bool)
    mask[query_idx] = False  # exclude self

    query_team = metadata_df.iloc[query_idx]["team"]

    if exclude_same_franchise:
        same_team_mask = (metadata_df["team"] == query_team).values
        mask = mask & ~same_team_mask

    # Get top-k indices
    valid_indices = np.where(mask)[0]
    valid_sims = sims[valid_indices]
    top_k_local = min(top_k, len(valid_sims))
    top_local_indices = np.argpartition(valid_sims, -top_k_local)[-top_k_local:]
    top_local_indices = top_local_indices[np.argsort(valid_sims[top_local_indices])[::-1]]
    top_global_indices = valid_indices[top_local_indices]

    # Build result DataFrame
    results = metadata_df.iloc[top_global_indices][
        ["team", "season", "w_pct"]
    ].copy()
    results["cosine_similarity"] = sims[top_global_indices]
    results["rank"] = range(1, len(results) + 1)
    results = results.reset_index(drop=True)

    return results



