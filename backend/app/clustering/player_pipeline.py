"""
Player Archetype Clustering Pipeline (functional)
==================================================
Full pipeline: 8 CSVs → merge → filter 5+ seasons → career means →
7-block features → era adjust → PCA → UMAP → cluster → evaluate → label → visualize.

Reuses clustering, similarity, evaluation, and dimensionality modules from
the team pipeline; shared caching/profile helpers live in pipeline_common.

Usage:
    from app.clustering.player_pipeline import run_player_pipeline, get_similar_players

    result = run_player_pipeline(data_dir="data/nba-aba-baa-stats/versions/56")
    similar = get_similar_players(result, "LeBron James")
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .player_feature_engineering import run_player_feature_engineering
from .dimensionality import run_dimensionality_reduction
from .clustering import run_clustering
from .similarity import compute_similarity_matrix, find_similar_teams as find_similar
from .evaluation import evaluate_clusters
from .player_labeling import generate_player_cluster_names
from .player_visualization import run_all_player_visualizations
from .pipeline_common import load_cache, save_cache, top_zscore_features
from ..validation import validate_output_dir, generate_data_quality_report


@dataclass(slots=True, frozen=True)
class PlayerPipelineResult:
    """All artifacts produced by one player pipeline run.

    Immutable container passed explicitly to the query functions below.
    Each row = one player (career means over 5+ seasons).
    """

    player_df: pd.DataFrame
    metadata_df: pd.DataFrame
    X_scaled: np.ndarray
    X_pca: np.ndarray
    X_umap: np.ndarray
    labels: np.ndarray
    sim_matrix: np.ndarray
    cluster_profiles: dict[int, dict[str, Any]]
    eval_results: dict[str, Any]
    feature_names: list[str]
    scaler: Any
    pca_model: Any
    umap_model: Any
    figures: dict[str, Any]


def _adapt_metadata_for_eval(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Rename player metadata columns to the team-oriented names expected by
    the shared evaluation/similarity modules.

    The shared modules expect 'team', 'season', 'w_pct'. We map
    player→team, debut_season→season, and synthesize a
    binary w_pct proxy from HOF status. The misleading w_pct-derived metrics
    are stripped from eval results afterwards (see run_player_pipeline).
    """
    adapted = metadata_df.rename(
        columns={"player": "team", "debut_season": "season"}
    ).copy()
    adapted["w_pct"] = adapted["hof"].astype(float)
    return adapted


def run_player_pipeline(
    data_dir: str,
    output_dir: str = "./output_players",
    n_clusters: int = 12,
    pca_variance: float = 0.90,
    min_seasons: int = 5,
    block_weights: Optional[dict[str, float]] = None,
    use_hdbscan: bool = True,
    min_cluster_size: int = 25,
    show_plots: bool = True,
    enable_cache: bool = True,
) -> PlayerPipelineResult:
    """Execute the full player clustering pipeline."""
    print("=" * 70)
    print("  NBA PLAYER ARCHETYPE CLUSTERING PIPELINE")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(output_dir, ".cache")
    if enable_cache:
        os.makedirs(cache_dir, exist_ok=True)

    # ── Pre-flight: validate output directory ──
    out_check = validate_output_dir(output_dir)
    if not out_check.is_valid:
        raise PermissionError(f"Output directory not writable: {out_check.error}")

    # ── Step 1: Feature Engineering ──
    print("\n" + "─" * 50)
    print("STEP 1: PLAYER FEATURE ENGINEERING")
    print("─" * 50)
    fe_result = load_cache(cache_dir, "feature_engineering", enable_cache)
    if fe_result is None:
        fe_result = run_player_feature_engineering(
            data_dir,
            min_seasons=min_seasons,
            block_weights=block_weights,
        )
        save_cache(cache_dir, "feature_engineering", fe_result, enable_cache)
    player_df = fe_result["player_df"]
    X_scaled = fe_result["X_scaled"]
    scaler = fe_result["scaler"]
    feature_names = fe_result["feature_names"]
    metadata_df = fe_result["metadata_df"]

    # ── Data Quality Report ──
    quality_report = generate_data_quality_report(
        X_scaled, feature_names,
        dataset_label="player",
    )
    quality_path = os.path.join(output_dir, "data_quality.json")
    with open(quality_path, "w") as f:
        f.write(quality_report.model_dump_json(indent=2))
    print(f"[quality] Data quality report → {quality_path}")

    # ── Step 2: Dimensionality Reduction ──
    print("\n" + "─" * 50)
    print("STEP 2: DIMENSIONALITY REDUCTION")
    print("─" * 50)
    dim_result = run_dimensionality_reduction(X_scaled, pca_variance=pca_variance)
    X_pca = dim_result["X_pca"]
    X_umap = dim_result["X_umap"]
    pca_model = dim_result["pca_model"]
    umap_model = dim_result["umap_model"]

    # ── Step 3: Clustering ──
    print("\n" + "─" * 50)
    print("STEP 3: CLUSTERING")
    print("─" * 50)
    cluster_result = run_clustering(
        X_pca,
        n_clusters=n_clusters,
        use_hdbscan=use_hdbscan,
        min_cluster_size=min_cluster_size,
    )
    labels = cluster_result["labels_consensus"]

    # ── Step 4: Similarity ──
    print("\n" + "─" * 50)
    print("STEP 4: SIMILARITY COMPUTATION")
    print("─" * 50)
    sim_matrix = compute_similarity_matrix(X_pca)

    # ── Step 5: Labeling ──
    print("\n" + "─" * 50)
    print("STEP 5: ARCHETYPE LABELING")
    print("─" * 50)
    cluster_profiles = generate_player_cluster_names(
        X_scaled, labels, feature_names, metadata_df
    )

    # ── Step 6: Evaluation ──
    print("\n" + "─" * 50)
    print("STEP 6: EVALUATION")
    print("─" * 50)
    # Adapt metadata for evaluation module (expects 'w_pct', 'season', 'team')
    eval_meta = _adapt_metadata_for_eval(metadata_df)
    eval_results = evaluate_clusters(
        X_pca, labels, eval_meta, algorithm_name="Player Consensus"
    )
    # Patch: override champion_diversity with HOF diversity
    eval_results["hof_diversity"] = _compute_hof_diversity(metadata_df, labels)
    # Remove misleading w_pct-based champion metric
    eval_results.pop("champion_diversity", None)
    eval_results.pop("win_pct_by_cluster", None)

    # ── Step 7: Visualization ──
    print("\n" + "─" * 50)
    print("STEP 7: VISUALIZATION")
    print("─" * 50)
    figures = run_all_player_visualizations(
        X_umap, labels,
        metadata_df, cluster_profiles,
        output_dir=output_dir, show=show_plots,
    )

    result = PlayerPipelineResult(
        player_df=player_df,
        metadata_df=metadata_df,
        X_scaled=X_scaled,
        X_pca=X_pca,
        X_umap=X_umap,
        labels=labels,
        sim_matrix=sim_matrix,
        cluster_profiles=cluster_profiles,
        eval_results=eval_results,
        feature_names=feature_names,
        scaler=scaler,
        pca_model=pca_model,
        umap_model=umap_model,
        figures=figures,
    )

    # ── Save artifacts ──
    _save_artifacts(result, output_dir)
    _print_summary(result)

    return result


# ═══════════════════════════════════════════════════════════════
# QUERY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def find_player_idx(result: PlayerPipelineResult, player_name: str) -> Optional[int]:
    """Find the row index of a player by (case-insensitive) name."""
    mask = result.metadata_df["player"].str.lower() == player_name.lower()
    indices = np.where(mask)[0]
    return int(indices[0]) if len(indices) > 0 else None


def get_similar_players(
    result: PlayerPipelineResult, player_name: str, top_k: int = 10
) -> pd.DataFrame:
    """Find players most stylistically similar to the given player."""
    idx = find_player_idx(result, player_name)
    if idx is None:
        raise ValueError(f"Player '{player_name}' not found")
    # Adapt metadata to match similarity module expectations
    sim_meta = _adapt_metadata_for_eval(result.metadata_df)
    return find_similar(idx, result.sim_matrix, sim_meta, top_k=top_k)


def get_player_profile(result: PlayerPipelineResult, player_name: str) -> dict[str, Any]:
    """Get archetype assignment and feature profile for a player."""
    idx = find_player_idx(result, player_name)
    if idx is None:
        raise ValueError(f"Player '{player_name}' not found")

    label = int(result.labels[idx])
    cluster_info = result.cluster_profiles.get(label, {"name": "Hybrid/Transitional"})

    from .player_labeling import FEATURE_LABELS
    top_features = top_zscore_features(result.X_scaled, result.feature_names, idx)
    for f in top_features:
        f["label"] = FEATURE_LABELS.get(
            (f["feature"], f["direction"]), f"{f['feature']}-{f['direction']}"
        )

    row = result.metadata_df.iloc[idx]
    return {
        "player": str(row["player"]),
        "position": str(row["primary_pos"]),
        "height": float(row["ht_in_in"]) if pd.notna(row.get("ht_in_in")) else None,
        "weight": float(row["wt"]) if pd.notna(row.get("wt")) else None,
        "hof": bool(row["hof"]),
        "debut_season": int(row["debut_season"]),
        "total_seasons": int(row["total_seasons"]),
        "archetype_id": label if label != -1 else None,
        "archetype_name": cluster_info.get("name", "Hybrid/Transitional"),
        "umap_2d": [float(result.X_umap[idx, 0]), float(result.X_umap[idx, 1])],
        "top_features": top_features,
    }


def compare_players(
    result: PlayerPipelineResult, player_a: str, player_b: str
) -> dict[str, Any]:
    """Side-by-side archetype comparison of two players."""
    idx_a = find_player_idx(result, player_a)
    idx_b = find_player_idx(result, player_b)
    if idx_a is None or idx_b is None:
        raise ValueError("One or both players not found")

    sim = float(result.sim_matrix[idx_a, idx_b])
    return {
        "player_a": get_player_profile(result, player_a),
        "player_b": get_player_profile(result, player_b),
        "cosine_similarity": sim,
        "same_archetype": int(result.labels[idx_a]) == int(result.labels[idx_b]),
    }


# ═══════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════

def _save_artifacts(result: PlayerPipelineResult, output_dir: str) -> None:
    """Save pipeline artifacts to disk."""
    # Cluster profiles as JSON
    with open(os.path.join(output_dir, "cluster_profiles.json"), "w") as f:
        json.dump(result.cluster_profiles, f, indent=2, default=str)
    print(f"[save] Cluster profiles → {output_dir}/cluster_profiles.json")

    # Evaluation
    with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
        json.dump(result.eval_results, f, indent=2, default=str)
    print(f"[save] Evaluation → {output_dir}/evaluation.json")

    # Export with labels
    export_df = result.metadata_df.copy()
    export_df["archetype_label"] = result.labels
    export_df["umap_x"] = result.X_umap[:, 0]
    export_df["umap_y"] = result.X_umap[:, 1]
    for i in range(min(5, result.X_pca.shape[1])):
        export_df[f"pca_{i+1}"] = result.X_pca[:, i]
    export_df.to_csv(
        os.path.join(output_dir, "players_with_archetypes.csv"),
        index=False,
    )
    print(f"[save] Player export → {output_dir}/players_with_archetypes.csv")

    # Similarity matrix
    np.save(
        os.path.join(output_dir, "similarity_matrix.npy"),
        result.sim_matrix,
    )
    print(f"[save] Similarity matrix → {output_dir}/similarity_matrix.npy")


def _compute_hof_diversity(metadata_df: pd.DataFrame, labels: np.ndarray) -> dict[str, Any]:
    """Measure how Hall of Famers are distributed across archetypes."""
    df = metadata_df.copy()
    df["label"] = labels
    hof_players = df[df["hof"] == True]
    hof_labels = np.asarray(hof_players["label"].to_numpy(), dtype=np.int64)
    hof_labels = hof_labels[hof_labels != -1]
    unique_hof_clusters = len(set(hof_labels.tolist()))

    return {
        "n_hof_players": len(hof_players),
        "n_clusters_with_hof": unique_hof_clusters,
        "hof_distribution": (
            pd.Series(hof_labels).value_counts().to_dict()
        ),
        "healthy": unique_hof_clusters >= 3,
    }


def _print_summary(result: PlayerPipelineResult) -> None:
    """Print a readable summary of player archetypes."""
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE — PLAYER ARCHETYPE SUMMARY")
    print("=" * 70)

    for cl in sorted(result.cluster_profiles.keys()):
        prof = result.cluster_profiles[cl]
        top_pos = sorted(prof["position_breakdown"].items(),
                         key=lambda x: -x[1])[:3]
        pos_str = "/".join(p for p, _ in top_pos)
        print(f"\n  ▸ Archetype {cl}: {prof['name']}")
        print(f"    Players: {prof['size']} | "
              f"HOF: {prof['hof_rate']:.1%} ({prof['hof_count']}) | "
              f"Pos: {pos_str}")
        print(f"    Avg Ht: {prof['avg_height']}\" | Avg Wt: {prof['avg_weight']} lbs")
        print(f"    Exemplars: {', '.join(prof['exemplar_players'][:5])}")
        top_feats = prof["top_features"][:3]
        feat_str = " | ".join(
            f"{f['label']} ({f['z_score']:+.2f}σ)" for f in top_feats
        )
        print(f"    Defining: {feat_str}")

    n_hybrid = int(np.sum(result.labels == -1))
    print(f"\n  ◇ Hybrid / Transitional: {n_hybrid} players "
          f"({n_hybrid/len(result.labels):.1%})")
    print("=" * 70)
