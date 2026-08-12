"""
Player Career Archetype Feature Engineering
============================================
Merges 8 player-level CSVs, filters to 5+ season careers, computes career
means, era-adjusts, and builds 7 semantic feature blocks for clustering
NBA players by playing style (not quality).

Key design decisions:
- One row per player (career means over all seasons)
- Era adjustment by debut decade (handles pace inflation, 3PT era shift)
- Shooting + Play-by-Play data only from 1997 — imputed per-era for older players
- Height, weight, position, HOF excluded from clustering (metadata only)
"""

import numpy as np
import pandas as pd
import os
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

warnings.filterwarnings("ignore", message="You are merging on int and float")

# ═══════════════════════════════════════════════════════════════
# ERA BUCKETING (by debut)
# ═══════════════════════════════════════════════════════════════

from ..constants import assign_player_era as assign_era


# ═══════════════════════════════════════════════════════════════
# DATA LOADING & MERGING (adapted from overralplayerStats.py)
# ═══════════════════════════════════════════════════════════════

def load_and_merge_player_data(data_dir: str) -> pd.DataFrame:
    """
    Load all 8 per-season player CSVs, merge on (season, player_id + context keys),
    and return one wide table with every player-season row.
    """
    advanced      = pd.read_csv(os.path.join(data_dir, "Advanced.csv"))
    totals        = pd.read_csv(os.path.join(data_dir, "Player Totals.csv"))
    per_game      = pd.read_csv(os.path.join(data_dir, "Player Per Game.csv"))
    shooting      = pd.read_csv(os.path.join(data_dir, "Player Shooting.csv"))
    play_by_play  = pd.read_csv(os.path.join(data_dir, "Player Play By Play.csv"))
    per_100       = pd.read_csv(os.path.join(data_dir, "Per 100 Poss.csv"))
    per_36        = pd.read_csv(os.path.join(data_dir, "Per 36 Minutes.csv"))
    season_info   = pd.read_csv(os.path.join(data_dir, "Player Season Info.csv"))

    # Standardise season to int
    for df in [advanced, totals, per_game, shooting, play_by_play, per_100, per_36, season_info]:
        if "season" in df.columns:
            df["season"] = df["season"].astype(int)

    # Per Game uses 'mp_per_game' instead of 'mp'
    per_game = per_game.rename(columns={"mp_per_game": "mp"})

    CORE_KEYS = ["season", "player_id"]
    CONTEXT_KEYS = ["lg", "player", "age", "team", "pos", "g", "gs", "mp"]

    datasets = [advanced, totals, per_game, shooting, play_by_play, per_100, per_36, season_info]

    merged = datasets[0]
    for df in datasets[1:]:
        merge_on = [k for k in (CORE_KEYS + CONTEXT_KEYS) if k in merged.columns and k in df.columns]
        overlap_cols = [c for c in df.columns if c in merged.columns and c not in merge_on]
        df_dedup = df.drop(columns=overlap_cols)
        merged = merged.merge(df_dedup, on=merge_on, how="outer")

    merged = merged.copy()
    print(f"[load] {len(datasets)} datasets merged → {merged.shape[1]} cols, {len(merged)} player-seasons")
    return merged


def filter_and_aggregate_players(
    merged: pd.DataFrame,
    data_dir: str,
    min_seasons: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Filter to players with ≥ min_seasons, then compute career means.

    Returns (player_df, feature_df, player_ids) where each row = one player.
    """
    # ── Filter: 5+ seasons ──
    seasons_per_player = merged.groupby("player_id")["season"].nunique()
    valid_players = seasons_per_player[seasons_per_player >= min_seasons].index
    n_before = merged["player_id"].nunique()
    merged = merged[merged["player_id"].isin(valid_players)]
    n_after = merged["player_id"].nunique()
    print(f"[filter] {n_after} players with ≥ {min_seasons} seasons (removed {n_before - n_after})")

    # ── Capture first/last season and debut era per player ──
    player_meta = merged.groupby("player_id").agg(
        debut_season=("season", "min"),
        final_season=("season", "max"),
        total_seasons=("season", "nunique"),
        primary_pos=("pos", lambda x: x.mode().iloc[0] if not x.mode().empty else "UNK"),
        player_name=("player", "first"),
    ).reset_index()
    player_meta["debut_era"] = player_meta["debut_season"].apply(assign_era)

    # ── Group by player_id → career means ──
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    agg_cols = [c for c in numeric_cols if c != "player_id"]

    player_df = merged.groupby("player_id")[agg_cols].mean().reset_index().copy()
    player_df = player_df.dropna(subset=[c for c in agg_cols if c in player_df.columns][:5], axis=0)

    # Merge metadata
    player_df = player_df.merge(player_meta, on="player_id", how="left")
    player_df["player"] = player_df["player_name"]

    print(f"[aggregate] {len(player_df)} unique players, {player_df.shape[1]} features")

    # Attach career info (height, weight, HOF) 
    # Path resolved relative to the data_dir passed from pipeline config
    career_info_path = os.path.join(data_dir, "Player Career Info.csv")
    if not os.path.exists(career_info_path):
        # Fallback: try relative to this file (for legacy usage)
        career_info_path = os.path.join(
            os.path.dirname(__file__),
            "../../../data/nba-aba-baa-stats/versions/56/Player Career Info.csv"
        )
    career_info = pd.read_csv(career_info_path)
    player_df = player_df.merge(
        career_info[["player_id", "ht_in_in", "wt", "hof"]],
        on="player_id", how="left"
    )
    player_df["hof"] = player_df["hof"].fillna(False).astype(bool)

    # Extract feature only DataFrame
    player_ids = player_df["player_id"].copy()
    feature_df = player_df.drop(columns=["player_id", "player", "player_name",
                                          "primary_pos", "ht_in_in", "wt", "hof",
                                          "debut_season", "final_season",
                                          "total_seasons", "debut_era"], errors="ignore")
    feature_df = feature_df.select_dtypes(include=[np.number])
    feature_df = feature_df.dropna(axis=1, how="all")
    feature_df = feature_df.loc[:, feature_df.nunique() > 1]
    feature_df = feature_df.fillna(feature_df.median())

    print(f"[filter] {feature_df.shape[1]} numeric columns ready for clustering")
    return player_df, feature_df, player_ids


# ═══════════════════════════════════════════════════════════════
# 7 FEATURE BLOCKS
# ═══════════════════════════════════════════════════════════════

def _find_cols(df: pd.DataFrame, *patterns: str) -> list[str]:
    """Return columns whose names contain any of the given substrings."""
    matches = []
    for pat in patterns:
        matches.extend([c for c in df.columns if pat.lower() in c.lower()])
    return list(dict.fromkeys(matches))


def build_feature_blocks(feature_df: pd.DataFrame) -> dict:
    """
    Partition features into 7 semantic blocks.

    Returns dict with block names → list of column names.
    """
    blocks = {}

    # ── Block 1: Scoring Profile (volume + efficiency) ──
    blocks["scoring"] = _find_cols(
        feature_df,
        "pts_per_game", "pts_per_100_poss", "pts_per_36_min",
        "usg_percent", "ts_percent", "fg_percent",
        "fg_per_game", "fga_per_game",
        "x3p_per_game", "x3pa_per_game",
        "ft_per_game", "fta_per_game",
        "x2p_per_game", "x2pa_per_game",
    )
    # Deduplicate — keep first occurrence of each base concept
    blocks["scoring"] = _deduplicate_concepts(blocks["scoring"],
        prefer=["pts_per_game", "usg_percent", "ts_percent", "fg_percent",
                "fg_per_game", "fga_per_game", "x3p_per_game", "x3pa_per_game",
                "ft_per_game", "fta_per_game", "x2p_per_game", "x2pa_per_game"])

    # ── Block 2: Playmaking ──
    blocks["playmaking"] = _find_cols(
        feature_df,
        "ast_per_game", "ast_percent", "ast_per_100_poss", "ast_per_36_min",
        "tov_per_game", "tov_percent",
        "points_generated_by_assists",
    )
    blocks["playmaking"] = _deduplicate_concepts(blocks["playmaking"],
        prefer=["ast_per_game", "ast_percent", "tov_per_game", "tov_percent",
                "points_generated_by_assists"])

    # ── Block 3: Rebounding ──
    blocks["rebounding"] = _find_cols(
        feature_df,
        "orb_per_game", "drb_per_game", "trb_per_game",
        "orb_percent", "drb_percent", "trb_percent",
        "orb_per_100_poss", "drb_per_100_poss", "trb_per_100_poss",
        "orb_per_36_min", "drb_per_36_min", "trb_per_36_min",
    )
    blocks["rebounding"] = _deduplicate_concepts(blocks["rebounding"],
        prefer=["orb_percent", "drb_percent", "trb_percent",
                "orb_per_game", "drb_per_game", "trb_per_game"])

    # ── Block 4: Defense ──
    blocks["defense"] = _find_cols(
        feature_df,
        "stl_per_game", "blk_per_game",
        "stl_percent", "blk_percent",
        "stl_per_100_poss", "blk_per_100_poss",
        "dbpm", "dws",
    )
    blocks["defense"] = _deduplicate_concepts(blocks["defense"],
        prefer=["stl_percent", "blk_percent", "dbpm", "dws",
                "stl_per_game", "blk_per_game"])

    # ── Block 5: Shooting Profile (shot diet) ──
    blocks["shooting"] = _find_cols(
        feature_df,
        "avg_dist_fga",
        "percent_fga_from", "fg_percent_from",
        "x3p_ar", "f_tr", "e_fg_percent",
        "x3p_percent", "ft_percent",
    )
    blocks["shooting"] = _deduplicate_concepts(blocks["shooting"],
        prefer=["avg_dist_fga", "x3p_ar", "f_tr", "e_fg_percent",
                "x3p_percent", "ft_percent"])

    # ── Block 6: Positional Tendency (play-by-play positions) ──
    blocks["positional"] = _find_cols(
        feature_df,
        "pg_percent", "sg_percent", "sf_percent",
        "pf_percent", "c_percent",
    )

    # ── Block 7: Advanced / Value metrics (used sparingly) ──
    blocks["advanced"] = [
        c for c in ["per", "bpm", "obpm", "dbpm", "vorp", "ws", "ows", "dws", "ws_48"]
        if c in feature_df.columns
    ]

    # Report
    for name, cols in blocks.items():
        print(f"[blocks] {name}: {len(cols)} features")

    total = sum(len(v) for v in blocks.values())
    print(f"[blocks] Total: {total} features across 7 blocks")

    return blocks


def _deduplicate_concepts(cols: list[str], prefer: list[str]) -> list[str]:
    """
    Keep only one variant of each stat concept using the explicit
    PLAYER_DEDUP_MAP from constants. Prefers the canonical form of each stat
    (e.g., pts_per_game over pts_per_100_poss, orb_percent over orb_per_game).
    """
    from ..constants import PLAYER_DEDUP_MAP

    kept = []
    replaced = set()

    for col in cols:
        if col in PLAYER_DEDUP_MAP:
            # This column is a duplicate of a preferred form
            replaced.add(col)
            continue
        kept.append(col)

    if replaced:
        print(f"[dedup] Removed {len(replaced)} duplicate concepts: {sorted(replaced)}")

    return kept


# ═══════════════════════════════════════════════════════════════
# BLOCK 7BIS: DERIVED ARCHETYPE SCORES
# ═══════════════════════════════════════════════════════════════

def compute_archetype_scores(
    feature_df: pd.DataFrame,
    blocks: dict,
) -> pd.DataFrame:
    """
    Compute composite Z-score indices for 6 player dimensions:
    scoring, playmaking, defense, rebounding, spacing, versatility.
    """
    from sklearn.preprocessing import StandardScaler

    Z_all = pd.DataFrame(
        StandardScaler().fit_transform(feature_df.fillna(feature_df.median())),
        columns=feature_df.columns,
        index=feature_df.index,
    )

    # Scoring: high PPG, high USG%, high PER
    scoring_cols = [c for c in blocks.get("scoring", []) + blocks.get("advanced", [])
                    if c in Z_all.columns and
                    any(k in c.lower() for k in ["pts", "usg", "per", "fg_per_game", "fga"])]
    scoring_cols = list(dict.fromkeys(scoring_cols))

    # Playmaking: high AST, high AST%, points generated by assists
    playmaking_cols = [c for c in blocks.get("playmaking", [])
                       if c in Z_all.columns]

    # Defense: high STL%, high BLK%, high DBPM, high DWS
    defense_cols = [c for c in blocks.get("defense", []) + ["dbpm", "dws"]
                    if c in Z_all.columns]
    defense_cols = list(dict.fromkeys(defense_cols))

    # Rebounding: high rebound rates
    rebounding_cols = [c for c in blocks.get("rebounding", [])
                       if c in Z_all.columns]

    # Spacing: high 3PA rate, high average shot distance, high 3P%
    spacing_cols = [c for c in blocks.get("shooting", [])
                    if c in Z_all.columns and
                    any(k in c.lower() for k in ["x3p_ar", "avg_dist", "x3p_percent", "e_fg"])]
    spacing_cols = list(dict.fromkeys(spacing_cols))

    # Versatility: positional diversity (entropy across PG/SG/SF/PF/C)
    pos_cols = [c for c in blocks.get("positional", []) if c in Z_all.columns]

    def _composite(cols):
        avail = [c for c in cols if c in Z_all.columns]
        return Z_all[avail].mean(axis=1) if avail else pd.Series(0.0, index=Z_all.index)

    archetype_df = pd.DataFrame({
        "scoring_score":      _composite(scoring_cols),
        "playmaking_score":   _composite(playmaking_cols),
        "defense_score":      _composite(defense_cols),
        "rebounding_score":   _composite(rebounding_cols),
        "spacing_score":      _composite(spacing_cols),
        "versatility_score":  _composite(pos_cols),
    }, index=feature_df.index)

    print(f"[archetype] 6 composite scores: {list(archetype_df.columns)}")
    return archetype_df


# ═══════════════════════════════════════════════════════════════
# ERA ADJUSTMENT
# ═══════════════════════════════════════════════════════════════

def era_adjust(
    player_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Z-score each feature within the player's debut era.

    Debut era = when the player entered the league (not per-season).
    This prevents a player who played across eras from being averaged out.
    """
    adjusted = player_df.copy()
    era_adj_cols = []

    for col in feature_cols:
        if col not in player_df.columns:
            continue
        era_means = player_df.groupby("debut_era")[col].transform("mean")
        era_stds  = player_df.groupby("debut_era")[col].transform("std").replace(0, 1.0)
        new_col = f"{col}_era_adj"
        adjusted[new_col] = (player_df[col] - era_means) / era_stds
        era_adj_cols.append(new_col)

    print(f"[era-adjust] {len(era_adj_cols)} features Z-scored within debut era")
    return adjusted, era_adj_cols


# ═══════════════════════════════════════════════════════════════
# FEATURE MATRIX CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

def build_feature_matrix(
    player_df: pd.DataFrame,
    era_adj_cols: list[str],
    block_weights: Optional[dict[str, float]] = None,
) -> tuple[np.ndarray, RobustScaler, list[str]]:
    """
    Build the final scaled feature matrix with NaN imputation and RobustScaler.
    """
    if block_weights is None:
        block_weights = {}

    available = [c for c in era_adj_cols if c in player_df.columns]
    print(f"[features] {len(available)}/{len(era_adj_cols)} era-adjusted columns available")

    X_raw = player_df[available].values.astype(np.float64)

    # Impute NaN
    n_nan = int(np.isnan(X_raw).sum())
    if n_nan > 0:
        imputer = SimpleImputer(strategy="median")
        X_raw = imputer.fit_transform(X_raw)
        print(f"[features] Imputed {n_nan} NaN values (median strategy)")

    # Robust scaling
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    X_scaled = scaler.fit_transform(X_raw)

    # Apply block weights
    block_map = _build_block_column_map(era_adj_cols)
    for i, col in enumerate(available):
        block = block_map.get(col, "default")
        w = block_weights.get(block, 1.0)
        X_scaled[:, i] *= w

    print(f"[features] Final matrix: {X_scaled.shape[0]} rows × {X_scaled.shape[1]} cols")
    return X_scaled, scaler, available


def _build_block_column_map(era_adj_cols: list[str]) -> dict[str, str]:
    """Map each era-adjusted column to its semantic block for weighting."""
    mapping = {}
    for col in era_adj_cols:
        base = col.replace("_era_adj", "")
        if any(k in base for k in ["pts", "usg", "ts_percent", "fg_percent", "fg_per_game"]):
            mapping[col] = "scoring"
        elif any(k in base for k in ["ast", "tov", "points_generated"]):
            mapping[col] = "playmaking"
        elif any(k in base for k in ["orb", "drb", "trb"]):
            mapping[col] = "rebounding"
        elif any(k in base for k in ["stl", "blk", "dbpm", "dws"]):
            mapping[col] = "defense"
        elif any(k in base for k in ["x3p_ar", "avg_dist", "percent_fga", "fg_percent_from"]):
            mapping[col] = "shooting"
        elif any(k in base for k in ["pg_percent", "sg_percent", "sf_percent", "pf_percent", "c_percent"]):
            mapping[col] = "positional"
        elif any(k in base for k in ["per", "bpm", "vorp", "ws", "obpm"]):
            mapping[col] = "advanced"
        else:
            mapping[col] = "default"
    return mapping


# ═══════════════════════════════════════════════════════════════
# FULL FEATURE ENGINEERING PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_player_feature_engineering(
    data_dir: str,
    min_seasons: int = 5,
    block_weights: Optional[dict[str, float]] = None,
) -> dict:
    """
    Execute the complete player feature engineering pipeline.

    Returns dict with:
        - player_df: full DataFrame with all features + metadata
        - feature_df: numeric-only DataFrame before era adjustment
        - X_scaled: final feature matrix (n_players, n_features)
        - scaler: fitted RobustScaler
        - feature_names: list of final column names
        - era_adj_cols: list of era-adjusted column names
        - metadata_df: DataFrame with player_id, name, pos, height, weight, HOF, debut_era
        - blocks: dict of feature block name → column names
        - archetype_df: DataFrame of 6 composite archetype scores
    """
    # Load + merge
    merged = load_and_merge_player_data(data_dir)

    # Filter + aggregate
    player_df, feature_df, player_ids = filter_and_aggregate_players(merged, data_dir, min_seasons)

    # Build feature blocks
    blocks = build_feature_blocks(feature_df)

    # Compute archetype scores
    archetype_df = compute_archetype_scores(feature_df, blocks)

    # Flatten block columns for era adjustment
    all_block_cols = []
    for cols in blocks.values():
        all_block_cols.extend(cols)
    all_block_cols = list(dict.fromkeys(all_block_cols))

    # Also include archetype scores
    all_block_cols.extend(archetype_df.columns.tolist())

    # Add archetype scores to player_df
    player_df = pd.concat([player_df, archetype_df], axis=1)

    available_features = [c for c in all_block_cols if c in player_df.columns]
    print(f"[feat-eng] {len(available_features)}/{len(all_block_cols)} clustering features available")

    # Era adjustment
    player_df, era_adj_cols = era_adjust(player_df, available_features)

    # Build final matrix
    X_scaled, scaler, feature_names = build_feature_matrix(player_df, era_adj_cols, block_weights)

    # Metadata for labeling/evaluation
    meta_cols = ["player_id", "player", "primary_pos", "ht_in_in", "wt", "hof",
                  "debut_season", "final_season", "total_seasons", "debut_era"]
    metadata_df = player_df[meta_cols].copy()

    return {
        "player_df": player_df,
        "feature_df": feature_df,
        "X_scaled": X_scaled,
        "scaler": scaler,
        "feature_names": feature_names,
        "era_adj_cols": era_adj_cols,
        "metadata_df": metadata_df,
        "blocks": blocks,
        "archetype_df": archetype_df,
    }
