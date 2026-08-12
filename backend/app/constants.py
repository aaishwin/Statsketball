"""
Shared Constants for NBA Clustering Pipelines
==============================================
Single source of truth for era buckets, feature blocks, label priorities,
color palettes, and all other hardcoded values used across the codebase.
"""

# ═══════════════════════════════════════════════════════════════
# ERA BUCKETS
# ═══════════════════════════════════════════════════════════════

PLAYER_ERA_BUCKETS = {
    "Pre-3PT":       (1900, 1979),
    "Early-3PT":     (1980, 1989),
    "Expansion-90s": (1990, 1999),
    "Dead-Ball":     (2000, 2011),
    "Modern":        (2012, 2019),
    "Post-COVID":    (2020, 2100),
}

def assign_player_era(season: int) -> str:
    """Map a season year to its stylistic era bucket (player pipeline)."""
    for name, (lo, hi) in PLAYER_ERA_BUCKETS.items():
        if lo <= season <= hi:
            return name
    return "Unknown"


# ═══════════════════════════════════════════════════════════════
# PLAYER FEATURE BLOCKS
# ═══════════════════════════════════════════════════════════════

PLAYER_BLOCK_WEIGHT_KEYS = {
    "scoring", "playmaking", "rebounding", "defense",
    "shooting", "positional", "advanced",
}

PLAYER_ARCHETYPE_SCORES = [
    "scoring_score", "playmaking_score", "defense_score",
    "rebounding_score", "spacing_score", "versatility_score",
]

# Explicit deduplication mapping for player features:
# When both "pts_per_game" and "pts_per_100_poss" exist, keep the preferred one.
PLAYER_DEDUP_MAP = {
    # Scoring → prefer per-game over rate stats for volume
    "pts_per_100_poss":       "pts_per_game",
    "pts_per_36_min":          "pts_per_game",
    "fg_per_100_poss":         "fg_per_game",
    "fga_per_100_poss":        "fga_per_game",
    "x3p_per_100_poss":        "x3p_per_game",
    "x3pa_per_100_poss":       "x3pa_per_game",
    "x2p_per_100_poss":        "x2p_per_game",
    "x2pa_per_100_poss":       "x2pa_per_game",
    "ft_per_100_poss":         "ft_per_game",
    "fta_per_100_poss":        "fta_per_game",
    # Playmaking
    "ast_per_100_poss":        "ast_per_game",
    "ast_per_36_min":          "ast_per_game",
    # Rebounding → prefer rates over raw counts for better comparison
    "orb_per_game":            "orb_percent",
    "drb_per_game":            "drb_percent",
    "trb_per_game":            "trb_percent",
    "orb_per_100_poss":        "orb_percent",
    "drb_per_100_poss":        "drb_percent",
    "trb_per_100_poss":        "trb_percent",
    "orb_per_36_min":          "orb_percent",
    "drb_per_36_min":          "drb_percent",
    "trb_per_36_min":          "trb_percent",
    # Defense
    "stl_per_100_poss":        "stl_percent",
    "blk_per_100_poss":        "blk_percent",
}


# ═══════════════════════════════════════════════════════════════
# COLOR PALETTES
# ═══════════════════════════════════════════════════════════════

CLUSTER_COLORS_PLAYER = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F8C471", "#82E0AA", "#F1948A", "#85929E", "#AED6F1",
    "#E8A87C", "#A8D8EA",
]

NOISE_COLOR = "#B0B0B0"

# ═══════════════════════════════════════════════════════════════
# PLAYER FEATURE → ARCHETYPE LABEL MAPPING
# ═══════════════════════════════════════════════════════════════

PLAYER_FEATURE_LABELS = {
    # Scoring
    ("pts_per_game", "high"): "High-Volume Scorer",
    ("pts_per_game", "low"): "Low-Volume Scorer",
    ("pts_per_100_poss", "high"): "Efficient Scorer",
    ("pts_per_100_poss", "low"): "Low-Impact Scorer",
    ("usg_percent", "high"): "High-Usage",
    ("usg_percent", "low"): "Low-Usage",
    ("per", "high"): "High-PER",
    ("per", "low"): "Low-PER",
    ("fg_per_game", "high"): "High-FGA",
    ("fg_per_game", "low"): "Low-FGA",
    ("fga_per_game", "high"): "Volume Shooter",
    ("fga_per_game", "low"): "Selective Shooter",
    ("x2p_per_game", "high"): "Paint Scorer",
    ("x2pa_per_game", "high"): "Paint Attacker",
    ("ft_per_game", "high"): "Foul Drawer",
    ("fta_per_game", "high"): "Contact Seeker",
    ("fta_per_game", "low"): "Avoids Contact",
    # Efficiency
    ("ts_percent", "high"): "Efficient Scorer",
    ("ts_percent", "low"): "Inefficient Scorer",
    ("e_fg_percent", "high"): "Efficient Shooter",
    ("e_fg_percent", "low"): "Inefficient Shooter",
    ("fg_percent", "high"): "High-FG%",
    ("fg_percent", "low"): "Low-FG%",
    ("x2p_percent", "high"): "Inside-Finisher",
    ("x2p_percent", "low"): "Poor-Finisher",
    # Playmaking
    ("ast_per_game", "high"): "Floor General",
    ("ast_per_game", "low"): "Non-Passer",
    ("ast_percent", "high"): "High-Assist-Rate",
    ("ast_percent", "low"): "Low-Assist-Rate",
    ("ast_per_100_poss", "high"): "Playmaker",
    ("points_generated_by_assists", "high"): "Offense Creator",
    ("tov_per_game", "high"): "Turnover-Prone",
    ("tov_per_game", "low"): "Ball-Secure",
    ("tov_percent", "high"): "High-TOV%",
    ("tov_percent", "low"): "Careful",
    # Rebounding
    ("trb_per_game", "high"): "Glass Cleaner",
    ("trb_per_game", "low"): "Weak Rebounder",
    ("orb_per_game", "high"): "Offensive Rebounder",
    ("orb_per_game", "low"): "Non-Crasher",
    ("drb_per_game", "high"): "Defensive Rebounder",
    ("drb_per_game", "low"): "Poor-DRB",
    ("orb_percent", "high"): "ORB Specialist",
    ("orb_percent", "low"): "Leaks Out",
    ("drb_percent", "high"): "DRB Anchor",
    ("drb_percent", "low"): "Weak-on-Glass",
    ("trb_percent", "high"): "Elite Rebounder",
    ("trb_percent", "low"): "Below-Avg Rebounder",
    # Defense
    ("stl_per_game", "high"): "Ball Hawk",
    ("stl_per_game", "low"): "Low-Steal",
    ("blk_per_game", "high"): "Shot Blocker",
    ("blk_per_game", "low"): "Non-Blocker",
    ("stl_percent", "high"): "Passing-Lane Disruptor",
    ("stl_percent", "low"): "Low-STL%",
    ("blk_percent", "high"): "Rim Protector",
    ("blk_percent", "low"): "Low-BLK%",
    ("dbpm", "high"): "Defensive Plus",
    ("dbpm", "low"): "Defensive Minus",
    ("dws", "high"): "Defensive Anchor",
    ("dws", "low"): "Low-Defensive-WS",
    # 3-Point Shooting
    ("x3p_per_game", "high"): "3PT Marksman",
    ("x3p_per_game", "low"): "Non-Shooter",
    ("x3pa_per_game", "high"): "Volume 3PT Shooter",
    ("x3pa_per_game", "low"): "Rare 3PT Shooter",
    ("x3p_percent", "high"): "Sharpshooter",
    ("x3p_percent", "low"): "Poor-Shooter",
    ("x3p_ar", "high"): "3PT-Reliant",
    ("x3p_ar", "low"): "Paint-Focused",
    ("avg_dist_fga", "high"): "Perimeter Player",
    ("avg_dist_fga", "low"): "Interior Player",
    # Shot diet zones
    ("percent_fga_from_x0_3_range", "high"): "Rim Attacker",
    ("percent_fga_from_x0_3_range", "low"): "Rim-Averse",
    ("percent_fga_from_x3_10_range", "high"): "Floater Game",
    ("percent_fga_from_x10_16_range", "high"): "Mid-Range Specialist",
    ("percent_fga_from_x16_3p_range", "high"): "Long-2 Specialist",
    ("percent_fga_from_x3p_range", "high"): "3PT Specialist",
    ("percent_fga_from_x3p_range", "low"): "Non-3PT Shooter",
    ("fg_percent_from_x0_3_range", "high"): "Elite Finisher",
    ("fg_percent_from_x3_10_range", "high"): "Floater Ace",
    ("fg_percent_from_x10_16_range", "high"): "Mid-Range Ace",
    ("fg_percent_from_x16_3p_range", "high"): "Long-2 Ace",
    ("fg_percent_from_x3p_range", "high"): "3PT Sniper",
    # Positional / Versatility
    ("pg_percent", "high"): "Point Guard",
    ("sg_percent", "high"): "Shooting Guard",
    ("sf_percent", "high"): "Small Forward",
    ("pf_percent", "high"): "Power Forward",
    ("c_percent", "high"): "Center",
    ("pg_percent", "low"): "Non-PG",
    ("c_percent", "low"): "Non-C",
    # Advanced
    ("bpm", "high"): "High-BPM",
    ("bpm", "low"): "Low-BPM",
    ("obpm", "high"): "Offensive Star",
    ("obpm", "low"): "Offensive Minus",
    ("vorp", "high"): "High-Value",
    ("vorp", "low"): "Low-Value",
    ("ws", "high"): "Win Producer",
    ("ws", "low"): "Low-WS",
    ("ws_48", "high"): "Efficient Contributor",
    ("ows", "high"): "Offensive Anchor",
    ("dws", "high"): "Defensive Anchor",
    # Composite Archetype Scores
    ("scoring_score", "high"): "Scoring Threat",
    ("scoring_score", "low"): "Non-Scorer",
    ("playmaking_score", "high"): "Creator",
    ("playmaking_score", "low"): "Finisher",
    ("defense_score", "high"): "Defensive Specialist",
    ("defense_score", "low"): "Low-Effort Defender",
    ("rebounding_score", "high"): "Board Man",
    ("rebounding_score", "low"): "Non-Rebounder",
    ("spacing_score", "high"): "Floor Spacer",
    ("spacing_score", "low"): "Non-Spacer",
    ("versatility_score", "high"): "Versatile",
    ("versatility_score", "low"): "One-Dimensional",
    # Free throws
    ("ft_percent", "high"): "Clutch FT Shooter",
    ("ft_percent", "low"): "Poor FT Shooter",
    ("f_tr", "high"): "Draws Contact",
    ("f_tr", "low"): "Finesse Player",
}

PLAYER_LABEL_PRIORITY = [
    "scoring_score", "playmaking_score", "defense_score", "rebounding_score",
    "spacing_score", "versatility_score",
    "pts_per_game", "usg_percent", "ast_per_game", "trb_per_game",
    "stl_per_game", "blk_per_game", "x3p_per_game", "x3pa_per_game",
    "x3p_percent", "x3p_ar", "avg_dist_fga", "ts_percent", "e_fg_percent",
    "ast_percent", "tov_percent", "orb_percent", "drb_percent",
    "stl_percent", "blk_percent", "dbpm", "dws",
    "percent_fga_from_x3p_range", "percent_fga_from_x0_3_range",
    "percent_fga_from_x3_10_range", "percent_fga_from_x10_16_range",
    "fg_percent_from_x0_3_range", "fg_percent_from_x3p_range",
    "pg_percent", "sg_percent", "sf_percent", "pf_percent", "c_percent",
    "per", "bpm", "obpm", "vorp", "ws", "ws_48",
    "ft_percent", "f_tr", "fta_per_game",
]
