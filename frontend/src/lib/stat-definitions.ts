/* ═══════════════════════════════════════════════════════════════
   Stat Definitions — feature name → plain-English definition.
   Keys mirror the backend feature names in
   backend/app/faiss_index/ranking.py (PLAYER_BLOCK_PATTERNS).
   Used for tooltips in the "What drives the similarity" section.
   ═══════════════════════════════════════════════════════════════ */

export const STAT_DEFINITIONS: Record<string, string> = {
  /* ── Scoring ── */
  pts_per_game: "Average points scored per game.",
  usg_percent:
    "This is the usage rate, or the percentage of team possessions a player uses while on the floor.",
  ts_percent:
    "This is the true shooting percentage, which is shooting efficiency accounting for 2s, 3s, and free throws.",
  fg_percent: "Field goal percentage, is the share of all field goal attempts made.",
  fg_per_game: "Field goals made per game.",
  fga_per_game: "Field goal attempts per game.",
  x3p_per_game: "Three-pointers made per game.",
  x3pa_per_game: "Three-point attempts per game.",
  x2p_per_game: "Two-pointers made per game.",
  x2pa_per_game: "Two-point attempts per game.",
  ft_per_game: "Free throws made per game.",
  fta_per_game: "Free throw attempts per game.",
  scoring_score: "Composite scoring archetype score (volume + efficiency).",

  /* ── Playmaking ── */
  ast_per_game: "Assists per game.",
  ast_percent:
    "Assist percentage, is the share of teammate field goals a player assisted while on the floor.",
  tov_per_game: "Turnovers per game.",
  tov_percent: "Turnover percentage, is the percentage of a team's turnovers that a player has while on the court.",
  points_generated_by_assists:
    "Estimated points created for teammates via assists.",
  playmaking_score: "Composite playmaking archetype score (creation vs. turnovers).",

  /* ── Rebounding ── */
  orb_percent:
    "Offensive rebound percentage, is the share of available offensive rebounds grabbed.",
  drb_percent:
    "Defensive rebound percentage, is the share of available defensive rebounds grabbed.",
  trb_percent:
    "Total rebound percentage, is the share of all available rebounds grabbed.",
  orb_per_game: "Offensive rebounds per game.",
  drb_per_game: "Defensive rebounds per game.",
  trb_per_game: "Total rebounds per game.",
  rebounding_score: "Composite rebounding archetype score.",

  /* ── Defense ── */
  stl_percent:
    "Steal percentage, is the opponent possessions ending in a steal by the player.",
  blk_percent:
    "Block percentage, is the opponent shot attempts blocked by the player.",
  dbpm: "Defensive box plus/minus, is the defensive impact per 100 possessions vs. league average.",
  dws: "Defensive win shares, is the wins credited to a player from defense.",
  stl_per_game: "Steals per game.",
  blk_per_game: "Blocks per game.",
  defense_score: "Composite defensive archetype score (steals, blocks, impact).",

  /* ── Shooting profile ── */
  avg_dist_fga: "Average shot distance (feet) on field goal attempts.",
  x3p_ar:
    "Three-point attempt rate, is the share of field goal attempts taken from three.",
  f_tr: "Free throw rate, is the amount of free throw attempts per field goal attempt.",
  e_fg_percent:
    "Effective field goal percentage, is the FG% adjusted so threes count 1.5×.",
  x3p_percent: "Three-point percentage.",
  ft_percent: "Free throw percentage",
  spacing_score: "Composite spacing archetype score (3-point volume + accuracy).",

  /* ── Positional ── */
  pg_percent: "Share of minutes played at point guard.",
  sg_percent: "Share of minutes played at shooting guard.",
  sf_percent: "Share of minutes played at small forward.",
  pf_percent: "Share of minutes played at power forward.",
  c_percent: "Share of minutes played at center.",
  versatility_score:
    "Composite positional versatility score — how many positions a player covers.",

  /* ── Advanced ── */
  per: "Player efficiency rating, is the players' per-minute production, the league average is 15.",
  bpm: "Box plus/minus, is the overall impact per 100 possessions vs. league average.",
  obpm: "Offensive box plus/minus, is the offensive impact per 100 possessions vs. league average.",
  vorp: "Value over replacement player, is the total value vs. a replacement-level player.",
  ws: "Win shares, is the estimated wins contributed by the player.",
  ows: "Offensive win shares, are wins credited to a player from offense.",
  ws_48: "Win shares per 48 minutes, is the players' win shares rate. The league average is ~0.100.",
};

/** Look up a stat definition; returns undefined when unknown. */
export function getStatDefinition(feature: string): string | undefined {
  return STAT_DEFINITIONS[feature];
}
