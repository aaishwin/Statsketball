"""
CSV Schema Definitions for NBA Clustering Pipelines
====================================================
Defines the expected columns for every CSV used by the player clustering
pipeline. Used by validation.py for pre-flight checks.

Each schema specifies:
- file_name: The CSV filename
- required_columns: Columns that MUST exist (pipeline will crash without them)
- min_rows: Minimum expected row count (sanity check)
- description: Human-readable purpose
"""

from typing import Optional
from .validation import CSVSchema


# ═══════════════════════════════════════════════════════════════
# PLAYER PIPELINE CSVs (9 files)
# ═══════════════════════════════════════════════════════════════

PLAYER_SCHEMAS = [
    CSVSchema(
        file_name="Advanced.csv",
        required_columns=[
            "season", "player_id", "player", "lg", "team", "pos", "age", "g", "mp",
            "per", "ts_percent", "x3p_ar", "f_tr",
            "orb_percent", "drb_percent", "trb_percent",
            "ast_percent", "stl_percent", "blk_percent",
            "tov_percent", "usg_percent",
            "ows", "dws", "ws", "ws_48",
            "obpm", "dbpm", "bpm", "vorp",
        ],
        min_rows=500,
        description="Advanced player stats per season",
    ),
    CSVSchema(
        file_name="Player Totals.csv",
        required_columns=[
            "season", "player_id", "player", "lg", "team", "pos", "age", "g", "mp",
        ],
        min_rows=500,
        description="Player season totals",
    ),
    CSVSchema(
        file_name="Player Per Game.csv",
        required_columns=[
            "season", "player_id", "player", "lg", "team", "pos", "age", "g",
            "mp_per_game",
        ],
        min_rows=500,
        description="Player per-game averages",
    ),
    CSVSchema(
        file_name="Player Shooting.csv",
        required_columns=[
            "season", "player_id", "player", "lg", "team", "pos", "age", "g", "mp",
            "avg_dist_fga",
            "percent_fga_from_x0_3_range",
            "percent_fga_from_x3_10_range",
            "percent_fga_from_x10_16_range",
            "percent_fga_from_x16_3p_range",
            "percent_fga_from_x3p_range",
        ],
        min_rows=500,
        description="Player shot zone distribution",
    ),
    CSVSchema(
        file_name="Player Play By Play.csv",
        required_columns=[
            "season", "player_id", "player", "lg", "team", "pos", "age", "g", "mp",
        ],
        min_rows=500,
        description="Player position percentages from play-by-play",
    ),
    CSVSchema(
        file_name="Per 100 Poss.csv",
        required_columns=[
            "season", "player_id", "player", "lg", "team", "pos", "age", "g", "mp",
        ],
        min_rows=500,
        description="Player stats per 100 possessions",
    ),
    CSVSchema(
        file_name="Per 36 Minutes.csv",
        required_columns=[
            "season", "player_id", "player", "lg", "team", "pos", "age", "g", "mp",
        ],
        min_rows=500,
        description="Player stats per 36 minutes",
    ),
    CSVSchema(
        file_name="Player Season Info.csv",
        required_columns=[
            "season", "player_id", "player", "lg", "team", "pos", "age",
        ],
        min_rows=500,
        description="Player season contextual info",
    ),
    CSVSchema(
        file_name="Player Career Info.csv",
        required_columns=[
            "player_id", "ht_in_in", "wt", "hof",
        ],
        min_rows=100,
        description="Player career-level metadata (height, weight, HOF)",
    ),
]


# ═══════════════════════════════════════════════════════════════
# HELPER: get schema by filename
# ═══════════════════════════════════════════════════════════════

def get_player_schema(file_name: str) -> Optional[CSVSchema]:
    """Get the schema for a player pipeline CSV by filename."""
    for schema in PLAYER_SCHEMAS:
        if schema.file_name == file_name:
            return schema
    return None


def validate_all_csvs(
    data_dir: str,
    schemas: list[CSVSchema],
) -> "CSVValidationReport":
    """
    Validate ALL CSVs in a directory against their schemas.

    Reports ALL missing columns across ALL files at once,
    rather than crashing on the first missing file.

    Args:
        data_dir: Directory containing the CSV files.
        schemas: List of CSVSchema definitions to validate.

    Returns:
        CSVValidationReport with per-file results.
    """
    import os
    from .validation import CSVValidationReport, CSVValidationResult, validate_csv_exists

    report = CSVValidationReport()

    for schema in schemas:
        file_path = os.path.join(data_dir, schema.file_name)
        result = validate_csv_exists(file_path, schema)
        report.add_result(result)

    # Print summary
    if report.all_valid:
        print(f"[validate] ✓ All {len(schemas)} CSVs passed schema checks")
    else:
        print(f"[validate] ❌ CSV schema validation FAILED:")
        for r in report.results:
            if not r.is_valid:
                if not r.exists:
                    print(f"  - {r.file_name}: FILE NOT FOUND")
                if r.missing_columns:
                    print(f"  - {r.file_name}: MISSING columns: {r.missing_columns}")
                if not r.row_count_ok:
                    print(f"  - {r.file_name}: Row count {r.row_count} below minimum")

    return report
