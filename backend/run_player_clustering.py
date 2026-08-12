#!/usr/bin/env python3
"""
NBA Player Archetype Clustering — Main Entry Point
===================================================
Run the full player pipeline: merge 8 CSVs → filter 5+ seasons → career means →
7-block features → era adjust → PCA → UMAP → cluster → evaluate → label → visualize.

Usage:
    python run_player_clustering.py
    python run_player_clustering.py --n-clusters 14 --no-plots
    python run_player_clustering.py --similar "LeBron James"
    python run_player_clustering.py --compare "Stephen Curry" "Ray Allen"
    python run_player_clustering.py --player-profile "Nikola Jokic"

Output:
    - output_players/cluster_profiles.json       : Named archetype profiles
    - output_players/evaluation.json             : Validation metrics
    - output_players/players_with_archetypes.csv : Full dataset with labels
    - output_players/*.html                      : Interactive visualizations
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.clustering.player_pipeline import (
    run_player_pipeline,
    get_similar_players,
    compare_players,
    get_player_profile,
    find_player_idx,
)
from app.validation import (
    PipelineParams,
    resolve_and_validate,
    CSVSchemaError,
)
from app.schema_definitions import PLAYER_SCHEMAS


def main():
    parser = argparse.ArgumentParser(
        description="NBA Player Archetype Clustering Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_player_clustering.py
  python run_player_clustering.py --n-clusters 14
  python run_player_clustering.py --similar "LeBron James"
  python run_player_clustering.py --compare "Stephen Curry" "Ray Allen"
  python run_player_clustering.py --player-profile "Nikola Jokic"
        """,
    )

    parser.add_argument("--n-clusters", type=int, default=12,
                        help="Target number of archetypes (default: 12)")
    parser.add_argument("--pca-variance", type=float, default=0.90,
                        help="PCA variance to retain (default: 0.90)")
    parser.add_argument("--min-seasons", type=int, default=5,
                        help="Minimum seasons played (default: 5)")
    parser.add_argument("--min-cluster-size", type=int, default=25,
                        help="HDBSCAN min cluster size (default: 25)")
    parser.add_argument("--no-hdbscan", action="store_true",
                        help="Disable HDBSCAN")
    parser.add_argument("--no-plots", action="store_true",
                        help="Do not display plots in browser")
    parser.add_argument("--output-dir", type=str, default="./output_players",
                        help="Output directory (default: ./output_players)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to CSV data directory")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable intermediate result caching")

    # Query modes
    parser.add_argument("--similar", type=str, metavar="PLAYER",
                        help="Find similar players to PLAYER")
    parser.add_argument("--compare", nargs=2, metavar=("PLAYER_A", "PLAYER_B"),
                        help="Compare two players' archetypes")
    parser.add_argument("--player-profile", type=str, metavar="PLAYER",
                        help="Show full archetype profile for a player")

    args = parser.parse_args()

    # ── Resolve default data directory ──
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data_dir = os.path.join(base, "data", "nba-aba-baa-stats", "versions", "56")

    # ── Validate parameters ──
    try:
        params = PipelineParams(
            n_clusters=args.n_clusters,
            pca_variance=args.pca_variance,
            min_seasons=args.min_seasons,
            min_cluster_size=args.min_cluster_size,
            use_hdbscan=not args.no_hdbscan,
        )
    except Exception as e:
        print(f"❌ Parameter validation failed: {e}")
        sys.exit(1)

    # ── Resolve data dir + validate CSV schemas (raises on failure) ──
    try:
        data_dir = resolve_and_validate(
            args.data_dir, default_data_dir, params, PLAYER_SCHEMAS,
        )
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except CSVSchemaError as e:
        print(f"\n❌ {e} Fix the above issues and retry.")
        sys.exit(1)

    # ── Run pipeline ──
    print(f"\n📁 Data: {data_dir}")
    print(f"📁 Output: {args.output_dir}\n")

    result = run_player_pipeline(
        data_dir=data_dir,
        output_dir=args.output_dir,
        enable_cache=not args.no_cache,
        n_clusters=args.n_clusters,
        pca_variance=args.pca_variance,
        min_seasons=args.min_seasons,
        use_hdbscan=not args.no_hdbscan,
        min_cluster_size=args.min_cluster_size,
        show_plots=not args.no_plots,
    )

    # ── Handle query modes (single error boundary for all modes) ──
    try:
        if args.similar:
            player = args.similar
            print(f"\n{'='*60}")
            print(f"  SIMILAR TO: {player}")
            print(f"{'='*60}")
            results = get_similar_players(result, player, top_k=10)
            for _, row in results.iterrows():
                print(f"  {int(row['rank']):2d}. {row['team']} | "
                      f"sim={row['cosine_similarity']:.4f}")

        if args.compare:
            player_a, player_b = args.compare
            print(f"\n{'='*60}")
            print(f"  COMPARING: {player_a} vs {player_b}")
            print(f"{'='*60}")
            comparison = compare_players(result, player_a, player_b)
            print(f"  Cosine similarity: {comparison['cosine_similarity']:.4f}")
            print(f"  Same archetype: {comparison['same_archetype']}")
            for key, label in [("player_a", player_a), ("player_b", player_b)]:
                p = comparison[key]
                print(f"\n  {label}:")
                print(f"    Archetype: {p['archetype_name']}")
                print(f"    Position: {p['position']} | HOF: {p['hof']}")
                print(f"    Top features: {', '.join(f['label'] for f in p['top_features'])}")

        if args.player_profile:
            player = args.player_profile
            print(f"\n{'='*60}")
            print(f"  PLAYER PROFILE: {player}")
            print(f"{'='*60}")
            profile = get_player_profile(result, player)
            print(f"  Position: {profile['position']} | HOF: {profile['hof']}")
            print(f"  Height: {profile['height']}\" | Weight: {profile['weight']} lbs")
            print(f"  Debut: {profile['debut_season']} | Seasons: {profile['total_seasons']}")
            print(f"  Archetype: {profile['archetype_name']}")
            print(f"  UMAP: ({profile['umap_2d'][0]:.3f}, {profile['umap_2d'][1]:.3f})")
            print(f"  Top defining features:")
            for f in profile['top_features']:
                print(f"    {f['label']}: {f['z_score']:+.3f}σ")
    except ValueError as e:
        print(f"  ❌ {e}")

    print(f"\n✅ Done. All outputs in: {args.output_dir}/")


if __name__ == "__main__":
    main()
