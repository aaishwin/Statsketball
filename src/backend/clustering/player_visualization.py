"""
Player Archetype Visualization Suite
=====================================
7 interactive Plotly charts for exploring NBA player archetypes:
1. UMAP Archetype Map (main view)
2. Cluster Profile Radar Chart
3. Cluster Size Bar Chart
4. Position Distribution Heatmap
5. Era Distribution Stacked Bar
6. HOF Rate by Cluster
7. Feature Importance by Cluster
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional
import warnings

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

warnings.filterwarnings("ignore", category=FutureWarning)

from ..constants import CLUSTER_COLORS_PLAYER as CLUSTER_COLORS, NOISE_COLOR

def get_color_map(labels: np.ndarray) -> dict:
    unique = sorted(set(labels))
    color_map = {}
    for i, lab in enumerate(unique):
        if lab == -1:
            color_map[lab] = NOISE_COLOR
        else:
            color_map[lab] = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
    return color_map


# ═══════════════════════════════════════════════════════════════
# 1. UMAP ARCHETYPE MAP (MAIN VIEW)
# ═══════════════════════════════════════════════════════════════

def plot_umap_archetype_map(
    X_umap: np.ndarray,
    labels: np.ndarray,
    metadata_df: pd.DataFrame,
    cluster_profiles: dict,
    title: str = "NBA Player Archetype Map",
    highlight_hof: bool = True,
) -> go.Figure:
    """
    Main 2D UMAP scatter plot with cluster coloring and rich hover.
    Hall of Famers shown as star markers.
    """
    color_map = get_color_map(labels)

    plot_df = metadata_df.copy()
    plot_df["x"] = X_umap[:, 0]
    plot_df["y"] = X_umap[:, 1]
    plot_df["label"] = labels
    plot_df["cluster_name"] = plot_df["label"].apply(
        lambda lab: cluster_profiles.get(int(lab), {}).get("name", "Hybrid/Transitional")
        if lab != -1 else "Hybrid / Transitional"
    )
    plot_df["color"] = plot_df["label"].map(color_map)

    fig = go.Figure()

    # Noise points
    noise_df = plot_df[plot_df["label"] == -1]
    if len(noise_df) > 0:
        fig.add_trace(go.Scatter(
            x=noise_df["x"], y=noise_df["y"],
            mode="markers",
            marker=dict(size=4, color=NOISE_COLOR, opacity=0.25, symbol="circle-open"),
            name="Hybrid / Transitional",
            hovertemplate="<b>%{customdata[0]}</b><br>Style: Hybrid/Transitional<extra></extra>",
            customdata=noise_df[["player"]].values,
        ))

    # Clustered players
    for lab in sorted(set(labels) - {-1}):
        cluster_df = plot_df[plot_df["label"] == lab]
        cluster_name = cluster_profiles.get(int(lab), {}).get("name", f"Archetype {lab}")

        # Non-HOF players
        non_hof = cluster_df[~cluster_df["hof"]]
        if len(non_hof) > 0:
            fig.add_trace(go.Scatter(
                x=non_hof["x"], y=non_hof["y"],
                mode="markers",
                marker=dict(size=5, color=color_map[lab], opacity=0.6,
                             line=dict(width=0.3, color="white")),
                name=f"A{lab}: {cluster_name}",
                legendgroup=f"cluster_{lab}",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Style: %{customdata[1]}<br>"
                    "Pos: %{customdata[2]} | Ht: %{customdata[3]}\"<br>"
                    "Debut: %{customdata[4]}<extra></extra>"
                ),
                customdata=non_hof[["player", "cluster_name", "primary_pos",
                                     "ht_in_in", "debut_season"]].values,
            ))

        # HOF players (stars)
        hof_df = cluster_df[cluster_df["hof"]]
        if len(hof_df) > 0:
            fig.add_trace(go.Scatter(
                x=hof_df["x"], y=hof_df["y"],
                mode="markers",
                marker=dict(size=12, color=color_map[lab], opacity=0.95,
                             symbol="star", line=dict(width=1.5, color="gold")),
                name=f"A{lab}: {cluster_name} (HOF)",
                legendgroup=f"cluster_{lab}",
                showlegend=False,
                hovertemplate=(
                    "<b>⭐ %{customdata[0]} (HOF)</b><br>"
                    "Style: %{customdata[1]}<br>"
                    "Pos: %{customdata[2]} | Ht: %{customdata[3]}\"<br>"
                    "Debut: %{customdata[4]}<extra></extra>"
                ),
                customdata=hof_df[["player", "cluster_name", "primary_pos",
                                    "ht_in_in", "debut_season"]].values,
            ))

    # Layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center",
                    font=dict(size=22, color="#cdd6f4")),
        template="plotly_dark",
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#cdd6f4"),
        legend=dict(
            title="Player Archetypes",
            bgcolor="rgba(30,30,46,0.9)",
            bordercolor="#45475a",
            borderwidth=1,
            itemsizing="constant",
            yanchor="top", y=0.99,
            xanchor="left", x=1.01,
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="closest",
        dragmode="pan",
        width=1500,
        height=950,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, title="")
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, title="")

    return fig


# ═══════════════════════════════════════════════════════════════
# 2. CLUSTER PROFILE RADAR
# ═══════════════════════════════════════════════════════════════

RADAR_DIMENSIONS = [
    ("scoring_score", "Scoring"),
    ("playmaking_score", "Playmaking"),
    ("defense_score", "Defense"),
    ("rebounding_score", "Rebounding"),
    ("spacing_score", "Spacing"),
    ("versatility_score", "Versatility"),
    ("ast_per_game", "Passing"),
    ("x3p_per_game", "3PT Shooting"),
]

def plot_player_radar(
    cluster_profiles: dict[int, dict],
) -> go.Figure:
    """Radar chart comparing archetype profiles across 8 dimensions."""
    valid_clusters = sorted([k for k in cluster_profiles.keys()])
    fig = go.Figure()

    for cl in valid_clusters:
        prof = cluster_profiles[cl]
        zs = prof["all_feature_z_scores"]

        values = []
        dim_labels = []
        for feat_key, dim_label in RADAR_DIMENSIONS:
            val = zs.get(feat_key, 0)
            values.append(val)
            dim_labels.append(dim_label)

        values.append(values[0])
        dim_labels.append(dim_labels[0])

        fig.add_trace(go.Scatterpolar(
            r=values, theta=dim_labels,
            name=f"A{cl}: {prof['name'][:35]}",
            fill="toself", opacity=0.3, line=dict(width=2),
        ))

    fig.update_layout(
        title=dict(text="Player Archetype Profiles (Z-scores vs. Global Mean)",
                    x=0.5, font=dict(size=18, color="#cdd6f4")),
        polar=dict(
            radialaxis=dict(visible=True, range=[-3, 3], gridcolor="#45475a"),
            angularaxis=dict(gridcolor="#45475a"),
            bgcolor="#1e1e2e",
        ),
        template="plotly_dark", paper_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
        legend=dict(bgcolor="rgba(30,30,46,0.9)", bordercolor="#45475a", borderwidth=1),
        width=1000, height=750,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# 3. CLUSTER SIZE BAR CHART
# ═══════════════════════════════════════════════════════════════

def plot_player_cluster_sizes(
    labels: np.ndarray,
    cluster_profiles: dict[int, dict],
) -> go.Figure:
    """Horizontal bar chart of players per archetype."""
    valid = sorted([k for k in cluster_profiles.keys()])
    names = [f"A{k}: {cluster_profiles[k]['name'][:50]}" for k in valid]
    sizes = [cluster_profiles[k]["size"] for k in valid]
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(valid))]

    n_noise = int(np.sum(labels == -1))
    if n_noise > 0:
        names.append("Hybrid / Transitional")
        sizes.append(n_noise)
        colors.append(NOISE_COLOR)

    fig = go.Figure(go.Bar(
        x=sizes[::-1], y=names[::-1], orientation="h",
        marker=dict(color=colors[::-1]),
        text=sizes[::-1], textposition="outside",
        hovertemplate="%{y}: %{x} players<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Players per Archetype", x=0.5, font=dict(size=18, color="#cdd6f4")),
        template="plotly_dark", paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
        xaxis=dict(title="Number of Players", gridcolor="#45475a"),
        yaxis=dict(title=""),
        margin=dict(l=20, r=40, t=60, b=20),
        width=1000, height=600,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# 4. POSITION DISTRIBUTION HEATMAP
# ═══════════════════════════════════════════════════════════════

def plot_position_distribution(
    cluster_profiles: dict[int, dict],
) -> go.Figure:
    """Heatmap: Archetypes × Positions. Shows how each archetype maps to
    traditional positions."""
    valid = sorted([k for k in cluster_profiles.keys()])
    positions = ["PG", "SG", "SF", "PF", "C", "G", "F", "F-C", "G-F"]

    # Collect position counts
    pos_matrix = []
    cluster_labels = []
    for cl in valid:
        pos_counts = cluster_profiles[cl]["position_breakdown"]
        total = sum(pos_counts.values())
        row = [pos_counts.get(p, 0) / total if total > 0 else 0 for p in positions]
        pos_matrix.append(row)
        cluster_labels.append(f"A{cl}: {cluster_profiles[cl]['name'][:35]}")

    pos_matrix = np.array(pos_matrix)

    fig = go.Figure(go.Heatmap(
        z=pos_matrix,
        x=positions,
        y=cluster_labels,
        colorscale="YlOrRd",
        text=np.round(pos_matrix, 2),
        texttemplate="%{text:.0%}",
        textfont=dict(size=11),
        hovertemplate="%{y}<br>%{x}: %{z:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Position Distribution by Archetype", x=0.5,
                    font=dict(size=18, color="#cdd6f4")),
        template="plotly_dark", paper_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
        xaxis=dict(title="Position", tickangle=0),
        yaxis=dict(title=""),
        width=900, height=600,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# 5. HOF RATE BY CLUSTER
# ═══════════════════════════════════════════════════════════════

def plot_hof_rate(
    cluster_profiles: dict[int, dict],
) -> go.Figure:
    """Bar chart of Hall of Fame rate per archetype + HOF count annotations."""
    valid = sorted([k for k in cluster_profiles.keys()])
    names = [f"A{k}: {cluster_profiles[k]['name'][:45]}" for k in valid]
    hof_rates = [cluster_profiles[k]["hof_rate"] for k in valid]
    hof_counts = [cluster_profiles[k]["hof_count"] for k in valid]
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(valid))]

    fig = go.Figure(go.Bar(
        x=hof_rates, y=names, orientation="h",
        marker=dict(color=colors),
        text=[f"{r:.1%} ({c} HOF)" for r, c in zip(hof_rates, hof_counts)],
        textposition="outside",
        hovertemplate="%{y}<br>HOF Rate: %{x:.1%}<br>HOF Count: %{customdata}<extra></extra>",
        customdata=hof_counts,
    ))
    fig.update_layout(
        title=dict(text="Hall of Fame Rate by Archetype", x=0.5,
                    font=dict(size=18, color="#cdd6f4")),
        template="plotly_dark", paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
        xaxis=dict(title="HOF Rate", tickformat=".0%", gridcolor="#45475a"),
        yaxis=dict(title=""),
        width=1000, height=600,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# 7. FEATURE IMPORTANCE BY CLUSTER
# ═══════════════════════════════════════════════════════════════

def plot_top_features_by_cluster(
    cluster_profiles: dict[int, dict],
    top_n: int = 5,
) -> go.Figure:
    """Grouped horizontal bar chart: top-N defining features per archetype."""
    valid = sorted([k for k in cluster_profiles.keys()])

    fig = make_subplots(
        rows=len(valid), cols=1,
        subplot_titles=[
            f"A{cl}: {cluster_profiles[cl]['name'][:60]}" for cl in valid
        ],
        vertical_spacing=0.03,
    )

    for row_idx, cl in enumerate(valid, start=1):
        prof = cluster_profiles[cl]
        top_feats = prof["top_features"][:top_n]
        feat_names = [f["label"] for f in top_feats][::-1]
        feat_zs = [f["z_score"] for f in top_feats][::-1]
        colors = ["#FF6B6B" if z > 0 else "#4ECDC4" for z in feat_zs]

        fig.add_trace(go.Bar(
            x=feat_zs, y=feat_names, orientation="h",
            marker=dict(color=colors),
            text=[f"{z:+.2f}σ" for z in feat_zs],
            textposition="outside",
            showlegend=False,
            hovertemplate="%{y}: %{x:+.2f}σ<extra></extra>",
        ), row=row_idx, col=1)

    fig.update_layout(
        title=dict(text="Defining Features by Archetype (Z-scores vs. Global Mean)",
                    x=0.5, font=dict(size=18, color="#cdd6f4")),
        template="plotly_dark", paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
        height=300 * len(valid),
        width=1000,
    )
    fig.update_xaxes(range=[-3.5, 3.5], gridcolor="#45475a")

    return fig


# ═══════════════════════════════════════════════════════════════
# MASTER VISUALIZATION RUNNER
# ═══════════════════════════════════════════════════════════════

def run_all_player_visualizations(
    X_umap: np.ndarray,
    labels: np.ndarray,
    metadata_df: pd.DataFrame,
    cluster_profiles: dict[int, dict],
    output_dir: str = "./output_players",
    show: bool = True,
) -> dict[str, go.Figure]:
    """Generate all 7 player archetype visualizations and save to HTML."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    figures = {}

    print("[viz] Generating UMAP archetype map...")
    figures["umap_archetype_map"] = plot_umap_archetype_map(
        X_umap, labels, metadata_df, cluster_profiles,
        title="NBA Player Archetype Map — Career Playing Styles"
    )

    print("[viz] Generating archetype radar chart...")
    figures["archetype_radar"] = plot_player_radar(cluster_profiles)

    print("[viz] Generating archetype size chart...")
    figures["archetype_sizes"] = plot_player_cluster_sizes(labels, cluster_profiles)

    print("[viz] Generating position distribution heatmap...")
    figures["position_distribution"] = plot_position_distribution(cluster_profiles)

    print("[viz] Generating HOF rate chart...")
    figures["hof_rate"] = plot_hof_rate(cluster_profiles)

    print("[viz] Generating feature importance chart...")
    figures["feature_importance"] = plot_top_features_by_cluster(cluster_profiles)

    for name, fig in tqdm(figures.items(), desc="Saving HTML"):
        path = os.path.join(output_dir, f"{name}.html")
        fig.write_html(path)
        print(f"[viz] Saved: {path}")

    if show:
        figures["umap_archetype_map"].show()

    print(f"\n[viz] ✅ {len(figures)} visualizations generated in {output_dir}/")
    return figures
