"use client";

/*
 * UmapExplorer — the full UMAP projection of all ~1,790 players.
 *
 * Every player is a point; stylistic similarity is spatial distance.
 * Clusters carry chart-family colors; HDBSCAN noise recedes to
 * ink-tertiary. Amber appears exactly once: on the searched player.
 * Cluster chips isolate one archetype — the rest dim rather than
 * vanish, so the shape of the space stays legible.
 *
 * Rendered as scattergl (WebGL) — 1,790 points at 60fps.
 * Axes are hidden on purpose: UMAP dimensions are artifacts of the
 * reduction, not stats (DESIGN.md, Data Visualization).
 */

import { useCallback, useMemo, useState } from "react";
import type { Layout, PlotData } from "plotly.js";
import type { ArchetypeData } from "@/lib/types";
import {
  BASE_LAYOUT,
  HIDDEN_AXIS,
  INK,
  NOISE_COLOR,
  PRIMARY,
  clusterColor,
} from "@/lib/chart-theme";
import PlotlyChart from "./PlotlyChart";
import PlayerSearchInput from "./PlayerSearchInput";

interface UmapExplorerProps {
  archetypes: ArchetypeData;
}

export default function UmapExplorer({ archetypes }: UmapExplorerProps) {
  const [activeCluster, setActiveCluster] = useState<number | null>(null);
  const [highlighted, setHighlighted] = useState<{
    entity_id: string;
    entity_name: string;
  } | null>(null);

  const clusterNames = useMemo(() => {
    const m = new Map<number, string>();
    for (const c of archetypes.clusters) m.set(c.cluster_id, c.cluster_name);
    return m;
  }, [archetypes.clusters]);

  const { data, layout } = useMemo(() => {
    const players = archetypes.players;

    const xs: number[] = [];
    const ys: number[] = [];
    const colors: string[] = [];
    const sizes: number[] = [];
    const hover: string[] = [];

    const highlightedPoint = highlighted
      ? players.find((p) => p.entity_id === highlighted.entity_id) ?? null
      : null;

    for (const p of players) {
      if (highlightedPoint && p.entity_id === highlightedPoint.entity_id) {
        continue; // drawn in its own amber trace on top
      }
      const base = clusterColor(p.cluster_id);
      const dimmed =
        (activeCluster !== null && p.cluster_id !== activeCluster) ||
        highlightedPoint !== null;
      xs.push(p.umap_x);
      ys.push(p.umap_y);
      colors.push(dimmed ? "rgba(111,110,109,0.25)" : base);
      sizes.push(dimmed ? 4 : 6);
      hover.push(
        `<b>${p.entity_name}</b><br>${p.position}` +
          `<br>${clusterNames.get(p.cluster_id) ?? "Unclassified"}` +
          `<br><span style="color:#6f6e6d">${p.debut_season}–${p.final_season}${p.hof ? " · HOF" : ""}</span>`
      );
    }

    const traces: Partial<PlotData>[] = [
      {
        type: "scattergl",
        mode: "markers",
        x: xs,
        y: ys,
        marker: { size: sizes, color: colors, line: { width: 0 } },
        hovertext: hover,
        hovertemplate: "%{hovertext}<extra></extra>",
      },
    ];

    if (highlightedPoint) {
      traces.push({
        type: "scattergl",
        mode: "text+markers",
        x: [highlightedPoint.umap_x],
        y: [highlightedPoint.umap_y],
        text: [highlightedPoint.entity_name],
        textposition: "top center",
        textfont: { size: 12, color: INK },
        marker: {
          size: 16,
          color: PRIMARY,
          line: { color: "rgba(255,255,255,0.9)", width: 2 },
        },
        hovertext: [
          `<b>${highlightedPoint.entity_name}</b><br>${highlightedPoint.position}<br>${clusterNames.get(highlightedPoint.cluster_id) ?? "Unclassified"}`,
        ],
        hovertemplate: "%{hovertext}<extra></extra>",
      });
    }

    const chartLayout: Partial<Layout> = {
      ...BASE_LAYOUT,
      xaxis: { ...HIDDEN_AXIS },
      yaxis: { ...HIDDEN_AXIS, scaleanchor: "x" },
    };

    return { data: traces, layout: chartLayout };
  }, [archetypes.players, activeCluster, highlighted, clusterNames]);

  const handleSelect = useCallback(
    (p: { entity_id: string; entity_name: string }) => {
      setHighlighted(p);
      setActiveCluster(null);
    },
    []
  );

  const sortedClusters = useMemo(
    () =>
      [...archetypes.clusters].sort((a, b) =>
        a.cluster_id === -1 ? 1 : b.cluster_id === -1 ? -1 : b.size - a.size
      ),
    [archetypes.clusters]
  );

  return (
    <div className="glass-outer">
      <div className="glass-inner p-4 sm:p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <PlayerSearchInput
            placeholder="Find a player in the map"
            onSelect={handleSelect}
          />
          {highlighted && (
            <button
              type="button"
              onClick={() => setHighlighted(null)}
              className="font-mono text-[11px] text-ink-muted hover:text-ink transition-colors duration-150 tracking-wide"
            >
              CLEAR SELECTION ✕
            </button>
          )}
        </div>

        {/* Cluster filter chips — filter, never sort */}
        <div
          className="flex flex-wrap gap-1.5 mb-4"
          role="group"
          aria-label="Filter by archetype"
        >
          {sortedClusters.map((c) => {
            const active = activeCluster === c.cluster_id;
            const swatch =
              c.cluster_id === -1 ? NOISE_COLOR : clusterColor(c.cluster_id);
            return (
              <button
                key={c.cluster_id}
                type="button"
                aria-pressed={active}
                onClick={() => {
                  setActiveCluster(active ? null : c.cluster_id);
                  setHighlighted(null);
                }}
                className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[12px] font-medium transition-all duration-300 ease-[cubic-bezier(0.22,1,0.36,1)] border ${
                  active
                    ? "border-primary/60 bg-primary/15 text-ink"
                    : "border-border bg-transparent text-ink-soft hover:text-ink hover:bg-[oklch(1_0_0/0.04)]"
                }`}
              >
                <span
                  className="w-2 h-2 rounded-full shrink-0"
                  style={{ background: swatch }}
                  aria-hidden
                />
                {c.cluster_name}
                <span className="font-mono text-[10px] text-ink-muted tabular-nums">
                  {c.size}
                </span>
              </button>
            );
          })}
        </div>

        <PlotlyChart
          data={data}
          layout={layout}
          className="h-120 w-full"
          ariaLabel={`UMAP projection of ${archetypes.total_players} players. Each point is a player positioned by stylistic similarity and colored by archetype cluster.`}
        />

        <p className="mt-3 text-[12px] text-ink-muted">
          Each point is a player. Distance is determined by how similar their playstyle is, not their volume. 
          You should not focus on the axes, but instead focus on the clusters.
        </p>
      </div>
    </div>
  );
}
