"use client";

/*
 * ArchetypeGraph — Obsidian-style similarity mind map.
 *
 * Search a player → they anchor the graph in amber. Their FAISS
 * k-nearest neighbors orbit them, colored by archetype cluster,
 * connected by edges whose width and opacity encode match strength
 * (distance is real, so it's drawn as distance). Click any node to
 * expand its own neighborhood into the graph.
 *
 * Layout: d3-force (charge + link-distance from similarity score).
 * Render: Plotly scattergl traces (edges as line segments, nodes as
 * markers + labels). Plotly has no physics engine, so d3 supplies
 * the geometry and Plotly supplies the interaction surface.
 */

import { useCallback, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCollide,
  forceX,
  forceY,
  type SimulationNodeDatum,
  type SimulationLinkDatum,
} from "d3";
import type { Layout, PlotData } from "plotly.js";
import { getPlayerGraph } from "@/lib/api";
import type { GraphEdge, GraphNode, PlayerGraph } from "@/lib/types";
import {
  BASE_LAYOUT,
  HIDDEN_AXIS,
  INK,
  INK_MUTED,
  PRIMARY,
  clusterColor,
} from "@/lib/chart-theme";
import PlotlyChart, { type PlotlyClickPoint } from "./PlotlyChart";
import PlayerSearchInput from "./PlayerSearchInput";

const DEFAULT_PLAYER = { entity_id: "jamesle01", entity_name: "LeBron James" };
const NEIGHBORS_PER_NODE = 10;

interface SimNode extends SimulationNodeDatum {
  id: string;
  node: GraphNode;
  expanded: boolean;
}

interface MergedGraph {
  nodes: Map<string, GraphNode>;
  edges: GraphEdge[];
  centerId: string;
  expandedIds: Set<string>;
}

function mergeGraphs(graphs: PlayerGraph[], centerId: string): MergedGraph {
  const nodes = new Map<string, GraphNode>();
  const edges: GraphEdge[] = [];
  const seenEdges = new Set<string>();
  const expandedIds = new Set<string>();

  for (const g of graphs) {
    expandedIds.add(g.center_id);
    for (const n of g.nodes) {
      const existing = nodes.get(n.entity_id);
      // Center flag belongs only to the root query player
      if (!existing) {
        nodes.set(n.entity_id, {
          ...n,
          is_center: n.entity_id === centerId,
        });
      }
    }
    for (const e of g.edges) {
      const key =
        e.source < e.target
          ? `${e.source}|${e.target}`
          : `${e.target}|${e.source}`;
      if (!seenEdges.has(key)) {
        seenEdges.add(key);
        edges.push(e);
      }
    }
  }
  return { nodes, edges, centerId, expandedIds };
}

/** Run d3-force synchronously and return settled positions. */
function computeLayout(
  merged: MergedGraph
): Map<string, { x: number; y: number }> {
  const simNodes: SimNode[] = [...merged.nodes.values()].map((n) => ({
    id: n.entity_id,
    node: n,
    expanded: merged.expandedIds.has(n.entity_id),
  }));
  const simLinks: (SimulationLinkDatum<SimNode> & { score: number })[] =
    merged.edges.map((e) => ({ source: e.source, target: e.target, score: e.score }));

  const sim = forceSimulation(simNodes)
    .force(
      "link",
      forceLink<SimNode, SimulationLinkDatum<SimNode> & { score: number }>(simLinks)
        .id((d) => d.id)
        // Higher similarity → shorter edge. Score ∈ [0,1].
        .distance((l) => 40 + (1 - l.score) * 260)
        .strength(0.6)
    )
    .force("charge", forceManyBody().strength(-320))
    .force("collide", forceCollide(26))
    .force("x", forceX(0).strength(0.04))
    .force("y", forceY(0).strength(0.04))
    .stop();

  // 300 ticks ≈ alpha decay to rest; deterministic enough at this size.
  for (let i = 0; i < 300; i++) sim.tick();

  const out = new Map<string, { x: number; y: number }>();
  for (const n of simNodes) out.set(n.id, { x: n.x ?? 0, y: n.y ?? 0 });
  return out;
}

export default function ArchetypeGraph({
  clusterNames,
}: {
  clusterNames: Map<number, string>;
}) {
  const [root, setRoot] = useState(DEFAULT_PLAYER);
  const [expandedIds, setExpandedIds] = useState<string[]>([]);
  const queryClient = useQueryClient();

  const allIds = useMemo(
    () => [root.entity_id, ...expandedIds],
    [root.entity_id, expandedIds]
  );

  const { data: graphs, isPending, isError } = useQuery<PlayerGraph[]>({
    queryKey: ["player-graph", allIds],
    queryFn: () =>
      Promise.all(allIds.map((id) => getPlayerGraph(id, NEIGHBORS_PER_NODE))),
    staleTime: 5 * 60_000,
    placeholderData: (prev) => prev,
  });

  const handleSelect = useCallback(
    (p: { entity_id: string; entity_name: string }) => {
      setRoot(p);
      setExpandedIds([]);
    },
    []
  );

  const merged = useMemo(
    () => (graphs ? mergeGraphs(graphs, root.entity_id) : null),
    [graphs, root.entity_id]
  );

  const { data, layout } = useMemo(() => {
    if (!merged) return { data: [], layout: {} };
    const positions = computeLayout(merged);

    // ── Edge segments, bucketed so width/opacity encode similarity ──
    const edgeTraces: Partial<PlotData>[] = [];
    const buckets: { min: number; width: number; alpha: number }[] = [
      { min: 0.9, width: 2.2, alpha: 0.5 },
      { min: 0.8, width: 1.4, alpha: 0.3 },
      { min: 0, width: 0.8, alpha: 0.16 },
    ];
    for (const b of buckets) {
      const xs: (number | null)[] = [];
      const ys: (number | null)[] = [];
      for (const e of merged.edges) {
        const inBucket =
          e.score >= b.min &&
          !buckets.some((o) => o.min > b.min && e.score >= o.min);
        if (!inBucket) continue;
        const s = positions.get(e.source);
        const t = positions.get(e.target);
        if (!s || !t) continue;
        xs.push(s.x, t.x, null);
        ys.push(s.y, t.y, null);
      }
      if (xs.length === 0) continue;
      edgeTraces.push({
        type: "scatter",
        mode: "lines",
        x: xs as number[],
        y: ys as number[],
        line: { color: `rgba(255,255,255,${b.alpha})`, width: b.width },
        hoverinfo: "skip",
      });
    }

    // ── Nodes ──
    const nodeList = [...merged.nodes.values()];
    const nx: number[] = [];
    const ny: number[] = [];
    const colors: string[] = [];
    const sizes: number[] = [];
    const lineColors: string[] = [];
    const lineWidths: number[] = [];
    const labels: string[] = [];
    const hover: string[] = [];
    const custom: string[] = [];

    for (const n of nodeList) {
      const pos = positions.get(n.entity_id);
      if (!pos) continue;
      const isCenter = n.entity_id === merged.centerId;
      const isExpanded = merged.expandedIds.has(n.entity_id);
      nx.push(pos.x);
      ny.push(pos.y);
      colors.push(isCenter ? PRIMARY : clusterColor(n.cluster_id));
      sizes.push(isCenter ? 22 : isExpanded ? 16 : 12);
      lineColors.push(isCenter ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.25)");
      lineWidths.push(isCenter ? 2 : 1);
      labels.push(n.entity_name);
      const cluster = clusterNames.get(n.cluster_id) ?? "Unclassified";
      hover.push(
        `<b>${n.entity_name}</b><br>${n.position}<br>${cluster}` +
          (isCenter ? "" : "<br><i>Click to expand</i>")
      );
      custom.push(n.entity_id);
    }

    const nodeTrace: Partial<PlotData> = {
      type: "scatter",
      mode: "text+markers",
      x: nx,
      y: ny,
      text: labels,
      textposition: "bottom center",
      textfont: { size: 10, color: INK_MUTED },
      marker: {
        size: sizes,
        color: colors,
        line: { color: lineColors, width: lineWidths },
      },
      customdata: custom,
      hovertext: hover,
      hovertemplate: "%{hovertext}<extra></extra>",
    };

    const chartLayout: Partial<Layout> = {
      ...BASE_LAYOUT,
      xaxis: { ...HIDDEN_AXIS },
      yaxis: { ...HIDDEN_AXIS, scaleanchor: "x" },
      font: { ...BASE_LAYOUT.font, color: INK },
    };

    return { data: [...edgeTraces, nodeTrace], layout: chartLayout };
  }, [merged, clusterNames]);

  const handlePointClick = useCallback(
    (p: PlotlyClickPoint) => {
      const id = typeof p.customdata === "string" ? p.customdata : null;
      if (!id || !merged) return;
      if (merged.expandedIds.has(id)) return; // already expanded
      // Prefetch so the layout transition feels immediate
      void queryClient.prefetchQuery({
        queryKey: ["player-graph-single", id],
        queryFn: () => getPlayerGraph(id, NEIGHBORS_PER_NODE),
      });
      setExpandedIds((prev) => (prev.includes(id) ? prev : [...prev, id]));
    },
    [merged, queryClient]
  );

  return (
    <div className="glass-outer">
      <div className="glass-inner p-4 sm:p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <PlayerSearchInput
            placeholder="Search a player to map"
            onSelect={handleSelect}
          />
          <p className="font-mono text-[11px] text-ink-muted tracking-wide">
            {merged
              ? `${merged.nodes.size} NODES · ${merged.edges.length} EDGES`
              : ""}
          </p>
        </div>

        {isError ? (
          <div className="h-105 flex items-center justify-center">
            <p className="text-[14px] text-ink-muted">
              Graph unavailable. Confirm the API is running, then search again.
            </p>
          </div>
        ) : isPending && !merged ? (
          <div
            className="h-105 rounded-xl bg-surface-raised animate-pulse"
            aria-label="Loading similarity graph"
          />
        ) : (
          <PlotlyChart
            data={data}
            layout={layout}
            onPointClick={handlePointClick}
            className="h-105 w-full"
            ariaLabel={`Similarity mind map centered on ${root.entity_name}. Nodes are players colored by archetype; edges connect similar players.`}
          />
        )}

        <p className="mt-3 text-[12px] text-ink-muted">
          <span className="text-primary font-medium">{root.entity_name}</span>
          {" anchors the map. Edge weight is match strength from the hybrid FAISS scorer. Click any node to expannd it, and see further connections. scroll to zoom, drag to pan, and double-click to reset the position on the map. To reset the extra nodes, search for the player again."}
        </p>
      </div>
    </div>
  );
}
