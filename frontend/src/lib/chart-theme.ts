/* ═══════════════════════════════════════════════════════════════
   Chart Theme — DESIGN.md tokens translated for Plotly
   ═══════════════════════════════════════════════════════════════
   Plotly's WebGL/SVG renderers don't parse oklch() strings, so the
   OKLCH design tokens are pre-converted to sRGB hex (computed via
   the CSS Color 4 reference conversion; see DESIGN.md palette).

   Rules honored here:
   - Chart family: fixed lightness/chroma band (L=0.70, C=0.10),
     hue is the only variable. No hue at or near 290°.
   - Amber (--primary) is never a category color. It marks exactly
     one thing per view: the selected / queried player.
   - HDBSCAN noise (-1) recedes to ink-tertiary.
═══════════════════════════════════════════════════════════════ */

import type { Config, Layout, LayoutAxis } from "plotly.js";

/** Category colors for the 12 HDBSCAN clusters, keyed by cluster_id. */
export const CLUSTER_COLORS: Record<number, string> = {
  0: "#40b1b7", // oklch(0.70 0.10 200) steel
  1: "#5cb28f", // oklch(0.70 0.10 165) teal
  2: "#54aad1", // oklch(0.70 0.10 230) indigo
  3: "#c287bc", // oklch(0.70 0.10 330) rose
  4: "#c99159", // oklch(0.70 0.10 65)  ochre
  5: "#76af77", // oklch(0.70 0.10 145) green
  6: "#d68585", // oklch(0.70 0.10 20)  clay
  7: "#a2a457", // oklch(0.70 0.10 110) olive
  8: "#6da3da", // oklch(0.70 0.10 250) sky
  9: "#d084a1", // oklch(0.70 0.10 355) blush
  10: "#bb9951", // oklch(0.70 0.10 85) sand
  11: "#46b3a6", // oklch(0.70 0.10 185) sea
};

/** HDBSCAN noise / unclassified — recedes, never competes. */
export const NOISE_COLOR = "#6f6e6d"; // oklch(0.54 0.002 85) ink-tertiary

/** The one accent: live, selected, or true right now. */
export const PRIMARY = "#ee560c"; // oklch(0.65 0.20 40)

export const INK = "#f2f2f2"; // oklch(0.96 0 0)
export const INK_SOFT = "#b1b1b0"; // oklch(0.76 0.002 85)
export const INK_MUTED = "#6f6e6d"; // oklch(0.54 0.002 85)

export function clusterColor(clusterId: number): string {
  return CLUSTER_COLORS[clusterId] ?? NOISE_COLOR;
}

/** Shared Plotly layout base: transparent, chromeless, mono hover. */
export const BASE_LAYOUT: Partial<Layout> = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: {
    family:
      "var(--font-geist-sans), 'Geist', system-ui, -apple-system, sans-serif",
    color: INK_SOFT,
    size: 12,
  },
  margin: { l: 0, r: 0, t: 0, b: 0 },
  showlegend: false,
  dragmode: "pan",
  hoverlabel: {
    bgcolor: "#161513", // glass overlay approximation
    bordercolor: "rgba(255,255,255,0.12)",
    font: {
      family:
        "var(--font-geist-mono), 'Geist Mono', ui-monospace, monospace",
      color: INK,
      size: 12,
    },
  },
};

/** Axes for UMAP space: raw axes are meaningless, so hide everything. */
export const HIDDEN_AXIS: Partial<LayoutAxis> = {
  visible: false,
  showgrid: false,
  zeroline: false,
  showticklabels: false,
  fixedrange: false,
};

export const PLOT_CONFIG: Partial<Config> = {
  displayModeBar: false,
  scrollZoom: true,
  doubleClick: "reset",
  responsive: true,
};
