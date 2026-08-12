"use client";

/*
 * PlotlyChart — thin typed wrapper around plotly.js-dist-min.
 *
 * Why not react-plotly.js: it pins an older React peer range and adds
 * a class-component layer we don't need. This wrapper gives us
 * react()-based updates, proper cleanup, and typed event hooks in
 * ~80 lines, with the plotly bundle loaded only on the client.
 */

import { useEffect, useRef } from "react";
import type { Config, Layout, PlotData } from "plotly.js";
import { PLOT_CONFIG } from "@/lib/chart-theme";

// plotly.js-dist-min ships no types of its own; @types/plotly.js covers
// the API surface. The dynamic import keeps ~1MB of plotly out of SSR
// and out of the initial bundle.
type PlotlyModule = {
  react: (
    root: HTMLElement,
    data: Partial<PlotData>[],
    layout: Partial<Layout>,
    config?: Partial<Config>
  ) => Promise<unknown>;
  purge: (root: HTMLElement) => void;
};

let plotlyPromise: Promise<PlotlyModule> | null = null;

function loadPlotly(): Promise<PlotlyModule> {
  if (!plotlyPromise) {
    plotlyPromise = import("plotly.js-dist-min").then(
      (mod) => (mod.default ?? mod) as unknown as PlotlyModule
    );
  }
  return plotlyPromise;
}

export interface PlotlyClickPoint {
  curveNumber: number;
  pointIndex: number;
  customdata?: unknown;
}

interface PlotlyChartProps {
  data: Partial<PlotData>[];
  layout: Partial<Layout>;
  config?: Partial<Config>;
  onPointClick?: (point: PlotlyClickPoint) => void;
  className?: string;
  ariaLabel: string;
}

type PlotlyHTMLElement = HTMLDivElement & {
  on: (event: string, handler: (data: unknown) => void) => void;
  removeAllListeners?: (event: string) => void;
};

export default function PlotlyChart({
  data,
  layout,
  config,
  onPointClick,
  className,
  ariaLabel,
}: PlotlyChartProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  const clickRef = useRef(onPointClick);

  useEffect(() => {
    clickRef.current = onPointClick;
  }, [onPointClick]);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    let cancelled = false;

    loadPlotly().then((Plotly) => {
      if (cancelled || !rootRef.current) return;
      void Plotly.react(rootRef.current, data, layout, {
        ...PLOT_CONFIG,
        ...config,
      }).then(() => {
        if (cancelled || !rootRef.current) return;
        const el = rootRef.current as PlotlyHTMLElement;
        el.removeAllListeners?.("plotly_click");
        el.on("plotly_click", (evt: unknown) => {
          const e = evt as { points?: PlotlyClickPoint[] };
          const p = e.points?.[0];
          if (p && clickRef.current) clickRef.current(p);
        });
      });
    });

    return () => {
      cancelled = true;
    };
  }, [data, layout, config]);

  // Purge on unmount only — react() handles updates in place.
  useEffect(() => {
    const root = rootRef.current;
    return () => {
      if (root) loadPlotly().then((Plotly) => Plotly.purge(root));
    };
  }, []);

  return (
    <div
      ref={rootRef}
      className={className}
      role="img"
      aria-label={ariaLabel}
    />
  );
}
