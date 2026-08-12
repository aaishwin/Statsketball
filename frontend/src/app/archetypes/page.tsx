"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "motion/react";
import dynamic from "next/dynamic";
import { getArchetypes } from "@/lib/api";
import type { ArchetypeData } from "@/lib/types";

/*
 * Archetypes page — live archetype exploration on real clustering data.
 *
 * Structure:
 *   Hero — heading + description
 *   Stats strip — live index metadata in a glass bar
 *   Similarity Graph — Obsidian-style mind map of FAISS k-NN neighborhoods
 *   Archetype Map — full UMAP projection, cluster chips, player search
 *
 * Charts are client-only (Plotly is a browser library), loaded via
 * next/dynamic so the ~1MB plotly bundle never blocks first paint.
 */

const ArchetypeGraph = dynamic(
  () => import("@/components/charts/ArchetypeGraph"),
  {
    ssr: false,
    loading: () => (
      <div className="glass-outer">
        <div className="glass-inner p-5">
          <div className="h-105 rounded-xl bg-surface-raised animate-pulse" />
        </div>
      </div>
    ),
  }
);

const UmapExplorer = dynamic(
  () => import("@/components/charts/UmapExplorer"),
  {
    ssr: false,
    loading: () => (
      <div className="glass-outer">
        <div className="glass-inner p-5">
          <div className="h-120 rounded-xl bg-surface-raised animate-pulse" />
        </div>
      </div>
    ),
  }
);

const EASE = [0.22, 1, 0.36, 1] as const;

export default function ArchetypesPage() {
  const {
    data: archetypes,
    isPending: archetypesPending,
    isError: archetypesError,
  } = useQuery<ArchetypeData>({
    queryKey: ["archetypes"],
    queryFn: getArchetypes,
    staleTime: 10 * 60_000,
  });

  const clusterNames = useMemo(() => {
    const m = new Map<number, string>();
    for (const c of archetypes?.clusters ?? []) m.set(c.cluster_id, c.cluster_name);
    return m;
  }, [archetypes?.clusters]);

  return (
    <div className="flex flex-col flex-1">
      <section className="px-5 pt-28 pb-24 sm:pt-36 sm:pb-32">
        <div className="mx-auto max-w-240">
          {/* ── Hero ── */}
          <motion.div
            initial={{ opacity: 0, y: 20, filter: "blur(4px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 0.7, ease: EASE }}
          >
            <h1 className="display-hero text-ink">
              Archetypes
            </h1>
            <p className="mt-3 max-w-145 text-[16px] leading-relaxed text-ink-soft">
              UMAP projection and HDBSCAN clustering help us find natural player
              archetypes and playstyles across NBA history. Instead of positions, or box
              scores, the model reads how players actually play. Looking at things like the shots
              they take, how often they pass, where they score, how they
              defend, whether they take charges, etc.
            </p>
          </motion.div>


          {/* ── Similarity Graph — the mind map ── */}
          <motion.div
            initial={{ opacity: 0, y: 16, filter: "blur(4px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{
              duration: 0.7,
              ease: EASE,
              delay: 0.14,
            }}
            className="mt-12"
          >
            <h2 className="text-[18px] font-bold tracking-[-0.015em] mb-2">
              Similarity Graph
            </h2>
            <p className="mb-5 max-w-145 text-[14px] leading-relaxed text-ink-soft">
              Every edge is a real nearest neighbor match from the search
              index. Search any player, then click on the other players next to them to see more neighbours. 
            </p>
            <ArchetypeGraph clusterNames={clusterNames} />
          </motion.div>

          {/* ── Archetype Map — full UMAP projection ── */}
          <motion.div
            initial={{ opacity: 0, y: 16, filter: "blur(4px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{
              duration: 0.7,
              ease: EASE,
              delay: 0.2,
            }}
            className="mt-12"
          >
            <h2 className="text-[18px] font-bold tracking-[-0.015em] mb-2">
              Archetype Map
            </h2>
            <p className="mb-5 max-w-145 text-[14px] leading-relaxed text-ink-soft">
              {archetypes
                ? `All ${archetypes.total_players.toLocaleString()} qualifying players in one projection. Pick an archetype to isolate it, or search a player to place them in the space.`
                : "The full player space, projected to two dimensions."}
            </p>

            {archetypesError ? (
              <div className="glass-outer">
                <div className="glass-inner p-8 text-center">
                  <p className="text-[14px] text-ink-soft">
                    Archetype data is offline. Start the API and refresh to
                    load the map.
                  </p>
                </div>
              </div>
            ) : archetypesPending || !archetypes ? (
              <div className="glass-outer">
                <div className="glass-inner p-5">
                  <div className="h-120 rounded-xl bg-surface-raised animate-pulse" />
                </div>
              </div>
            ) : (
              <UmapExplorer archetypes={archetypes} />
            )}
          </motion.div>

        </div>
      </section>
    </div>
  );
}
