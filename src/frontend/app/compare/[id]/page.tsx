"use client";

import { useQuery } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import { motion } from "motion/react";
import { searchPlayers } from "@/lib/api";
import type { SearchResponse } from "@/lib/types";
import Link from "next/link";
import { ArrowLeftIcon } from "@/lib/icons";
import { Tooltip } from "@/components/motion/tooltip";
import { PlayerAvatar, ScorePill, RankBadge } from "@/components/player";
import { getStatDefinition } from "@/lib/stat-definitions";

/*
 * Compare page — cinematic split-screen glass comparison.
 *
 * Layout:
 *   Back link → Query player hero → Match card (double-bezel, centered)
 *   → Match bar (spring-animated) → Feature contributions (glass bars)
 *   → All comparisons (numbered glass rows)
 *
 * The "match score" is the hero — large, centered, cinematic reveal.
 */

export default function ComparePage() {
  const { id } = useParams<{ id: string }>();
  const playerId = decodeURIComponent(id);
  const { data, isLoading, isError } = useQuery<SearchResponse>({
    queryKey: ["cmp", playerId],
    queryFn: () => searchPlayers(playerId, 10),
    enabled: !!playerId,
  });

  /* ── Loading ── */
  if (isLoading) {
    return (
      <div className="mx-auto max-w-225 px-5 pt-28 pb-24 space-y-2">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="result-row">
            <div className="w-5 h-4 skeleton" />
            <div className="w-10 h-10 rounded-full skeleton" />
            <div className="flex-1 space-y-2">
              <div className="h-4 w-36 skeleton" />
              <div className="h-3 w-20 skeleton" />
            </div>
            <div className="h-5 w-14 skeleton" />
          </div>
        ))}
      </div>
    );
  }

  /* ── Error ── */
  if (isError) {
    return (
      <div className="mx-auto max-w-225 px-5 pt-28 pb-24 text-center">
        <div className="glass-panel inline-block px-8 py-6">
          <p className="text-[15px] text-ink-soft mb-3">
            Couldn&apos;t load comparisons. The search service may be unavailable.
          </p>
          <Link href="/" className="btn-subtle inline-flex">
            <ArrowLeftIcon size={14} />
            Back to search
          </Link>
        </div>
      </div>
    );
  }

  /* ── Not found ── */
  if (!data || data.results.length === 0) {
    return (
      <div className="mx-auto max-w-225 px-5 pt-28 pb-24 text-center">
        <div className="glass-panel inline-block px-8 py-6">
          <p className="text-[15px] text-ink-muted mb-3">
            No comparisons available.
          </p>
          <Link
            href="/"
            className="btn-subtle inline-flex"
          >
            <ArrowLeftIcon size={14} />
            Back to search
          </Link>
        </div>
      </div>
    );
  }

  const q = data.query_entity;
  const t = data.results[0];
  const pct = (t.score * 100).toFixed(0);

  /* Largest single contributor gets the one amber fill (DESIGN.md Data Viz) */
  const maxContribution = Math.max(
    ...(t.top_contributing_features ?? []).map((f) => Math.abs(f.contribution)),
    0,
  );

  return (
    <div className="mx-auto max-w-225 px-5 pt-28 pb-24">
      {/* ── Back link ── */}
      <motion.div
        initial={{ opacity: 0, x: -8 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      >
        <Link
          href="/"
          className="btn-subtle mb-8 inline-flex"
        >
          <ArrowLeftIcon size={14} />
          Back
        </Link>
      </motion.div>

      {/* ── Hero: Query Player ── */}
      <motion.div
        initial={{ opacity: 0, y: 20, filter: "blur(4px)" }}
        animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
        transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
      >
        
        <h1 className="text-[clamp(2.4rem,5vw,4rem)] font-bold leading-[0.96] tracking-tight text-balance">
          {q.entity_name}
        </h1>
        <p className="mt-2 text-[14px] text-ink-muted">
          Closest stylistic comparisons
        </p>
      </motion.div>

      {/* ── Top Match Card — Double-Bezel, Cinematic ── */}
      <motion.div
        initial={{ opacity: 0, y: 24, filter: "blur(4px)" }}
        animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
        transition={{
          duration: 0.7,
          ease: [0.22, 1, 0.36, 1],
          delay: 0.10,
        }}
        className="mt-8"
      >
        <div className="glass-outer">
          <div className="glass-inner-elevated p-7">
            <div className="flex flex-col sm:flex-row sm:items-center gap-6">
              {/* Query player avatar */}
              <div className="flex items-center gap-4">
                <PlayerAvatar
                  name={q.entity_name}
                  entityId={q.entity_id}
                  highlight
                  className="w-14 h-14 text-[16px]"
                />
                <div>
                  <p className="text-[12px] font-semibold uppercase tracking-[0.08em] text-ink-muted mb-0.5">
                    Player
                  </p>
                  <p className="text-[18px] font-bold tracking-[-0.015em]">
                    {q.entity_name}
                  </p>
                </div>
              </div>

              {/* Central match score — Constellation line + score on midpoint */}
              <div className="flex-1 flex flex-col items-center justify-center py-2 relative">
                {/* SVG connecting line — draws in over 600ms */}
                <svg
                  className="absolute inset-0 w-full h-full pointer-events-none"
                  preserveAspectRatio="none"
                  viewBox="0 0 100 100"
                  aria-hidden="true"
                >
                  <motion.line
                    x1="0"
                    y1="50"
                    x2="100"
                    y2="50"
                    stroke="oklch(0.65 0.20 40 / 0.4)"
                    strokeWidth="0.5"
                    strokeDasharray="100"
                    initial={{ strokeDashoffset: 100 }}
                    animate={{ strokeDashoffset: 0 }}
                    transition={{
                      duration: 0.6,
                      ease: [0.22, 1, 0.36, 1],
                      delay: 0.15,
                    }}
                  />
                </svg>
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{
                    type: "spring",
                    stiffness: 200,
                    damping: 18,
                    delay: 0.55,
                  }}
                  className="relative z-10 px-4 bg-surface-raised"
                >
                  <span className="scoreboard text-primary">
                    {pct}
                    <span className="text-[0.4em]">%</span>
                  </span>
                </motion.div>
                <span className="text-[12px] font-semibold uppercase tracking-[0.08em] text-ink-muted mt-1 relative z-10">
                  Style Match
                </span>
              </div>

              {/* Match player avatar */}
              <div className="flex items-center gap-4 sm:flex-row-reverse sm:text-right">
                <PlayerAvatar
                  name={t.entity_name}
                  entityId={t.entity_id}
                  highlight
                  className="w-14 h-14 text-[16px]"
                />
                <div>
                  <p className="text-[12px] font-semibold uppercase tracking-[0.08em] text-ink-muted mb-0.5">
                    #{t.rank} Match
                  </p>
                  <p className="text-[18px] font-bold tracking-[-0.015em]">
                    {t.entity_name}
                  </p>
                </div>
              </div>
            </div>

            {/* Match progress bar */}
            <div className="mt-5 flex items-center gap-3">
              <span className="text-[11px] font-semibold text-ink-muted tabular-nums">
                0%
              </span>
              <div className="flex-1 h-2 rounded-full overflow-hidden bg-surface">
                <motion.div
                  className="h-full rounded-full bg-primary"
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{
                    type: "spring",
                    stiffness: 120,
                    damping: 16,
                    delay: 0.40,
                  }}
                />
              </div>
              <span className="text-[11px] font-semibold text-ink-muted tabular-nums">
                100%
              </span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* ── Feature Contributions ── */}
      {t.top_contributing_features?.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 16, filter: "blur(4px)" }}
          animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          transition={{
            duration: 0.7,
            ease: [0.22, 1, 0.36, 1],
            delay: 0.18,
          }}
          className="mt-8"
        >
          <h3 className="text-[16px] font-bold tracking-[-0.015em] mb-5">
            What drives the similarity
          </h3>

          <div className="glass-outer">
            <div className="glass-inner p-5 space-y-4">
              {t.top_contributing_features.map((f, i) => {
                const w = Math.min(Math.abs(f.contribution) * 100, 100);
                const isTop = Math.abs(f.contribution) === maxContribution;
                const definition = getStatDefinition(f.feature);
                const label = (
                  <span className="text-[13px] font-semibold tracking-[-0.005em] text-ink-soft capitalize">
                    {f.feature.replace(/_/g, " ")}
                  </span>
                );
                return (
                  <div key={f.feature}>
                    <div className="flex justify-between mb-1.5">
                      {definition ? (
                        <Tooltip
                          content={definition}
                          side="top"
                          className="w-max max-w-70 whitespace-normal text-pretty border-[oklch(1_0_0/0.1)] bg-[oklch(0.15_0.005_85/0.95)] px-3 py-2 leading-relaxed text-ink-soft backdrop-blur-xl"
                        >
                          <span className="cursor-help underline decoration-dotted decoration-[oklch(0.65_0.20_40/0.5)] underline-offset-4">
                            {label}
                          </span>
                        </Tooltip>
                      ) : (
                        label
                      )}
                      <span className="text-[12px] font-bold tabular-nums text-ink-muted">
                        {w.toFixed(0)}%
                      </span>
                    </div>
                    <div className="h-2 rounded-full overflow-hidden bg-surface">
                      <motion.div
                        className={
                          isTop
                            ? "h-full rounded-full bg-primary"
                            : "h-full rounded-full bg-ink-soft"
                        }
                        initial={{ width: 0 }}
                        animate={{ width: `${w}%` }}
                        transition={{
                          type: "spring",
                          stiffness: 160,
                          damping: 18,
                          delay: 0.30 + i * 0.05,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </motion.div>
      )}

      {/* ── All Comparisons ── */}
      <motion.div
        initial={{ opacity: 0, y: 16, filter: "blur(4px)" }}
        animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
        transition={{
          duration: 0.7,
          ease: [0.22, 1, 0.36, 1],
          delay: 0.26,
        }}
        className="mt-10"
      >
        <h3 className="text-[16px] font-bold tracking-[-0.015em] mb-4">
          All comparisons · {data.results.length}
        </h3>

        <div className="glass-outer">
          <div className="glass-inner p-1.5">
            {data.results.map((r, i) => {
              const ps =
                typeof r.metadata?.position === "string"
                  ? r.metadata.position
                  : null;
              const isFirst = r.rank === 1;

              return (
                <motion.div
                  key={r.entity_id}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{
                    type: "spring",
                    stiffness: 300,
                    damping: 26,
                    delay: i * 0.03,
                  }}
                >
                  <Link
                    href={`/compare/${r.entity_id}`}
                    className="result-row"
                    data-cursor="selectable"
                  >
                    <RankBadge rank={r.rank} highlight={isFirst} />
                    <PlayerAvatar name={r.entity_name} entityId={r.entity_id} highlight={isFirst} />

                    <div className="flex-1 min-w-0">
                      <p className="truncate text-[15px] font-semibold tracking-[-0.01em]">
                        {r.entity_name}
                      </p>
                      <div className="flex items-center gap-2 mt-0.5 text-[13px] text-ink-muted">
                        {ps && <span>{ps}</span>}
                      </div>
                    </div>

                    <ScorePill score={r.score} highlight={isFirst} className="shrink-0" />
                  </Link>
                </motion.div>
              );
            })}
          </div>
        </div>
      </motion.div>
    </div>
  );
}
