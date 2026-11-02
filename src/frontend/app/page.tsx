"use client";

import { useState, useRef, useEffect, useCallback, useId } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "motion/react";
import { searchPlayers, suggestPlayers } from "@/lib/api";
import type { SearchResponse, PlayerSuggestion } from "@/lib/types";
import Link from "next/link";
import { SearchIcon, ArrowRightIcon } from "@/lib/icons";
import { WordReveal } from "@/components/word-reveal";
import { PlayerAvatar, ScorePill, RankBadge } from "@/components/player";

/*
 * Search Page — Courtside at midnight.
 *
 * Spotlight-style search pill with a real ARIA combobox, position
 * filter chips (wired to the backend's position params),
 * trending searches, and results: rank 1 gets the Scoreboard card,
 * the rest are numbered rows.
 */

const POSITIONS = ["PG", "SG", "SF", "PF", "C"];

const TRENDING = [
  { id: "jamesle01", name: "LeBron James" },
  { id: "curryst01", name: "Stephen Curry" },
  { id: "doncilu01", name: "Luka Dončić" },
  { id: "antetgi01", name: "Giannis Antetokounmpo" },
  { id: "jokicni01", name: "Nikola Jokić" },
];

export default function SearchPage() {
  const [queryId, setQueryId] = useState<string | null>(null);
  const [pos, setPos] = useState<string | null>(null);
  const listboxId = useId();

  // ── Autocomplete state ──
  const [inputValue, setInputValue] = useState("");
  const [debouncedInput, setDebouncedInput] = useState("");
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [activeSuggestion, setActiveSuggestion] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  // Debounce input for autocomplete
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedInput(inputValue.trim());
    }, 200);
    return () => clearTimeout(timer);
  }, [inputValue]);

  // Autocomplete query
  const { data: suggestions, isLoading: suggestionsLoading } = useQuery<PlayerSuggestion[]>({
    queryKey: ["suggest", debouncedInput],
    queryFn: () => suggestPlayers(debouncedInput, 8),
    enabled: debouncedInput.length >= 2 && showSuggestions,
    staleTime: 30_000,
  });

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setShowSuggestions(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const selectSuggestion = useCallback((s: PlayerSuggestion) => {
    setInputValue(s.entity_name);
    setQueryId(s.entity_id);
    setShowSuggestions(false);
    setActiveSuggestion(0);
  }, []);

  const suggestionsOpen = showSuggestions && debouncedInput.length >= 2;

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!suggestionsOpen || !suggestions || suggestions.length === 0) {
      if (e.key === "Escape") setShowSuggestions(false);
      return;
    }

    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveSuggestion((prev) =>
        prev < suggestions.length - 1 ? prev + 1 : 0
      );
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveSuggestion((prev) =>
        prev > 0 ? prev - 1 : suggestions.length - 1
      );
    } else if (e.key === "Enter" && suggestions[activeSuggestion]) {
      e.preventDefault();
      selectSuggestion(suggestions[activeSuggestion]);
    } else if (e.key === "Escape") {
      setShowSuggestions(false);
    }
  };

  const { data, isLoading, isError, refetch } = useQuery<SearchResponse>({
    queryKey: ["search", queryId, pos],
    queryFn: () =>
      searchPlayers(queryId!, 10, {
        position: pos ?? undefined,
      }),
    enabled: queryId !== null,
    retry: false,
  });

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (suggestionsOpen && suggestions && suggestions[activeSuggestion]) {
      selectSuggestion(suggestions[activeSuggestion]);
      return;
    }
    const v = inputValue.trim();
    if (v) {
      setQueryId(v);
      setShowSuggestions(false);
    }
  };

  const triggerTrending = (id: string, name: string) => {
    setInputValue(name);
    setQueryId(id);
    setShowSuggestions(false);
  };

  // Split results: rank 1 gets Scoreboard treatment, rest are rows
  const topResult = data?.results?.[0];
  const remainingResults = data?.results?.slice(1) ?? [];

  return (
    <div className="flex flex-col flex-1">
      {/* ═══ Hero Section ═══ */}
      <section className="px-5 pt-28 pb-12 sm:pt-36 sm:pb-16">
        <div className="mx-auto max-w-275">
          <div className="max-w-180">
            <motion.div
              initial={{ opacity: 0, y: 20, filter: "blur(4px)" }}
              animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
              transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
            >
              <h1 className="display-hero text-ink headline-ambient">
                <WordReveal text="Who really plays like" delay={0.1} />
                <br />
                <WordReveal text="LeBron?" delay={0.4} />
              </h1>

              <motion.p
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1], delay: 0.9 }}
                className="mt-5 max-w-120 text-[16px] leading-relaxed text-ink-soft"
              >
                Compare players by how they actually play. 
              </motion.p>
            </motion.div>

            {/* ── Search Pill — Spotlight-style, ARIA combobox ── */}
            <motion.div
              initial={{ opacity: 0, y: 16, filter: "blur(4px)" }}
              animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
              transition={{
                duration: 0.7,
                ease: [0.22, 1, 0.36, 1],
                delay: 0.10,
              }}
              className="mt-8 relative"
              ref={containerRef}
            >
              <form onSubmit={submit} role="search">
                <div className="search-pill">
                  <SearchIcon size={16} className="shrink-0 text-ink-muted" />
                  <input
                    type="text"
                    role="combobox"
                    aria-expanded={suggestionsOpen}
                    aria-controls={listboxId}
                    aria-activedescendant={
                      suggestionsOpen && suggestions?.[activeSuggestion]
                        ? `${listboxId}-${activeSuggestion}`
                        : undefined
                    }
                    aria-autocomplete="list"
                    aria-label="Search by player name"
                    value={inputValue}
                    onChange={(e) => {
                      setInputValue(e.target.value);
                      setShowSuggestions(true);
                      setActiveSuggestion(0);
                    }}
                    onFocus={() => {
                      if (inputValue.trim().length >= 2) {
                        setShowSuggestions(true);
                      }
                    }}
                    onKeyDown={handleKeyDown}
                    placeholder="Search by player name…"
                    className="search-pill-input"
                    autoComplete="off"
                    spellCheck={false}
                  />
                  <button type="submit" className="search-submit">
                    {isLoading ? "Searching…" : "Compare"}
                    <span className="search-submit-icon">
                      <ArrowRightIcon size={12} />
                    </span>
                  </button>
                </div>
              </form>

              {/* ── Autocomplete Dropdown — heavy glass, not clipped ── */}
              <AnimatePresence>
                {suggestionsOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: -4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    transition={{ duration: 0.15 }}
                    className="absolute left-0 right-0 top-full z-50 mt-2"
                  >
                    <div className="glass-inner-heavy overflow-hidden p-1">
                      {suggestionsLoading && (
                        <div className="px-4 py-3 text-[13px] text-ink-muted">
                          Searching…
                        </div>
                      )}
                      {!suggestionsLoading && suggestions && suggestions.length === 0 && (
                        <div className="px-4 py-3 text-[13px] text-ink-muted">
                          No players found. Try a different name.
                        </div>
                      )}
                      {suggestions && suggestions.length > 0 && (
                        <ul
                          id={listboxId}
                          role="listbox"
                          aria-label="Player suggestions"
                          className="max-h-80 overflow-y-auto"
                        >
                          {suggestions.map((s, i) => {
                            const sPos =
                              typeof s.metadata?.primary_position === "string"
                                ? s.metadata.primary_position
                                : typeof s.metadata?.position === "string"
                                  ? s.metadata.position
                                  : null;
                            return (
                              <li
                                key={s.entity_id}
                                id={`${listboxId}-${i}`}
                                role="option"
                                aria-selected={i === activeSuggestion}
                              >
                                <button
                                  type="button"
                                  onClick={() => selectSuggestion(s)}
                                  onMouseEnter={() => setActiveSuggestion(i)}
                                  className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-colors ${
                                    i === activeSuggestion
                                      ? "bg-white/8"
                                      : "hover:bg-white/5"
                                  }`}
                                >
                                  <PlayerAvatar name={s.entity_name} entityId={s.entity_id} size="sm" />
                                  <div className="flex-1 min-w-0">
                                    <p className="truncate text-[14px] font-semibold tracking-[-0.01em] text-ink">
                                      {s.entity_name}
                                    </p>
                                    <div className="flex items-center gap-1.5 mt-0.5 text-[12px] text-ink-muted">
                                      {sPos && <span>{sPos}</span>}
                                    </div>
                                  </div>
                                  <span className="text-[11px] font-mono text-ink-muted/50 shrink-0">
                                    {s.entity_id}
                                  </span>
                                </button>
                              </li>
                            );
                          })}
                        </ul>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* ── Filters — position chips, wired to the API ── */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2, duration: 0.5 }}
              className="mt-5 flex flex-wrap items-center gap-2"
              role="group"
              aria-label="Filter results"
            >
              {POSITIONS.map((p) => (
                <button
                  key={p}
                  type="button"
                  aria-pressed={pos === p}
                  onClick={() => setPos((prev) => (prev === p ? null : p))}
                  className={pos === p ? "chip chip-active" : "chip"}
                >
                  {p}
                </button>
              ))}
            </motion.div>

            {/* ── Trending Searches ── */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.30, duration: 0.5 }}
              className="mt-5 flex flex-wrap items-center gap-x-5 gap-y-2"
            >
              <span className="text-[12px] font-medium text-ink-disabled">
                Try
              </span>
              {TRENDING.map((t, i) => (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => triggerTrending(t.id, t.name)}
                  className="trending-chip"
                >
                  <span className="trending-chip-num">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  {t.name}
                  <ArrowRightIcon size={12} className="trending-chip-arrow" />
                </button>
              ))}
            </motion.div>
          </div>
        </div>
      </section>

      {/* ═══ Results Section ═══ */}
      <section className="flex-1 px-5 pb-32">
        <div className="mx-auto max-w-275">
          {/* No mode="wait": these states are mutually exclusive and the
              wait handoff can stall when other always-on animations (particle
              canvas, cursor) keep the tree busy, leaving content stuck at
              opacity 0. Plain AnimatePresence swaps immediately. */}
          <AnimatePresence>
            {/* ── Error State ── */}
            {isError && (
              <motion.div
                key="error"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.4 }}
                className="py-24 text-center"
              >
                <div className="glass-panel inline-block px-8 py-6 border-danger/30">
                  <p className="text-[15px] text-ink-soft mb-3">
                    Couldn&apos;t load results. The search service may be unavailable.
                  </p>
                  <button
                    type="button"
                    className="btn-subtle inline-flex"
                    onClick={() => refetch()}
                  >
                    Retry
                  </button>
                </div>
              </motion.div>
            )}

            {/* ── Empty State ── */}
            {!queryId && (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                className="py-24 text-center"
              >
                <p className="text-[15px] text-ink-muted">
                  Search for a player to see stylistic comparisons
                </p>
              </motion.div>
            )}

            {/* ── Loading Skeletons ── */}
            {isLoading && (
              <motion.div
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
                className="space-y-2 mt-2"
              >
                {Array.from({ length: 6 }).map((_, i) => (
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
              </motion.div>
            )}

            {/* ── Results ── */}
            {data && data.results.length > 0 && (
              <motion.div
                key="results"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
              >
                {/* Result header */}
                <div className="flex items-baseline justify-between mb-3 px-4">
                  <p className="text-[13px] text-ink-muted" aria-live="polite">
                    <span className="font-bold text-ink">
                      {data.results.length}
                    </span>{" "}
                    results{" "}
                   
                  </p>
                </div>

                {/* ── Rank 1: Scoreboard card ── */}
                {topResult && (
                  <motion.div
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{
                      type: "spring",
                      stiffness: 300,
                      damping: 26,
                    }}
                    className="mb-3"
                  >
                    <Link
                      href={`/compare/${topResult.entity_id}`}
                      className="glass-outer block scoreboard-card"
                      data-cursor="selectable"
                      aria-label={`Top match: ${topResult.entity_name}, ${(topResult.score * 100).toFixed(0)} percent similarity`}
                    >
                      <div className="glass-inner p-6 sm:p-8">
                        <div className="flex flex-col sm:flex-row sm:items-center gap-4 sm:gap-8">
                          {/* Scoreboard number */}
                          <div className="shrink-0">
                            <motion.div
                              initial={{ opacity: 0, scale: 0.9 }}
                              animate={{ opacity: 1, scale: 1 }}
                              transition={{
                                type: "spring",
                                stiffness: 200,
                                damping: 20,
                                delay: 0.4,
                              }}
                            >
                              <p className="scoreboard text-primary">
                                {(topResult.score * 100).toFixed(0)}
                                <span className="text-[0.4em]">%</span>
                              </p>
                              <p className="mt-1 text-[11px] font-semibold uppercase tracking-[0.12em] text-ink-muted">
                                Similarity in play style
                              </p>
                            </motion.div>
                          </div>

                          {/* Player identity */}
                          <div className="flex items-center gap-4 flex-1 min-w-0">
                            <PlayerAvatar
                              name={topResult.entity_name}
                              entityId={topResult.entity_id}
                              size="lg"
                              highlight
                            />
                            <div className="min-w-0">
                              <p className="truncate text-[20px] font-bold tracking-[-0.02em] text-ink">
                                {topResult.entity_name}
                              </p>
                              <div className="flex items-center gap-2 mt-0.5 text-[13px] text-ink-muted">
                                {typeof topResult.metadata?.position === "string" && (
                                  <span>{topResult.metadata.position as string}</span>
                                )}
                              </div>
                            </div>
                          </div>

                          <RankBadge rank={1} highlight className="shrink-0" />
                        </div>
                      </div>
                    </Link>
                  </motion.div>
                )}

                {/* ── Remaining results — numbered rows ── */}
                {remainingResults.length > 0 && (
                  <div className="glass-outer">
                    <div className="glass-inner p-1.5">
                      {remainingResults.map((r, i) => {
                        const posS =
                          typeof r.metadata?.position === "string"
                            ? r.metadata.position
                            : null;
                        return (
                          <motion.div
                            key={r.entity_id}
                            initial={{ opacity: 0, y: 8 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{
                              type: "spring",
                              stiffness: 300,
                              damping: 26,
                              delay: (i + 1) * 0.035,
                            }}
                          >
                            <Link
                              href={`/compare/${r.entity_id}`}
                              className="result-row"
                              data-cursor="selectable"
                            >
                              <RankBadge rank={r.rank} />
                              <PlayerAvatar name={r.entity_name} entityId={r.entity_id} />

                              <div className="flex-1 min-w-0">
                                <p className="truncate text-[15px] font-semibold tracking-[-0.01em]">
                                  {r.entity_name}
                                </p>
                                <div className="flex items-center gap-2 mt-0.5 text-[13px] text-ink-muted">
                                  {posS && <span>{posS}</span>}
                                </div>
                              </div>

                              <ScorePill score={r.score} className="shrink-0" />
                              <span className="score-bar-track" aria-hidden>
                                <span
                                  className="score-bar-fill"
                                  style={{ width: `${(r.score * 100).toFixed(0)}%` }}
                                />
                              </span>
                            </Link>
                          </motion.div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </motion.div>
            )}

            {/* ── No Results ── */}
            {data && data.results.length === 0 && !isLoading && (
              <motion.div
                key="none"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.4 }}
                className="py-24 text-center"
              >
                <p className="text-[15px] text-ink-muted">
                  No matches above threshold. Try a different player.
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </section>
    </div>
  );
}
