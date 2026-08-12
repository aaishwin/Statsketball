/**
 * NBA player headshot URL helpers.
 *
 * The backend runs a Scrapy-Playwright spider (see `backend/app/scraping/`)
 * that scrapes https://www.nba.com/players (with "Show Historic" enabled)
 * and extracts every player's headshot <img src> URL from the rendered
 * roster table. The resulting mapping ({ player_name → CDN URL }) is served
 * via `GET /api/v1/headshots`.
 *
 * This module fetches the full mapping once at app load, caches it in a
 * module-level Map, and provides a synchronous lookup function. Name
 * normalization (diacritic stripping, suffix removal, case-folding) is
 * performed on both sides so that "Goran Dragić" matches "Goran Dragic".
 *
 * If the backend is unavailable or the mapping hasn't been generated yet,
 * `playerHeadshotUrl` returns `undefined` and the Avatar component falls
 * back to initials gracefully.
 */

import { useSyncExternalStore } from "react";

import { API_BASE } from "./api";

/** Normalized player name → headshot URL. */
let headshotCache: Map<string, string> | null = null;

/** In-flight fetch promise (prevents duplicate requests). */
let fetchPromise: Promise<Map<string, string>> | null = null;

/**
 * Normalize a player name for matching.
 *
 * Strips diacritics (é→e, č→c, ş→s), lowercases, removes generational
 * suffixes (Jr., Sr., III), removes periods/apostrophes, and collapses
 * whitespace. Mirrors the backend normalization in `headshot_store.py`.
 */
function normalizeName(name: string): string {
  return name
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "") // strip combining marks
    .toLowerCase()
    .trim()
    .replace(/\s+(jr|sr|ii|iii|iv)\.?$/, "")
    .replace(/[.'"]/g, "")
    .replace(/\s+/g, " ");
}

/**
 * Fetch the headshot mapping from the backend and cache it.
 *
 * Called automatically on first `playerHeadshotUrl` lookup. Safe to call
 * multiple times — the result is cached and subsequent calls return the
 * cached Map immediately.
 */
async function ensureHeadshotsLoaded(): Promise<Map<string, string>> {
  if (headshotCache) return headshotCache;
  if (fetchPromise) return fetchPromise;

  fetchPromise = fetch(`${API_BASE}/headshots`)
    .then((res) => {
      if (!res.ok) {
        throw new Error(`Headshot endpoint returned ${res.status}`);
      }
      return res.json();
    })
    .then((data: { headshots: Record<string, string> }) => {
      const map = new Map<string, string>();
      for (const [name, url] of Object.entries(data.headshots)) {
        map.set(normalizeName(name), url);
      }
      headshotCache = map;
      return map;
    })
    .catch((err) => {
      console.warn("[headshots] Failed to load headshot mapping:", err);
      // Cache an empty map so we don't retry on every lookup
      headshotCache = new Map();
      return headshotCache;
    })
    .finally(() => {
      fetchPromise = null;
    });

  return fetchPromise;
}

/**
 * Get a player's headshot URL by name.
 *
 * Returns `undefined` if:
 *   - The mapping hasn't been loaded yet (first call triggers async fetch)
 *   - The player name doesn't match any entry in the mapping
 *   - The backend is unavailable
 *
 * The Avatar component handles `undefined` by rendering initials.
 *
 * NOTE: The first call triggers an async fetch. If the mapping isn't loaded
 * yet, this returns `undefined` for that render cycle. The component will
 * re-render with the headshot once the fetch completes (see `useEffect` in
 * components that call this).
 */
export function playerHeadshotUrl(name: string): string | undefined {
  if (!headshotCache) {
    // Trigger the fetch (fire-and-forget); return undefined for this cycle
    void ensureHeadshotsLoaded();
    return undefined;
  }
  return headshotCache.get(normalizeName(name));
}

/**
 * Pre-load the headshot mapping. Call this in a top-level layout effect
 * to ensure headshots are available before the first avatar renders.
 */
export async function preloadHeadshots(): Promise<void> {
  await ensureHeadshotsLoaded();
}

/**
 * Check whether the headshot mapping has been loaded.
 * Useful for components that need to re-render after the initial fetch.
 */
export function headshotsLoaded(): boolean {
  return headshotCache !== null;
}

/**
 * React hook: returns a headshot URL for the given player name, or
 * `undefined` while the mapping is loading. Triggers a re-render when
 * the mapping finishes loading.
 *
 * Usage:
 *   const url = useHeadshotUrl("LeBron James");
 */
export function useHeadshotUrl(name: string): string | undefined {
  return useSyncExternalStore(
    subscribeHeadshotStore,
    () => playerHeadshotUrl(name) ?? null,
    () => null,
  ) ?? undefined;
}

// ── External store for useSyncExternalStore ──

const headshotListeners: Set<() => void> = new Set();

function subscribeHeadshotStore(callback: () => void): () => void {
  // If already loaded, no need to subscribe
  if (headshotCache) {
    return () => {};
  }
  headshotListeners.add(callback);
  // Trigger fetch if not already in progress
  void ensureHeadshotsLoaded().then(() => {
    // Notify all listeners that the cache is now populated
    headshotListeners.forEach((cb) => cb());
    headshotListeners.clear();
  });
  return () => {
    headshotListeners.delete(callback);
  };
}
