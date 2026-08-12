/* ═══════════════════════════════════════════════════════════════
   API Client — typed fetch wrapper for FastAPI backend
   ═══════════════════════════════════════════════════════════════ */

import type {
  SearchResponse,
  IndexInfo,
  EntityType,
  ApiError,
  PlayerSuggestion,
  ArchetypeData,
  PlayerGraph,
} from "./types";

// API origin resolution:
// 1. NEXT_PUBLIC_API_URL (inlined at build time) — explicit override.
// 2. Production builds default to the Render backend directly. Netlify's
//    Next.js runtime applies next.config.ts rewrites, but with the env var
//    unset they target http://localhost:8000 and fail with a 500, so the
//    client must not rely on the same-origin proxy in production. CORS on
//    the backend is configured for this site's origin.
// 3. Local dev uses the relative /api/v1 path through the next dev rewrite
//    to http://localhost:8000.
const API_BASE = `${
  process.env.NEXT_PUBLIC_API_URL ??
  (process.env.NODE_ENV === "production"
    ? "https://statsketball-api.onrender.com"
    : "")
}`.replace(/\/+$/, "") + "/api/v1";

export { API_BASE };

class ApiClientError extends Error {
  status: number;
  detail: ApiError;

  constructor(status: number, detail: ApiError) {
    super(detail.detail ?? detail.error);
    this.name = "ApiClientError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    ...options,
  });

  if (!res.ok) {
    let detail: ApiError;
    try {
      detail = await res.json();
    } catch {
      detail = {
        error: "UNKNOWN",
        detail: res.statusText,
        status_code: res.status,
      };
    }
    throw new ApiClientError(res.status, detail);
  }

  return res.json();
}

/* ── Search ── */

export async function suggestPlayers(
  q: string,
  limit = 10
): Promise<PlayerSuggestion[]> {
  const params = new URLSearchParams({ q, limit: String(limit) });
  const data = await request<{ query: string; suggestions: PlayerSuggestion[]; total: number }>(
    `/search/players?${params}`
  );
  return data.suggestions;
}

export async function searchPlayers(
  playerId: string,
  k = 10,
  filters?: { position?: string }
): Promise<SearchResponse> {
  const params = new URLSearchParams({ k: String(k) });
  if (filters?.position) params.set("position", filters.position);

  return request<SearchResponse>(
    `/search/player/${encodeURIComponent(playerId)}?${params}`
  );
}

/* ── Archetypes / Graph ── */

export async function getArchetypes(): Promise<ArchetypeData> {
  return request<ArchetypeData>("/archetypes");
}

export async function getPlayerGraph(
  playerId: string,
  k = 12
): Promise<PlayerGraph> {
  const params = new URLSearchParams({ k: String(k) });
  return request<PlayerGraph>(
    `/graph/player/${encodeURIComponent(playerId)}?${params}`
  );
}

/* ── Index Info ── */

export async function getIndexInfo(
  entityType: EntityType
): Promise<IndexInfo> {
  return request<IndexInfo>(`/index/info/${entityType}`);
}

export { ApiClientError };
