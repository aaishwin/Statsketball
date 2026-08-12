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

// Always use relative path so browser requests go through the Next.js
// rewrite proxy (configured in next.config.ts), which forwards /api/v1/*
// to the backend. This avoids CORS issues entirely.
const API_BASE = "/api/v1";

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
