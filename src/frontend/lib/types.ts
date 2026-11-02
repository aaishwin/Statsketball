/* ═══════════════════════════════════════════════════════════════
   API Types — mirrors backend Pydantic models
   ═══════════════════════════════════════════════════════════════ */

export interface SearchResult {
  entity_id: string;
  entity_name: string;
  score: number;
  cosine_similarity: number;
  rank: number;
  metadata: Record<string, unknown>;
  top_contributing_features: FeatureContribution[];
}

export interface FeatureContribution {
  feature: string;
  contribution: number;
  query_value: number;
  candidate_value: number;
}

export interface SearchResponse {
  query_id: string;
  query_entity: {
    entity_id: string;
    entity_name: string;
    entity_type: "player";
  };
  results: SearchResult[];
  total_candidates_searched: number;
  filters_applied: Record<string, string>;
  timing_ms: number;
  index_version: string;
  cache_hit: boolean;
}

export interface IndexInfo {
  entity_type: "player";
  version: string;
  built_at: string;
  dimension: number;
  vector_count: number;
  feature_names: string[];
}

export interface PlayerSuggestion {
  entity_id: string;
  entity_name: string;
  metadata: {
    position?: string;
    [key: string]: unknown;
  };
}

export interface ArchetypeCluster {
  cluster_id: number;
  cluster_name: string;
  size: number;
  description: string;
  key_traits: string[];
  example_players: string[];
}

export interface ArchetypeData {
  clusters: ArchetypeCluster[];
  players: PlayerArchetypeRecord[];
  total_players: number;
}

export interface PlayerArchetypeRecord {
  entity_id: string;
  entity_name: string;
  cluster_id: number;
  umap_x: number;
  umap_y: number;
  position: string;
  hof: boolean;
  debut_season: number;
  final_season: number;
}

export interface GraphNode {
  entity_id: string;
  entity_name: string;
  cluster_id: number;
  position: string;
  hof: boolean;
  is_center: boolean;
}

export interface GraphEdge {
  source: string;
  target: string;
  score: number;
}

export interface PlayerGraph {
  center_id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export type EntityType = "player";

export interface ApiError {
  error: string;
  detail: string;
  status_code: number;
}
