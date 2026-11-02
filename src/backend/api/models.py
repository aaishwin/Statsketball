"""
API Models for FAISS Similarity Search
=======================================
Pydantic v2 request/response models for the FastAPI search endpoints.

Design (from faiss_similarity_search_design.md §4):
- SearchResult, SearchResponse, RawVectorQuery
- RebuildRequest/Response, IndexInfoResponse
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from enum import Enum


class EntityType(str, Enum):
    PLAYER = "player"


class SearchResult(BaseModel):
    """A single similarity search result."""
    entity_id: str = Field(..., description="Player ID")
    entity_name: str = Field(..., description="Display name")
    score: float = Field(..., description="Hybrid similarity score (cosine ∈ [0,1])")
    cosine_similarity: float = Field(..., description="Raw cosine similarity from FAISS")
    rank: int = Field(..., ge=1)
    metadata: dict = Field(default_factory=dict, description="Era, position, season, etc.")
    top_contributing_features: list[dict] = Field(
        default_factory=list,
        description="Top-5 features driving this similarity, each with feature, contribution, query_value, candidate_value",
    )


class SearchResponse(BaseModel):
    """Standard search response wrapper."""
    query_id: str = Field(..., description="Unique query ID for tracing")
    query_entity: dict = Field(..., description="{entity_id, entity_name, entity_type}")
    results: list[SearchResult]
    total_candidates_searched: int
    filters_applied: dict = Field(default_factory=dict)
    timing_ms: float = Field(..., description="Total server-side search time in ms")
    index_version: str = Field(..., description="SHA of the active index for debuggability")
    cache_hit: bool = Field(default=False)


class ArchetypeClusterModel(BaseModel):
    """One HDBSCAN archetype cluster summary."""
    cluster_id: int = Field(..., description="HDBSCAN label; -1 is noise/unclassified")
    cluster_name: str
    size: int = Field(..., ge=0)
    description: str
    key_traits: list[str] = Field(default_factory=list)
    example_players: list[str] = Field(default_factory=list)


class PlayerArchetypePoint(BaseModel):
    """One player's coordinates in UMAP archetype space."""
    entity_id: str
    entity_name: str
    cluster_id: int
    umap_x: float
    umap_y: float
    position: str
    hof: bool
    debut_season: int
    final_season: int


class ArchetypeDataResponse(BaseModel):
    """Full archetype dataset: cluster summaries + all player points."""
    clusters: list[ArchetypeClusterModel]
    players: list[PlayerArchetypePoint]
    total_players: int


class GraphNode(BaseModel):
    """A node in the player similarity graph."""
    entity_id: str
    entity_name: str
    cluster_id: int
    position: str
    hof: bool
    is_center: bool = Field(default=False, description="True for the queried player")


class GraphEdge(BaseModel):
    """A similarity edge between two players."""
    source: str = Field(..., description="entity_id of the query player")
    target: str = Field(..., description="entity_id of the similar player")
    score: float = Field(..., description="Hybrid similarity score in [0,1]")


class PlayerGraphResponse(BaseModel):
    """k-NN neighborhood of one player as a graph."""
    center_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class RawVectorQuery(BaseModel):
    """Query with a raw embedding vector — for programmatic consumers."""
    vector: list[float] = Field(..., min_length=1, max_length=200)
    k: int = Field(default=10, ge=1, le=100)
    entity_type: EntityType
    normalize: bool = Field(default=True, description="L2-normalize input before search")

    @field_validator("vector")
    @classmethod
    def check_finite(cls, v):
        import math
        if any(not math.isfinite(x) for x in v):
            raise ValueError("Vector contains non-finite values (NaN or Inf)")
        return v


class BlockWeights(BaseModel):
    """
    Per-block weight multipliers for hybrid scoring.
    Only the specified blocks are adjusted; unspecified blocks use weight=1.0.
    """
    scoring: float = Field(default=1.0, ge=0.0, le=5.0, description="Scoring profile weight")
    playmaking: float = Field(default=1.0, ge=0.0, le=5.0, description="Playmaking weight")
    defense: float = Field(default=1.0, ge=0.0, le=5.0, description="Defense weight")
    rebounding: float = Field(default=1.0, ge=0.0, le=5.0, description="Rebounding weight")
    shooting: float = Field(default=1.0, ge=0.0, le=5.0, description="Shooting/spacing weight")
    advanced: float = Field(default=1.0, ge=0.0, le=5.0, description="Advanced metrics weight")


class RebuildRequest(BaseModel):
    """Trigger an index rebuild."""
    entity_type: EntityType
    force: bool = Field(default=False, description="Rebuild even if no data has changed")


class RebuildResponse(BaseModel):
    """Result of an index rebuild."""
    status: Literal["accepted", "rejected", "completed"]
    index_version: str
    build_time_ms: float
    vector_count: int
    dimension: int


class IndexInfoResponse(BaseModel):
    """Metadata about the currently active index."""
    entity_type: EntityType
    index_version: str
    index_type: str = Field(..., description="e.g., 'IndexFlatIP'")
    vector_count: int
    dimension: int
    built_at: str = Field(..., description="ISO 8601 timestamp")
    memory_bytes: int
    last_data_update: Optional[str] = Field(default=None)


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str
    status_code: int


class PlayerSuggestion(BaseModel):
    """A single player suggestion for autocomplete."""
    entity_id: str = Field(..., description="Player entity ID (e.g., 'jamesle01')")
    entity_name: str = Field(..., description="Display name")
    metadata: dict = Field(default_factory=dict, description="Position, era, etc.")


class PlayerSuggestionResponse(BaseModel):
    """Response for player autocomplete/suggest endpoint."""
    query: str = Field(..., description="The query string")
    suggestions: list[PlayerSuggestion]
    total: int


class HeadshotMapResponse(BaseModel):
    """Bulk headshot URL mapping for all known players."""
    headshots: dict[str, str] = Field(
        ..., description="Player name → CDN headshot URL (1040x760)"
    )
    total: int = Field(..., description="Number of headshot entries")
