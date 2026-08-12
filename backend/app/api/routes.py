"""
FastAPI Routes for FAISS Similarity Search
===========================================
Endpoints for player-to-player similarity search,
raw vector search, and index management.

Design (from faiss_similarity_search_design.md §4):
- GET  /search/player/{player_id}  — player similarity search
- POST /search/query               — raw vector search
- POST /index/rebuild              — trigger index rebuild
- GET  /index/info/{entity_type}   — index metadata
"""

import logging
from fastapi import APIRouter, HTTPException, Query, Path, Depends, Request
from fastapi.concurrency import run_in_threadpool
from typing import Optional

from .models import (
    SearchResponse,
    SearchResult,
    RawVectorQuery,
    RebuildRequest,
    RebuildResponse,
    IndexInfoResponse,
    ErrorResponse,
    PlayerSuggestion,
    PlayerSuggestionResponse,
    ArchetypeClusterModel,
    PlayerArchetypePoint,
    ArchetypeDataResponse,
    GraphNode,
    GraphEdge,
    PlayerGraphResponse,
    HeadshotMapResponse,
    EntityType as EntityTypeEnum,
)
from .archetype_store import get_store, ArchetypeStore, PlayerPoint
from ..faiss_index import service
from ..faiss_index.service import (
    EntityType,
    acquire_rebuild_lock,
    release_rebuild_lock,
)
from ..security import require_admin_key, log_and_generic_503, service_unavailable, not_found, bad_request
from ..rate_limit import limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["similarity-search"])


# ═══════════════════════════════════════════════════════════════
# PLAYER AUTOCOMPLETE / SUGGEST
# ═══════════════════════════════════════════════════════════════

@router.get(
    "/search/players",
    response_model=PlayerSuggestionResponse,
    responses={503: {"model": ErrorResponse}},
)
async def suggest_players(
    q: str = Query(..., min_length=1, max_length=100, description="Player name or ID prefix"),
    limit: int = Query(default=10, ge=1, le=50, description="Max suggestions"),
):
    """
    Autocomplete endpoint for player search.

    Returns matching players by prefix or substring match on name.
    Used by the frontend search bar typeahead.
    """
    try:
        suggestions = service.suggest_players(q, limit=limit)
        return PlayerSuggestionResponse(
            query=q,
            suggestions=[PlayerSuggestion(**s) for s in suggestions],
            total=len(suggestions),
        )
    except RuntimeError as e:
        raise log_and_generic_503(
            e, "SERVICE_UNAVAILABLE", "Player suggestion service is unavailable."
        )


# ═══════════════════════════════════════════════════════════════
# SEARCH ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@router.get(
    "/search/player/{player_id}",
    response_model=SearchResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def search_similar_players(
    player_id: str = Path(..., description="Player entity ID (e.g., 'jamesle01')"),
    k: int = Query(default=10, ge=1, le=50, description="Number of results"),
    position: Optional[str] = Query(default=None, description="Filter by position (e.g., 'PG', 'C')"),
):
    """
    Find the top-k most stylistically similar players.

    Uses hybrid scoring: 60% cosine similarity + 35% block-weighted similarity + 5% role bonus.
    Results are post-filtered by position if specified.

    The player_id path parameter accepts either a basketball-reference entity ID
    (e.g., 'jamesle01') or a player name (e.g., 'LeBron James'). Names are
    resolved via prefix matching against the player metadata index.
    """
    try:
        filters = {}
        if position:
            filters["position"] = position

        # Resolve name → entity_id if the input isn't a direct entity_id
        resolved_id = service.resolve_player(player_id)
        if resolved_id is None:
            raise not_found(
                "NOT_FOUND",
                f"Player '{player_id}' not found. Try the /search/players endpoint for suggestions.",
            )

        result = service.search_player(
            player_id=resolved_id,
            k=k,
            filters=filters if filters else None,
        )

        # Convert to Pydantic model
        search_results = [
            SearchResult(
                entity_id=r["entity_id"],
                entity_name=r["entity_name"],
                score=r["score"],
                cosine_similarity=r["cosine_similarity"],
                rank=r["rank"],
                metadata=r["metadata"],
                top_contributing_features=r["top_contributing_features"],
            )
            for r in result["results"]
        ]

        return SearchResponse(
            query_id=result["query_id"],
            query_entity=result["query_entity"],
            results=search_results,
            total_candidates_searched=result["total_candidates_searched"],
            filters_applied=result["filters_applied"],
            timing_ms=result["timing_ms"],
            index_version=result["index_version"],
            cache_hit=result.get("cache_hit", False),
        )

    except KeyError:
        raise not_found("NOT_FOUND", f"Player '{player_id}' not found in index.")
    except HTTPException:
        raise
    except RuntimeError as e:
        raise log_and_generic_503(
            e, "SERVICE_UNAVAILABLE", "Similarity search is currently unavailable."
        )


@router.post(
    "/search/query",
    response_model=SearchResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
@limiter.limit("30/minute")
async def search_by_raw_vector(
    request: Request,
    query: RawVectorQuery,
):
    """
    Search using a raw embedding vector.

    For programmatic consumers, custom embedding experiments,
    and debug consoles. The vector is L2-normalized by default
    before searching.
    """
    try:
        entity_type = EntityType.PLAYER

        # Validate dimension
        info = service.get_index_info(entity_type)
        if info is None:
            raise service_unavailable(
                "SERVICE_UNAVAILABLE", f"No active {entity_type} index."
            )

        if len(query.vector) != info["dimension"]:
            raise bad_request(
                "INVALID_DIMENSION",
                f"Expected {info['dimension']}-dim vector, got {len(query.vector)}",
            )

        result = service.search_by_vector(
            vector=query.vector,
            entity_type=entity_type,
            k=query.k,
            normalize=query.normalize,
        )

        search_results = [
            SearchResult(
                entity_id=r["entity_id"],
                entity_name=r["entity_name"],
                score=r["score"],
                cosine_similarity=r["cosine_similarity"],
                rank=r["rank"],
                metadata=r["metadata"],
                top_contributing_features=r["top_contributing_features"],
            )
            for r in result["results"]
        ]

        return SearchResponse(
            query_id=result["query_id"],
            query_entity=result["query_entity"],
            results=search_results,
            total_candidates_searched=result["total_candidates_searched"],
            filters_applied=result["filters_applied"],
            timing_ms=result["timing_ms"],
            index_version=result["index_version"],
            cache_hit=result.get("cache_hit", False),
        )

    except ValueError as e:
        raise bad_request("INVALID_INPUT", str(e))
    except HTTPException:
        raise
    except RuntimeError as e:
        raise log_and_generic_503(
            e, "SERVICE_UNAVAILABLE", "Similarity search is currently unavailable."
        )


# ═══════════════════════════════════════════════════════════════
# INDEX MANAGEMENT ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@router.post(
    "/index/rebuild",
    response_model=RebuildResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
@limiter.limit("2/hour")
async def rebuild_index(
    request: Request,
    request_body: RebuildRequest,
    _admin_key: str = Depends(require_admin_key),
):
    """
    Trigger a full index rebuild for the specified entity type.

    Requires an ``X-Admin-Key`` header matching the ``ADMIN_API_KEY`` env var.
    If that env var is unset, the endpoint returns 503 (fail-closed).

    The rebuild runs off the event loop (via a threadpool) so other requests
    stay responsive, and a global non-blocking lock serializes concurrent
    rebuilds — a second concurrent request gets 409 REBUILD_IN_PROGRESS
    instead of racing file writes in ``faiss_output/``.
    """
    # Non-blocking acquire: refuse to start a second concurrent rebuild.
    if not acquire_rebuild_lock():
        raise HTTPException(
            status_code=409,
            detail={
                "error": "REBUILD_IN_PROGRESS",
                "detail": "An index rebuild is already in progress. Retry later.",
                "status_code": 409,
            },
        )

    try:
        entity_type = EntityType.PLAYER

        # Run the heavy CPU/IO-bound rebuild off the event loop so the API
        # stays responsive to /search/* during the build (tens of seconds+).
        result = await run_in_threadpool(
            service.rebuild_player_index, force=request_body.force
        )

        return RebuildResponse(**result)

    except (RuntimeError, FileNotFoundError) as e:
        raise log_and_generic_503(
            e, "REBUILD_FAILED", "Index rebuild failed. See server logs for details."
        )
    finally:
        release_rebuild_lock()


@router.get(
    "/index/info/{entity_type}",
    response_model=IndexInfoResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_index_info(
    entity_type: EntityTypeEnum = Path(..., description="Entity type: 'player'"),
):
    """
    Return metadata about the currently active index:
    version, type, vector count, dimension, memory footprint, build timestamp.
    """
    et = EntityType.PLAYER
    info = service.get_index_info(et)

    if info is None:
        raise not_found("NOT_FOUND", f"No active index for {entity_type.value}.")

    return IndexInfoResponse(
        entity_type=entity_type,
        index_version=info["index_version"],
        index_type=info["index_type"],
        vector_count=info["vector_count"],
        dimension=info["dimension"],
        built_at=info["built_at"],
        memory_bytes=info["memory_bytes"],
    )


# ═══════════════════════════════════════════════════════════════
# ARCHETYPE / GRAPH ENDPOINTS
# ═══════════════════════════════════════════════════════════════

def _archetype_store_or_503() -> ArchetypeStore:
    """Load the archetype store, mapping load failures to a generic 503."""
    try:
        return get_store()
    except (FileNotFoundError, ValueError) as e:
        raise log_and_generic_503(
            e, "ARCHETYPE_DATA_UNAVAILABLE", "Archetype data is currently unavailable."
        )


@router.get(
    "/archetypes",
    response_model=ArchetypeDataResponse,
    responses={503: {"model": ErrorResponse}},
)
async def get_archetypes():
    """
    Full archetype dataset for the UMAP explorer:
    cluster summaries plus every player's 2D projection.

    Cached client-side for 1 hour (data only changes on manual rebuild/rescrape).
    """
    store = _archetype_store_or_503()

    clusters = [
        ArchetypeClusterModel(
            cluster_id=c.cluster_id,
            cluster_name=c.name,
            size=c.size,
            description=c.description,
            key_traits=list(c.key_traits),
            example_players=list(c.example_players),
        )
        for c in sorted(store.clusters.values(), key=lambda c: c.cluster_id)
    ]
    players = [
        PlayerArchetypePoint(
            entity_id=p.entity_id,
            entity_name=p.entity_name,
            cluster_id=p.cluster_id,
            umap_x=p.umap_x,
            umap_y=p.umap_y,
            position=p.position,
            hof=p.hof,
            debut_season=p.debut_season,
            final_season=p.final_season,
        )
        for p in store.players.values()
    ]
    return ArchetypeDataResponse(
        clusters=clusters, players=players, total_players=len(players)
    )


def _graph_node(point: PlayerPoint, is_center: bool = False) -> GraphNode:
    return GraphNode(
        entity_id=point.entity_id,
        entity_name=point.entity_name,
        cluster_id=point.cluster_id,
        position=point.position,
        hof=point.hof,
        is_center=is_center,
    )


@router.get(
    "/graph/player/{player_id}",
    response_model=PlayerGraphResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def get_player_graph(
    player_id: str = Path(..., description="Player entity ID or name"),
    k: int = Query(default=12, ge=1, le=30, description="Neighbors per node"),
):
    """
    k-nearest-neighbor similarity neighborhood for one player,
    shaped as nodes + weighted edges for the archetype mind map.

    Edges come from the same hybrid FAISS scorer as /search/player.
    Nodes are enriched with archetype cluster labels; neighbors that
    exist in the FAISS index but not in the clustering output are
    still returned with cluster_id -1.
    """
    store = _archetype_store_or_503()

    try:
        resolved_id = service.resolve_player(player_id)
        if resolved_id is None:
            raise not_found("NOT_FOUND", f"Player '{player_id}' not found.")

        result = service.search_player(player_id=resolved_id, k=k)
    except KeyError:
        raise not_found("NOT_FOUND", f"Player '{player_id}' not found in index.")
    except HTTPException:
        raise
    except RuntimeError as e:
        raise log_and_generic_503(
            e, "SERVICE_UNAVAILABLE", "Similarity search is currently unavailable."
        )

    def point_for(entity_id: str, entity_name: str, metadata: dict) -> PlayerPoint:
        existing = store.players.get(entity_id)
        if existing is not None:
            return existing
        # Neighbor known to FAISS but absent from clustering output
        return PlayerPoint(
            entity_id=entity_id,
            entity_name=entity_name,
            cluster_id=-1,
            umap_x=0.0,
            umap_y=0.0,
            position=str(metadata.get("primary_position") or "—"),
            hof=bool(metadata.get("hof", False)),
            debut_season=int(metadata.get("debut_season") or 0),
            final_season=int(metadata.get("final_season") or 0),
            total_seasons=int(metadata.get("total_seasons") or 0),
        )

    query_entity = result["query_entity"]
    center = point_for(
        query_entity["entity_id"], query_entity["entity_name"], {}
    )

    nodes: list[GraphNode] = [_graph_node(center, is_center=True)]
    edges: list[GraphEdge] = []
    seen: set[str] = {center.entity_id}

    for r in result["results"]:
        neighbor = point_for(r["entity_id"], r["entity_name"], r["metadata"])
        if neighbor.entity_id not in seen:
            seen.add(neighbor.entity_id)
            nodes.append(_graph_node(neighbor))
        edges.append(
            GraphEdge(
                source=center.entity_id,
                target=neighbor.entity_id,
                score=float(r["score"]),
            )
        )

    return PlayerGraphResponse(center_id=center.entity_id, nodes=nodes, edges=edges)


# ═══════════════════════════════════════════════════════════════
# PLAYER HEADSHOTS
# ═══════════════════════════════════════════════════════════════

@router.get(
    "/headshots",
    response_model=HeadshotMapResponse,
    responses={503: {"model": ErrorResponse}},
)
async def get_headshot_map():
    """
    Return the full player name → headshot URL mapping.

    The frontend fetches this once at app load and caches it client-side
    for synchronous avatar lookups. The mapping is produced by the
    Scrapy-Playwright spider (see ``backend/app/scraping/``) and stored
    as ``backend/data/nba_player_headshots.json``.
    """
    from .headshot_store import get_store

    store = get_store()
    if not store.loaded:
        raise service_unavailable(
            "SERVICE_UNAVAILABLE",
            "Headshot mapping not loaded. Run `python run_scrape.py` to generate it.",
        )

    all_headshots = store.get_all()
    return HeadshotMapResponse(
        headshots=all_headshots,
        total=len(all_headshots),
    )


# ═══════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════

@router.get("/health")
async def health_check():
    """Simple health check with index status."""
    indices = service.active_indices()
    return {
        "status": "healthy" if any(indices.values()) else "degraded",
        "indices": indices,
        "cache_size": service.cache_size(),
    }
