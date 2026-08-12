# FAISS Similarity Search System — Design Document

## Assumptions

- **Base scale**: ~1,800 players (career aggregates) and ~1,900 team-seasons, matching the existing pipeline output. Per-game expansion would push to ~300K–1.2M rows depending on history depth (post-1980 ~45 seasons × 30 teams × 82 games ≈ 111K; all eras including ABA/BAA ≈ 200K; adding stint/lineup granularity easily pushes to low millions).
- **Changed from Context**: I'm anchoring the initial index type recommendation to the **actual** base-scale numbers observed in the pipeline output (1,790 players, 1,907 team-seasons), not an abstract "hundreds of thousands." Flat is unequivocally sufficient at this scale. I also note what changes at each order-of-magnitude threshold.
- **Changed from Context**: The embedding dimension after PCA won't be a fixed 32–64. The existing pipeline uses `pca_variance=0.90`, which empirically produces 18–28 components for teams and 35–55 for players. The FAISS design must accommodate variable dimensionality from the PCA step — we don't pad/truncate to a fixed size.
- **Feature availability**: Shot-zone data (`percent_fga_from_x0_3_range`, `avg_dist_fga`, etc.) exists from 1997 onward (not mentioned in Context, but true per the existing pipeline). Pre-1997, these are median-imputed per era. Player-tracking data (speed, distance, touches) only from 2013–14. This gap is the primary reason player↔team cross-search is infeasible — there is no shared feature subspace that spans all eras.
- **No GPU dependency at base scale**: Flat index on ~2K vectors of ~30–50 dims is sub-millisecond. Reaching for faiss-gpu at this scale would be an architectural mistake. GPU only becomes relevant at the extended scale (>500K vectors).
- **Nightly incremental updates are bulk-rebuilds at base scale**: With <2K vectors, rebuilding the full index is faster than implementing incremental insertion logic. At extended scale, this changes.

---

## 1. Data Representation

### 1.1 Player Embedding Features

The embedding is constructed on the **existing pipeline's era-adjusted, RobustScaler-transformed feature matrix before PCA**. This is the same `X_scaled` matrix from `player_feature_engineering.py`, which already produces the search-quality representation. We bypass the PCA step for the FAISS index — see §2 for why.

| Block | Features (post-dedup, pre-era-adjust) | Count | Missing-era handling |
|-------|--------------------------------------|-------|---------------------|
| Scoring Profile | `pts_per_game`, `usg_percent`, `ts_percent`, `fg_percent`, `fg_per_game`, `fga_per_game`, `x3p_per_game`, `x3pa_per_game`, `ft_per_game`, `fta_per_game`, `x2p_per_game`, `x2pa_per_game` | 12 | All present in box scores back to BAA (1946) |
| Playmaking | `ast_per_game`, `ast_percent`, `tov_per_game`, `tov_percent`, `points_generated_by_assists` | 5 | `points_generated_by_assists` available from ~1997; median-imputed per debut-era earlier |
| Rebounding | `orb_percent`, `drb_percent`, `trb_percent`, `orb_per_game`, `drb_per_game`, `trb_per_game` | 6 | All present back to 1946 |
| Defense | `stl_percent`, `blk_percent`, `dbpm`, `dws`, `stl_per_game`, `blk_per_game` | 6 | STL/BLK tracked from 1973–74; DBPM from 1973–74; imputed per-era earlier |
| Shooting Profile | `avg_dist_fga`, `x3p_ar`, `f_tr`, `e_fg_percent`, `x3p_percent`, `ft_percent`, `percent_fga_from_x0_3_range`, `percent_fga_from_x3_10_range`, `percent_fga_from_x10_16_range`, `percent_fga_from_x16_3p_range`, `percent_fga_from_x3p_range`, `fg_percent_from_x0_3_range`, `fg_percent_from_x3_10_range`, `fg_percent_from_x10_16_range`, `fg_percent_from_x16_3p_range`, `fg_percent_from_x3p_range` | 16 | Shot-zone columns available from 1997; imputed per-era for older players using era-median |
| Positional | `pg_percent`, `sg_percent`, `sf_percent`, `pf_percent`, `c_percent` | 5 | Play-by-play data from 1997; imputed per-era |
| Advanced | `per`, `bpm`, `obpm`, `dbpm`, `vorp`, `ws`, `ows`, `dws`, `ws_48` | 9 | BPM from 1973–74; PER from 1951–52; imputed per-era |
| Archetype Scores | `scoring_score`, `playmaking_score`, `defense_score`, `rebounding_score`, `spacing_score`, `versatility_score` | 6 | Computed from available features per player; always present |

**Total: ~65 features before era-adjustment.** After era-adjustment, the same count (one column per feature + `_era_adj` suffix). These are the raw embedding.

#### Feature Engineering Strategy

1. **Era-adjustment**: Within-debut-era Z-score (`player_feature_engineering.py:era_adjust`). This is the critical step — a 1980s power forward taking 2 threes per game was a spacing outlier; by 2025 standards they'd look like a non-shooter. Era-relative Z-scores capture "how extreme was this player's style *for their time*." Debut era (not per-season) is used so a player's career is normalized against the era they entered the league in.

2. **RobustScaler**: `RobustScaler(quantile_range=(5.0, 95.0))` clips extreme outliers. Chosen over `StandardScaler` because era-adjusted features can still have extreme values (e.g., Wilt Chamberlain's minutes and rebound rates remain outliers even within his era). RobustScaler prevents these from dominating L2 distance.

3. **No PCA in the embedding**: The FAISS index stores the full ~65-dim era-adjusted + RobustScaler-transformed vector. PCA is orthogonal — it's used downstream for clustering and visualization but removed from the search embedding for a specific reason: PCA projects onto global variance-maximizing axes, which *amplifies* the dominance of usage rate and pace (the highest-variance features). This is the exact failure mode described in §6. The era-adjusted + scaled raw space preserves feature-level independence, enabling the re-ranking strategies in §6.

**Justification for no learned encoder**: At 1,790 training examples, a learned encoder (VAE, contrastive) would overfit to player identity. The existing pipeline's era-adjusted + RobustScaler representation is already validated by the clustering pipeline (silhouette ≈ 0, DBI ≈ 2.1 on 12 archetypes) and produces interpretable feature-level attributions (see §8). A learned encoder would trade interpretability for a marginal embedding-quality gain that can't be validated at this data volume.

#### Normalization for FAISS

After RobustScaler: **L2-normalize every vector to unit length** before inserting into FAISS. This makes L2 distance equivalent to cosine similarity:

```
cos_sim(u, v) = 1 - 0.5 * ||u/||u|| - v/||v||||²
```

We store L2-normalized vectors and use `faiss.METRIC_INNER_PRODUCT` (which on normalized vectors equals cosine similarity). This is more efficient than computing cosine distance explicitly inside FAISS.

**Alternative rejected**: Storing raw vectors and using `faiss.METRIC_L2`. This would mix magnitude with direction — a high-usage player would appear "close" to another high-usage player regardless of stylistic similarity. L2-normalization removes magnitude, keeping only direction (style).

### 1.2 Team Embedding Features

| Block | Features | Count |
|-------|----------|-------|
| Offense (per 100 poss) | `pts_per_100_poss`, `fg_per_100_poss`, `fga_per_100_poss`, `fg_percent`, `x3p_per_100_poss`, `x3pa_per_100_poss`, `x3p_percent`, `x2p_per_100_poss`, `x2pa_per_100_poss`, `x2p_percent`, `ft_per_100_poss`, `fta_per_100_poss`, `ft_percent`, `orb_per_100_poss`, `drb_per_100_poss`, `trb_per_100_poss`, `ast_per_100_poss`, `stl_per_100_poss`, `blk_per_100_poss`, `tov_per_100_poss`, `pf_per_100_poss` | 21 |
| Defense (opponent per 100 poss) | All 21 above prefixed `opp_` | 21 |
| Advanced | `o_rtg`, `d_rtg`, `n_rtg`, `pace`, `ts_percent`, `e_fg_percent`, `tov_percent`, `orb_percent`, `ft_fga`, `x3p_ar`, `f_tr`, `opp_e_fg_percent`, `opp_tov_percent`, `drb_percent`, `opp_ft_fga` | 15 |
| Derived | `ast_ratio`, `x3p_share`, `paint_vs_perimeter`, `ast_tov_ratio`, `disruption_rate`, `defensive_pressure`, `orb_drb_ratio`, `ft_rate`, `opp_ft_rate`, `eff_scoring_margin`, `transition_index`, `halfcourt_grind` | 12 |

**Total: ~69 features, era-adjusted within season-era bucket (5 era buckets: Pre-3PT, Early-3PT, Mid-Era, Modern, Post-COVID).** Same RobustScaler + L2-normalize pipeline as players.

Missing-era handling for teams: All team-level box score stats exist for the entire dataset. The only gaps are in advanced metrics for very early seasons (pre-1974), handled by the same median-imputation-per-era strategy already in `feature_engineering.py:build_feature_matrix`.

### 1.3 Separate vs. Shared Embedding Space

**Decision: Separate embedding spaces. Players and teams do NOT share a vector space.**

Justification:

1. **Different feature schemas**: Player features include play-by-play positional data (`pg_percent`, `c_percent`), shot-zone granularity, and individual advanced metrics (PER, BPM, VORP). Team features include opponent stats, pace, net rating, and scheme-level derived metrics (transition index, disruption rate). There is no natural 1:1 mapping between them.

2. **Different era coverage**: Player shot-zone data starts in 1997, team advanced metrics are mostly complete. Any shared space would have to either drop features to the intersection (losing most of the signal) or impute heavily (diluting quality).

3. **Different conceptual granularity**: A team-season is a system-level entity (5 players on court, coaching scheme). A player is an individual. Conflating them in one vector space produces nearest-neighbor results that are hard to interpret — "this team is similar to Michael Jordan" is not a meaningful statement without a theory of how individual style maps to team style.

**What would make a shared space feasible**: If player-tracking data (spacing maps, touch networks, defensive matchup matrices) existed across all eras, you could build a joint embedding where players are represented by their on-court impact vectors and teams by their aggregate impact vectors. But player-tracking only exists from 2013–14, covering <10% of the historical player base. For pre-2013 players, you'd be imputing the entire shared subspace, making the representation noise. This is the concrete blocker.

---

## 2. FAISS Index Design

### 2.1 Index Type Selection

| Index | Build time | Query time (10-NN) | Memory | Suitable at base scale? | Suitable at extended scale? |
|-------|-----------|---------------------|--------|------------------------|---------------------------|
| `IndexFlatIP` | O(1) | O(nd) = 1,790 × 65 ≈ 116K flops | nd × 4B = ~450 KB | ✅ Yes | ❌ >100K vectors |
| `IndexHNSWFlat` | O(n log n) | O(log n) ≈ 11 hops | (n·d + n·M·4) × 4B | ✅ Overkill but works | ✅ Up to ~10M |
| `IndexIVFFlat` | O(n log n) | O(√n · d) | n·d × 4B + centroids | ❌ Training overhead | ✅ 100K–10M |
| `IndexIVFPQ` | O(n log n) | O(√n · coded) | n·coded × 1B + centroids | ❌ Massive overkill | ✅ >1M |

### 2.2 Recommended Index Types

#### Player Search: `faiss.IndexFlatIP` at base scale

**Choice**: Flat inner-product index on L2-normalized 65-dim vectors.

**Justification tied to actual numbers**: 1,790 vectors × 65 dims × 4 bytes = **465 KB** in memory. A brute-force top-10 search is 1,790 × 65 = 116,350 multiply-adds, which completes in **<0.1 ms** on a single modern CPU core. There is no latency justification for an approximate index at this cardinality. Flat also guarantees exact nearest neighbors — no recall degradation, no training step, no hyperparameters to tune.

**Alternative rejected — `IndexHNSWFlat`**: HNSW would reduce query time from O(n) to O(log n), but at n=1,790 the difference is ~116K flops vs. ~11 hops × M=32 neighbors × 65 dims ≈ 23K flops — both are sub-millisecond. HNSW adds build complexity (M, efConstruction, efSearch tuning), 2× memory overhead for the graph, and non-deterministic results between builds. At this scale, Flat's simplicity dominates.

#### Team Search: `faiss.IndexFlatIP` at base scale

Same reasoning. 1,907 vectors × 69 dims × 4 bytes = **527 KB**. Brute-force top-10 is sub-0.1 ms.

#### Extended Scale Migration Path

| Vector count | Switch to | Reason |
|-------------|-----------|--------|
| >50K | `IndexIVFFlat, IndexFlatIP` (two-tier) | Flat exceeds ~20 MB, query time >5 ms. IVF with `nlist=4*sqrt(n)` ≈ 900 centroids brings query to <1 ms |
| >500K | `IndexIVFPQ` | Memory becomes the bottleneck. 500K × 65 × 4B = 130 MB raw. PQ with M=32, nbits=8 compresses to 500K × 32 = 16 MB. Recall@10 drops to ~0.92–0.95, acceptable for analytics use case |
| >5M | `IndexIVFPQ` + GPU | CPU-PQ query time exceeds 100 ms target. GPU-accelerated PQ (faiss-gpu) brings it back under 10 ms |

**Key design rule**: When migrating from Flat, always keep a `IndexFlatIP` as the "source of truth" for exact search during the transition. The nightly rebuild writes to the flat index first, then builds the approximate index from it. This means exact search is always available (at higher latency) even if the approximate index has a bug.

### 2.3 Distance Metric

**Choice**: Inner product (`faiss.METRIC_INNER_PRODUCT`) on L2-normalized vectors. This equals cosine similarity.

**Why not L2**: The existing pipeline uses cosine similarity in PCA space (`sklearn.metrics.pairwise.cosine_similarity`). The system's notion of "similar" is *directional* (playing style), not *magnitudinal* (quality/volume). L2 distance would conflate the two. For example, two low-usage defensive specialists would appear "close" under L2 because both have low magnitude — but they might play completely different defensive styles. Cosine isolates direction.

**Normalization interaction**: L2-normalization is applied *after* RobustScaler, *before* FAISS insertion. The RobustScaler already handled outlier clipping and feature scaling. Normalization then projects every vector onto the unit hypersphere. The FAISS index stores and searches these unit vectors with inner product.

---

## 3. Pipeline Design

### 3.1 Step-by-Step (Tied to Update Cadence)

#### Nightly Incremental Update (in-season)

```
┌─────────────────────────────────────────────────────────────┐
│ NIGHTLY UPDATE PIPELINE                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │ New/Upd  │───▶│ Feature Eng. │───▶│ Era-Adjust +      │  │
│  │ CSV Rows │    │ (incremental)│    │ RobustScale (full) │  │
│  └──────────┘    └──────────────┘    └───────┬───────────┘  │
│                                              │              │
│                    ┌─────────────────────────┘              │
│                    ▼                                        │
│  ┌──────────────────────────────────────┐                   │
│  │ Re-fit RobustScaler on full dataset  │                   │
│  │ (new rows may shift percentiles)     │                   │
│  └──────────────┬───────────────────────┘                   │
│                 ▼                                           │
│  ┌──────────────────────────────────────┐                   │
│  │ Transform ALL vectors through new    │                   │
│  │ scaler → L2-normalize → build new    │                   │
│  │ FAISS index in background            │                   │
│  └──────────────┬───────────────────────┘                   │
│                 ▼                                           │
│  ┌──────────────────────────────────────┐                   │
│  │ Atomically swap active index         │                   │
│  │ (blue-green via file/pointer swap)   │                   │
│  └──────────────────────────────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why re-fit scaler and rebuild full index?** At base scale (<2K vectors), the entire rebuild takes <1 second. Incremental insertion into FAISS (`IndexFlatIP.add()`) is technically possible, but the RobustScaler parameters (5th/95th percentile) must be refit when new data arrives — which changes the scale of ALL vectors. If the scaler changes, every vector in the index is stale. The only correct approach is a full rebuild.

**Why re-compute era-adjustment?** Adding new rows changes per-era means and standard deviations. However, re-computing era-Z-scores is O(n) per feature and trivial. We do it on every nightly.

#### Season Rollover (Full Rebuild)

```
┌─────────────────────────────────────────────────────────────┐
│ SEASON ROLLOVER PIPELINE                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Run full feature engineering from scratch               │
│     (reload all CSVs, recompute all derived features)       │
│                                                             │
│  2. Re-compute era buckets (new season may shift            │
│     era boundaries — currently hardcoded, but if             │
│     Post-COVID era boundary moves, all era-adj recalc)      │
│                                                             │
│  3. Re-fit RobustScaler on complete historical dataset      │
│                                                             │
│  4. Re-run PCA (if used for anything outside FAISS)         │
│                                                             │
│  5. Build new player + team FAISS indices                   │
│                                                             │
│  6. Run regression test suite (see §7.3) against            │
│     previous season's index before promoting                │
│                                                             │
│  7. Promote: swap active index, archive previous            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Query Flow

```
GET /search/player/123?k=10&era=Modern&position=G
                        │
                        ▼
┌──────────────────────────────────────────┐
│ 1. Lookup entity 123 in metadata store   │
│    → retrieve its embedding vector v     │
└──────────────┬───────────────────────────┘
               ▼
┌──────────────────────────────────────────┐
│ 2. FAISS search: index.search(v, k*3)    │
│    (overfetch 3× for post-filter)        │
│    Returns: [(id, score), ...]           │
└──────────────┬───────────────────────────┘
               ▼
┌──────────────────────────────────────────┐
│ 3. Metadata filter (position, era, etc.) │
│    Apply filters to overfetched results  │
│    Discard filtered-out candidates       │
└──────────────┬───────────────────────────┘
               ▼
┌──────────────────────────────────────────┐
│ 4. Re-rank (see §6)                      │
│    Blend cosine similarity + explicit    │
│    feature weights into hybrid score     │
└──────────────┬───────────────────────────┘
               ▼
┌──────────────────────────────────────────┐
│ 5. Return top-k with scores +            │
│    feature attribution (see §8.2)        │
└──────────────────────────────────────────┘
```

**Why overfetch 3×?** Metadata filtering happens after FAISS retrieval (post-filtering). If the user wants top-10 point guards from the modern era, and only 3 of the top-10 cosine neighbors match these filters, the user sees 3 results. Overfetching by 3× gives the filter a larger candidate pool. At <2K vectors, we could also retrieve ALL candidates and filter — but the overfetch pattern scales to extended scale without behavior change.

---

## 4. API Design (FastAPI)

### 4.1 Pydantic Models

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from enum import Enum
import uuid


class EntityType(str, Enum):
    PLAYER = "player"
    TEAM = "team"


class SearchResult(BaseModel):
    """A single similarity search result."""
    entity_id: str = Field(..., description="Player ID or team-season composite ID")
    entity_name: str = Field(..., description="Display name")
    score: float = Field(..., description="Similarity score (cosine similarity ∈ [0,1])")
    rank: int = Field(..., ge=1)
    metadata: dict = Field(default_factory=dict, description="Era, position, season, etc.")

    # Feature attribution — which features drove this similarity
    top_contributing_features: list[dict[str, float]] = Field(
        default_factory=list,
        description="[{'feature': 'x3p_share', 'contribution': 0.23}, ...]"
    )


class PlayerSearchResult(SearchResult):
    """Player-specific result fields."""
    primary_position: str
    debut_era: str
    hof: bool = False


class TeamSearchResult(SearchResult):
    """Team-specific result fields."""
    season: int
    era_bucket: str
    win_pct: float


class SearchResponse(BaseModel):
    """Standard search response wrapper."""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_entity: dict  # {id, name, type}
    results: list[SearchResult]
    total_candidates_searched: int
    filters_applied: dict = Field(default_factory=dict)
    timing_ms: float = Field(..., description="Total server-side search time in ms")
    index_version: str = Field(..., description="SHA of the active index for debuggability")


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
            raise ValueError("Vector contains non-finite values")
        return v


class SearchFilterParams(BaseModel):
    """Optional metadata filters applied post-retrieval."""
    era_bucket: Optional[str] = None  # e.g., "Modern", "Post-COVID"
    position: Optional[str] = None    # e.g., "PG", "SG", "SF", "PF", "C" — players only
    min_season: Optional[int] = None  # teams only
    max_season: Optional[int] = None  # teams only
    exclude_self: bool = Field(default=True)
    exclude_same_franchise: bool = Field(default=False)  # teams only


class RebuildRequest(BaseModel):
    """Trigger an index rebuild."""
    entity_type: EntityType
    force: bool = Field(default=False, description="Rebuild even if no data has changed")


class RebuildResponse(BaseModel):
    status: Literal["accepted", "rejected", "completed"]
    index_version: str
    build_time_ms: float
    vector_count: int
    dimension: int


class IndexInfoResponse(BaseModel):
    entity_type: EntityType
    index_version: str
    index_type: str  # e.g., "IndexFlatIP"
    vector_count: int
    dimension: int
    built_at: str  # ISO 8601
    memory_bytes: int
    last_data_update: str
```

### 4.2 Endpoints

```python
from fastapi import FastAPI, HTTPException, Query, Path
from app.faiss_service import FaissService  # singleton service

app = FastAPI(title="NBA Similarity Search API")

faiss_service: FaissService  # injected at startup


@app.get("/search/player/{player_id}", response_model=SearchResponse)
async def search_similar_players(
    player_id: str = Path(..., description="Player ID (Basketball-Reference style)"),
    k: int = Query(default=10, ge=1, le=50),
    era_bucket: Optional[str] = Query(default=None),
    position: Optional[str] = Query(default=None),
):
    """
    Find the top-k most stylistically similar players.
    
    Filters applied post-retrieval. Overfetch is 3×k internally.
    """
    try:
        result = await faiss_service.search_player(
            player_id=player_id,
            k=k,
            filters=SearchFilterParams(
                era_bucket=era_bucket,
                position=position,
            ),
        )
        return result
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Player '{player_id}' not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/search/team/{team_id}", response_model=SearchResponse)
async def search_similar_teams(
    team_id: str = Path(..., description="Team-season composite ID, e.g., 'GSW-2017'"),
    k: int = Query(default=10, ge=1, le=50),
    era_bucket: Optional[str] = Query(default=None),
    min_season: Optional[int] = Query(default=None),
    max_season: Optional[int] = Query(default=None),
):
    """Find the top-k most stylistically similar team-seasons."""
    try:
        result = await faiss_service.search_team(
            team_id=team_id, k=k,
            filters=SearchFilterParams(
                era_bucket=era_bucket,
                min_season=min_season,
                max_season=max_season,
            ),
        )
        return result
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found")


@app.post("/search/query", response_model=SearchResponse)
async def search_by_raw_vector(query: RawVectorQuery):
    """
    Search using a raw embedding vector. Used by programmatic consumers,
    custom embedding experiments, and the debug console.
    """
    if query.entity_type not in faiss_service.active_indices:
        raise HTTPException(status_code=400, detail=f"No active index for {query.entity_type}")

    result = await faiss_service.search_by_vector(
        vector=query.vector,
        k=query.k,
        entity_type=query.entity_type,
        normalize=query.normalize,
    )
    return result


@app.post("/index/rebuild", response_model=RebuildResponse)
async def rebuild_index(request: RebuildRequest):
    """
    Trigger a full index rebuild for the specified entity type.
    Used by the nightly cron job and the admin UI.
    """
    if request.entity_type == EntityType.PLAYER:
        result = await faiss_service.rebuild_player_index(force=request.force)
    else:
        result = await faiss_service.rebuild_team_index(force=request.force)
    return result


@app.get("/index/info/{entity_type}", response_model=IndexInfoResponse)
async def get_index_info(entity_type: EntityType):
    """Return metadata about the currently active index."""
    info = faiss_service.get_index_info(entity_type)
    if info is None:
        raise HTTPException(status_code=404, detail=f"No active index for {entity_type}")
    return info
```

### 4.3 Error Handling for Unknown/Invalid IDs

- **Unknown entity ID**: Returns `404` with detail `"Player 'xyz' not found"`. The metadata store (a simple in-memory dict keyed by entity ID) is checked before FAISS search.
- **Invalid vector dimensions**: `RawVectorQuery` validator checks against the active index dimension at request time. If `len(vector) != index.d`, return `400` with `"Expected 65-dim vector, got 47"`.
- **Index not ready**: If the nightly rebuild is in progress (index is being swapped), return `503` with `"Index temporarily unavailable — rebuild in progress"`. The blue-green swap means this window is <100 ms.

---

## 5. Performance Considerations

### 5.1 CPU vs. GPU FAISS

| Scale | CPU (`faiss-cpu`) | GPU (`faiss-gpu`) | Decision |
|-------|-------------------|-------------------|----------|
| Base (1.8K–2K vectors) | <0.1 ms query, <1s rebuild | Setup overhead > query time | **CPU only** — GPU adds no value |
| Extended (100K–500K) | IVF: 1–3 ms query | IVF: 0.3–0.5 ms | **CPU sufficient** — 1–3 ms within 100 ms p95 budget |
| Extended (1M+) | IVFPQ: 20–50 ms CPU | IVFPQ: 2–5 ms GPU | **Consider GPU** if burst QPS exceeds CPU capacity |
| Extended (10M+) | IVFPQ: 200 ms+ CPU | IVFPQ: 5–10 ms GPU | **GPU required** to stay under 100 ms |

**Decision for this system**: `faiss-cpu` only. At base scale, Flat is sub-millisecond. At extended scale up to ~500K, IVF on CPU is well within the 100 ms p95 target. Only introduce `faiss-gpu` when you either (a) exceed 1M vectors, or (b) need to sustain >500 QPS on 500K+ vectors. Both are far beyond the extended scale estimate.

### 5.2 Caching Strategy

**Cache key**: `f"search:{entity_type}:{entity_id}:{k}:{md5(json.dumps(filters, sort_keys=True))}"`

The key includes all request parameters that affect the result set. MD5-hashing the normalized filter dict ensures deterministic cache keys without storing raw JSON.

**Cache policy**:
- **TTL**: 1 hour for player/team similarity queries. Results don't change until the nightly rebuild.
- **Eviction**: LRU with a max of 10,000 entries (~20 MB for full response objects). At 10–50 QPS steady state, the cache hit rate should exceed 95% for common queries (top players/teams are repeatedly searched).
- **In-memory**: Use `cachetools.TTLCache` — no Redis needed. The cache is process-local and lost on restart, which is acceptable for an analytics product.

**Invalidation**: The nightly rebuild atomically increments the `index_version`. The cache layer checks the active `index_version` against the cached response's `index_version`. On mismatch, the entry is evicted. Additionally, the rebuild handler calls `cache.clear()` to purge all entries immediately.

**Why not Redis**: The cache is small (<10K entries, <20 MB), the data is read-only between rebuilds, and there's no multi-process coordination requirement at the stated QPS. Adding Redis for this would violate the constraint against introducing unnecessary infrastructure.

### 5.3 Blue-Green Index Swap (Zero-Downtime Updates)

```python
import faiss
import threading
import hashlib
from datetime import datetime, timezone


class FaissService:
    """
    Manages FAISS indices with blue-green swapping for zero-downtime updates.
    
    Two index slots per entity type: _active and _standby.
    Queries always read from _active.
    Rebuilds write to _standby, then atomically swap.
    """

    def __init__(self):
        self._lock = threading.Lock()  # protects the swap, not queries
        self._indices: dict[EntityType, dict] = {
            EntityType.PLAYER: {"active": None, "standby": None, "metadata": None},
            EntityType.TEAM:   {"active": None, "standby": None, "metadata": None},
        }
        # Metadata store: entity_id → {embedding_idx, name, era, position, ...}
        self._metadata: dict[EntityType, dict[str, dict]] = {
            EntityType.PLAYER: {},
            EntityType.TEAM: {},
        }

    def rebuild_player_index(self, vectors: np.ndarray, metadata: list[dict]) -> str:
        """
        Build a new player index in the standby slot, then swap.

        vectors: (n_players, d) L2-normalized float32 array
        metadata: list of dicts with entity_id, name, primary_pos, debut_era, hof
        Returns the new index version (SHA256 hash).
        """
        d = vectors.shape[1]
        new_index = faiss.IndexFlatIP(d)
        new_index.add(vectors.astype(np.float32))

        # Build metadata lookup: entity_id → row index
        # (entity_id is stable across rebuilds; row index may shift)
        new_metadata = {}
        id_to_idx = {}
        for idx, meta in enumerate(metadata):
            eid = meta["entity_id"]
            new_metadata[eid] = meta
            id_to_idx[eid] = idx

        version = hashlib.sha256(vectors.tobytes()).hexdigest()[:12]

        with self._lock:
            # Move current active → old (for rollback), standby → active
            self._indices[EntityType.PLAYER]["standby"] = {
                "index": new_index,
                "version": version,
                "built_at": datetime.now(timezone.utc).isoformat(),
                "id_to_idx": id_to_idx,
                "dimension": d,
                "vector_count": vectors.shape[0],
            }
            # Atomically promote standby → active
            self._indices[EntityType.PLAYER]["active"] = \
                self._indices[EntityType.PLAYER]["standby"]
            self._metadata[EntityType.PLAYER] = new_metadata

        return version

    def search_player(self, player_id: str, k: int, filters: SearchFilterParams):
        """Query the active player index."""
        active = self._indices[EntityType.PLAYER]["active"]
        if active is None:
            raise RuntimeError("No active player index")

        idx_map = active["id_to_idx"]
        if player_id not in idx_map:
            raise KeyError(f"Player '{player_id}' not found")

        query_vec = active["index"].reconstruct(idx_map[player_id]).reshape(1, -1)
        
        # Overfetch for post-filtering
        fetch_k = k * 3 if (filters.era_bucket or filters.position) else k
        scores, indices = active["index"].search(query_vec, fetch_k)

        # ... post-filter, re-rank, format response ...
```

The swap is safe because:
1. Python's GIL makes pointer assignment atomic for single-threaded reads.
2. Queries read `self._indices[entity_type]["active"]` without acquiring the lock (the dict read is atomic).
3. The lock only protects the swap, which is a single dict assignment — held for microseconds.

**Alternative rejected — file-based swap (writing index to disk, restarting)**: Adds I/O latency and a service restart window. At base scale the in-memory rebuild is <1 second.

---

## 6. Ranking Strategy

### 6.1 Re-Ranking Signal

**Raw FAISS output alone is insufficient.** Cosine similarity on the full feature vector is dominated by the highest-variance features — usage rate, pace, and scoring volume. This produces results that are "correct" by cosine distance but misleading to users: "LeBron James is similar to Russell Westbrook" because both have high USG% and AST%, ignoring that their shot diets and defensive roles are radically different.

**Re-ranking formula**:

```
hybrid_score = α · cos_sim(query, candidate)
             + β · block_sim(query, candidate, user_weights)
             + γ · role_bonus(query, candidate)
```

Where:

- **`cos_sim`**: Raw cosine similarity from FAISS (0 to 1).
- **`block_sim`**: Weighted cosine similarity computed over feature *blocks* (scoring, playmaking, defense, etc.), with user-specified or default block weights. This allows "find similar scorers" vs. "find similar defenders."
- **`role_bonus`**: Binary or scaled bonus for matching position/role. For players: +0.05 if same primary position. For teams: +0.05 if same era bucket. This is intentionally small — it breaks ties rather than dominating results.

**Default weights**: `α=0.6, β=0.35, γ=0.05`. These are tunable per-endpoint and can be exposed as query parameters for power users.

```python
def compute_hybrid_score(
    query_vec: np.ndarray,       # L2-normalized
    candidate_vec: np.ndarray,   # L2-normalized
    feature_names: list[str],
    block_assignments: dict[str, list[str]],  # block_name → [feature names]
    block_weights: dict[str, float],          # user-specified or default
    role_bonus: float = 0.0,
    alpha: float = 0.6,
    beta: float = 0.35,
    gamma: float = 0.05,
) -> float:
    """
    Compute hybrid similarity score blending full cosine, block-weighted
    cosine, and role bonus.
    """
    # Full cosine similarity (already have this from FAISS)
    cos_full = float(np.dot(query_vec, candidate_vec))

    # Block-weighted similarity
    block_scores = []
    total_weight = 0.0
    for block_name, feat_indices in block_assignments.items():
        w = block_weights.get(block_name, 1.0)
        q_block = query_vec[feat_indices]
        c_block = candidate_vec[feat_indices]
        # Re-normalize sub-vectors for block-specific cosine
        q_norm = q_block / (np.linalg.norm(q_block) + 1e-8)
        c_norm = c_block / (np.linalg.norm(c_block) + 1e-8)
        block_scores.append(w * float(np.dot(q_norm, c_norm)))
        total_weight += w

    block_sim = sum(block_scores) / total_weight if total_weight > 0 else 0.0

    return alpha * cos_full + beta * block_sim + gamma * role_bonus
```

### 6.2 Position/Role Weighting

For players, expose a `role_weights` parameter:

```python
class RoleWeights(BaseModel):
    scoring: float = Field(default=1.0, ge=0.0, le=5.0)
    playmaking: float = Field(default=1.0, ge=0.0, le=5.0)
    defense: float = Field(default=1.0, ge=0.0, le=5.0)
    rebounding: float = Field(default=1.0, ge=0.0, le=5.0)
    shooting: float = Field(default=1.0, ge=0.0, le=5.0)
```

These map to the feature blocks from §1.1 and feed directly into the `block_weights` parameter of `compute_hybrid_score`. A user searching "players like LeBron James but focused on playmaking" would set `playmaking=3.0, scoring=0.5`.

### 6.3 Concrete Failure Mode and Mitigation

**Failure mode**: Usage rate dominance. Two players with high USG% (30%+) appear "similar" in cosine space because USG% has high variance and correlates with many other features (PPG, FGA, AST, TOV). Example: Luka Dončić and Russell Westbrook both have USG% >30%, high AST%, high TOV%. Cosine similarity places them as top-5 similar players. But Dončić is a heliocentric half-court creator; Westbrook was a transition-attacking slasher. Their shot diets and defensive impacts are radically different.

**How this design avoids it**:
1. **Era-adjustment** dampens USG%'s raw magnitude by normalizing within era (USG% has inflated league-wide over time).
2. **L2-normalization** converts magnitude to direction — high-USG players point in a similar direction, but the remaining 64 dimensions capture *how* they use those possessions.
3. **Feature block weighting** in the re-ranker (§6.1) allows the `shooting` block (shot diet features: `avg_dist_fga`, `x3p_ar`, `percent_fga_from_*`) to counterbalance `scoring` block dominance. Even with default weights, the 7-block structure ensures no single block dominates.
4. **Feature attribution** in the response (see §8.2) makes the dominance visible: if USG% contributed 40% of the similarity score, the user sees that and can adjust weights.

---

## 7. Evaluation & Validation

### 7.1 Offline: Building Labeled Ground Truth

**Approach**: Construct a labeled set of "known similar" pairs from domain knowledge, then measure recall@k.

**Data sources for labeling**:

| Source | Example pairs | Count estimate |
|--------|--------------|----------------|
| MVP vote shares | Same-archetype MVPs in adjacent years (e.g., Jokić ↔ Embiid as heliocentric centers) | ~50 pairs |
| All-NBA team overlap | Players repeatedly selected to same All-NBA team | ~80 pairs |
| Basketball-Reference similarity scores | B-R's built-in similarity tool provides a noisy-but-independent baseline | ~200 pairs |
| Manual expert curation | 2–3 domain experts label obvious pairs (e.g., Ray Allen ↔ Reggie Miller) | ~100 pairs |
| Cluster co-membership consensus | Pairs assigned to the same cluster by ≥2 of 3 algorithms (KMeans, HDBSCAN, Agglomerative) at high confidence (>0.8 probability) | ~500 pairs |

**Total**: ~800–1000 labeled similar pairs.

**Metrics**:

```python
def evaluate_recall_at_k(
    index: faiss.Index,
    vectors: np.ndarray,
    labeled_pairs: list[tuple[int, int]],  # (query_idx, expected_similar_idx)
    ks: list[int] = [1, 5, 10, 20, 50],
) -> dict[int, float]:
    """
    For each labeled pair, check if expected_similar_idx appears in the
    top-k FAISS results for query_idx (excluding self).
    """
    recalls = {k: 0 for k in ks}
    for query_idx, expected_idx in labeled_pairs:
        scores, indices = index.search(vectors[query_idx:query_idx+1], max(ks) + 1)
        # Exclude self
        retrieved = [i for i in indices[0] if i != query_idx]
        for k in ks:
            if expected_idx in retrieved[:k]:
                recalls[k] += 1

    n = len(labeled_pairs)
    return {k: count / n for k, count in recalls.items()}
```

**Target**: recall@10 ≥ 0.85 on the labeled set. Below 0.70 indicates a feature engineering problem (wrong features or normalization).

### 7.2 Online: User Signal

**Primary signal**: Click-through rate on "similar player" links in the analytics product. If users click through to view a similar player's profile, that's a weak positive signal. If they immediately bounce back, that's a weak negative.

**Secondary signal**: Explicit feedback (thumbs up/down on similarity results). This is the gold standard but requires UI support.

**Lag metric**: Time-on-page after a similarity search. Low time-on-page + low CTR suggests results aren't compelling.

**Operational metric**: Track the distribution of returned similarity scores. If the median cosine similarity drifts significantly between index versions (>0.05 shift), something changed in the feature pipeline — investigate before users notice.

### 7.3 Regression Testing Index Rebuilds

**Goal**: Detect "silent drift" in top-K results when the index is rebuilt.

```python
import hashlib
import json

class IndexRegressionTest:
    """
    Regression test suite for index rebuilds.
    
    Stores a fingerprint of top-K results for a fixed set of query entities.
    On rebuild, compares new results against the stored baseline.
    """

    def __init__(self, query_entities: list[str], k: int = 10):
        self.query_entities = query_entities  # e.g., ["LeBron James", "Stephen Curry", ...]
        self.k = k
        self._baseline: dict[str, list[str]] = {}  # entity → [top-k entity IDs]

    def capture_baseline(self, service: FaissService, entity_type: EntityType):
        """Capture the current top-K results as the baseline."""
        for entity_id in self.query_entities:
            results = service.search(entity_type, entity_id, k=self.k)
            self._baseline[entity_id] = [r.entity_id for r in results.results]
        print(f"[regression] Captured baseline for {len(self._baseline)} entities")

    def compare(self, service: FaissService, entity_type: EntityType) -> dict:
        """
        Compare new results against baseline.

        Returns:
        - jaccard_similarity per entity (intersection / union of top-K sets)
        - rank_correlation (Spearman's ρ on overlapping IDs)
        - list of entities that changed by >threshold
        """
        results = {}
        for entity_id in self.query_entities:
            new_results = service.search(entity_type, entity_id, k=self.k)
            new_ids = [r.entity_id for r in new_results.results]
            old_ids = self._baseline.get(entity_id, [])

            intersection = set(new_ids) & set(old_ids)
            union = set(new_ids) | set(old_ids)
            jaccard = len(intersection) / len(union) if union else 0.0

            results[entity_id] = {
                "jaccard": jaccard,
                "overlap_count": len(intersection),
                "new_only": list(set(new_ids) - set(old_ids)),
                "removed": list(set(old_ids) - set(new_ids)),
            }

        # Aggregate
        avg_jaccard = sum(r["jaccard"] for r in results.values()) / len(results)
        alerts = [eid for eid, r in results.items() if r["jaccard"] < 0.70]

        return {
            "avg_jaccard": avg_jaccard,
            "alerts": alerts,
            "per_entity": results,
            "pass": avg_jaccard >= 0.85 and len(alerts) == 0,
        }
```

**Integration into rebuild**: The season-rollover pipeline (§3.1) runs `capture_baseline` before promoting the new index, then `compare`. If `pass=False`, the new index is NOT promoted — it's flagged for manual review and the previous index stays active.

**What triggers a regression**: Changes in feature availability (a new CSV version dropping columns), changes in era boundaries (adding new buckets), or bugs in the normalization pipeline. The Jaccard threshold of 0.85 allows for legitimate drift (new players entering, a season's stats shifting slightly) while catching bugs.

### 7.4 Monitoring Dashboard Metrics

| Metric | Source | Alert threshold |
|--------|--------|----------------|
| p95 query latency | FastAPI middleware | >100 ms |
| p99 query latency | FastAPI middleware | >200 ms |
| Cache hit rate | `cachetools` stats | <0.80 |
| Index memory usage | `faiss.index.ntotal * d * 4` | >1 GB |
| Recall@10 (offline eval) | Regression test | <0.85 |
| Top-K Jaccard drift | Regression test | <0.70 per entity |
| Rebuild duration | `build_time_ms` in RebuildResponse | >30s (base scale: absurd) |

---

## 8. Optional Enhancements

### 8.1 Hybrid Retrieval: FAISS + Metadata Filtering

**Implementation**: Overfetch from FAISS (3×k) then apply metadata filters in Python. At base scale, this is trivial. At extended scale, two options:

1. **Post-filter (status quo)**: FAISS returns 3×k candidates, filter in Python, return k. Risk: if the filter is highly restrictive (e.g., "point guards from the 1960s"), the candidate pool may be too small even at 3×k. Mitigation: if filtered results < k, re-query with 10×k overfetch.

2. **Pre-filter with sub-indices**: Split the index by position (PG/SG/SF/PF/C) or era. Query the relevant sub-index directly. This avoids the overfetch problem entirely. The cost is managing 5–6 sub-indices instead of one. At base scale this is unnecessary; at extended scale (>100K vectors), prefer option 2.

**Decision**: Implement post-filter now (option 1). Add sub-indices (option 2) only when filter-restrictiveness becomes a user-facing problem.

### 8.2 Explainability: Feature-Level Attribution

**Implementation**: For each similar pair returned, compute per-feature contribution to the cosine similarity.

```python
def compute_feature_attributions(
    query_vec: np.ndarray,      # L2-normalized
    candidate_vec: np.ndarray,  # L2-normalized
    feature_names: list[str],
) -> list[dict]:
    """
    Compute per-feature contribution to cosine similarity.
    
    Cosine similarity decomposes as: sum(query_i * candidate_i)
    Each feature contributes: query_i * candidate_i / cos_sim
    """
    cos_sim = float(np.dot(query_vec, candidate_vec))
    if cos_sim <= 0:
        return []

    raw_contributions = query_vec * candidate_vec
    # Normalize so contributions sum to 1.0
    normalized = raw_contributions / cos_sim

    # Return top-5 contributing features
    top_idx = np.argsort(-normalized)[:5]
    return [
        {
            "feature": feature_names[i],
            "contribution": float(normalized[i]),
            "query_value": float(query_vec[i]),
            "candidate_value": float(candidate_vec[i]),
        }
        for i in top_idx
    ]
```

This is included in every search response (`top_contributing_features` in `SearchResult`). For the frontend, this enables UI like "87% similarity — mostly because both players have high AST% and low avg shot distance."

**Cost**: O(d) per candidate pair — negligible at <100 dims.

### 8.3 Sub-Indexes by Position/Role

**When to add**: If user feedback shows that position dominates similarity results (e.g., "every similar player to Chris Paul is a point guard — I want to see similar playmakers regardless of position"), sub-indexes by position become valuable.

**Implementation**: Build one FAISS index per position group (PG, SG, SF, PF, C). Query the relevant sub-index based on the query player's position (or all sub-indices, merging results). This gives position-aware retrieval without post-filtering.

**Decision**: Do NOT implement now. The feature block weighting in the re-ranker (§6.1) addresses the same problem more flexibly. A user can down-weight "positional" block features to get cross-position results. Sub-indexes are a fallback if re-ranking proves insufficient.

---

## 9. Player ↔ Team Style-Similarity Search: GO / NO-GO

**Decision: NO-GO for the initial system.**

**Justification**:

1. **No shared feature space exists.** Player features (PER, BPM, shot zones, positional percentages) and team features (opponent stats, pace, net rating, scheme-derived metrics) measure fundamentally different things. You cannot embed two entities with non-overlapping feature sets into the same vector space without a joint model.

2. **Building a joint model requires data that doesn't exist across eras.** The only features that could theoretically bridge players and teams are player-tracking metrics — on-court impact on team spacing, defensive matchup data, touch networks. These exist only from 2013–14. A joint embedding trained on 2014–2025 data would have no representation for 80% of historical players and 60% of historical team-seasons.

3. **Even with tracking data, the mapping is underspecified.** A team is an *emergent* system. Five players with individually moderate defensive stats can form an elite defense through scheme and chemistry (e.g., the 2004 Pistons). A joint embedding would need to learn this emergent mapping, which requires far more data than exists.

**What would need to change to make it feasible**:

1. **Complete player-tracking data back to at least 1997** (shot-zone era). This gives ~28 years of shared features for training a joint embedding. Even then, pre-1997 players have no representation.
2. **A two-tower architecture**: One encoder for players (maps individual stats → latent style vector), one encoder for teams (maps aggregate stats → latent style vector), trained with a contrastive loss where positive pairs are "Player X's teams" and "Team Y containing similar players." This requires constructing a training set of player-team associations — feasible for modern tracking-data era but imputed/noisy for earlier eras.
3. **Accept that pre-1997 players and pre-2014 teams are excluded** from cross-type search. This fragments the product (some players support cross-search, some don't) and may be worse than not offering the feature at all.

**Recommendation**: Revisit this decision when player-tracking data covers ≥20 years of history (projected 2033). Until then, implement a simpler proxy: allow users to search "teams with a similar statistical profile to this player's teams" by looking up the teams the player played on, then doing team-to-team similarity on those. This is a two-hop search (player → their teams → similar teams) that doesn't require a shared embedding space but provides a related user experience.
