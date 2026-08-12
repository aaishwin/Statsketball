"""
NBA Similarity Search API — FastAPI Application
================================================
Production-grade FAISS-based similarity search for NBA players.

Startup:
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

Or with the run script:
    python run_api.py

Endpoints:
    GET  /api/v1/search/player/{player_id}
    POST /api/v1/search/query
    POST /api/v1/index/rebuild   (requires X-Admin-Key header)
    GET  /api/v1/index/info/{entity_type}
    GET  /api/v1/health
    GET  /docs  (OpenAPI Swagger UI — disabled unless ENABLE_DOCS=1)

Security configuration (env vars):
    ADMIN_API_KEY          — shared secret gating /index/rebuild (fail-closed if unset)
    CORS_ALLOWED_ORIGINS   — comma-separated allowlist (default: http://localhost:3000)
    ENABLE_DOCS            — "1"/"true" to expose /docs and /redoc (default: on in dev)
"""

import os
import sys
import logging
import time
from contextlib import asynccontextmanager

# Resolve OpenMP conflict between faiss-cpu and sklearn/hdbscan
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Ensure the backend package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.routes import router
from app.faiss_index import service as faiss_service
from app.rate_limit import limiter

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_cors_origins() -> list[str]:
    """Parse CORS_ALLOWED_ORIGINS env var into a list of origin strings.

    Defaults to the frontend dev server. An empty/whitespace entry is dropped.
    A literal "*" is permitted only when no credentials are used (we never set
    allow_credentials=True, so this is safe but discouraged).
    """
    raw = os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:3000")
    # Starlette matches allow_origins by exact string equality, and browsers
    # never send a trailing slash in Origin — strip it so env values like
    # "https://app.example.com/" still work.
    origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    return origins or ["http://localhost:3000"]


def _docs_enabled() -> bool:
    """Whether to expose /docs and /redoc.

    Default ON (dev convenience). Set ENABLE_DOCS=0 to disable in deployment.
    """
    val = os.environ.get("ENABLE_DOCS", "1").strip().lower()
    return val not in {"0", "false", "no", "off"}


# ═══════════════════════════════════════════════════════════════
# LIFESPAN: Initialize FAISS indices on startup
# ═══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: initialize FAISS indices on startup,
    clean up on shutdown.
    """
    logger.info("=" * 60)
    logger.info("NBA Similarity Search API — Starting up")
    logger.info("=" * 60)

    # Resolve data directory (workspace root)
    # main.py is at backend/app/main.py, data is at ../data/... from workspace root
    _app_dir = os.path.dirname(os.path.abspath(__file__))        # backend/app/
    _backend_dir = os.path.dirname(_app_dir)                      # backend/
    _workspace_root = os.path.dirname(_backend_dir)               # workspace root

    data_dir = os.environ.get(
        "NBA_DATA_DIR",
        os.path.join(_workspace_root, "data", "nba-aba-baa-stats", "versions", "56"),
    )

    faiss_output_dir = os.environ.get(
        "FAISS_OUTPUT_DIR",
        os.path.join(_backend_dir, "faiss_output"),
    )

    try:
        # Store data_dir/output_dir for rebuild operations
        faiss_service.configure(data_dir=data_dir, output_dir=faiss_output_dir)

        logger.info(f"Data directory: {data_dir}")
        logger.info(f"FAISS output directory: {faiss_output_dir}")

        # Check if pre-built index exists
        player_index_path = os.path.join(faiss_output_dir, "faiss_player.index")

        if os.path.exists(player_index_path):
            logger.info("Loading pre-built FAISS index from disk...")
            faiss_service.load_prebuilt_indices(faiss_output_dir)
        else:
            logger.info("No pre-built index found — building from scratch...")
            t0 = time.time()
            result = faiss_service.initialize(data_dir=data_dir, output_dir=faiss_output_dir)
            logger.info(
                f"Index built in {time.time() - t0:.1f}s: "
                f"player={result['player']['vector_count']} vectors"
            )

    except Exception as e:
        logger.error(f"Failed to initialize FAISS indices: {e}")
        logger.warning("API starting in degraded mode — rebuild required")
        # Don't crash — allow the API to start and serve /index/rebuild

    logger.info("API ready to serve requests")
    yield  # Application runs here

    logger.info("NBA Similarity Search API — Shutting down")


# ═══════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="NBA Similarity Search API",
    description="""
Production-grade FAISS-based similarity search for NBA players.

## Features
- **Player-to-player similarity**: Find stylistically similar players
- **Raw vector search**: Submit custom embedding vectors
- **Hybrid scoring**: Blends cosine similarity with block-weighted scoring
- **Feature attribution**: See which stats drive each similarity match
- **Metadata filtering**: Filter by position
- **Blue-green index swap**: Zero-downtime nightly index updates
""",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled() else None,
    redoc_url="/redoc" if _docs_enabled() else None,
)

# ── CORS ──
# Wildcard origin + credentials is an invalid combination per the Fetch spec
# and Starlette reflects the request Origin, making every site a trusted
# origin. We use an explicit allowlist from CORS_ALLOWED_ORIGINS and never
# enable allow_credentials (no cookies/sessions are used).
_cors_origins = _parse_cors_origins()
logger.info("CORS allowed origins: %s", _cors_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Admin-Key"],
)

# ── Rate limiting middleware ──
# slowapi requires the limiter state on the app and the SlowAPIMiddleware
# to intercept rate-limited requests and return 429.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ── Include routes ──
app.include_router(router)


# ── Response timing + cache headers middleware ──
# Bulk endpoints (/archetypes, /headshots) return the entire dataset on every
# call with no caching headers — cheap amplification. Their data only changes
# on manual rebuild/rescrape, so we set a 1-hour Cache-Control to let clients
# and CDNs cache and cut amplification.
_CACHEABLE_BULK_PATHS = {"/api/v1/archetypes", "/api/v1/headshots"}
_CACHE_MAX_AGE = 3600  # seconds


@app.middleware("http")
async def add_response_headers(request: Request, call_next):
    """Add X-Process-Time to all responses and Cache-Control to bulk endpoints."""
    start = time.time()
    response = await call_next(request)
    process_time = (time.time() - start) * 1000
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    if request.url.path in _CACHEABLE_BULK_PATHS and response.status_code == 200:
        response.headers["Cache-Control"] = f"public, max-age={_CACHE_MAX_AGE}"
    return response


# ── Root redirect ──
@app.get("/")
async def root():
    """Redirect to API docs."""
    return {
        "service": "NBA Similarity Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
