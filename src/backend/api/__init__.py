"""API module for FAISS similarity search."""
from .routes import router
from .models import (
    SearchResult,
    SearchResponse,
    RawVectorQuery,
    BlockWeights,
    RebuildRequest,
    RebuildResponse,
    IndexInfoResponse,
    ErrorResponse,
    EntityType,
)

__all__ = [
    "router",
    "SearchResult",
    "SearchResponse",
    "RawVectorQuery",
    "BlockWeights",
    "RebuildRequest",
    "RebuildResponse",
    "IndexInfoResponse",
    "ErrorResponse",
    "EntityType",
]
