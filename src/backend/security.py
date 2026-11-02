"""
Security primitives for the NBA Similarity Search API.

Centralizes:
- Admin API-key authentication for privileged endpoints (e.g. /index/rebuild).
- Generic, leak-free error responses (no internal paths/tracebacks to clients).

Design (per SECURITY_AUDIT_PLAN.md Phase 1 & 2):
- Admin key is read from the ``ADMIN_API_KEY`` env var. The value is supplied
  out-of-band and never logged. If the env var is unset, admin endpoints fail
  CLOSED (503) — an unset key is not an open door.
- Key comparison uses ``secrets.compare_digest`` (constant-time) to prevent
  timing attacks.
- Error helpers log full detail server-side and return only a stable error
  code + generic message to the client.
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Any, Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# Header name clients must send to authenticate admin endpoints.
# Documented in the OpenAPI schema via Security(...).
_ADMIN_KEY_HEADER = APIKeyHeader(name="X-Admin-Key", auto_error=False)


def _admin_key_configured() -> bool:
    """Return True iff an ADMIN_API_KEY env var is set to a non-empty value."""
    key = os.environ.get("ADMIN_API_KEY")
    return bool(key) and len(key) > 0


def _configured_admin_key() -> Optional[str]:
    """Return the configured admin key, or None if unset. Never logged."""
    key = os.environ.get("ADMIN_API_KEY")
    if key:
        return key
    return None


async def require_admin_key(
    api_key: Optional[str] = Security(_ADMIN_KEY_HEADER),
) -> str:
    """FastAPI dependency: enforce a valid X-Admin-Key header.

    Fail-closed semantics:
    - If ``ADMIN_API_KEY`` is unset → 503 (admin is disabled, not open).
    - If the header is missing or empty → 401.
    - If the header does not match the configured key → 403.

    Comparison is constant-time via ``secrets.compare_digest``.
    """
    configured = _configured_admin_key()

    if configured is None:
        # Admin endpoints are disabled entirely when no key is configured.
        # Fail closed — never expose the endpoint as open.
        logger.warning("Admin endpoint blocked: ADMIN_API_KEY not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "ADMIN_DISABLED",
                "detail": "Admin operations are not configured on this server.",
                "status_code": 503,
            },
        )

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "UNAUTHORIZED",
                "detail": "Missing X-Admin-Key header.",
                "status_code": 401,
            },
            headers={"WWW-Authenticate": "X-Admin-Key"},
        )

    # Constant-time comparison to prevent timing side-channels.
    if not secrets.compare_digest(api_key, configured):
        logger.warning("Admin endpoint blocked: invalid X-Admin-Key")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "FORBIDDEN",
                "detail": "Invalid admin key.",
                "status_code": 403,
            },
        )

    return api_key


# ── Generic error responses (Phase 2: stop leaking internals) ──

def service_unavailable(error_code: str, message: str) -> HTTPException:
    """Build a 503 with a generic message. Internal details stay server-side."""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"error": error_code, "detail": message, "status_code": 503},
    )


def not_found(error_code: str, message: str) -> HTTPException:
    """Build a 404. The message may echo a client-supplied identifier only."""
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": error_code, "detail": message, "status_code": 404},
    )


def bad_request(error_code: str, message: str) -> HTTPException:
    """Build a 400 for invalid client input."""
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": error_code, "detail": message, "status_code": 400},
    )


def log_and_generic_503(exc: Exception, error_code: str, message: str) -> HTTPException:
    """Log the real exception server-side, return a generic 503 to the client.

    Use this in route handlers where an internal failure (RuntimeError,
    FileNotFoundError, etc.) must not leak filesystem paths or library
    internals to the caller.
    """
    logger.error("%s: %s", error_code, exc, exc_info=True)
    return service_unavailable(error_code, message)
