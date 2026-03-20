"""CORS origin validation and policy resolution.

Ported from src/api/cors-policy.ts - determines whether a cross-origin
request should be allowed based on configured origins, request Origin
header, and Host header.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CorsDecision:
    """Result of a CORS policy evaluation."""
    allowed: bool
    allow_origin: str | None = None
    allow_credentials: bool = False


def _normalize(value: str | None) -> str | None:
    if not value:
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


def _is_same_origin(request_origin: str, request_host: str) -> bool:
    return (
        request_origin == f"http://{request_host}"
        or request_origin == f"https://{request_host}"
    )


def _parse_allowed_origins(configured_origin: str) -> list[str]:
    return [entry.strip() for entry in configured_origin.split(",") if entry.strip()]


def resolve_cors_decision(
    *,
    configured_origin: str | None = None,
    request_origin: str | None = None,
    request_host: str | None = None,
) -> CorsDecision:
    """Evaluate CORS policy.

    Rules:
    - No configured origin (default): same-origin only, cross-origin denied.
    - ``"*"``: allow all origins (credentials not allowed).
    - Comma-separated list: allow exact matches only (credentials allowed).
    - No browser Origin header: not a CORS request, allow.
    """
    configured = _normalize(configured_origin)
    origin = _normalize(request_origin)
    host = _normalize(request_host)

    # No Origin header => not a browser CORS request
    if not origin:
        return CorsDecision(allowed=True)

    # No configured origin => same-origin only
    if not configured:
        if host and _is_same_origin(origin, host):
            return CorsDecision(allowed=True)
        return CorsDecision(allowed=False)

    allowed_origins = _parse_allowed_origins(configured)
    if not allowed_origins:
        return CorsDecision(allowed=False)

    if "*" in allowed_origins:
        return CorsDecision(allowed=True, allow_origin="*", allow_credentials=False)

    if origin in allowed_origins:
        return CorsDecision(allowed=True, allow_origin=origin, allow_credentials=True)

    return CorsDecision(allowed=False)


def get_allowed_origins_from_env() -> str | None:
    """Read allowed origins from environment (RUNE_CORS_ORIGINS)."""
    return os.environ.get("RUNE_CORS_ORIGINS")
