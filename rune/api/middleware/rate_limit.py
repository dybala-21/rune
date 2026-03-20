"""Token bucket rate limiter middleware for FastAPI.

Provides per-client rate limiting using the token bucket algorithm.
Tokens refill at a steady rate; each request consumes one token.
When the bucket is empty, requests receive HTTP 429.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class _Bucket:
    """Per-client token bucket."""
    tokens: float
    last_refill: float
    max_tokens: float
    refill_rate: float  # tokens per second

    def consume(self) -> bool:
        """Try to consume one token. Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class RateLimiter:
    """Token bucket rate limiter.

    Example usage with FastAPI::

        from fastapi import FastAPI, Request, Response
        from rune.api.middleware.rate_limit import RateLimiter

        limiter = RateLimiter(max_tokens=60, refill_rate=1.0)
        app = FastAPI()

        @app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            client_ip = request.client.host if request.client else "unknown"
            if not limiter.allow(client_ip):
                return Response(
                    content='{"detail":"Rate limit exceeded"}',
                    status_code=429,
                    media_type="application/json",
                    headers={"Retry-After": str(limiter.retry_after(client_ip))},
                )
            return await call_next(request)

    Args:
        max_tokens: Maximum burst size (bucket capacity).
        refill_rate: Tokens added per second.
        cleanup_interval: Seconds between stale bucket cleanup.
    """

    def __init__(
        self,
        max_tokens: float = 60.0,
        refill_rate: float = 1.0,
        cleanup_interval: float = 300.0,
    ) -> None:
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.cleanup_interval = cleanup_interval
        self._buckets: dict[str, _Bucket] = {}
        self._last_cleanup = time.monotonic()

    def _get_bucket(self, client_id: str) -> _Bucket:
        bucket = self._buckets.get(client_id)
        if bucket is None:
            bucket = _Bucket(
                tokens=self.max_tokens,
                last_refill=time.monotonic(),
                max_tokens=self.max_tokens,
                refill_rate=self.refill_rate,
            )
            self._buckets[client_id] = bucket
        return bucket

    def allow(self, client_id: str) -> bool:
        """Check if a request from *client_id* is allowed.

        Returns ``True`` if there are tokens available, ``False`` otherwise.
        """
        self._maybe_cleanup()
        bucket = self._get_bucket(client_id)
        return bucket.consume()

    def retry_after(self, client_id: str) -> int:
        """Suggested Retry-After value in seconds for a rate-limited client."""
        bucket = self._buckets.get(client_id)
        if bucket is None or bucket.tokens >= 1.0:
            return 0
        deficit = 1.0 - bucket.tokens
        return max(1, int(deficit / self.refill_rate) + 1)

    def _maybe_cleanup(self) -> None:
        """Remove stale buckets periodically."""
        now = time.monotonic()
        if now - self._last_cleanup < self.cleanup_interval:
            return
        self._last_cleanup = now

        # Remove buckets that have been full for a while (idle clients)
        stale_threshold = now - self.cleanup_interval
        stale_keys = [
            k
            for k, b in self._buckets.items()
            if b.last_refill < stale_threshold and b.tokens >= b.max_tokens
        ]
        for k in stale_keys:
            del self._buckets[k]

        if stale_keys:
            log.info("rate_limit_cleanup", removed=len(stale_keys))


def create_rate_limit_middleware(
    max_tokens: float = 60.0,
    refill_rate: float = 1.0,
) -> Any:
    """Create a FastAPI middleware function for rate limiting.

    Returns an async middleware callable suitable for use with
    ``app.middleware("http")``.

    Example::

        app.middleware("http")(
            create_rate_limit_middleware(max_tokens=100, refill_rate=2.0)
        )
    """
    limiter = RateLimiter(max_tokens=max_tokens, refill_rate=refill_rate)

    async def middleware(request: Any, call_next: Callable[..., Any]) -> Any:
        from fastapi.responses import JSONResponse

        client_ip = request.client.host if request.client else "unknown"

        # Skip rate limiting for health endpoint
        if request.url.path == "/health":
            return await call_next(request)

        if not limiter.allow(client_ip):
            retry = limiter.retry_after(client_ip)
            log.warning("rate_limited", client_ip=client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": str(retry)},
            )

        return await call_next(request)

    return middleware
