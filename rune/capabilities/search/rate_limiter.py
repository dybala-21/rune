"""Semaphore-based rate limiter for search providers.

Ported from src/capabilities/search/rate-limiter.ts - limits concurrent
search requests to avoid hitting provider rate limits.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class SearchRateLimiter:
    """Semaphore-based concurrency limiter for search requests."""

    def __init__(self, max_concurrent: int) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent

    async def acquire(self) -> None:
        """Acquire a slot (blocks if all slots are in use)."""
        await self._semaphore.acquire()

    def release(self) -> None:
        """Release a slot."""
        self._semaphore.release()

    async def wrap(self, fn: Callable[[], Awaitable[T]]) -> T:
        """Execute *fn* while holding a concurrency slot."""
        async with self._semaphore:
            return await fn()

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent
