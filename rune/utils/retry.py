"""Exponential backoff retry utility.

Ported from src/utils/retry.ts - with jitter (0-20% symmetric).
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from rune.utils.logger import get_logger

log = get_logger(__name__)

T = TypeVar("T")


async def retry[T](
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.2,
    should_retry: Callable[[Exception], bool] | None = None,
) -> T:
    """Execute *fn* with exponential backoff.

    Args:
        fn: Async callable to retry.
        max_retries: Maximum number of retry attempts (0 = no retry).
        base_delay: Initial delay in seconds.
        max_delay: Cap on delay between retries.
        jitter: Symmetric jitter fraction (0.2 = ±20%).
        should_retry: Predicate that receives the exception; return False to abort early.

    Returns:
        The return value of *fn* on success.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as exc:
            last_error = exc

            if attempt >= max_retries:
                break

            if should_retry is not None and not should_retry(exc):
                break

            delay = min(base_delay * (2 ** attempt), max_delay)
            # Apply symmetric jitter: delay * (1 ± jitter)
            jitter_factor = 1.0 + random.uniform(-jitter, jitter)
            delay *= jitter_factor

            log.warning(
                "retry_attempt",
                attempt=attempt + 1,
                max_retries=max_retries,
                delay=round(delay, 2),
                error=str(exc),
            )
            await asyncio.sleep(delay)

    assert last_error is not None
    raise last_error
