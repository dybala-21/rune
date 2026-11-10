"""What a web run is told it remembers: the same memory context the CLI and daemon inject."""

from __future__ import annotations

import asyncio

from rune.utils.logger import get_logger

log = get_logger(__name__)

# Memory is looked up before the model sees the task, so it must never hold the run up for long.
MEMORY_CONTEXT_TIMEOUT_S = 3.0


async def memory_context_for(goal: str) -> str:
    """Learned rules, past experience and durable facts relevant to *goal*, or "" when there are none.

    A failure or a slow lookup is logged and the run goes on without memory.
    """
    if not goal.strip():
        return ""
    try:
        from rune.memory.manager import get_memory_manager

        manager = get_memory_manager()
        async with asyncio.timeout(MEMORY_CONTEXT_TIMEOUT_S):
            await manager.initialize()
            return await manager.build_memory_context(goal)
    except TimeoutError:
        log.warning("web_memory_context_timeout", seconds=MEMORY_CONTEXT_TIMEOUT_S)
    except Exception as exc:
        log.warning("web_memory_context_failed", error=f"{type(exc).__name__}: {str(exc)[:200]}")
    return ""
