"""Async EventEmitter for RUNE.

Replaces Node.js EventEmitter pattern used by 20+ classes in the TS codebase.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any

type AsyncHandler = Callable[..., Coroutine[Any, Any, None]]
type SyncHandler = Callable[..., None]
type Handler = AsyncHandler | SyncHandler


class EventEmitter:
    """Async-first event emitter compatible with Node.js EventEmitter semantics."""

    __slots__ = ("_handlers",)

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)

    def on(self, event: str, handler: Handler) -> None:
        """Register a handler for *event*."""
        self._handlers[event].append(handler)

    def off(self, event: str, handler: Handler) -> None:
        """Remove a specific handler for *event*."""
        handlers = self._handlers.get(event)
        if handlers:
            with contextlib.suppress(ValueError):
                handlers.remove(handler)

    def once(self, event: str, handler: AsyncHandler) -> None:
        """Register a handler that auto-removes after first invocation."""
        async def wrapper(*args: Any, **kwargs: Any) -> None:
            self.off(event, wrapper)
            await handler(*args, **kwargs)
        self.on(event, wrapper)

    async def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Emit an event, awaiting async handlers sequentially."""
        for handler in list(self._handlers.get(event, [])):
            result = handler(*args, **kwargs)
            if asyncio.iscoroutine(result):
                await result

    def emit_nowait(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Fire-and-forget: schedule all handlers as tasks (no await)."""
        for handler in list(self._handlers.get(event, [])):
            result = handler(*args, **kwargs)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)

    def remove_all(self, event: str | None = None) -> None:
        """Remove all handlers for *event*, or all events if None."""
        if event is None:
            self._handlers.clear()
        else:
            self._handlers.pop(event, None)

    def listener_count(self, event: str) -> int:
        """Return the number of handlers registered for *event*."""
        return len(self._handlers.get(event, []))
