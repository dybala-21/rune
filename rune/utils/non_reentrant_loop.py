"""NonReentrantLoop - prevents concurrent execution of the same async task.

Ported from src/utils/non-reentrant-loop.ts - runs an async tick function
at a fixed interval, guaranteeing that the next tick only starts after the
previous one has settled.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable


class NonReentrantLoop:
    """Run an async callback on a fixed interval without overlapping executions.

    Parameters
    ----------
    interval_sec:
        Delay between the *end* of one tick and the *start* of the next.
    tick:
        The async (or sync) callable to invoke each cycle.
    on_error:
        Optional error handler invoked when *tick* raises.
    """

    def __init__(
        self,
        *,
        interval_sec: float,
        tick: Callable[[], Awaitable[None] | None],
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._interval = interval_sec
        self._tick = tick
        self._on_error = on_error
        self._task: asyncio.Task[None] | None = None
        self._active = False
        self._tick_running = False
        self._trigger_event: asyncio.Event = asyncio.Event()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Start the loop. No-op if already running."""
        if self._active:
            return
        self._active = True
        self._trigger_event.clear()
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._run_loop())

    def stop(self) -> None:
        """Stop the loop. The currently running tick (if any) is allowed to finish."""
        self._active = False
        self._trigger_event.set()  # unblock any pending wait
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_tick_running(self) -> bool:
        return self._tick_running

    def trigger_now(self) -> None:
        """Request an immediate tick (skipping the interval wait)."""
        if not self._active or self._tick_running:
            return
        self._trigger_event.set()

    # -- internal -----------------------------------------------------------

    async def _run_loop(self) -> None:
        while self._active:
            try:
                # Wait for the interval OR an early trigger
                self._trigger_event.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._trigger_event.wait(),
                        timeout=self._interval,
                    )

                if not self._active:
                    break

                await self._run_tick()

            except asyncio.CancelledError:
                break

    async def _run_tick(self) -> None:
        if not self._active or self._tick_running:
            return

        self._tick_running = True
        try:
            result = self._tick()
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                await result
        except Exception as exc:  # noqa: BLE001
            if self._on_error:
                self._on_error(exc)
        finally:
            self._tick_running = False
