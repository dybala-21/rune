"""Graceful Shutdown Handling for RUNE.

Ported from src/cli/shutdown.ts -- provides signal-based shutdown
coordination with timeout-guarded cleanup, idempotent execution,
and force-exit on repeated signals.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
from collections.abc import Awaitable, Callable

from rune.utils.logger import get_logger

log = get_logger(__name__)


# ============================================================================
# Once-async runner
# ============================================================================


def create_once_async_runner(
    task: Callable[[], Awaitable[None]],
) -> Callable[[], Awaitable[None]]:
    """Return a coroutine function that runs *task* at most once.

    Subsequent calls return the same awaitable as the first invocation,
    ensuring the underlying work is never duplicated.
    """
    _promise: asyncio.Task[None] | None = None

    async def run() -> None:
        nonlocal _promise
        if _promise is None:
            _promise = asyncio.create_task(task())
        await _promise

    return run


# ============================================================================
# Shutdown Controller
# ============================================================================


class ShutdownController:
    """Coordinates graceful shutdown with signal handling.

    Parameters
    ----------
    cleanup:
        Async callable to release application resources.
    stop:
        Async callable to stop the main server / event loop.
    on_shutdown_message:
        Optional sync callback invoked when the first shutdown signal
        is received (useful for printing a message).
    timeout_sec:
        Maximum time in seconds to wait for cleanup before force-exiting.
    """

    def __init__(
        self,
        cleanup: Callable[[], Awaitable[None]],
        stop: Callable[[], Awaitable[None]],
        *,
        on_shutdown_message: Callable[[], None] | None = None,
        timeout_sec: float = 5.0,
    ) -> None:
        self._cleanup = cleanup
        self._stop = stop
        self._on_shutdown_message = on_shutdown_message
        self._timeout_sec = timeout_sec

        self._shutting_down = False
        self._completed: asyncio.Future[None] | None = None
        self._registered_signals: list[signal.Signals] = []

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    # Signal registration

    def register_signals(
        self,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """Register SIGINT and SIGTERM handlers on the running event loop."""
        if loop is None:
            loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._signal_received)
            self._registered_signals.append(sig)

    def cleanup_signal_listeners(self) -> None:
        """Remove previously registered signal handlers."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        for sig in self._registered_signals:
            with contextlib.suppress(OSError, ValueError):
                loop.remove_signal_handler(sig)
        self._registered_signals.clear()

    # Shutdown sequence

    def _signal_received(self) -> None:
        """Handle an incoming shutdown signal."""
        if self._shutting_down:
            # Second signal -> force exit immediately.
            log.warning("force_exit_on_repeated_signal")
            sys.exit(1)

        self._shutting_down = True
        if self._on_shutdown_message:
            self._on_shutdown_message()

        # Schedule the async shutdown work.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            sys.exit(1)

        self._completed = loop.create_future()
        loop.create_task(self._run_shutdown(loop))

    async def _run_shutdown(self, loop: asyncio.AbstractEventLoop) -> None:
        """Execute cleanup and stop with a safety timeout."""
        try:
            await asyncio.wait_for(
                self._cleanup_then_stop(),
                timeout=self._timeout_sec,
            )
        except TimeoutError:
            log.warning("shutdown_timeout_exceeded")
        except Exception:
            log.exception("shutdown_error")
        finally:
            self.cleanup_signal_listeners()
            if self._completed and not self._completed.done():
                self._completed.set_result(None)

    async def _cleanup_then_stop(self) -> None:
        try:
            await self._cleanup()
        except Exception:
            log.exception("cleanup_error")
        try:
            await self._stop()
        except Exception:
            log.exception("stop_error")

    async def wait_for_completion(self) -> None:
        """Block until the shutdown sequence has finished."""
        if self._completed is not None:
            await self._completed


# ============================================================================
# Factory helper (mirrors TS ``createWebShutdownController``)
# ============================================================================


def create_web_shutdown_controller(
    cleanup: Callable[[], Awaitable[None]],
    stop: Callable[[], Awaitable[None]],
    *,
    on_shutdown_message: Callable[[], None] | None = None,
    timeout_sec: float = 5.0,
) -> ShutdownController:
    """Create and return a :class:`ShutdownController` instance.

    This is a convenience factory that mirrors the TS
    ``createWebShutdownController`` export.
    """
    return ShutdownController(
        cleanup,
        stop,
        on_shutdown_message=on_shutdown_message,
        timeout_sec=timeout_sec,
    )
