"""Apply lifecycle deadlines and continue cleanup after individual failures."""

import asyncio
import inspect
import os
import threading
import time

from rune.utils.logger import get_logger

log = get_logger(__name__)
STARTUP_TIMEOUT = 30
CANCEL_TIMEOUT = 1
_unfinished: set[asyncio.Future] = set()


def _track(task: asyncio.Future) -> None:
    _unfinished.add(task)

    def finished(result):
        _unfinished.discard(result)
        if not result.cancelled() and (error := result.exception()) is not None:
            log.warning("cancelled_work_failed", error=str(error))

    task.add_done_callback(finished)


class IncompleteShutdown(RuntimeError):
    """Work is still running; retain process ownership until exit."""


async def bounded(awaitable, seconds: float):
    task = asyncio.ensure_future(awaitable)
    try:
        done, _ = await asyncio.wait((task,), timeout=max(0, seconds))
        if not done:
            raise TimeoutError("Operation did not finish before its deadline")
        return task.result()
    finally:
        if not task.done():
            task.cancel()
            _track(task)


async def start_until_stopped(start, stopped: asyncio.Event) -> bool:
    startup = asyncio.create_task(start())
    stop = asyncio.create_task(stopped.wait())
    try:
        done, _ = await asyncio.wait((startup, stop), timeout=STARTUP_TIMEOUT,
                                     return_when=asyncio.FIRST_COMPLETED)
        if stopped.is_set():
            return False
        if startup not in done:
            raise TimeoutError("Rune startup did not finish within 30 seconds")
        await startup
        return True
    finally:
        for task in (startup, stop):
            if not task.done():
                task.cancel()
        _, pending = await asyncio.wait((startup, stop), timeout=CANCEL_TIMEOUT)
        for task in (startup, stop):
            if task in pending:
                _track(task)
            elif not task.cancelled():
                task.exception()
        if pending:
            raise IncompleteShutdown("Startup did not stop after cancellation")


class Cleanup:
    def __init__(self, seconds: float = 23) -> None:
        self.deadline = time.monotonic() + seconds
        self.errors: list[str] = []

    async def run(self, name: str, action, timeout: float = 2) -> None:
        async def invoke():
            result = action()
            if inspect.isawaitable(result):
                return await result
            return result

        try:
            await bounded(invoke(), min(timeout, max(0, self.deadline - time.monotonic())))
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                raise
            self.errors.append(name)
            log.warning("subsystem_shutdown_cancelled", name=name)
        except Exception as exc:
            self.errors.append(name)
            log.warning("subsystem_shutdown_failed", name=name, error=str(exc))

    async def cancel(self, name: str, task: asyncio.Task | None) -> None:
        if task is not None:
            task.cancel()
            await self.run(name, lambda: asyncio.gather(task, return_exceptions=True))

    def check(self) -> None:
        if self.errors:
            raise RuntimeError(f"Shutdown could not finish: {', '.join(self.errors)}")


class ProcessDeadline:
    """Exit a stuck daemon when async cleanup cannot finish."""

    def __init__(self) -> None:
        self._timer = None
        self._lock = threading.Lock()
        self._generation = 0

    def arm(self, seconds: float) -> None:
        with self._lock:
            self._generation += 1
            generation = self._generation
            if self._timer:
                self._timer.cancel()

            def expire():
                with self._lock:
                    if generation == self._generation:
                        try:
                            os.write(2, b"Rune could not stop safely; exiting to release process ownership.\n")
                        finally:
                            os._exit(1)

            self._timer = threading.Timer(seconds, expire)
            self._timer.daemon = True
            self._timer.start()

    def disarm(self) -> None:
        with self._lock:
            self._generation += 1
            if self._timer:
                self._timer.cancel()
                self._timer = None


def run_foreground(daemon) -> None:
    guard = ProcessDeadline()
    daemon._process_deadline = guard
    guard.arm(STARTUP_TIMEOUT + 35)
    try:
        asyncio.run(daemon.serve_forever())
    finally:
        guard.disarm()
