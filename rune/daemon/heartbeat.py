"""Heartbeat and cron scheduler for RUNE.

Ported from src/daemon/heartbeat.ts - periodic health checks and
basic cron expression matching for scheduled tasks.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from rune.utils.logger import get_logger

log = get_logger(__name__)

type CronCallback = Callable[[], Awaitable[None]]


@dataclass(slots=True)
class _ScheduledTask:
    """A task scheduled via cron expression."""

    name: str
    cron_expr: str
    callback: CronCallback
    last_run: datetime | None = None
    running: asyncio.Task[None] | None = None
    failures: int = 0
    last_success: float | None = None


class HeartbeatScheduler:
    """Periodic heartbeat loop with cron-style task scheduling.

    The scheduler wakes up at the configured interval, checks each
    registered cron task, and fires matching callbacks.
    """

    def __init__(self, interval_seconds: float = 60.0) -> None:
        self._interval = interval_seconds
        self._tasks: dict[str, _ScheduledTask] = {}
        self._running = False
        self._loop_task: asyncio.Task[None] | None = None
        self._last_tick: datetime | None = None
        self._inflight: set[asyncio.Task] = set()

    @property
    def is_running(self) -> bool:
        return bool(self._running and self._loop_task and not self._loop_task.done())

    def healthy(self, name: str) -> bool:
        task = self._tasks.get(name)
        return bool(task and not task.failures and task.last_success is not None
                    and time.monotonic() - task.last_success <= max(180, self._interval * 3))

    async def start(self) -> None:
        """Start the heartbeat loop."""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._heartbeat_loop())
        log.info("heartbeat_started", interval=self._interval)

    async def stop(self) -> None:
        """Stop the heartbeat loop."""
        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None
        pending = list(self._inflight)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        log.info("heartbeat_stopped")

    def add_task(
        self, name: str, cron_expr: str, callback: CronCallback
    ) -> None:
        """Register a scheduled task with a cron expression.

        Cron format: ``minute hour day month weekday``
        Each field supports:
            - ``*``       (any)
            - ``N``       (exact value)
            - ``N,M``     (list)
            - ``N-M``     (range)
            - ``*/N``     (step)
        """
        self.remove_task(name)
        self._tasks[name] = _ScheduledTask(
            name=name, cron_expr=cron_expr, callback=callback
        )
        log.info("cron_task_added", name=name, expr=cron_expr)

    def remove_task(self, name: str) -> None:
        """Remove a scheduled task by name."""
        removed = self._tasks.pop(name, None)
        if removed:
            if removed.running is not None:
                removed.running.cancel()
            log.info("cron_task_removed", name=name)

    async def _invoke(self, task: _ScheduledTask) -> None:
        try:
            async with asyncio.timeout(120):
                await task.callback()
        except Exception as exc:
            task.failures += 1
            log.error("cron_task_failed", name=task.name, error=str(exc))
        else:
            task.failures = 0
            task.last_success = time.monotonic()

    def tick(self, now: datetime | None = None) -> None:
        now = (now or datetime.now()).replace(second=0, microsecond=0)
        if self._last_tick is not None and now <= self._last_tick:
            return
        first = max(self._last_tick or now - timedelta(minutes=1), now - timedelta(minutes=5))
        self._last_tick = now
        for task in list(self._tasks.values()):
            if task.running is not None and not task.running.done():
                continue
            due = now
            while due > first:
                if self._matches_cron(task.cron_expr, due):
                    if task.last_run is None or due > task.last_run:
                        task.last_run = due
                        task.running = asyncio.create_task(self._invoke(task))
                        self._inflight.add(task.running)
                        task.running.add_done_callback(self._inflight.discard)
                    break
                due -= timedelta(minutes=1)

    async def _heartbeat_loop(self) -> None:
        while self._running:
            self.tick()
            await asyncio.sleep(self._interval)

    @staticmethod
    def _matches_cron(expr: str, now: datetime) -> bool:
        """Check if a cron expression matches the given datetime.

        Supports the standard 5-field cron format:
            minute  hour  day-of-month  month  day-of-week

        Each field supports: ``*``, exact number, comma-separated list,
        range (``N-M``), and step (``*/N``).
        """
        parts = expr.strip().split()
        if len(parts) != 5:
            log.warning("invalid_cron_expression", expr=expr)
            return False

        time_values = [
            now.minute,
            now.hour,
            now.day,
            now.month,
            now.isoweekday() % 7,  # 0 = Sunday to match cron convention
        ]

        for field_expr, value in zip(parts, time_values, strict=True):
            if not _matches_cron_field(field_expr, value):
                return False

        return True


def _matches_cron_field(field_expr: str, value: int) -> bool:
    """Check if a single cron field matches a value."""
    # Wildcard
    if field_expr == "*":
        return True

    # Step: */N
    if field_expr.startswith("*/"):
        try:
            step = int(field_expr[2:])
            return step > 0 and value % step == 0
        except ValueError:
            return False

    # Comma-separated list: N,M,O
    if "," in field_expr:
        return any(
            _matches_cron_field(part.strip(), value)
            for part in field_expr.split(",")
        )

    # Range: N-M
    if "-" in field_expr:
        try:
            low, high = field_expr.split("-", 1)
            return int(low) <= value <= int(high)
        except ValueError:
            return False

    # Exact match
    try:
        return int(field_expr) == value
    except ValueError:
        return False
