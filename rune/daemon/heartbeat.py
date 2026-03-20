"""Heartbeat and cron scheduler for RUNE.

Ported from src/daemon/heartbeat.ts - periodic health checks and
basic cron expression matching for scheduled tasks.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

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
        self._tasks[name] = _ScheduledTask(
            name=name, cron_expr=cron_expr, callback=callback
        )
        log.info("cron_task_added", name=name, expr=cron_expr)

    def remove_task(self, name: str) -> None:
        """Remove a scheduled task by name."""
        removed = self._tasks.pop(name, None)
        if removed:
            log.info("cron_task_removed", name=name)

    async def _heartbeat_loop(self) -> None:
        """Main loop that runs at the configured interval."""
        try:
            while self._running:
                now = datetime.now()

                for task in list(self._tasks.values()):
                    if self._matches_cron(task.cron_expr, now):
                        # Avoid running the same task twice in the same minute
                        if (
                            task.last_run is not None
                            and task.last_run.replace(second=0, microsecond=0)
                            == now.replace(second=0, microsecond=0)
                        ):
                            continue

                        task.last_run = now
                        try:
                            await task.callback()
                            log.debug("cron_task_executed", name=task.name)
                        except Exception as exc:
                            log.error(
                                "cron_task_failed",
                                name=task.name,
                                error=str(exc),
                            )

                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass

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
