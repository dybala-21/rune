"""Agent Scheduler - priority-aware async task scheduler.

Ported from src/agent/scheduler.ts (263 lines).
Provides a bounded priority queue with preemption support, stale-task cleanup,
and concurrency limiting for the agent system.
"""

from __future__ import annotations

import asyncio
import heapq
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any
from uuid import uuid4

from rune.utils.events import EventEmitter
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Types

class TaskPriority(IntEnum):
    """Lower numeric value = higher priority."""

    INTERACTIVE = 0
    GATEWAY = 1
    PROACTIVE = 2
    BACKGROUND = 3


@dataclass(slots=True)
class ScheduledTask:
    """A unit of work submitted to the scheduler."""

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    priority: TaskPriority = TaskPriority.BACKGROUND
    sender_id: str = ""
    execute: Callable[[], Coroutine[Any, Any, Any]] = field(
        default_factory=lambda: _noop,
    )
    created_at: float = field(default_factory=time.monotonic)

    # heapq comparison - lower priority value wins, then older first
    def __lt__(self, other: ScheduledTask) -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at


async def _noop() -> None:
    """Default no-op coroutine for ScheduledTask."""


@dataclass(slots=True)
class SchedulerConfig:
    max_concurrency: int = 2
    max_queue_depth: int = 50
    stale_threshold_seconds: float = 300.0


# AgentScheduler

class AgentScheduler(EventEmitter):
    """Priority-aware, concurrency-limited async task scheduler.

    Tasks are enqueued with a :class:`TaskPriority`.  The scheduler runs up to
    *max_concurrency* tasks simultaneously, always preferring higher-priority
    items.  Stale tasks (older than *stale_threshold_seconds*) are cleaned up
    periodically.
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        super().__init__()
        self._config = config or SchedulerConfig()
        self._queue: list[ScheduledTask] = []  # min-heap
        self._active: dict[str, asyncio.Task[Any]] = {}
        self._aborted: set[str] = set()
        self._lock = asyncio.Lock()
        self._processing = False
        self._total_enqueued: int = 0
        self._total_completed: int = 0
        self._total_failed: int = 0
        self._total_aborted: int = 0

    # -- Public API ---------------------------------------------------------

    async def enqueue(self, task: ScheduledTask) -> bool:
        """Add *task* to the scheduler queue.

        Returns ``True`` if the task was accepted, ``False`` if the queue is
        full and the task cannot preempt anything.
        """
        async with self._lock:
            # Cleanup stale before accepting new work
            self._cleanup_stale()

            if len(self._queue) >= self._config.max_queue_depth:
                # Attempt preemption: if the new task is higher priority than
                # the lowest-priority queued item, replace it.
                if self._should_preempt(task):
                    # Remove the lowest-priority (highest numeric value) item
                    worst = max(self._queue, key=lambda t: (t.priority, -t.created_at))
                    self._queue.remove(worst)
                    heapq.heapify(self._queue)
                    log.info(
                        "scheduler_preempt",
                        evicted=worst.id,
                        new_task=task.id,
                    )
                else:
                    log.warning("scheduler_queue_full", task_id=task.id)
                    return False

            heapq.heappush(self._queue, task)
            self._total_enqueued += 1
            log.debug(
                "scheduler_enqueued",
                task_id=task.id,
                priority=task.priority.name,
                queue_depth=len(self._queue),
            )
            await self.emit("enqueued", task)

        # Kick the processing loop (fire-and-forget)
        if not self._processing:
            asyncio.create_task(self._process_queue())

        return True

    def abort(self, task_id: str) -> None:
        """Cancel a task by *task_id*.

        If the task is currently executing, its asyncio.Task is cancelled.
        If it is still queued, it is removed from the queue.
        """
        self._aborted.add(task_id)

        # Cancel if active
        aio_task = self._active.get(task_id)
        if aio_task is not None and not aio_task.done():
            aio_task.cancel()
            log.info("scheduler_abort_active", task_id=task_id)
            return

        # Remove from queue
        self._queue = [t for t in self._queue if t.id != task_id]
        heapq.heapify(self._queue)
        self._total_aborted += 1
        log.info("scheduler_abort_queued", task_id=task_id)

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of scheduler statistics."""
        return {
            "queue_depth": len(self._queue),
            "active": len(self._active),
            "max_concurrency": self._config.max_concurrency,
            "total_enqueued": self._total_enqueued,
            "total_completed": self._total_completed,
            "total_failed": self._total_failed,
            "total_aborted": self._total_aborted,
        }

    # -- Internal -----------------------------------------------------------

    async def _process_queue(self) -> None:
        """Worker loop: pull tasks from the priority queue and execute them."""
        if self._processing:
            return
        self._processing = True
        try:
            while True:
                async with self._lock:
                    # Remove any aborted tasks still in the queue
                    if self._aborted:
                        self._queue = [
                            t for t in self._queue if t.id not in self._aborted
                        ]
                        heapq.heapify(self._queue)

                    if not self._queue:
                        break
                    if len(self._active) >= self._config.max_concurrency:
                        break

                    task = heapq.heappop(self._queue)

                if task.id in self._aborted:
                    self._aborted.discard(task.id)
                    self._total_aborted += 1
                    continue

                # Launch the task
                aio_task = asyncio.create_task(
                    self._run_task(task), name=f"sched-{task.id}"
                )
                self._active[task.id] = aio_task

                # If we still have capacity, keep pulling
                if len(self._active) >= self._config.max_concurrency:
                    # Wait for at least one active task to finish
                    if self._active:
                        done, _ = await asyncio.wait(
                            self._active.values(),
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        # Cleanup done tasks (handled in _run_task callbacks)

        finally:
            self._processing = False

        # If there are still queued items and capacity, recurse
        if self._queue and len(self._active) < self._config.max_concurrency:
            asyncio.create_task(self._process_queue())

    async def _run_task(self, task: ScheduledTask) -> None:
        """Execute a single scheduled task with error handling."""
        try:
            log.debug("scheduler_task_start", task_id=task.id)
            await self.emit("task_start", task)
            await task.execute()
            self._total_completed += 1
            log.debug("scheduler_task_complete", task_id=task.id)
            await self.emit("task_complete", task)
        except asyncio.CancelledError:
            self._total_aborted += 1
            log.info("scheduler_task_cancelled", task_id=task.id)
            await self.emit("task_cancelled", task)
        except Exception as exc:
            self._total_failed += 1
            log.error("scheduler_task_error", task_id=task.id, error=str(exc))
            await self.emit("task_error", task, exc)
        finally:
            self._active.pop(task.id, None)
            self._aborted.discard(task.id)

    def _should_preempt(self, new_task: ScheduledTask) -> bool:
        """Return ``True`` if *new_task* should evict the lowest-priority
        queued item."""
        if not self._queue:
            return False
        worst = max(self._queue, key=lambda t: (t.priority, -t.created_at))
        return new_task.priority < worst.priority

    def _cleanup_stale(self) -> None:
        """Remove tasks that have been queued longer than the stale threshold."""
        now = time.monotonic()
        threshold = self._config.stale_threshold_seconds
        before = len(self._queue)
        self._queue = [
            t for t in self._queue
            if (now - t.created_at) < threshold
        ]
        removed = before - len(self._queue)
        if removed:
            heapq.heapify(self._queue)
            log.info("scheduler_stale_cleanup", removed=removed)


# Singleton

_scheduler_instance: AgentScheduler | None = None


def get_agent_scheduler(config: SchedulerConfig | None = None) -> AgentScheduler:
    """Return the global :class:`AgentScheduler` singleton."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = AgentScheduler(config)
    return _scheduler_instance
