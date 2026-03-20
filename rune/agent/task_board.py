"""Shared Task Board - concurrent work-stealing board for multi-agent orchestration.

Ported from src/agent/task-board.ts (275 lines).
Provides atomic claim/complete/fail semantics, dependency tracking, circular
dependency validation, and an asyncio.Event-based change notification so workers
can sleep until new work is available.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)


# Forward-reference: SubTask / SubTaskResult live in orchestrator.py but we
# need them here.  To avoid circular imports we re-declare compatible
# dataclasses and accept either via duck-typing.

class ClaimableStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    DONE = "done"
    FAILED = "failed"


@dataclass(slots=True)
class SubTask:
    """Minimal SubTask representation used by the task board.

    Kept in sync with :class:`rune.agent.orchestrator.SubTask`.
    """

    id: str
    description: str = ""
    role: str = "executor"
    dependencies: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 60_000
    priority: int = 0  # Lower value = higher priority (0 = highest)


@dataclass(slots=True)
class SubTaskResult:
    """Result of a completed (or failed) sub-task."""

    task_id: str
    success: bool
    output: str = ""
    error: str | None = None
    duration_ms: float = 0.0


@dataclass(slots=True)
class ClaimableTask:
    """A :class:`SubTask` enriched with board-level lifecycle metadata."""

    id: str
    description: str = ""
    role: str = "executor"
    dependencies: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 60_000
    priority: int = 0

    status: ClaimableStatus = ClaimableStatus.PENDING
    claimed_by: str | None = None
    result: SubTaskResult | None = None
    started_at: float | None = None
    completed_at: float | None = None

    @classmethod
    def from_subtask(cls, task: SubTask) -> ClaimableTask:
        return cls(
            id=task.id,
            description=task.description,
            role=task.role,
            dependencies=list(task.dependencies),
            params=dict(task.params),
            timeout_ms=task.timeout_ms,
            priority=task.priority,
        )


# SharedTaskBoard

class SharedTaskBoard:
    """Thread-safe (asyncio-safe) task board with dependency-aware scheduling.

    Workers call :meth:`claim` to atomically take the next ready task,
    :meth:`complete` / :meth:`fail` to report results, and
    :meth:`wait_for_change` to sleep until there is new work.
    """

    __slots__ = ("_tasks", "_lock", "_change_event")

    def __init__(self, tasks: list[SubTask] | None = None) -> None:
        self._tasks: dict[str, ClaimableTask] = {}
        self._lock = asyncio.Lock()
        self._change_event = asyncio.Event()

        if tasks:
            self._validate_no_cycles(tasks)
            for t in tasks:
                self._tasks[t.id] = ClaimableTask.from_subtask(t)

    # -- Validation ---------------------------------------------------------

    @staticmethod
    def _validate_no_cycles(tasks: list[SubTask]) -> None:
        """Raise :class:`ValueError` if there is a circular dependency."""
        ids = {t.id for t in tasks}
        adj: dict[str, list[str]] = {t.id: list(t.dependencies) for t in tasks}

        # Validate that all dependency references exist
        for t in tasks:
            for dep in t.dependencies:
                if dep not in ids:
                    raise ValueError(
                        f"Task {t.id!r} depends on unknown task {dep!r}"
                    )

        # Kahn's algorithm for topological sort - cycle detection
        in_degree: dict[str, int] = {tid: 0 for tid in ids}
        for tid, deps in adj.items():
            in_degree[tid] = len(deps)

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        visited = 0

        while queue:
            node = queue.pop()
            visited += 1
            for tid, deps in adj.items():
                if node in deps:
                    in_degree[tid] -= 1
                    if in_degree[tid] == 0:
                        queue.append(tid)

        if visited != len(ids):
            raise ValueError("Circular dependency detected among tasks")

    # -- Core operations ----------------------------------------------------

    async def claim(self, task_id: str, worker_id: str) -> ClaimableTask | None:
        """Atomically claim a pending task.  Returns ``None`` if the task is
        not pending or does not exist."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != ClaimableStatus.PENDING:
                return None
            # Verify all dependencies are done
            for dep_id in task.dependencies:
                dep = self._tasks.get(dep_id)
                if dep is None or dep.status != ClaimableStatus.DONE:
                    return None
            task.status = ClaimableStatus.CLAIMED
            task.claimed_by = worker_id
            task.started_at = time.monotonic()
            log.debug("task_claimed", task_id=task_id, worker_id=worker_id)
            self._notify()
            return task

    async def complete(self, task_id: str, result: SubTaskResult) -> None:
        """Mark a claimed task as done with a successful result."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"Unknown task: {task_id!r}")
            task.status = ClaimableStatus.DONE
            task.result = result
            task.completed_at = time.monotonic()
            log.debug("task_completed", task_id=task_id, success=result.success)
            self._notify()

    async def fail(self, task_id: str, error: str) -> None:
        """Mark a task as failed and propagate failure to all dependents."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"Unknown task: {task_id!r}")
            task.status = ClaimableStatus.FAILED
            task.result = SubTaskResult(
                task_id=task_id, success=False, error=error,
            )
            task.completed_at = time.monotonic()

            # Cascade failure to direct and transitive dependents
            self._propagate_failure(task_id, error)
            log.warning("task_failed", task_id=task_id, error=error)
            self._notify()

    def _propagate_failure(self, failed_id: str, root_error: str) -> None:
        """Mark all tasks that transitively depend on *failed_id* as failed."""
        stack = [failed_id]
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for tid, task in self._tasks.items():
                if current in task.dependencies and task.status in (
                    ClaimableStatus.PENDING,
                    ClaimableStatus.CLAIMED,
                ):
                    task.status = ClaimableStatus.FAILED
                    task.result = SubTaskResult(
                        task_id=tid,
                        success=False,
                        error=f"Dependency {failed_id!r} failed: {root_error}",
                    )
                    task.completed_at = time.monotonic()
                    stack.append(tid)

    def get_ready_tasks(self) -> list[ClaimableTask]:
        """Return all pending tasks whose dependencies are satisfied, sorted by priority.

        Lower priority value = higher priority (executed first).
        """
        ready: list[ClaimableTask] = []
        for task in self._tasks.values():
            if task.status != ClaimableStatus.PENDING:
                continue
            deps_done = all(
                self._tasks.get(dep_id) is not None
                and self._tasks[dep_id].status == ClaimableStatus.DONE
                for dep_id in task.dependencies
            )
            if deps_done:
                ready.append(task)
        # Sort by priority (lower value = higher priority)
        ready.sort(key=lambda t: t.priority)
        return ready

    def is_done(self) -> bool:
        """Return ``True`` when every task has reached a terminal state."""
        return all(
            t.status in (ClaimableStatus.DONE, ClaimableStatus.FAILED)
            for t in self._tasks.values()
        )

    async def add_task(self, task: SubTask) -> None:
        """Dynamically add a new task to the board."""
        async with self._lock:
            if task.id in self._tasks:
                raise ValueError(f"Duplicate task id: {task.id!r}")
            # Validate deps exist
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise ValueError(
                        f"Task {task.id!r} depends on unknown task {dep_id!r}"
                    )
            self._tasks[task.id] = ClaimableTask.from_subtask(task)
            log.debug("task_added", task_id=task.id)
            self._notify()

    def get_dependency_context(self, task_id: str) -> dict[str, Any]:
        """Return the outputs of all resolved dependencies for *task_id*."""
        task = self._tasks.get(task_id)
        if task is None:
            return {}
        context: dict[str, Any] = {}
        for dep_id in task.dependencies:
            dep = self._tasks.get(dep_id)
            if dep is not None and dep.result is not None:
                context[dep_id] = {
                    "success": dep.result.success,
                    "output": dep.result.output,
                    "error": dep.result.error,
                }
        return context

    def get_results(self) -> dict[str, SubTaskResult]:
        """Return a mapping of task_id → :class:`SubTaskResult` for all
        completed or failed tasks."""
        results: dict[str, SubTaskResult] = {}
        for tid, task in self._tasks.items():
            if task.result is not None:
                results[tid] = task.result
        return results

    def summary(self) -> dict[str, int]:
        """Return a summary of task statuses."""
        counts: dict[str, int] = {
            "total": len(self._tasks),
            "pending": 0,
            "claimed": 0,
            "completed": 0,
            "failed": 0,
        }
        for task in self._tasks.values():
            match task.status:
                case ClaimableStatus.PENDING:
                    counts["pending"] += 1
                case ClaimableStatus.CLAIMED:
                    counts["claimed"] += 1
                case ClaimableStatus.DONE:
                    counts["completed"] += 1
                case ClaimableStatus.FAILED:
                    counts["failed"] += 1
        return counts

    async def wait_for_change(self, timeout: float | None = None) -> None:
        """Block until the board state changes or *timeout* seconds elapse."""
        self._change_event.clear()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._change_event.wait(), timeout=timeout)

    # -- Internal helpers ---------------------------------------------------

    def _notify(self) -> None:
        """Signal waiting workers that the board state has changed."""
        self._change_event.set()
