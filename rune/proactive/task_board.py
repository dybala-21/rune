"""Task board with priority-based sorting for proactive suggestions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Priority = Literal["critical", "high", "medium", "low"]

_PRIORITY_ORDER: dict[str, int] = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@dataclass(slots=True)
class Task:
    id: str
    title: str
    description: str = ""
    priority: Priority = "medium"
    status: Literal["pending", "in_progress", "completed", "cancelled"] = "pending"
    created_at: float = 0.0
    deadline: float | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


class TaskBoard:
    """Priority-sorted task board."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def add(self, task: Task) -> None:
        self._tasks[task.id] = task

    def remove(self, task_id: str) -> Task | None:
        return self._tasks.pop(task_id, None)

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def update_priority(self, task_id: str, priority: Priority) -> None:
        task = self._tasks.get(task_id)
        if task:
            task.priority = priority

    def update_status(self, task_id: str, status: str) -> None:
        task = self._tasks.get(task_id)
        if task:
            task.status = status  # type: ignore[assignment]

    def list_by_priority(self, *, status: str | None = None) -> list[Task]:
        """Return tasks sorted by priority (critical first), then by created_at."""
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        tasks.sort(key=lambda t: (_PRIORITY_ORDER.get(t.priority, 99), t.created_at))
        return tasks

    def pending_count(self) -> int:
        return sum(1 for t in self._tasks.values() if t.status == "pending")

    @property
    def size(self) -> int:
        return len(self._tasks)
