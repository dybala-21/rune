"""Task operations capabilities for RUNE.

Ported from src/capabilities/task-ops.ts - create, update, and list
tasks on the agent task board.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)


# In-memory task store

@dataclass(slots=True)
class TaskRecord:
    """A user-facing task tracked by the agent."""
    id: str
    goal: str
    category: str = "project"
    status: str = "pending"  # pending | running | completed | failed
    progress: float = 0.0  # 0-100
    notes: str = ""
    deadline: str = ""
    created_at: str = ""
    updated_at: str = ""


_tasks: dict[str, TaskRecord] = {}


# Parameter schemas

class TaskCreateParams(BaseModel):
    goal: str = Field(description="Task goal / description")
    category: str = Field(default="project",
                          description="Category (project/personal/maintenance)")
    deadline: str = Field(default="", description="Optional deadline (ISO format)")
    sub_tasks: list[str] = Field(default_factory=list, description="Optional list of sub-task descriptions")


class TaskUpdateParams(BaseModel):
    task_id: str = Field(description="Task ID to update")
    status: str | None = Field(default=None,
                               description="New status (pending/running/completed/failed)")
    progress: float | None = Field(default=None,
                                   description="Progress percentage (0-100)")
    notes: str | None = Field(default=None, description="Additional notes")


class TaskListParams(BaseModel):
    status: str = Field(default="", description="Filter by status")
    category: str = Field(default="", description="Filter by category")


# Implementations

_VALID_STATUSES = {"pending", "running", "completed", "failed"}


async def task_create(params: TaskCreateParams) -> CapabilityResult:
    """Create a new task."""
    log.debug("task_create", goal=params.goal[:80], category=params.category)

    now = datetime.now().isoformat(timespec="seconds")
    task_id = uuid4().hex[:12]

    task = TaskRecord(
        id=task_id,
        goal=params.goal,
        category=params.category,
        deadline=params.deadline,
        created_at=now,
        updated_at=now,
    )
    _tasks[task_id] = task

    log.info("task_created", task_id=task_id, goal=params.goal[:80])

    # Create sub-tasks if provided
    sub_task_ids: list[str] = []
    for sub_desc in params.sub_tasks:
        sub_id = uuid4().hex[:12]
        sub_task = TaskRecord(
            id=sub_id,
            goal=sub_desc,
            category=params.category,
            deadline=params.deadline,
            notes=f"Sub-task of {task_id}",
            created_at=now,
            updated_at=now,
        )
        _tasks[sub_id] = sub_task
        sub_task_ids.append(sub_id)
        log.info("sub_task_created", task_id=sub_id, parent=task_id, goal=sub_desc[:80])

    output_lines = [
        f"Task created: {task_id}",
        f"Goal: {params.goal}",
        f"Category: {params.category}",
        f"Deadline: {params.deadline or '(none)'}",
    ]
    if sub_task_ids:
        output_lines.append(f"Sub-tasks ({len(sub_task_ids)}):")
        for sid, sdesc in zip(sub_task_ids, params.sub_tasks, strict=False):
            output_lines.append(f"  {sid}: {sdesc}")

    return CapabilityResult(
        success=True,
        output="\n".join(output_lines),
        metadata={
            "task_id": task_id,
            "goal": params.goal,
            "category": params.category,
            "sub_task_ids": sub_task_ids,
        },
    )


async def task_update(params: TaskUpdateParams) -> CapabilityResult:
    """Update an existing task's status, progress, or notes."""
    log.debug("task_update", task_id=params.task_id)

    task = _tasks.get(params.task_id)
    if task is None:
        return CapabilityResult(
            success=False,
            error=f"Task not found: {params.task_id}",
        )

    changes: list[str] = []

    if params.status is not None:
        if params.status not in _VALID_STATUSES:
            return CapabilityResult(
                success=False,
                error=f"Invalid status '{params.status}'. "
                      f"Valid: {', '.join(sorted(_VALID_STATUSES))}",
            )
        task.status = params.status
        changes.append(f"status={params.status}")

    if params.progress is not None:
        task.progress = max(0.0, min(100.0, params.progress))
        changes.append(f"progress={task.progress:.0f}%")

    if params.notes is not None:
        task.notes = params.notes
        changes.append("notes updated")

    task.updated_at = datetime.now().isoformat(timespec="seconds")

    if not changes:
        return CapabilityResult(
            success=True,
            output=f"No changes specified for task {params.task_id}",
            metadata={"task_id": params.task_id, "changed": False},
        )

    log.info("task_updated", task_id=params.task_id, changes=changes)

    return CapabilityResult(
        success=True,
        output=f"Task {params.task_id} updated: {', '.join(changes)}",
        metadata={
            "task_id": params.task_id,
            "changed": True,
            "status": task.status,
            "progress": task.progress,
        },
    )


async def task_list(params: TaskListParams) -> CapabilityResult:
    """List tasks with optional status and category filters."""
    log.debug("task_list", status=params.status, category=params.category)

    tasks = list(_tasks.values())

    if params.status:
        tasks = [t for t in tasks if t.status == params.status]
    if params.category:
        tasks = [t for t in tasks if t.category == params.category]

    if not tasks:
        filters: list[str] = []
        if params.status:
            filters.append(f"status={params.status}")
        if params.category:
            filters.append(f"category={params.category}")
        filter_msg = f" ({', '.join(filters)})" if filters else ""
        return CapabilityResult(
            success=True,
            output=f"No tasks found{filter_msg}.",
            metadata={"count": 0},
        )

    lines: list[str] = [f"Tasks ({len(tasks)}):"]
    for t in tasks:
        progress_str = f" {t.progress:.0f}%" if t.progress > 0 else ""
        deadline_str = f" (due: {t.deadline})" if t.deadline else ""
        lines.append(f"  [{t.status}]{progress_str} {t.id}: {t.goal}{deadline_str}")
        if t.notes:
            lines.append(f"    Notes: {t.notes}")
        lines.append(f"    Category: {t.category}  Updated: {t.updated_at}")

    return CapabilityResult(
        success=True,
        output="\n".join(lines),
        metadata={
            "count": len(tasks),
            "tasks": [
                {
                    "id": t.id,
                    "goal": t.goal,
                    "status": t.status,
                    "progress": t.progress,
                    "category": t.category,
                }
                for t in tasks
            ],
        },
    )


# Registration

def register_task_ops_capabilities(registry: CapabilityRegistry) -> None:
    """Register task operation capabilities."""
    registry.register(CapabilityDefinition(
        name="task_create",
        description="Create a new task",
        domain=Domain.GENERAL,
        risk_level=RiskLevel.LOW,
        group="safe",
        parameters_model=TaskCreateParams,
        execute=task_create,
    ))
    registry.register(CapabilityDefinition(
        name="task_update",
        description="Update a task's status, progress, or notes",
        domain=Domain.GENERAL,
        risk_level=RiskLevel.LOW,
        group="safe",
        parameters_model=TaskUpdateParams,
        execute=task_update,
    ))
    registry.register(CapabilityDefinition(
        name="task_list",
        description="List tasks with optional filters",
        domain=Domain.GENERAL,
        risk_level=RiskLevel.LOW,
        group="safe",
        parameters_model=TaskListParams,
        execute=task_list,
    ))
