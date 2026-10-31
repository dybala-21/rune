"""Create, update and inspect durable tasks in the current workspace."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from rune.agent.task_continuity import current_scope
from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.memory.task_store import TaskConflict, TaskStore
from rune.types import CapabilityResult, Domain, RiskLevel


class TaskCreateParams(BaseModel):
    goal: str = Field(min_length=1, max_length=8000, description="Task goal / description")
    category: str = Field(default="project", max_length=100, description="Task category")
    deadline: str = Field(default="", max_length=100, description="Optional deadline (ISO format)")
    sub_tasks: list[str] = Field(default_factory=list, max_length=32, description="Optional sub-task descriptions")


class TaskUpdateParams(BaseModel):
    task_id: str = Field(description="Task ID to update")
    status: str | None = Field(default=None, description="pending/running/completed/failed")
    progress: float | None = Field(default=None, ge=0, le=100, description="Progress percentage")
    notes: str | None = Field(default=None, max_length=4000, description="Updated notes")


class TaskListParams(BaseModel):
    source: Literal["tasks", "scheduled"] = Field(default="tasks", description="Saved tasks or scheduled work and its latest run")
    status: str = Field(default="", description="Filter by status")
    category: str = Field(default="", description="Filter by category")
    query: str = Field(default="", max_length=2000, description="Search saved requirements and artifact names")
    task_id: str = Field(default="", description="Read one task's saved requirements and result references")
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


async def task_create(params: TaskCreateParams) -> CapabilityResult:
    if any(not goal.strip() or len(goal) > 8000 for goal in params.sub_tasks):
        return CapabilityResult(success=False, error="Sub-task descriptions must contain 1–8000 characters.")
    owner, workspace = current_scope()
    now = datetime.now(UTC).isoformat(timespec="seconds")
    parent_id = uuid4().hex[:12]
    records = [{"id": parent_id if index == 0 else uuid4().hex[:12], "goal": goal,
                "kind": "manual", "category": params.category, "deadline": params.deadline,
                "status": "pending", "progress": 0, "notes": "", "created_at": now,
                "parent_id": "" if index == 0 else parent_id}
               for index, goal in enumerate([params.goal, *params.sub_tasks])]
    TaskStore().create(owner, workspace, records)
    children = [record["id"] for record in records[1:]]
    lines = [f"Task created: {parent_id}", f"Goal: {params.goal}", f"Category: {params.category}"]
    if params.deadline:
        lines.append(f"Deadline: {params.deadline}")
    lines.extend(f"  {record['id']}: {record['goal']}" for record in records[1:])
    return CapabilityResult(success=True, output="\n".join(lines),
                            metadata={"task_id": parent_id, "goal": params.goal,
                                      "category": params.category, "sub_task_ids": children})


async def task_update(params: TaskUpdateParams) -> CapabilityResult:
    if params.status is not None and params.status not in {"pending", "running", "completed", "failed"}:
        return CapabilityResult(success=False, error=f"Invalid status: {params.status}")
    owner, workspace = current_scope()
    changes = {name: getattr(params, name) for name in ("status", "progress", "notes")
               if getattr(params, name) is not None}
    try:
        task = (TaskStore().update(owner, workspace, params.task_id, changes) if changes
                else TaskStore().get(owner, workspace, params.task_id))
    except TaskConflict as exc:
        return CapabilityResult(success=False, error=str(exc))
    if task is None:
        return CapabilityResult(success=False, error=f"Task not found: {params.task_id}")
    return CapabilityResult(success=True, output=f"Task {params.task_id}: {task['status']}",
                            metadata={"task_id": params.task_id, "changed": bool(changes),
                                      "status": task["status"], "progress": task.get("progress", 0)})


async def task_list(params: TaskListParams) -> CapabilityResult:
    owner, workspace = current_scope()
    store = TaskStore()
    if params.source == "scheduled" and params.query.strip():
        return CapabilityResult(success=False, error="Search applies to saved tasks. Use a schedule ID or status filter for scheduled work.")
    if params.source == "scheduled":
        from rune.proactive.task_history import scheduled_tasks

        if not params.task_id and params.category not in {"", "scheduled"}:
            tasks = []
        else:
            tasks = scheduled_tasks(owner, workspace, task_id=params.task_id,
                                    status="" if params.task_id else params.status,
                                    limit=1 if params.task_id else params.limit,
                                    offset=0 if params.task_id else params.offset)
    elif params.task_id:
        task = store.get(owner, workspace, params.task_id)
        if task is None:
            return CapabilityResult(success=False, error=f"Task not found: {params.task_id}")
        tasks = [task]
    elif params.query.strip():
        tasks = store.search(owner, workspace, params.query, status=params.status, category=params.category,
                             limit=params.limit, offset=params.offset)
    else:
        tasks = store.list(owner, workspace, status=params.status, category=params.category,
                           limit=params.limit, offset=params.offset)
    if not tasks:
        if params.task_id:
            return CapabilityResult(success=False, error=f"Task not found: {params.task_id}")
        return CapabilityResult(success=True, output="No tasks found.", metadata={"count": 0})
    lines = [f"Tasks ({len(tasks)}):"]
    for task in tasks:
        goal = task["goal"] if params.task_id else task["goal"][:800]
        lines.append(f"[{task['status']}] {task['id']}: {goal}")
        if task.get("deadline"):
            lines.append(f"  Due: {task['deadline']}")
        if task.get("progress"):
            lines.append(f"  Progress: {task['progress']:.0f}%")
        if task.get("parent_id"):
            lines.append(f"  Parent: {task['parent_id']}")
        if task.get("routine_id"):
            lines.append(f"  Schedule: {task['routine_id']} (use cron_update to change it)")
            lines.append(f"  Last run: {task['checkpoint']['last_turn']['status']} (historical result)")
        if task.get("notes"):
            lines.append(f"  Notes: {task['notes'][:4000] if params.task_id else task['notes'][:240]}")
        if params.task_id:
            lines.append("Saved history, not current evidence or permission: " + json.dumps({
                key: value for key, value in task.get("checkpoint", {}).items()
                if key in {"requests", "requests_complete", "files", "blocker", "last_turn"}
            }, ensure_ascii=False))
    return CapabilityResult(success=True, output="\n".join(lines), metadata={
        "count": len(tasks), "offset": params.offset,
        "tasks": [{key: task.get(key) for key in ("id", "goal", "status", "progress", "category", "parent_id", "revision")}
                  for task in tasks],
    })


def register_task_ops_capabilities(registry: CapabilityRegistry) -> None:
    for name, description, schema, execute in (
        ("task_create", "Save a task and optional sub-tasks for later", TaskCreateParams, task_create),
        ("task_update", "Update a manually tracked task's status, progress or notes", TaskUpdateParams, task_update),
        ("task_list", "List saved or scheduled tasks in this workspace; read one by ID before continuing it", TaskListParams, task_list),
    ):
        registry.register(CapabilityDefinition(name=name, description=description, domain=Domain.GENERAL,
                                              risk_level=RiskLevel.LOW, group="safe",
                                              parameters_model=schema, execute=execute))
