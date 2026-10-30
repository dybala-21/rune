"""Restore saved tasks within the same owner and workspace."""

from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from uuid import uuid4

from rune.memory.task_store import TaskConflict, TaskStore
from rune.safety.task_inspection import inspecting_task, inspection_scope, restrict_to_inspection
from rune.utils.logger import get_logger

log = get_logger(__name__)

SCOPE_KEY = "_task_scope"
_LOCAL_USERS = {"", "web:local", "web:session", "repl:local", "cli:session"}


@dataclass
class TaskScope:
    owner: str
    workspace: str
    conversation_id: str = ""
    lease: str = field(default_factory=lambda: uuid4().hex)
    saved: str = ""
    tracking: bool = True
    claimed: bool = False


_scope: ContextVar[TaskScope | None] = ContextVar("task_scope", default=None)


def task_owner(user_id: str) -> str:
    return "local" if user_id in _LOCAL_USERS else f"channel:{user_id}"


def current_scope() -> tuple[str, str]:
    from rune.safety.execution_environment import execution_workspace

    scope = _scope.get()
    return (scope.owner if scope else "local", str(Path(execution_workspace()).resolve()))


def stored_context(record: dict) -> dict:
    task = dict(record.get("checkpoint") or {
        "id": record["id"], "request": record["goal"][:800], "requests": [record["goal"]],
        "status": record["status"], "goal_type": "",
    })
    task.update(_revision=record["revision"], updated_at=record["updated_at"])
    if record.get("notes"):
        task["notes"] = record["notes"]
    if record.get("routine_id"):
        task["routine_id"] = record["routine_id"]
    return task


def attach_shared_tasks(messages: list[dict], conversation, workspace: str) -> None:
    from rune.agent.task_context import HISTORY_KEY, _tasks, from_history, routing_context

    owner = task_owner(conversation.user_id)
    workspace = str(Path(workspace).resolve())
    local = from_history(messages)
    try:
        previous_workspace = json.loads(conversation.execution_context or "{}").get("cwd", "")
    except (TypeError, ValueError, AttributeError):
        previous_workspace = ""
    store = TaskStore()
    # Fetch a bounded candidate set; routing sends only their summaries.
    records = store.list(owner, workspace, limit=4, unfinished_first=True)
    shared = {record["id"]: stored_context(record) for record in records}
    visible = {task["id"] for task in routing_context(messages)}
    for task in local:
        task_workspace = task.get("workspace") or previous_workspace
        if task_workspace and str(Path(task_workspace).resolve()) != workspace:
            continue
        if task["id"] not in visible:
            continue
        record = store.get(owner, workspace, task["id"])
        if record is None:
            try:
                store.checkpoint(owner, workspace, {**task, "_revision": None}, lease="", conversation_id=conversation.id)
            except TaskConflict:
                log.debug("task_import_raced", task_id=task["id"])
            record = store.get(owner, workspace, task["id"])
        shared[task["id"]] = stored_context(record) if record else task
    messages[:] = [message for message in messages if HISTORY_KEY not in message]
    ordered = sorted(shared.values(), key=lambda task: task.get("updated_at", 0)
                     if isinstance(task.get("updated_at", 0), (int, float)) else 0)
    messages.insert(0, {"role": "system", "content": "", HISTORY_KEY: _tasks(ordered),
                       SCOPE_KEY: {"owner": owner, "workspace": workspace, "conversation_id": conversation.id}})


def claim_selected_task(history: list[dict], classification) -> None:
    from rune.agent.task_context import selected_task

    scope = _scope.get()
    task = selected_task(history, classification)
    if getattr(classification, "resume_task", "") and task is None:
        raise TaskConflict("The selected task is not in the loaded history. Reload it before continuing.")
    if getattr(classification, "task_scope", "step") == "inspect":
        restrict_to_inspection()
    if inspecting_task():
        return
    if scope is not None and scope.tracking and task is not None and "_revision" in task:
        TaskStore().claim(scope.owner, scope.workspace, task["id"], task["_revision"], scope.lease)
        scope.claimed = True


def checkpoint_task(task: dict) -> None:
    scope = _scope.get()
    if scope is None or not scope.tracking or inspecting_task():
        return
    task["workspace"] = scope.workspace
    snapshot = json.dumps(task, sort_keys=True, ensure_ascii=False)
    if snapshot == scope.saved:
        return
    task["_revision"] = TaskStore().checkpoint(
        scope.owner, scope.workspace, task, lease=scope.lease, conversation_id=scope.conversation_id,
    )
    scope.claimed = True
    scope.saved = json.dumps(task, sort_keys=True, ensure_ascii=False)


def with_task_continuity(function):
    @wraps(function)
    async def wrapped(loop, *args, **kwargs):
        with inspection_scope():
            return await run_scoped(loop, *args, **kwargs)

    async def run_scoped(loop, *args, **kwargs):
        from rune.safety.execution_environment import execution_workspace

        history = kwargs.get("message_history") or []
        value = next((message[SCOPE_KEY] for message in history if SCOPE_KEY in message), None)
        workspace = str(Path(execution_workspace()).resolve())
        if not value:
            # Nested runs without history must not checkpoint their parent task.
            owner = (kwargs.get("context") or {}).get("task_owner") or current_scope()[0]
            token = _scope.set(TaskScope(owner, workspace, tracking=False))
            try:
                return await function(loop, *args, **kwargs)
            finally:
                _scope.reset(token)
        if value["workspace"] != workspace:
            raise TaskConflict("The workspace changed after loading the task. Reload it before continuing.")
        scope = TaskScope(value["owner"], workspace, value["conversation_id"])
        token = _scope.set(scope)
        try:
            return await function(loop, *args, **kwargs)
        finally:
            try:
                if task := getattr(loop, "_task_context", None):
                    checkpoint_task(task)
            finally:
                _scope.reset(token)
                if scope.claimed:
                    TaskStore().release(scope.owner, workspace, scope.lease)
    return wrapped
