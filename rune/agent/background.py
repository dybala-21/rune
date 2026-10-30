"""Bounded background runs using the same agent and verification as chat."""

from __future__ import annotations

import asyncio
import time
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

from rune.agent.run_control import RunControl, control_scope
from rune.agent.run_outcome import run_outcome
from rune.safety.approval_context import approval_required
from rune.safety.execution_environment import environment_scope


@dataclass(frozen=True, slots=True)
class BackgroundTask:
    goal: str
    source: str
    workspace: str = ""
    verification: list[str] = field(default_factory=list)
    max_steps: int = 30
    timeout_seconds: float = 120
    token_budget: int = 50_000
    continuation: str = ""
    approved_action: dict | None = None
    file_changes: dict | None = None
    routing: dict | None = None
    owner: str = "local"


_current_task: ContextVar[BackgroundTask | None] = ContextVar("background_task", default=None)


def current_background_task() -> BackgroundTask | None:
    return _current_task.get()


async def run_background(task: BackgroundTask, *, loop_factory=None) -> dict[str, Any]:
    from rune.daemon.maintenance import check_storage
    check_storage()
    from rune.agent.loop import NativeAgentLoop
    from rune.types import AgentConfig
    from rune.utils.paths import normalize_path, user_workspace

    if task.max_steps < 1 or task.timeout_seconds <= 0 or task.token_budget < 1:
        raise ValueError("Background execution requires positive limits")
    workspace = str(normalize_path(task.workspace)) if task.workspace else str(user_workspace())
    cfg = AgentConfig(
        max_iterations=task.max_steps, timeout_seconds=task.timeout_seconds,
        token_budget_override=task.token_budget,
    )
    loop = (loop_factory or NativeAgentLoop)(config=cfg)
    blocked: list[str] = []
    pending = None
    from rune.agent.execution_journal import (
        active_journal,
        recovery_context,
        recovery_written_files,
    )
    from rune.agent.request_budget import BudgetExceeded
    from rune.safety.approval_request import current_request, execute_approved_action
    from rune.safety.execution_environment import execution_config

    journal = active_journal()

    async def needs_approval(capability: str, reason: str) -> bool:
        nonlocal pending
        blocked.append(capability)
        if pending is None:
            pending = {"action": current_request(), "reason": reason[:4000],
                       "environment": execution_config().model_dump(mode="json")}
        control.stop()
        return False

    loop.set_approval_callback(needs_approval)
    run_id = f"{task.source}_{uuid4().hex}"
    control = RunControl(run_id)
    if journal is not None:
        journal.approval = needs_approval
    started = time.monotonic()
    result: dict[str, Any] = {
        "run_id": run_id, "source": task.source, "workspace": workspace, "success": False,
        "verified": False, "status": "unverified", "output": "", "error": None,
    }
    scope = _current_task.set(task)
    try:
        with control_scope(control), approval_required(), environment_scope(workspace):
            async with asyncio.timeout(task.timeout_seconds):
                from rune.proactive.routine_routing import restore, routing_key, snapshot

                key = routing_key(task.goal, workspace)
                classification = restore(task.routing, key, task.goal) if journal is not None and journal.previous is not None else None
                async def remember_routing(value):
                    if saved := snapshot(key, value, task.goal):
                        result["routing"] = saved
                if on := getattr(loop, "on", None):
                    on("goal_classified", remember_routing)
                validate = None
                if task.verification:
                    from rune.agent.goal_validate import make_validate_fn
                    validate = make_validate_fn(cwd=workspace, timeout_s=task.timeout_seconds, auto_root=False)
                if task.approved_action is not None:
                    if journal is None:
                        raise ValueError("Approval continuation requires a durable execution journal")
                    await execute_approved_action(task.approved_action, journal, loop.emit)
                    result["approved_action"] = {"tool": task.approved_action["tool"],
                                                 "status": "executed", "verification": "pending"}
                context = {"workspace_root": workspace, "task_owner": task.owner}
                continuation = task.continuation
                if journal is not None and journal.previous is not None:
                    context["recovery_written_files"] = recovery_written_files(journal.previous)
                    continuation = recovery_context({"runId": journal.run_id, "fileChanges": task.file_changes,
                                                     "completedAction": result.get("approved_action")}, journal.previous)
                extra = {"extra_system_context": continuation} if continuation else {}
                if classification is not None:
                    extra["classification"] = classification
                trace = await loop.run(task.goal, context=context, max_steps=task.max_steps, **extra)
                outcome = run_outcome(trace)
                if validate and outcome.success:
                    verification, evidence = await validate(task.verification)
                    result["evidence"] = evidence
                    outcome = replace(outcome, verification_required=True,
                                      verification="passed" if verification else "failed")
                result.update(
                    success=outcome.success,
                    verified=outcome.verified,
                    status=outcome.status,
                    outcome=outcome.payload(),
                    iterations=int(getattr(trace, "final_step", 0) or 0),
                    output=getattr(trace, "answer", "") or getattr(loop, "_last_answer_text", ""),
                    error=getattr(trace, "error", None),
                )
                if outcome.reason == "request_budget_exhausted":
                    detail = (getattr(trace, "completion_check", None) or {}).get("detail")
                    result.update(status="budget_exhausted", error=detail or "The model request exceeded the remaining budget.")
                if result.get("approved_action") and outcome.completion == "completed":
                    result["approved_action"]["verification"] = outcome.verification
    except BudgetExceeded as exc:
        result.update(status="budget_exhausted", error=str(exc))
    except TimeoutError:
        result["error"] = "Time limit reached; inspect external state before retrying."
        result["execution_unknown"] = True
        result["status"] = "interrupted"
    finally:
        _current_task.reset(scope)
        control.stop()
        result["duration_ms"] = round((time.monotonic() - started) * 1000, 1)
        result["timings"] = getattr(loop, "_last_run_timings", {})
        if blocked and not result.get("execution_unknown"):
            result.update(status="needs_approval", success=False, verified=False,
                          error="User approval is required to continue.", blocked_tools=sorted(set(blocked)),
                          pending_approval=pending)
            if "outcome" in result:
                result["outcome"].update(completionStatus="incomplete", reason="approval_required")
    return result
