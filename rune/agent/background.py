"""Bounded background runs using the same agent and verification as chat."""

from __future__ import annotations

import asyncio
import time
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


async def run_background(task: BackgroundTask, *, loop_factory=None) -> dict[str, Any]:
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

    async def needs_approval(capability: str, reason: str) -> bool:
        blocked.append(capability)
        return False

    loop.set_approval_callback(needs_approval)
    run_id = f"{task.source}_{uuid4().hex}"
    control = RunControl(run_id)
    started = time.monotonic()
    result: dict[str, Any] = {
        "run_id": run_id, "source": task.source, "workspace": workspace, "success": False,
        "verified": False, "status": "unverified", "output": "", "error": None,
    }
    try:
        with control_scope(control), approval_required(), environment_scope(workspace):
            async with asyncio.timeout(task.timeout_seconds):
                validate = None
                if task.verification:
                    from rune.agent.goal_validate import make_validate_fn
                    validate = make_validate_fn(cwd=workspace, timeout_s=task.timeout_seconds, auto_root=False)
                trace = await loop.run(task.goal, context={"workspace_root": workspace}, max_steps=task.max_steps)
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
    except TimeoutError:
        result["error"] = "Time limit reached; inspect external state before retrying."
        result["execution_unknown"] = True
        result["status"] = "interrupted"
    finally:
        control.stop()
        result["duration_ms"] = round((time.monotonic() - started) * 1000, 1)
        result["timings"] = getattr(loop, "_last_run_timings", {})
        if blocked and not result["verified"] and not result.get("execution_unknown"):
            result.update(status="needs_approval", success=False, verified=False,
                          error="User approval is required to continue.", blocked_tools=sorted(set(blocked)))
            if "outcome" in result:
                result["outcome"].update(completionStatus="incomplete", reason="approval_required")
    return result
