"""Execution limits and durable claims for explicitly scheduled work."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, Field, field_validator

from rune.proactive.execution_store import ExecutionStore
from rune.utils.paths import rune_data


class RoutinePolicy(BaseModel):
    workspace: str = ""
    max_steps: int = Field(30, ge=1, le=200)
    timeout_seconds: int = Field(120, ge=1, le=1800)
    token_budget: int = Field(50_000, ge=1000, le=500_000)
    total_tokens: int | None = Field(None, ge=0, strict=True)
    total_model_requests: int | None = Field(None, ge=0, strict=True)
    total_cost_usd: float | None = Field(None, ge=0, allow_inf_nan=False, strict=True)
    deadline: AwareDatetime | None = None
    verification: list[str] = Field(default_factory=list, max_length=10)
    notify: Literal["always", "changes", "failures"] = "changes"
    input_paths: list[str] = Field(default_factory=list, max_length=32,
                                  description="Input files or folders, relative to workspace. Skip successful unchanged work only when these and tracked outputs are unchanged. Omit for time-dependent or external-data tasks.")
    output_paths: list[str] = Field(default_factory=list, max_length=32,
                                   description="Output files to compare for changes; missing or modified outputs require another run.")

    @field_validator("input_paths", "output_paths", "verification", mode="before")
    @classmethod
    def omit_blank_lines(cls, value):
        if isinstance(value, list):
            return [item for item in value if not isinstance(item, str) or item.strip()]
        return value

    @property
    def expired(self) -> bool:
        return self.deadline is not None and self.deadline <= datetime.now(UTC)

    def bind_workspace(self) -> RoutinePolicy:
        from rune.utils.paths import normalize_path, user_workspace
        workspace = normalize_path(self.workspace) if self.workspace else user_workspace()
        return self.model_copy(update={"workspace": str(workspace)})

    def patched(self, update: RoutinePolicy) -> RoutinePolicy:
        return RoutinePolicy.model_validate({**self.model_dump(), **update.model_dump(exclude_unset=True)}).bind_workspace()


def task_fingerprint(job) -> str:
    return hashlib.sha256(json.dumps([
        job.command, job.goal, job.schedule, job.notify_channel, job.policy.model_dump(mode="json"),
    ], sort_keys=True).encode()).hexdigest()


def claim_occurrence(job, scheduled_minute: int | None = None) -> tuple[ExecutionStore, str, str]:
    store = ExecutionStore(rune_data() / "routine-executions.db")
    minute = int(time.time() // 60) if scheduled_minute is None else scheduled_minute
    occurrence = f"cron:{job.id}:{minute}"
    # A job run by code only reads and reports, so a run of it that stopped part way holds nothing back.
    decision = store.claim(occurrence, task_fingerprint(job), 1000, resource=f"cron:{job.id}",
                           rerunnable=bool(job.kind))
    return store, occurrence, decision


def code_job(kind: str) -> Any:
    """What runs a job marked with ``kind`` in place of an agent or a shell; None when nothing does."""
    if kind == "briefing":
        from rune.briefing.job import run_briefing

        return run_briefing
    if kind == "watch":
        from rune.watch.job import run_watch

        return run_watch
    return None


def error_text(job: Any, exc: BaseException) -> str:
    """What a failed run says went wrong. A job run by code reads mail and pages, and an error's own
    wording can repeat them: only the kind of error is said, the rest stays in the log."""
    return type(exc).__name__ if job.kind else str(exc)


def rerunnable(job: Any, result: dict[str, Any]) -> dict[str, Any]:
    """A run that stopped part way holds its job back until the owner has looked at what it did
    (ExecutionStore.finish). A job run by code only reads and reports, so there is nothing to look
    at: it is recorded as failed and runs again at its next time."""
    if not job.kind or not (result.get("execution_unknown") or result.get("status") == "interrupted"):
        return result
    settled = {key: value for key, value in result.items() if key != "execution_unknown"}
    if settled.get("status") == "interrupted":
        settled.update(status="failed", error="It stopped before it finished. It runs again at its next scheduled time.")
    return settled


def should_notify(policy: RoutinePolicy, result: dict, previous: dict | None) -> bool:
    if result.get("nothing_to_tell") is True and result.get("success") is True:
        return False  # the run looked and found nothing to say: staying quiet is its answer
    if (result.get("status") == "model_budget" and previous is not None
            and previous.get("status") == "model_budget" and previous.get("error") == result.get("error")):
        return False  # the month's model budget is still used up: said once is enough
    if policy.notify == "always":
        return True
    if policy.notify == "failures" and result.get("status") not in (
        "failed", "unverified", "needs_approval", "interrupted", "budget_exhausted", "model_budget",
    ):
        return False
    if result.get("status") == "unchanged":
        return False
    if previous is not None:
        from rune.proactive.routine_observation import result_identity

        current_id, previous_id = result_identity(result), result_identity(previous)
        prior_status = previous.get("status_before_reuse", previous.get("status"))
        if current_id is not None and previous_id is not None:
            return (current_id != previous_id or result.get("status") != prior_status
                    or result.get("error") != previous.get("error"))
    # Detect changes from saved output, not the model's description.
    return previous is None or any(result.get(key) != previous.get(key) for key in ("status", "output", "error"))


async def run_while_current(job, store, execute) -> dict:
    import asyncio

    from rune.capabilities.cron import _row_to_cronjob

    def current():
        row = store.get_cron_job(job.id)
        return bool(row and _row_to_cronjob(row) == job and not job.policy.expired)

    if not current():
        return {"status": "cancelled", "success": False, "verified": False, "not_started": True,
                "output": "", "error": "The task changed before execution started."}
    task = asyncio.create_task(execute(job))
    try:
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=1)
            if not current():
                return {"status": "interrupted", "verified": False, "execution_unknown": True,
                        "output": "", "error": "The task changed, stopped or expired. Inspect its effects before resuming."}
            if done:
                break
        return await task
    finally:
        if not task.done():
            task.cancel()
        # Do not release the execution lease until the worker has stopped.
        try:
            await task
        except asyncio.CancelledError:
            pass
