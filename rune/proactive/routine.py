"""Execution limits and durable claims for explicitly scheduled work."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field, field_validator

from rune.proactive.execution_store import ExecutionStore
from rune.utils.paths import rune_data


class RoutinePolicy(BaseModel):
    workspace: str = ""
    max_steps: int = Field(30, ge=1, le=200)
    timeout_seconds: int = Field(120, ge=1, le=1800)
    token_budget: int = Field(50_000, ge=1000, le=500_000)
    deadline: AwareDatetime | None = None
    verification: list[str] = Field(default_factory=list, max_length=10)
    notify: Literal["always", "changes", "failures"] = "changes"
    input_paths: list[str] = Field(default_factory=list, max_length=32,
                                  description="Explicit input files, relative to workspace. Skip successful unchanged work only when these and tracked outputs are unchanged. Omit for time-dependent or external-data tasks.")
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


def claim_occurrence(job) -> tuple[ExecutionStore, str, str]:
    store = ExecutionStore(rune_data() / "routine-executions.db")
    fingerprint = hashlib.sha256(json.dumps([
        job.command, job.goal, job.schedule, job.notify_channel, job.policy.model_dump(mode="json"),
    ], sort_keys=True).encode()).hexdigest()
    occurrence = f"cron:{job.id}:{int(time.time() // 60)}"
    decision = store.claim(occurrence, fingerprint, 1000, resource=f"cron:{job.id}")
    return store, occurrence, decision


def should_notify(policy: RoutinePolicy, result: dict, previous: dict | None) -> bool:
    if policy.notify == "always":
        return True
    if policy.notify == "failures":
        return result.get("status") in ("failed", "unverified", "needs_approval", "interrupted")
    if result.get("status") == "unchanged":
        return False
    if previous is not None:
        from rune.proactive.routine_observation import result_identity

        current_id, previous_id = result_identity(result), result_identity(previous)
        prior_status = previous.get("status_before_reuse", previous.get("status"))
        if current_id is not None and previous_id is not None:
            return (current_id != previous_id or result.get("status") != prior_status
                    or result.get("error") != previous.get("error"))
    # Compare actual output; a model's claim that nothing changed is not an oracle.
    return previous is None or any(result.get(key) != previous.get(key) for key in ("status", "output"))


async def run_while_current(job, store, execute) -> dict:
    import asyncio

    from rune.capabilities.cron import _row_to_cronjob

    task = asyncio.create_task(execute(job))
    try:
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=1)
            if done:
                break
            row = store.get_cron_job(job.id)
            current = _row_to_cronjob(row) if row else None
            if current != job or job.policy.expired:
                return {"status": "interrupted", "verified": False, "execution_unknown": True,
                        "output": "", "error": "The task changed, stopped or expired. Inspect its effects before resuming."}
        return await task
    finally:
        if not task.done():
            task.cancel()
        # Do not release the execution lease until the worker has stopped.
        try:
            await task
        except asyncio.CancelledError:
            pass
