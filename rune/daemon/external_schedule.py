"""Let an outside scheduler own the clock for cron jobs.

Hosted VMs may be stopped between tasks, so their timers belong to a control plane that
wakes the VM and fires each occurrence. Set ``RUNE_EXTERNAL_SCHEDULER=1`` to stop the
local dispatcher from firing jobs; approval continuations still run locally. A run that stops
for a connector approval waits on the host's approval page, and the control plane continues it
with ``connector.decided`` once the owner decides.

Scheduled minutes are interpreted in this machine's local time zone, exactly like the
local dispatcher; run hosted machines in UTC so the control plane computes the same minutes.

An occurrence is identified by ``(job id, scheduled minute)``, the same key the local
dispatcher claims (``cron:<job>:<minute>``). That key is the idempotency key: firing an
occurrence twice runs it once, so at-least-once delivery from the scheduler is safe.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

_inflight: dict[str, asyncio.Task[None]] = {}
# Occurrences that finished without claiming a record (job changed, run limit reached,
# execution disabled meanwhile). Kept in memory only: after a restart a re-fire is
# re-evaluated from scratch, which is safe because nothing ran.
_not_run: dict[str, str] = {}
_stopping = False
# Reasons another fire of the same occurrence may still succeed (while within grace).
_RETRYABLE = frozenset({"busy", "rate_limited", "disabled", "failed_to_start"})
_REASONS = {
    "busy": "The previous run of this job is still holding it",
    "rate_limited": "The hourly routine run limit was reached",
    "disabled": "Scheduled execution was disabled before it started",
    "expired": "It waited for a free execution slot past its misfire grace period",
    "changed": "The job definition changed before it started",
    "deleted": "The job was deleted before it started",
    "inactive": "The job was paused or passed its deadline before it started",
    "run_limit": "The job reached its run limit before it started",
    "failed_to_start": "It failed before claiming the occurrence; see the daemon log",
}


def external_scheduler() -> bool:
    """True when a control plane, not the local dispatcher, fires cron occurrences."""
    return os.environ.get("RUNE_EXTERNAL_SCHEDULER", "").strip() == "1"


def occurrence_id(job_id: str, minute: int) -> str:
    return f"cron:{job_id}:{minute}"


def _parse(job_id: object, minute: object) -> tuple[str, int]:
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("jobId is required")
    if isinstance(minute, bool) or not isinstance(minute, int) or minute < 0:
        raise ValueError("scheduledMinute must be a Unix minute (seconds since the epoch // 60)")
    return job_id, minute


def occurrence(job_id: object, minute: object) -> dict[str, Any]:
    """Report one occurrence: its state and, once finished, the verified result."""
    from rune.proactive.execution_store import ExecutionStore
    from rune.utils.paths import rune_data

    job_id, minute = _parse(job_id, minute)
    key = occurrence_id(job_id, minute)
    store = ExecutionStore(rune_data() / "routine-executions.db")
    try:
        record = store.get(key)
        state = store.state(key)
    finally:
        store.close()
    task = _inflight.get(key)
    if record is None and task is not None and not task.done():
        state = "accepted"
    elif record is None and key in _not_run:
        reason = _not_run[key]
        return {"occurrenceId": key, "state": "not_run", "result": None, "reason": reason,
                "message": _REASONS.get(reason, reason), "retryable": reason in _RETRYABLE}
    return {"occurrenceId": key, "state": state, "result": record[1] if record else None}


async def fire(job_id: object, minute: object) -> dict[str, Any]:
    """Start one scheduled occurrence in the background, or report the one already claimed."""
    from rune.capabilities.cron import _get_store, _row_to_cronjob
    from rune.config.loader import get_config
    from rune.daemon.heartbeat import HeartbeatScheduler

    job_id, minute = _parse(job_id, minute)
    key = occurrence_id(job_id, minute)
    if _stopping:
        raise ValueError("The daemon is shutting down; nothing was started")
    current = occurrence(job_id, minute)
    if current["state"] != "unavailable" and not current.get("retryable"):
        return {**current, "replayed": True}

    config = get_config().cron_execution
    if not config.enabled:
        raise ValueError("Scheduled execution is disabled on this machine; nothing was started")
    row = _get_store().get_cron_job(job_id)  # type: ignore[no-untyped-call]
    if row is None:
        raise ValueError(f"Unknown cron job: {job_id}")
    job = _row_to_cronjob(row)
    if job.status != "active" or job.policy.expired:
        raise ValueError("The job is not active or its deadline has passed; nothing was started")
    if row.get("max_runs") is not None and row.get("run_count", 0) >= row["max_runs"]:
        raise ValueError("The job has reached its run limit; nothing was started")
    now = int(time.time() // 60)
    if minute > now:
        raise ValueError("scheduledMinute is in the future; fire occurrences when they are due")
    if now - minute > config.misfire_grace_seconds // 60:
        raise ValueError("The occurrence is past its misfire grace period and will not run")
    if not HeartbeatScheduler._matches_cron(job.schedule, datetime.fromtimestamp(minute * 60)):
        raise ValueError("scheduledMinute is not on this job's schedule")

    from rune.capabilities.cron import execute_cron_job

    def skipped(reason: str) -> None:
        _not_run[key] = reason
        while len(_not_run) > 1000:
            _not_run.pop(next(iter(_not_run)))

    async def run() -> None:
        try:
            reason = await execute_cron_job(job, scheduled_minute=minute)
            if reason is not None and occurrence(job_id, minute)["state"] in {"unavailable", "accepted"}:
                skipped(reason)
        except Exception as exc:
            # A claimed occurrence records its own outcome; one that failed earlier stays re-fireable.
            log.warning("external_cron_fire_failed", occurrence=key, error=type(exc).__name__)
            if occurrence(job_id, minute)["state"] in {"unavailable", "accepted"}:
                skipped("failed_to_start")
        finally:
            _inflight.pop(key, None)

    _not_run.pop(key, None)
    if key not in _inflight:
        _inflight[key] = asyncio.create_task(run(), name=f"external-cron:{key}")
    return {"occurrenceId": key, "state": "accepted", "result": None, "replayed": False}


def _parked(approval_id: str) -> tuple[Any, str, str, str | None, str | None] | None:
    """The scheduled run stopped on this broker approval, as (job, operation id, its status, the
    continuation's id and state), or None."""
    from rune.capabilities.cron import _get_store, _row_to_cronjob
    from rune.proactive.execution_store import ExecutionStore
    from rune.utils.paths import rune_data

    records = ExecutionStore(rune_data() / "routine-executions.db")
    try:
        for row in _get_store().list_cron_jobs():  # type: ignore[no-untyped-call]
            for run in records.recent(f"cron:{row['id']}:"):
                result = run["result"] or {}
                action = (result.get("pending_approval") or {}).get("action") or {}
                # A continued or dismissed run keeps its pending action, so it is still found here.
                if (result.get("status") in {"needs_approval", "continued", "dismissed"}
                        and action.get("brokerApprovalId") == approval_id):
                    child = records.continued(run["id"])
                    return (_row_to_cronjob(row), run["id"], result["status"], child,
                            records.state(child) if child else None)
        return None
    finally:
        records.close()


async def continue_decided(approval_id: object, approved: object) -> dict[str, Any]:
    """Act on the owner's decision on the approval page for the scheduled run that stopped on it.

    Approved: the run continues and sends the request again, and the broker runs it. Declined:
    the run is dismissed and its job paused, as when the owner declines in the app; nothing is
    sent. Safe to call again: while a continuation runs it reports ``accepted``, and afterwards
    the state it ended in. ``unknown`` means no run here is waiting on that approval.
    """
    if not isinstance(approval_id, str) or not approval_id.isalnum() or not 0 < len(approval_id) <= 64:
        raise ValueError("approvalId must be the broker's approval id")
    if not isinstance(approved, bool):
        raise ValueError("approved must be true or false")
    key = f"approval:{approval_id}"
    if (task := _inflight.get(key)) is not None and not task.done():
        return {"approvalId": approval_id, "state": "accepted"}
    found = _parked(approval_id)
    if found is None:
        return {"approvalId": approval_id, "state": "unknown"}
    job, operation_id, status, child, state = found
    if child is not None:
        return {"approvalId": approval_id, "operationId": child, "state": state}
    if status == "dismissed":
        return {"approvalId": approval_id, "operationId": operation_id, "state": "dismissed"}
    if not approved:
        _dismiss(job.id, operation_id)
        return {"approvalId": approval_id, "operationId": operation_id, "state": "dismissed"}
    if _stopping:
        raise ValueError("The daemon is shutting down; nothing was started")
    from rune.capabilities.cron import execute_cron_job

    async def run() -> None:
        try:
            await execute_cron_job(job, approval_from=operation_id)
        except Exception as exc:
            # The run is still waiting on its approval, so the host's next call tries again.
            log.warning("approval_continuation_failed", operation=operation_id, error=type(exc).__name__)
        finally:
            _inflight.pop(key, None)

    _inflight[key] = asyncio.create_task(run(), name=f"approval-continuation:{operation_id}")
    return {"approvalId": approval_id, "operationId": operation_id, "state": "accepted"}


def _dismiss(job_id: str, operation_id: str) -> None:
    """What declining in the app does: pause the job first, so nothing reclaims it, then close the run."""
    from rune.capabilities.cron import _get_store
    from rune.proactive.execution_store import ExecutionStore
    from rune.utils.paths import rune_data

    _get_store().update_cron_job(job_id, enabled=False)  # type: ignore[no-untyped-call]
    records = ExecutionStore(rune_data() / "routine-executions.db")
    try:
        records.dismiss_approval(f"cron:{job_id}", operation_id)
    finally:
        records.close()


async def shutdown() -> None:
    """Refuse new fires and cancel in-flight ones so they end as interrupted, not half-torn-down."""
    global _stopping
    _stopping = True
    tasks = list(_inflight.values())
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
