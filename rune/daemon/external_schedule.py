"""Let an outside scheduler own the clock for cron jobs.

Hosted VMs may be stopped between tasks, so their timers belong to a control plane that
wakes the VM and fires each occurrence. Set ``RUNE_EXTERNAL_SCHEDULER=1`` to stop the
local dispatcher from firing jobs; approval continuations still run locally.

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


async def shutdown() -> None:
    """Refuse new fires and cancel in-flight ones so they end as interrupted, not half-torn-down."""
    global _stopping
    _stopping = True
    tasks = list(_inflight.values())
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
