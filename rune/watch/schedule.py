"""The one scheduled job that wakes the watch, kept in step with the owner's check time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any

from rune.watch.job import KIND

NAME = "Watch"
_DESCRIPTION = "Prices on the pages the owner asked to watch"
# Above the job's own deadline (rune.watch.job), so a run ends itself and always gives a result.
_TIMEOUT_SECONDS = 600


@dataclass(frozen=True, slots=True)
class Standing:
    """How the job stands: "ok", or what keeps the watch from running ("missing", "paused",
    "moved" when its schedule is no longer the check time's)."""

    state: str
    channel: str = ""
    job_id: str = ""


def line(at: time) -> str:
    """The cron line for a check time.

    A machine of its own is woken every hour and reads once a day, so one that was off at the time
    still reads when it is back on. A hosted machine is woken by its host only when a job is due:
    at the time, and an hour later for the second reading of a price that moved.
    """
    from rune.daemon.external_schedule import external_scheduler

    if external_scheduler():
        return f"{at.minute} {at.hour},{(at.hour + 1) % 24} * * *"
    return f"{at.minute} * * * *"


def standing(at: time) -> Standing:
    row, job = _found()
    if row is None or job is None:
        return Standing("missing")
    state = "paused" if not row.get("enabled", True) else "ok" if job.schedule == line(at) else "moved"
    return Standing(state, job.notify_channel, str(row["id"]))


def keep(at: time) -> Standing:
    """Create the job, or put it back as it should be: running, on the check time's schedule."""
    from rune.capabilities import cron
    from rune.proactive.routine import RoutinePolicy

    store = cron._get_store()  # type: ignore[no-untyped-call]
    row, job = _found()
    if row is not None and row.get("enabled", True) and job.schedule == line(at):
        return Standing("ok", job.notify_channel, str(row["id"]))
    channel = (job.notify_channel if job is not None else "") or _briefing_channel() or "telegram"
    # "changes": a wake with nothing to tell says so itself and isn't sent (rune.watch.job), and a
    # run that failed is said once, not at every wake of the same day.
    policy = RoutinePolicy.model_validate({"notify": "changes", "timeout_seconds": _TIMEOUT_SECONDS}).bind_workspace()
    command = cron._pack_command("", "", channel, _DESCRIPTION, policy, kind=KIND)
    if row is None:
        job_id = store.create_cron_job(name=NAME, schedule=line(at), command=command)
    else:
        job_id = str(row["id"])
        store.update_cron_job(job_id, schedule=line(at), command=command, enabled=True)
    return Standing("ok", channel, job_id)


def _found() -> tuple[dict[str, Any] | None, Any]:
    """The watch's job, known by what it is marked with: a job that only shares its name is not it."""
    from rune.capabilities import cron

    for row in cron._get_store().list_cron_jobs():  # type: ignore[no-untyped-call]
        job = cron._row_to_cronjob(row)
        if job.kind == KIND:
            return row, job
    return None, None


def _briefing_channel() -> str:
    """Where the morning briefing goes, so the first watch is told in the same place."""
    from rune.briefing.job import KIND as BRIEFING
    from rune.capabilities import cron

    for row in cron._get_store().list_cron_jobs():  # type: ignore[no-untyped-call]
        job = cron._row_to_cronjob(row)
        if job.kind == BRIEFING and job.notify_channel:
            return str(job.notify_channel)
    return ""
