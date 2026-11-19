"""Set up the morning briefing from a chat: one sentence names the time, and it is scheduled."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from rune.capabilities import cron
from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.proactive.routine import RoutinePolicy
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)

NAME = "Morning briefing"
_CLOCK = re.compile(r"([01]?\d|2[0-3]):([0-5]\d)")


class BriefingSetParams(BaseModel):
    language: str = Field(description="The language the briefing is written in, as a BCP-47 tag such as ko, en or ja: "
                                      "the one the owner's own messages are written in (read it from their words, "
                                      "never from a time zone or place), or another one if they ask for it")
    time: str | None = Field(None, description="When to send it, 24-hour HH:MM in the owner's local time. Leave out "
                                               "to keep the time it has (07:00 for a new one)")
    days: Literal["daily", "weekdays"] | None = Field(None, description="Every day, or Monday to Friday only. Leave "
                                                                        "out to keep what it has (daily for a new one)")
    enabled: bool = Field(True, description="False stops the briefing")
    notify_channel: str | None = Field(None, description="Channel it is sent to (telegram/discord/slack). Leave out "
                                                         "to keep the one it has (telegram for a new one)")

    @field_validator("language")
    @classmethod
    def language_tag(cls, value: str) -> str:
        from rune.briefing.labels import tag

        if not tag(value):
            raise ValueError("Use a language tag such as ko, en or ja")
        return tag(value)

    @field_validator("time")
    @classmethod
    def clock(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _CLOCK.fullmatch(value.strip()):
            raise ValueError("Use 24-hour HH:MM, e.g. 07:00")
        return value.strip()


def _schedule(params: BriefingSetParams, current: str = "") -> str:
    """The cron line: what was asked for, and for anything left out, what the briefing already has."""
    fields = current.split()
    minute, hour, month_day, month, weekdays = fields if len(fields) == 5 else ("0", "7", "*", "*", "*")
    if params.time is not None:
        hour, minute = (str(int(part)) for part in params.time.split(":"))
    if params.days is not None:
        weekdays = "1-5" if params.days == "weekdays" else "*"
    return f"{minute} {hour} {month_day} {month} {weekdays}"


def _when(schedule: str) -> str:
    """The schedule as the owner is told it; one this tool didn't write is given as it stands."""
    minute, hour, month_day, month, weekdays = schedule.split()
    days = {"*": "every day", "1-5": "Monday to Friday"}.get(weekdays)
    if not (hour.isdigit() and minute.isdigit() and days and (month_day, month) == ("*", "*")):
        return f"the schedule {schedule!r}"
    return f"{int(hour):02d}:{int(minute):02d} {days}"


async def briefing_set(params: BriefingSetParams) -> CapabilityResult:
    """Create, move or stop the owner's morning briefing."""
    from rune.briefing import labels
    from rune.briefing.job import KIND

    if blocked := cron._background_change():
        return blocked
    try:
        store = cron._get_store()  # type: ignore[no-untyped-call]
        current = next((j for j in store.list_cron_jobs() if j["name"] == NAME), None)
        if current is None and not params.enabled:
            return CapabilityResult(success=True, output="There is no morning briefing to stop.")
        # Before anything is scheduled: if it can't be kept, no briefing goes out in a language
        # the owner didn't ask for.
        labels.set_language(params.language)
        had = cron._row_to_cronjob(current) if current is not None else None
        schedule = _schedule(params, had.schedule if had is not None else "")
        channel = (params.notify_channel or "").strip() or (had.notify_channel if had is not None else "") or "telegram"
        # "always", so a read that failed is always said. A day with nothing to report is the
        # job's own call (rune.briefing.job): it answers with nothing to tell, and that isn't sent.
        policy = RoutinePolicy.model_validate({"notify": "always", "timeout_seconds": 180}).bind_workspace()
        command = cron._pack_command("", "", channel, "Today's calendar, unread mail and headlines", policy, kind=KIND)
        if current is None:
            job_id = store.create_cron_job(name=NAME, schedule=schedule, command=command)
        else:
            job_id = current["id"]
            store.update_cron_job(job_id, schedule=schedule, command=command, enabled=params.enabled)
    except Exception as exc:
        return CapabilityResult(success=False, error=f"The briefing could not be scheduled: {exc}")
    log.info("briefing_scheduled", id=job_id, schedule=schedule, enabled=params.enabled, language=params.language)
    if not params.enabled:
        return CapabilityResult(success=True, output="The morning briefing is stopped.", metadata={"id": job_id})
    return CapabilityResult(success=True, output=(
        f"The morning briefing is set for {_when(schedule)} ({cron._zone_label()}), sent to {channel}, written in "
        f"the owner's language ({params.language}). It lists today's calendar, unread mail and headlines; an account "
        "that isn't connected comes with a button to connect it. On a day with nothing to report, nothing is sent."),
        metadata={"id": job_id, "schedule": schedule})


def register_briefing_capabilities(registry: CapabilityRegistry) -> None:
    registry.register(CapabilityDefinition(
        name="briefing_set",
        description=("Schedule, move or stop the owner's morning briefing: today's calendar, unread mail and news "
                     "headlines in one message at a set time, in the owner's language. Use this, not cron_create, "
                     "when they ask for a daily briefing or morning summary, or to change its time or language."),
        domain=Domain.SCHEDULE,
        risk_level=RiskLevel.MEDIUM,
        group="schedule",
        parameters_model=BriefingSetParams,
        execute=briefing_set,
    ))
