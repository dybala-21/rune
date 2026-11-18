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
    time: str = Field("07:00", description="When to send it, 24-hour HH:MM in the owner's local time")
    days: Literal["daily", "weekdays"] = Field("daily", description="Every day, or Monday to Friday only")
    enabled: bool = Field(True, description="False stops the briefing")
    notify_channel: str = Field("telegram", description="Channel it is sent to (telegram/discord/slack)")

    @field_validator("time")
    @classmethod
    def clock(cls, value: str) -> str:
        if not _CLOCK.fullmatch(value.strip()):
            raise ValueError("Use 24-hour HH:MM, e.g. 07:00")
        return value.strip()


def _schedule(params: BriefingSetParams) -> str:
    hour, minute = (int(part) for part in params.time.split(":"))
    return f"{minute} {hour} * * {'1-5' if params.days == 'weekdays' else '*'}"


async def briefing_set(params: BriefingSetParams) -> CapabilityResult:
    """Create, move or stop the owner's morning briefing."""
    from rune.briefing.job import KIND

    if blocked := cron._background_change():
        return blocked
    schedule = _schedule(params)
    try:
        store = cron._get_store()  # type: ignore[no-untyped-call]
        current = next((j for j in store.list_cron_jobs() if j["name"] == NAME), None)
        # Always sent: a briefing that only arrives when something changed isn't one.
        policy = RoutinePolicy.model_validate({"notify": "always", "timeout_seconds": 180}).bind_workspace()
        command = cron._pack_command("", "", params.notify_channel.strip(), "Today's calendar, unread mail and headlines",
                                     policy, kind=KIND)
        if current is None:
            if not params.enabled:
                return CapabilityResult(success=True, output="There is no morning briefing to stop.")
            job_id = store.create_cron_job(name=NAME, schedule=schedule, command=command)
        else:
            job_id = current["id"]
            store.update_cron_job(job_id, schedule=schedule, command=command, enabled=params.enabled)
    except Exception as exc:
        return CapabilityResult(success=False, error=f"The briefing could not be scheduled: {exc}")
    log.info("briefing_scheduled", id=job_id, schedule=schedule, enabled=params.enabled)
    if not params.enabled:
        return CapabilityResult(success=True, output="The morning briefing is stopped.", metadata={"id": job_id})
    when = "every day" if params.days == "daily" else "Monday to Friday"
    return CapabilityResult(success=True, output=(
        f"The morning briefing is set for {params.time} {when} ({cron._zone_label()}), sent to "
        f"{params.notify_channel}. It lists today's calendar, unread mail and headlines; an account that "
        "isn't connected comes with a button to connect it."), metadata={"id": job_id, "schedule": schedule})


def register_briefing_capabilities(registry: CapabilityRegistry) -> None:
    registry.register(CapabilityDefinition(
        name="briefing_set",
        description=("Schedule, move or stop the owner's morning briefing: today's calendar, unread mail and news "
                     "headlines in one message at a set time. Use this, not cron_create, when they ask for a "
                     "daily briefing or morning summary."),
        domain=Domain.SCHEDULE,
        risk_level=RiskLevel.MEDIUM,
        group="schedule",
        parameters_model=BriefingSetParams,
        execute=briefing_set,
    ))
