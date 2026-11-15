"""Start a long task, and end a step of one (rune.proactive.long_task)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from rune.capabilities import cron
from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.proactive.routine import RoutinePolicy
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)


class TaskStartParams(BaseModel):
    name: str = Field(description="Unique name for the task")
    goal: str = Field(description="What the whole task must achieve, written so each step can pick it up")
    steps: int = Field(20, ge=1, le=500, description="The most steps it may take; this machine may allow fewer")
    step_minutes: int = Field(15, ge=1, le=30, description="The longest one step runs")
    total_tokens: int | None = Field(None, ge=1000, description="Tokens over all steps; unset is this machine's most")
    notify_channel: str = Field(default="", description="Channel to tell the owner on (telegram/discord/slack); unset routes by priority")
    description: str = Field(default="", description="Human-readable description")


class TaskParkParams(BaseModel):
    progress: str = Field(description="What this step did and what the next step should do, for that step to read")
    resume_in_minutes: int = Field(1, ge=1, le=7 * 24 * 60,
                                   description="When the next step starts; longer when waiting on the outside world")
    notify: bool = Field(False, description="Tell the owner now: a finding, a milestone or half way. Not for routine progress")
    done: bool = Field(False, description="The whole task is done; progress is then its final answer, which the owner is told")



async def task_start(params: TaskStartParams) -> CapabilityResult:
    """Start a long task: steps of bounded work with parks in between, until it is done."""
    import time

    from rune.proactive import long_task

    if blocked := cron._background_change():
        return blocked
    if not params.goal.strip() or not params.name.strip():
        return CapabilityResult(success=False, error="A long task needs a name and a goal.")
    steps = min(params.steps, long_task.step_limit())
    tokens = min(params.total_tokens or long_task.token_limit(), long_task.token_limit())
    try:
        store = cron._get_store()
        if any(j["name"] == params.name for j in store.list_cron_jobs()):
            return CapabilityResult(success=False, error=f"A task or cron job named '{params.name}' already exists.")
        policy = RoutinePolicy(timeout_seconds=params.step_minutes * 60, total_tokens=tokens, notify="failures",
                               max_steps=60, token_budget=min(200_000, tokens)).bind_workspace()
        first = int(time.time() // 60)
        job_id = store.create_cron_job(
            name=params.name, schedule=long_task.SCHEDULE, max_runs=steps,
            command=cron._pack_command("", params.goal.strip(), params.notify_channel.strip(), params.description,
                                  policy, task={"wake_at": first}),
        )
    except Exception as exc:
        return CapabilityResult(success=False, error=f"Failed to start the task: {exc}")
    log.info("long_task_started", name=params.name, id=job_id, steps=steps)
    return CapabilityResult(success=True, output=(
        f"Long task '{params.name}' started (id: {job_id}). Its first step starts within a minute and runs "
        f"in the background, up to {params.step_minutes} minutes a step, {steps} steps and {tokens:,} tokens "
        "in all. The owner is told when it is done, stops or needs them. Stop it with cron_update "
        "(enabled=false) or cron_delete."), metadata={"id": job_id, "name": params.name, "steps": steps})


async def task_park(params: TaskParkParams) -> CapabilityResult:
    """End this step of a long task; the next one starts later with this note."""
    from rune.proactive import long_task

    try:
        resume_at = long_task.park(params.progress, params.resume_in_minutes, params.notify, params.done)
    except ValueError as exc:
        return CapabilityResult(success=False, error=str(exc), metadata={"action_status": "not_executed"})
    if params.done:
        return CapabilityResult(success=True, output="Done. The task ends with this answer, and the owner is told.")
    return CapabilityResult(success=True, output=(
        f"Parked. This step ends now; the next one starts at minute {resume_at} (in about "
        f"{params.resume_in_minutes} min) with this note."))


def register_task_capabilities(registry: CapabilityRegistry) -> None:
    registry.register(CapabilityDefinition(
        name="task_start",
        description=(
            "Start a long task that runs in the background for hours or days, as steps of bounded work "
            "with pauses in between (each step ends with task_park). Use it for work too big for one run: "
            "a large research, a long batch, something to watch until it changes. Use cron_create for work "
            "on a clock instead."
        ),
        domain=Domain.SCHEDULE,
        risk_level=RiskLevel.MEDIUM,
        group="schedule",
        parameters_model=TaskStartParams,
        execute=task_start,
    ))
    registry.register(CapabilityDefinition(
        name="task_park",
        description=(
            "In a long task's step: end this step once it has done a bounded chunk of work, with a note of "
            "what is done and what comes next. The next step starts later and reads the note. When the "
            "whole task is done, call it with done=true and the final answer as progress."
        ),
        domain=Domain.SCHEDULE,
        risk_level=RiskLevel.LOW,
        group="schedule",
        parameters_model=TaskParkParams,
        execute=task_park,
    ))
