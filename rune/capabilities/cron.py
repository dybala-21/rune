"""Create, list, update and delete tasks in the heartbeat scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.proactive.routine import RoutinePolicy, code_job
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)


import json as _json


@dataclass(slots=True)
class CronJob:
    """Scheduled job loaded from storage."""
    id: str
    name: str
    schedule: str
    command: str = ""
    goal: str = ""
    notify_channel: str = ""
    description: str = ""
    status: str = "active"
    last_run_at: str = ""
    policy: RoutinePolicy = field(default_factory=RoutinePolicy)
    continuation: str = field(default="", compare=False)
    approved_action: dict | None = field(default=None, compare=False)
    file_changes: dict | None = field(default=None, compare=False)
    routing: dict | None = field(default=None, compare=False)
    owner: str = "local"
    # A long task's state (rune.proactive.long_task): {"wake_at": minute | None, "ended": why}.
    task: dict | None = field(default=None, compare=False)
    # What a long task's step is told about the task, ahead of everything else (`step_brief`).
    preamble: str = field(default="", compare=False)
    # Run by code instead of bash or an agent: "briefing" or "watch" (rune.proactive.routine.code_job).
    kind: str = ""

    @property
    def wake_at(self) -> int | None:
        value = (self.task or {}).get("wake_at")
        return value if isinstance(value, int) and not isinstance(value, bool) else None


def _get_store():
    """Import the store lazily to avoid a circular dependency."""
    from rune.memory.store import get_memory_store
    return get_memory_store()


def _row_to_cronjob(row: dict) -> CronJob:
    """Read a job from either a legacy shell command or a JSON payload."""
    raw_cmd = row.get("command", "")
    goal = ""
    notify = ""
    desc = ""
    bash_cmd = raw_cmd
    from rune.config.loader import get_config
    default_timeout = get_config().cron_execution.default_timeout_seconds
    policy = RoutinePolicy(timeout_seconds=default_timeout)

    try:
        payload = _json.loads(raw_cmd)
    except (_json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict):
        bash_cmd = payload.get("bash", "")
        goal = payload.get("goal", "")
        notify = payload.get("notify_channel", "")
        desc = payload.get("description", "")
        policy = RoutinePolicy.model_validate({"timeout_seconds": default_timeout, **payload.get("policy", {})})

    return CronJob(
        id=row["id"],
        name=row["name"],
        schedule=row["schedule"],
        command=bash_cmd,
        goal=goal,
        notify_channel=notify,
        description=desc,
        status="active" if row.get("enabled", True) else "paused",
        last_run_at=row.get("last_run_at", "") or "",
        policy=policy,
        owner=payload.get("owner", "local") if isinstance(payload, dict) else "local",
        task=payload.get("task") if isinstance(payload, dict) and isinstance(payload.get("task"), dict) else None,
        kind=payload.get("kind", "") if isinstance(payload, dict) and isinstance(payload.get("kind"), str) else "",
    )


def _pack_command(command: str, goal: str, notify_channel: str, description: str,
                  policy: RoutinePolicy | None = None, *, owner: str | None = None, task: dict | None = None,
                  kind: str = "") -> str:
    """Keep the goal, owner, execution policy and a long task's state in the existing command column."""
    if goal or notify_channel or policy is not None or task is not None or kind:
        from rune.agent.task_continuity import current_scope

        packed = {
            "bash": command,
            "goal": goal,
            "notify_channel": notify_channel,
            "description": description,
            "policy": (policy or RoutinePolicy()).model_dump(mode="json"),
            "owner": owner if owner is not None else current_scope()[0],
        }
        if task is not None:
            packed["task"] = task
        if kind:
            packed["kind"] = kind
        return _json.dumps(packed, ensure_ascii=False)
    return command


def _zone_label() -> str:
    """The zone a schedule is read in, for the owner to check "7 am" landed where they meant."""
    from datetime import datetime

    from rune.daemon.external_schedule import schedule_timezone

    return schedule_timezone() or datetime.now().astimezone().tzname() or "local time"


# Parameter schemas

class CronCreateParams(BaseModel):
    policy: RoutinePolicy = Field(default_factory=RoutinePolicy, description="Execution limits, workspace, checks and notification policy")
    max_runs: int | None = Field(None, ge=1, description="Maximum started runs; unset means no count limit")
    name: str = Field(description="Unique name for the cron job")
    schedule: str = Field(
        description="Cron expression (minute hour day month weekday) in this machine's local time, "
                    "the zone shown with the current time in the Environment section"
    )
    command: str = Field(default="", description="Bash command to execute (use 'goal' for agent tasks)")
    goal: str = Field(
        default="",
        description="What one run does, written as a single piece of work to do then (e.g., 'Find Steam deals and "
                    "summarize'). Leave out when and how often: `schedule` says that. Runs a full agent loop.",
    )
    notify_channel: str = Field(
        default="",
        description="Channel to send results to (telegram/discord/slack). Only used with 'goal'.",
    )
    description: str = Field(default="", description="Human-readable description")


class CronListParams(BaseModel):
    status: str = Field(
        default="",
        description="Filter by status: active, paused, or empty for all",
    )


class CronDeleteParams(BaseModel):
    name: str = Field(description="Name of the cron job to delete")


class CronUpdateParams(BaseModel):
    """Parameters for updating a cron job."""
    job_id: str = Field(description="ID of the cron job to update")
    schedule: str | None = Field(default=None, description="New cron schedule expression, in this machine's local time")
    command: str | None = Field(default=None, description="New command to execute")
    goal: str | None = Field(default=None, description="New agent goal: what one run does, without when or how often")
    notify_channel: str | None = Field(default=None, description="Change notification channel (telegram/discord/slack/tui)")
    enabled: bool | None = Field(default=None, description="Enable or disable the job")
    policy: RoutinePolicy | None = None
    max_runs: int | None = Field(None, ge=1)


# Helpers

def _validate_cron_expr(expr: str) -> str | None:
    """Validate a cron expression. Returns error message or None if valid."""
    parts = expr.strip().split()
    if len(parts) != 5:
        return f"Cron expression must have 5 fields, got {len(parts)}"

    field_names = ("minute", "hour", "day", "month", "weekday")
    field_ranges = (
        (0, 59),
        (0, 23),
        (1, 31),
        (1, 12),
        (0, 6),
    )

    for part, name, (lo, hi) in zip(parts, field_names, field_ranges, strict=True):
        if part == "*":
            continue
        if part.startswith("*/"):
            try:
                step = int(part[2:])
                if step <= 0:
                    return f"Invalid step in {name}: {part}"
            except ValueError:
                return f"Invalid step in {name}: {part}"
            continue
        # Handle comma-separated and ranges
        for token in part.split(","):
            if "-" in token:
                try:
                    low, high = token.split("-", 1)
                    low_v, high_v = int(low), int(high)
                    if not (lo <= low_v <= hi and lo <= high_v <= hi):
                        return f"Range out of bounds for {name}: {token}"
                except ValueError:
                    return f"Invalid range in {name}: {token}"
            else:
                try:
                    val = int(token)
                    if not lo <= val <= hi:
                        return f"Value out of bounds for {name}: {token}"
                except ValueError:
                    return f"Invalid value in {name}: {token}"

    return None


# Implementations

def _background_change() -> CapabilityResult | None:
    from rune.agent.background import current_background_task
    from rune.safety.approval_context import was_approved

    if current_background_task() is not None and not was_approved():
        reason = "Background tasks need user approval to change schedules or execution limits."
        return CapabilityResult(success=False, error=reason,
                                metadata={"requires_approval": True, "action_status": "not_executed", "reason": reason})
    return None


async def cron_create(params: CronCreateParams) -> CapabilityResult:
    """Schedule a command or agent goal, optionally delivering its result to a channel."""
    log.debug("cron_create", name=params.name, schedule=params.schedule)
    if blocked := _background_change():
        return blocked

    if bool(params.command.strip()) == bool(params.goal.strip()):
        return CapabilityResult(
            success=False,
            error="Provide one 'command' (bash) or 'goal' (agent task).",
        )

    error = _validate_cron_expr(params.schedule)
    if error:
        return CapabilityResult(success=False, error=error)

    try:
        store = _get_store()
        # Check name uniqueness
        existing = store.list_cron_jobs()
        if any(j["name"] == params.name for j in existing):
            return CapabilityResult(
                success=False,
                error=f"Cron job '{params.name}' already exists. Delete it first.",
            )

        packed = _pack_command(
            params.command or "", params.goal or "",
            params.notify_channel or "", params.description or "",
            params.policy.bind_workspace(),
        )
        job_id = store.create_cron_job(
            name=params.name,
            schedule=params.schedule,
            command=packed,
            max_runs=params.max_runs,
        )

        log.info("cron_job_created", name=params.name, id=job_id,
                 mode="goal" if params.goal else "command")

        mode = "Agent goal" if params.goal else "Command"
        action = params.goal or params.command
        output_parts = [
            f"Cron job '{params.name}' created (id: {job_id}).",
            f"Schedule: {params.schedule} ({_zone_label()})",
            f"{mode}: {action}",
        ]
        if params.notify_channel:
            output_parts.append(f"Notify: {params.notify_channel}")
        if params.description:
            output_parts.append(f"Description: {params.description}")
        if params.goal:
            output_parts.append(
                "Note: each execution runs a full agent loop (LLM API cost applies per run)."
            )

        return CapabilityResult(
            success=True,
            output="\n".join(output_parts),
            metadata={"id": job_id, "name": params.name, "schedule": params.schedule},
        )

    except Exception as exc:
        return CapabilityResult(success=False, error=f"Failed to create cron job: {exc}")


async def cron_list(params: CronListParams) -> CapabilityResult:
    """List registered cron jobs from DB."""
    log.debug("cron_list", status=params.status)

    try:
        store = _get_store()
        rows = store.list_cron_jobs(enabled_only=(params.status == "active"))
        jobs = [_row_to_cronjob(r) for r in rows]

        if params.status and params.status != "active":
            jobs = [j for j in jobs if j.status == params.status]

        if not jobs:
            return CapabilityResult(success=True, output="No cron jobs found.", metadata={"count": 0})

        lines: list[str] = [f"Cron jobs ({len(jobs)}):"]
        for job in jobs:
            lines.append(f"  [{job.status}] {job.name} (id: {job.id})")
            if job.task is not None:
                ended = job.task.get("ended")
                lines.append(f"    Long task: {'ended (' + str(ended) + ')' if ended else 'next step at minute ' + str(job.wake_at) if job.wake_at else 'running or waiting'}")
            else:
                lines.append(f"    Schedule: {job.schedule}")
            if job.goal:
                lines.append(f"    Goal: {job.goal}")
            elif job.command:
                lines.append(f"    Command: {job.command}")
            if job.notify_channel:
                lines.append(f"    Notify: {job.notify_channel}")
            if job.description:
                lines.append(f"    Desc: {job.description}")
            if job.last_run_at:
                lines.append(f"    Last run: {job.last_run_at}")
            lines.append(f"    Limits: {job.policy.timeout_seconds}s, {job.policy.token_budget} tokens, {job.policy.max_steps} steps")
            if job.policy.workspace:
                lines.append(f"    Workspace: {job.policy.workspace}")
            lines.append("")

        return CapabilityResult(
            success=True,
            output="\n".join(lines).strip(),
            metadata={"count": len(jobs)},
        )
    except Exception as exc:
        return CapabilityResult(success=False, error=f"Failed to list cron jobs: {exc}")


async def cron_update(params: CronUpdateParams) -> CapabilityResult:
    """Update an existing cron job in DB."""
    log.debug("cron_update", job_id=params.job_id)
    if blocked := _background_change():
        return blocked

    try:
        store = _get_store()
        existing = store.get_cron_job(params.job_id)
        if existing is None:
            # Try by name
            for row in store.list_cron_jobs():
                if row["name"] == params.job_id:
                    existing = row
                    break
            if existing is None:
                return CapabilityResult(success=False, error=f"Cron job '{params.job_id}' not found.")

        job_id = existing["id"]
        updates = {}

        if params.schedule is not None:
            error = _validate_cron_expr(params.schedule)
            if error:
                return CapabilityResult(success=False, error=error)
            updates["schedule"] = params.schedule

        if params.enabled is not None:
            updates["enabled"] = params.enabled
        if "max_runs" in params.model_fields_set:
            updates["max_runs"] = params.max_runs
        try:
            task = task_change(existing, params.enabled, updates.get("max_runs", existing.get("max_runs")),
                               setting_max_runs="max_runs" in params.model_fields_set)
        except ValueError as exc:
            return CapabilityResult(success=False, error=str(exc))

        # Update command payload (goal/notify_channel/command), and a long task's state
        if (params.command is not None or params.goal is not None or params.notify_channel is not None
                or params.policy is not None or task is not None):
            current = _row_to_cronjob(existing)
            new_cmd = params.command if params.command is not None else current.command
            new_goal = params.goal if params.goal is not None else current.goal
            new_notify = params.notify_channel if params.notify_channel is not None else current.notify_channel
            new_desc = current.description
            if bool(new_cmd.strip()) == bool(new_goal.strip()):
                return CapabilityResult(success=False, error="Provide one command or agent goal")
            policy = current.policy.patched(params.policy) if params.policy else current.policy
            updates["command"] = _pack_command(new_cmd, new_goal, new_notify, new_desc, policy, owner=current.owner,
                                               task=task if task is not None else current.task, kind=current.kind)
        if updates:
            store.update_cron_job(job_id, **updates)

        log.info("cron_job_updated", job_id=job_id)
        return CapabilityResult(
            success=True,
            output=f"Cron job '{params.job_id}' updated.",
            metadata={"job_id": job_id},
        )
    except Exception as exc:
        return CapabilityResult(success=False, error=f"Failed to update: {exc}")


async def cron_delete(params: CronDeleteParams) -> CapabilityResult:
    """Delete a cron job from DB."""
    log.debug("cron_delete", name=params.name)
    if blocked := _background_change():
        return blocked

    try:
        store = _get_store()
        # Try by ID first, then by name
        deleted = store.delete_cron_job(params.name)
        if not deleted:
            for row in store.list_cron_jobs():
                if row["name"] == params.name:
                    store.delete_cron_job(row["id"])
                    deleted = True
                    break

        if not deleted:
            return CapabilityResult(success=False, error=f"Cron job '{params.name}' not found.")

        log.info("cron_job_deleted", name=params.name)
        return CapabilityResult(success=True, output=f"Cron job '{params.name}' deleted.")
    except Exception as exc:
        return CapabilityResult(success=False, error=f"Failed to delete: {exc}")


# Cron execution engine - runs pending jobs via heartbeat

async def execute_cron_job(job: CronJob, *, approval_from: str | None = None, refresh: bool = False,
                           scheduled_minute: int | None = None) -> str | None:
    """Run one occurrence. Returns why it did not start (e.g. ``"busy"``), or None once it ran."""
    from rune.daemon.cron_dispatcher import execution_slot

    async with execution_slot() as enabled:
        if not enabled and approval_from:
            raise ValueError("Scheduled execution is disabled. Enable it before continuing this task.")
        if enabled:
            from rune.proactive.long_task import is_task
            if scheduled_minute is not None and not is_task(job.schedule):
                import time

                from rune.config.loader import get_config
                if int(time.time() // 60) - scheduled_minute > get_config().cron_execution.misfire_grace_seconds // 60:
                    return "expired"
            from rune.daemon.maintenance import check_storage
            check_storage()
            return await _execute_cron_job(job, approval_from=approval_from, refresh=refresh,
                                           scheduled_minute=scheduled_minute)
        return "disabled"


async def _execute_cron_job(job: CronJob, *, approval_from: str | None = None, refresh: bool = False,
                            scheduled_minute: int | None = None) -> str | None:
    """Execute a single cron job (bash command or agent goal)."""
    import asyncio

    from rune.agent.execution_journal import ExecutionJournal, RecoveryBlocked, journal_scope
    from rune.agent.request_budget import BudgetExceeded
    from rune.agent.timing import capture_timing, timing_snapshot
    from rune.proactive.long_task import can_run, is_task, most_steps, step_brief, token_limit
    from rune.proactive.routine import (
        claim_occurrence,
        error_text,
        rerunnable,
        run_while_current,
        should_notify,
    )
    from rune.proactive.routine_budget import budget_owner
    from rune.proactive.routine_context import continuation
    from rune.proactive.routine_observation import changes, observe, reusable

    store = _get_store()
    row = store.get_cron_job(job.id)
    if row is None:
        return "deleted"
    if scheduled_minute is not None:
        from rune.proactive.routine import task_fingerprint
        if task_fingerprint(job) != task_fingerprint(_row_to_cronjob(row)):
            return "changed"
    job = _row_to_cronjob(row)
    if job.status != "active" or job.policy.expired:
        if approval_from:
            raise ValueError("Enable the task and check its deadline before continuing")
        return "inactive"
    if not approval_from and row.get("max_runs") is not None and row.get("run_count", 0) >= row["max_runs"]:
        return "run_limit"
    if is_task(job.schedule) and not approval_from and not can_run(job, int(row.get("run_count") or 0),
                                                                  row.get("max_runs")):
        # Out of steps under this machine's limit (lowered since): it ends before another starts.
        await _settle_task(job, {"status": "parked", "park": {"resume_at": 1}, "success": True,
                                 "output": "see its last step in the app."},
                           occurrence=(job.task or {}).get("step", ""))
        return "run_limit"
    # This machine's token limit holds for a task whatever its own policy was changed to. Only the
    # budget sees it: the job itself stays as stored, so it isn't taken for a changed one.
    budget_policy = job.policy
    if is_task(job.schedule):
        limit = token_limit()
        budget_policy = job.policy.model_copy(update={"total_tokens": min(job.policy.total_tokens or limit, limit)})
    prior, history = None, None
    if approval_from:
        from rune.proactive.execution_store import ExecutionStore
        from rune.proactive.routine_approval import prepare_continuation
        from rune.utils.paths import rune_data

        claims = ExecutionStore(rune_data() / "routine-executions.db")
        try:
            if claims.continued(approval_from):
                claims.close()
                return
            occurrence, prior, history = await prepare_continuation(job, claims, approval_from, refresh=refresh)
        except (ValueError, RecoveryBlocked, BudgetExceeded):
            continued = claims.continued(approval_from)
            claims.close()
            if continued:
                return
            raise
        except BaseException:
            claims.close()
            raise
        job.approved_action = None if refresh else prior["pending_approval"]["action"]
        job.routing = prior.get("routing")
        decision = "claimed"
    else:
        claims, occurrence, decision = claim_occurrence(job, scheduled_minute)
    started = False
    measured = None
    result = {}
    try:
        if decision != "claimed":
            log.info("cron_occurrence_skipped", job_id=job.id, reason=decision)
            return decision
        previous = prior or claims.latest_result(f"cron:{job.id}:")
        progress = prior or claims.latest_result(f"cron:{job.id}:", executed_only=True)
        observed = None
        if job.policy.input_paths or job.policy.output_paths:
            observed = await asyncio.to_thread(observe, job.policy)
        current = store.get_cron_job(job.id)
        if (current is None or _row_to_cronjob(current) != job or job.policy.expired
                or (not approval_from and current.get("max_runs") is not None and current.get("run_count", 0) >= current["max_runs"])):
            claims.finish(occurrence, {"status": "cancelled", "success": False, "verified": False,
                                       "not_started": True, "output": "", "error": "The task changed before execution started."})
            return
        if observed and not observed["complete"]:
            result = {"status": "unverified", "success": False, "verified": False, "not_started": True,
                      "output": "", "error": "Tracked files could not be inspected. Check their paths, permissions and size limits.",
                      "observations": observed}
            claims.finish(occurrence, result)
            if job.notify_channel and should_notify(job.policy, result, previous):
                await _send_to_channel(job.notify_channel, job.name, result["error"])
            return
        if observed and reusable(job, observed, progress):
            result = {**progress, "status": "unchanged",
                      "status_before_reuse": progress.get("status_before_reuse", progress["status"]),
                      "reused": True}
            claims.finish(occurrence, result)
            if job.notify_channel and should_notify(job.policy, result, previous):
                await _send_to_channel(job.notify_channel, job.name, "Tracked files are unchanged; the previous result was retained.")
            return
        try:
            owner = budget_owner(budget_policy, claims.usage(f"cron:{job.id}:", exclude=occurrence))
        except BudgetExceeded as exc:
            result = {"status": "budget_exhausted", "success": False, "verified": False,
                      "output": "", "error": str(exc), "not_started": True}
            claims.finish(occurrence, result)
            if is_task(job.schedule):
                await _settle_task(job, result, occurrence=occurrence)
            elif job.notify_channel and should_notify(job.policy, result, previous):
                await _send_to_channel(job.notify_channel, job.name, result["error"])
            return
        # Continuations share the original attempt's max_runs allowance.
        if not approval_from:
            store.record_cron_run(job.id)
        latest = store.get_cron_job(job.id)
        job.last_run_at = (latest or {}).get("last_run_at") or ""
        job.continuation = continuation(job, observed, progress)
        if is_task(job.schedule):
            job.preamble = step_brief(int((latest or {}).get("run_count") or 0),
                                      most_steps((latest or {}).get("max_runs")), job.policy.timeout_seconds // 60)
        job.file_changes = changes(observed, progress)
        from rune.utils.paths import user_workspace
        journal = ExecutionJournal(claims, occurrence, job.policy.workspace or str(user_workspace()), previous=history)
        from rune.proactive.routine_budget import restore_input_estimates
        owner.input_estimates = restore_input_estimates(prior)
        with capture_timing(owner) as measured, journal_scope(journal):
            try:
                started = True
                async with asyncio.timeout(job.policy.timeout_seconds):
                    result = await run_while_current(job, store, _executor(job))
            except BudgetExceeded as exc:
                result = {"status": "budget_exhausted", "success": False, "verified": False,
                          "output": "", "error": str(exc)}
            except TimeoutError:
                result = {"status": "interrupted", "success": False, "verified": False,
                          "execution_unknown": True, "output": "",
                          "error": "The scheduled task exceeded its execution timeout. Inspect its effects before resuming."}
            except Exception as exc:
                if job.kind:
                    log.warning("code_job_failed", job_id=job.id, kind=job.kind, error=type(exc).__name__, exc_info=True)
                result = {"status": "failed", "verified": False, "execution_unknown": True,
                          "output": "", "error": error_text(job, exc)}
            result["timings"] = timing_snapshot(measured)
            if measured["budget_blocked"] and not result.get("execution_unknown"):
                result.update(status="budget_exhausted", success=False, verified=False, error=measured["budget_blocked"])
        result.update(workspace=job.policy.workspace, routine_goal=job.goal, routine_command=job.command)
        if observed:
            after = await asyncio.to_thread(observe, job.policy)
            result["observations"] = after
            if (result.get("success") and not result.get("execution_unknown")
                    and (not after["complete"] or not observed["complete"] or observed["inputs"] != after["inputs"]
                         or any(item["state"] != "present" for group in ("inputs", "outputs") for item in after[group].values()))):
                result.update(status="unverified", success=False, verified=False,
                              error="Tracked files are missing, changed during execution or could not be inspected; the result needs review.")
        result = rerunnable(job, result)
        claims.finish(occurrence, result)
        if is_task(job.schedule):
            await _settle_task(job, result, occurrence=occurrence)
        elif job.notify_channel and should_notify(job.policy, result, previous):
            output = result.get("output") or result.get("error") or result["status"]
            if result.get("status") == "model_budget":
                output = result["error"]  # already says what happened and when it starts again
            elif not result.get("verified"):
                output = f"{result['status']}: {output}"
            # A run may name its own message (the briefing does, in the owner's language).
            title = result.get("title")
            await _send_to_channel(job.notify_channel, title if isinstance(title, str) and title else job.name,
                                   output, result.get("buttons"))
    except (asyncio.CancelledError, Exception) as exc:
        cancelled = isinstance(exc, asyncio.CancelledError)
        if decision == "claimed":
            stopped = {"status": ("interrupted" if started else "cancelled") if cancelled else "failed",
                       "success": False, "verified": False, "not_started": not started, "execution_unknown": started,
                       "output": result.get("output", ""), "error": "Task cancelled" if cancelled else error_text(job, exc),
                       "workspace": job.policy.workspace, "routine_goal": job.goal, "routine_command": job.command}
            if measured is not None:
                stopped["timings"] = timing_snapshot(measured)
            stopped = rerunnable(job, stopped)
            claims.finish(occurrence, stopped)
            if is_task(job.schedule):
                # Stopped part way: it doesn't go on by itself. Told unless the machine is stopping.
                await _settle_task(job, stopped, quiet=cancelled, occurrence=occurrence)
        if cancelled:
            raise
        log.warning("cron_execution_failed", job_id=job.id, error=str(exc))
    finally:
        claims.close()


def _executor(job: CronJob):
    return code_job(job.kind) or (_execute_goal_job if job.goal else _execute_bash_job)


async def _execute_bash_job(job: CronJob) -> dict:
    """Execute a cron job as a bash command."""
    from rune.capabilities.bash import BashParams
    from rune.capabilities.registry import get_capability_registry
    from rune.safety.approval_context import approval_required
    from rune.safety.approval_request import action_request
    from rune.safety.execution_environment import environment_scope, execution_config
    from rune.utils.paths import user_workspace

    params = BashParams(command=job.command, cwd=job.policy.workspace or str(user_workspace()),
                        timeout=job.policy.timeout_seconds * 1000).model_dump(mode="json", by_alias=True)
    with approval_required(), environment_scope(job.policy.workspace or str(user_workspace())):
        if job.approved_action:
            from rune.agent.execution_journal import active_journal
            from rune.safety.approval_request import execute_approved_action
            async def emit(*args):
                return None
            journal = active_journal()
            records = await execute_approved_action(job.approved_action, journal, emit)
            result = CapabilityResult(**records[-1]["result"])
        else:
            result = await get_capability_registry().execute("bash_execute", params)
        pending = {"action": action_request("bash_execute", params), "reason": result.error,
                   "environment": execution_config().model_dump(mode="json")} if result.metadata.get("requires_approval") else None
    status = "completed" if result.success else (
        "needs_approval" if result.metadata.get("requires_approval") else "failed"
    )
    return {"status": status, "success": result.success, "verified": False,
            "output": result.output, "error": result.error,
            "execution_unknown": result.metadata.get("action_status") == "unknown", "pending_approval": pending}


async def _execute_goal_job(job: CronJob) -> dict:
    """Run a scheduled agent task."""
    log.info("cron_goal_start", name=job.name, goal=job.goal[:100])

    from rune.agent.background import BackgroundTask, run_background
    from rune.proactive.long_task import is_task

    result = await run_background(BackgroundTask(
        goal=job.goal, source="task" if is_task(job.schedule) else "cron", workspace=job.policy.workspace,
        verification=job.policy.verification, max_steps=job.policy.max_steps,
        timeout_seconds=job.policy.timeout_seconds, token_budget=job.policy.token_budget,
        continuation=job.continuation,
        preamble=job.preamble,
        approved_action=job.approved_action,
        file_changes=job.file_changes,
        routing=job.routing,
        owner=job.owner,
    ))
    log.info("cron_goal_done", name=job.name, status=result["status"])
    return result


def task_change(row: dict, enabled: bool | None, max_runs: int | None, *, setting_max_runs: bool = False) -> dict | None:
    """A long task's new state when the owner changes it: turned on again, it goes on with its next
    step now. None: nothing to change. Raises why it can't go on, or past this machine's limit
    (checked only when the limit is being set, so a lowered machine limit never blocks turning a
    task off or editing it)."""
    from rune.proactive import long_task

    job = _row_to_cronjob(row)
    if not long_task.is_task(job.schedule):
        return None
    if setting_max_runs and max_runs is not None and max_runs > long_task.step_limit():
        raise ValueError(f"This machine allows a task at most {long_task.step_limit()} steps.")
    if not enabled or row.get("enabled", True) and (job.task or {}).get("wake_at") is not None:
        return None
    from rune.proactive.execution_store import ExecutionStore
    from rune.utils.paths import rune_data

    records = ExecutionStore(rune_data() / "routine-executions.db")
    try:
        runs = records.recent(f"cron:{job.id}:", limit=20)
    finally:
        records.close()
    # The latest run (an approval's continuation is `cron:<job>:resume:<hex>`), and the latest
    # minute that has a record, which the next step must come after.
    minutes = [int(r["id"].rsplit(":", 1)[-1]) for r in runs if r["id"].rsplit(":", 1)[-1].isdigit()]
    return long_task.resumed(job.task, int(row.get("run_count") or 0), max_runs,
                             runs[0]["state"] if runs else None, max(minutes) if minutes else None,
                             runs[0]["id"] if runs else None)


async def _settle_task(job: CronJob, result: dict, *, quiet: bool = False, occurrence: str = "") -> None:
    """After a long task's step: park it until its next step, or end it, and tell the owner if it matters.

    Never raises: the step's record is already written. A message that can't go now (`quiet`: the
    machine is stopping) is kept on the task and sent once it is back (`tell_task_notices`).
    """
    import asyncio

    from rune.proactive.long_task import after_step

    try:
        store = _get_store()
        row = store.get_cron_job(job.id)
        if row is None:
            return
        current = _row_to_cronjob(row)
        state, message = after_step(current, result, int(row.get("run_count") or 0), row.get("max_runs"),
                                    enabled=bool(row.get("enabled", True)))
        if message and quiet:
            state["notice"] = message[:4000]
        if occurrence:
            # The step this state follows from: a step recorded but not settled is settled later.
            state["step"] = occurrence
        updates: dict = {"command": _pack_command(current.command, current.goal, current.notify_channel,
                                                  current.description, current.policy, owner=current.owner,
                                                  task=state)}
        if state.get("ended"):
            updates["enabled"] = False
        store.update_cron_job(job.id, **updates)
        log.info("long_task_step", job_id=job.id, status=result.get("status"), wake_at=state.get("wake_at"),
                 ended=state.get("ended"))
        if message and not quiet:
            await _send_to_channel(current.notify_channel, current.name, message)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.warning("long_task_settle_failed", job_id=job.id, error=type(exc).__name__)


_tending = None


async def tell_task_notices() -> None:
    """Look after long tasks between steps: settle a step whose settling failed, and send what a
    task kept to tell the owner while the machine was stopping, once it can. One at a time; each
    task is read again just before it is changed."""
    import asyncio

    from rune.daemon.gateway import get_gateway
    from rune.proactive.execution_store import ExecutionStore
    from rune.proactive.long_task import is_task, most_steps
    from rune.utils.paths import rune_data

    global _tending
    if _tending is None:
        _tending = asyncio.Lock()
    if _tending.locked():
        return
    async with _tending:
        store = _get_store()
        for listed in store.list_cron_jobs():
            if not is_task(listed.get("schedule", "")):
                continue
            records = ExecutionStore(rune_data() / "routine-executions.db")
            try:
                last = records.recent(f"cron:{listed['id']}:", limit=1)
            finally:
                records.close()
            row = store.get_cron_job(listed["id"])
            if row is None:
                continue
            job = _row_to_cronjob(row)
            unsettled = (last and job.task is not None and job.task.get("step") != last[0]["id"]
                         and last[0]["state"] not in {"running", "accepted"})
            if unsettled and last[0]["result"] is not None:
                await _settle_task(job, last[0]["result"], occurrence=last[0]["id"])
                continue
            if unsettled and last[0]["state"] == "recovery_required":
                # The machine died during the step (killed, out of memory): no result was written.
                await _settle_task(job, {"status": "interrupted", "success": False, "execution_unknown": True,
                                         "output": "", "error": "The machine stopped during the step."},
                                   occurrence=last[0]["id"])
                continue
            if (row.get("enabled", True) and job.wake_at is not None and not (job.task or {}).get("ended")
                    and int(row.get("run_count") or 0) >= most_steps(row.get("max_runs"))):
                # Out of steps under a machine limit lowered while it was parked: no step will start.
                await _settle_task(job, {"status": "parked", "park": {"resume_at": 1}, "success": True,
                                         "output": "see its last step in the app."},
                                   occurrence=(job.task or {}).get("step", ""))
                continue
            notice = (job.task or {}).get("notice")
            if not notice or get_gateway() is None:
                continue
            task = {k: v for k, v in job.task.items() if k != "notice"}
            store.update_cron_job(job.id, command=_pack_command(job.command, job.goal, job.notify_channel,
                                                                job.description, job.policy, owner=job.owner, task=task))
            await _send_to_channel(job.notify_channel, job.name, str(notice))


async def task_stalled(job_id: object, minute: object) -> dict:
    """The host gave up starting a task's step (`task.stalled`): end the task and tell the owner.

    Only for the step the task waits on, with no record: one that ran, or that the task no longer
    waits on, is left alone.
    """
    from rune.proactive.execution_store import ExecutionStore
    from rune.proactive.long_task import is_task
    from rune.utils.paths import rune_data

    if not isinstance(job_id, str) or isinstance(minute, bool) or not isinstance(minute, int):
        raise ValueError("jobId and scheduledMinute are required")
    row = _get_store().get_cron_job(job_id)
    if row is None or not is_task(row.get("schedule", "")):
        return {"state": "unknown"}
    job = _row_to_cronjob(row)
    records = ExecutionStore(rune_data() / "routine-executions.db")
    try:
        ran = records.get(f"cron:{job_id}:{minute}") is not None
    finally:
        records.close()
    if job.wake_at != minute or ran:
        return {"state": "moved_on"}
    await _settle_task(job, {"status": "failed", "success": False, "output": "",
                             "error": "Its next step couldn't start, however often it was tried. Turn it on "
                                      "again to retry."}, occurrence=(job.task or {}).get("step", ""))
    return {"state": "ended"}


async def _send_to_channel(channel_name: str, job_name: str, text: str, buttons: list[dict] | None = None) -> None:
    """Route a job result by channel and priority, with a local notification fallback."""
    try:
        from rune.channels.types import LinkButton
        from rune.daemon.gateway import GatewayNotification, get_gateway

        gateway = get_gateway()
        if gateway is None:
            log.warning("cron_gateway_not_available")
            return

        notification = GatewayNotification(
            title=job_name,
            body=text,
            priority="high" if channel_name else "medium",
            source="cron",
            channel=channel_name,
            buttons=[LinkButton(str(b.get("text", "")), str(b.get("url", ""))) for b in buttons or [] if isinstance(b, dict)],
        )
        await gateway.route_notification(notification)
        log.info("cron_result_routed", name=job_name)
    except Exception as exc:
        log.warning("cron_result_route_failed", error=str(exc))


def get_active_cron_jobs() -> list[CronJob]:
    """Return all active cron jobs from DB (for scheduler integration)."""
    try:
        store = _get_store()
        rows = store.list_cron_jobs(enabled_only=True)
        jobs = []
        for row in rows:
            try:
                jobs.append(_row_to_cronjob(row))
            except (TypeError, ValueError, KeyError) as exc:
                log.warning("invalid_cron_job", job_id=row.get("id"), error=str(exc))
        return jobs
    except Exception as exc:
        log.debug("get_active_cron_jobs_failed", error=str(exc))
        return []


# Registration

def register_cron_capabilities(registry: CapabilityRegistry) -> None:
    """Register cron capabilities, and the long-task ones (rune.capabilities.task_tools)."""
    from rune.capabilities.briefing import register_briefing_capabilities
    from rune.capabilities.task_tools import register_task_capabilities
    from rune.capabilities.watch import register_watch_capabilities

    register_task_capabilities(registry)
    register_briefing_capabilities(registry)
    register_watch_capabilities(registry)
    registry.register(CapabilityDefinition(
        name="cron_create",
        description=(
            "Create a scheduled cron job. Use 'goal' for agent tasks "
            "(e.g., 'find Steam deals') and 'notify_channel' to send results "
            "to Telegram/Discord/Slack. Use 'command' for bash commands. To tell the owner "
            "when a price on a page changes, use watch_set, not this."
        ),
        domain=Domain.SCHEDULE,
        risk_level=RiskLevel.MEDIUM,
        group="schedule",
        parameters_model=CronCreateParams,
        execute=cron_create,
    ))
    registry.register(CapabilityDefinition(
        name="cron_list",
        description="List registered cron jobs",
        domain=Domain.SCHEDULE,
        risk_level=RiskLevel.LOW,
        group="schedule",
        parameters_model=CronListParams,
        execute=cron_list,
    ))
    registry.register(CapabilityDefinition(
        name="cron_update",
        description="Update an existing cron job",
        domain=Domain.SCHEDULE,
        risk_level=RiskLevel.MEDIUM,
        group="schedule",
        parameters_model=CronUpdateParams,
        execute=cron_update,
    ))
    registry.register(CapabilityDefinition(
        name="cron_delete",
        description="Delete a cron job",
        domain=Domain.SCHEDULE,
        risk_level=RiskLevel.MEDIUM,
        group="schedule",
        parameters_model=CronDeleteParams,
        execute=cron_delete,
    ))
