"""Create, list, update and delete tasks in the heartbeat scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.proactive.routine import RoutinePolicy
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Store command and goal metadata as JSON in the existing command column.

import json as _json


@dataclass(slots=True)
class CronJob:
    """In-memory view of a cron job (hydrated from DB row)."""
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


def _get_store():
    """Lazy import to avoid circular deps."""
    from rune.memory.store import get_memory_store
    return get_memory_store()


def _row_to_cronjob(row: dict) -> CronJob:
    """Convert a DB row dict into a CronJob, unpacking the JSON command."""
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
    )


def _pack_command(command: str, goal: str, notify_channel: str, description: str,
                  policy: RoutinePolicy | None = None) -> str:
    """Pack goal/notify/description into the command column as JSON."""
    if goal or notify_channel or policy is not None:
        return _json.dumps({
            "bash": command,
            "goal": goal,
            "notify_channel": notify_channel,
            "description": description,
            "policy": (policy or RoutinePolicy()).model_dump(mode="json"),
        }, ensure_ascii=False)
    return command


# Parameter schemas

class CronCreateParams(BaseModel):
    policy: RoutinePolicy = Field(default_factory=RoutinePolicy, description="Execution limits, workspace, checks and notification policy")
    max_runs: int | None = Field(None, ge=1, description="Maximum started runs; unset means no count limit")
    name: str = Field(description="Unique name for the cron job")
    schedule: str = Field(
        description="Cron expression (minute hour day month weekday)"
    )
    command: str = Field(default="", description="Bash command to execute (use 'goal' for agent tasks)")
    goal: str = Field(
        default="",
        description="Agent goal to execute (e.g., 'Find Steam deals and summarize'). Runs a full agent loop.",
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
    schedule: str | None = Field(default=None, description="New cron schedule expression")
    command: str | None = Field(default=None, description="New command to execute")
    goal: str | None = Field(default=None, description="New agent goal")
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
            f"Schedule: {params.schedule}",
            f"{mode}: {action}",
        ]
        if params.notify_channel:
            output_parts.append(f"Notify: {params.notify_channel}")
        if params.description:
            output_parts.append(f"Description: {params.description}")
        if params.goal:
            output_parts.append(
                "⚠️ Each execution runs a full agent loop (LLM API cost applies per run)."
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

        # Update command payload (goal/notify_channel/command)
        if params.command is not None or params.goal is not None or params.notify_channel is not None or params.policy is not None:
            current = _row_to_cronjob(existing)
            new_cmd = params.command if params.command is not None else current.command
            new_goal = params.goal if params.goal is not None else current.goal
            new_notify = params.notify_channel if params.notify_channel is not None else current.notify_channel
            new_desc = current.description
            if bool(new_cmd.strip()) == bool(new_goal.strip()):
                return CapabilityResult(success=False, error="Provide one command or agent goal")
            policy = current.policy.patched(params.policy) if params.policy else current.policy
            updates["command"] = _pack_command(new_cmd, new_goal, new_notify, new_desc, policy)
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
                           scheduled_minute: int | None = None) -> None:
    from rune.daemon.cron_dispatcher import execution_slot

    async with execution_slot() as enabled:
        if not enabled and approval_from:
            raise ValueError("Scheduled execution is disabled. Enable it before continuing this task.")
        if enabled:
            if scheduled_minute is not None:
                import time

                from rune.config.loader import get_config
                if int(time.time() // 60) - scheduled_minute > get_config().cron_execution.misfire_grace_seconds // 60:
                    return
            from rune.daemon.maintenance import check_storage
            check_storage()
            await _execute_cron_job(job, approval_from=approval_from, refresh=refresh,
                                    scheduled_minute=scheduled_minute)


async def _execute_cron_job(job: CronJob, *, approval_from: str | None = None, refresh: bool = False,
                            scheduled_minute: int | None = None) -> None:
    """Execute a single cron job (bash command or agent goal)."""
    import asyncio

    from rune.agent.execution_journal import ExecutionJournal, RecoveryBlocked, journal_scope
    from rune.agent.request_budget import BudgetExceeded
    from rune.agent.timing import capture_timing, timing_snapshot
    from rune.proactive.routine import claim_occurrence, run_while_current, should_notify
    from rune.proactive.routine_budget import budget_owner
    from rune.proactive.routine_context import continuation
    from rune.proactive.routine_observation import changes, observe, reusable

    store = _get_store()
    row = store.get_cron_job(job.id)
    if row is None:
        return
    if scheduled_minute is not None:
        from rune.proactive.routine import task_fingerprint
        if task_fingerprint(job) != task_fingerprint(_row_to_cronjob(row)):
            return
    job = _row_to_cronjob(row)
    if job.status != "active" or job.policy.expired:
        if approval_from:
            raise ValueError("Enable the task and check its deadline before continuing")
        return
    if not approval_from and row.get("max_runs") is not None and row.get("run_count", 0) >= row["max_runs"]:
        return
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
            return
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
            owner = budget_owner(job.policy, claims.usage(f"cron:{job.id}:", exclude=occurrence))
        except BudgetExceeded as exc:
            result = {"status": "budget_exhausted", "success": False, "verified": False,
                      "output": "", "error": str(exc), "not_started": True}
            claims.finish(occurrence, result)
            if job.notify_channel and should_notify(job.policy, result, previous):
                await _send_to_channel(job.notify_channel, job.name, result["error"])
            return
        # Continuations share the original attempt's max_runs allowance.
        if not approval_from:
            store.record_cron_run(job.id)
        latest = store.get_cron_job(job.id)
        job.last_run_at = (latest or {}).get("last_run_at") or ""
        job.continuation = continuation(job, observed, progress)
        job.file_changes = changes(observed, progress)
        from rune.utils.paths import user_workspace
        journal = ExecutionJournal(claims, occurrence, job.policy.workspace or str(user_workspace()), previous=history)
        from rune.proactive.routine_budget import restore_input_estimates
        owner.input_estimates = restore_input_estimates(prior)
        with capture_timing(owner) as measured, journal_scope(journal):
            try:
                started = True
                async with asyncio.timeout(job.policy.timeout_seconds):
                    result = await run_while_current(job, store, _execute_goal_job if job.goal else _execute_bash_job)
            except BudgetExceeded as exc:
                result = {"status": "budget_exhausted", "success": False, "verified": False,
                          "output": "", "error": str(exc)}
            except TimeoutError:
                result = {"status": "interrupted", "success": False, "verified": False,
                          "execution_unknown": True, "output": "",
                          "error": "The scheduled task exceeded its execution timeout. Inspect its effects before resuming."}
            except Exception as exc:
                result = {"status": "failed", "verified": False, "execution_unknown": True,
                          "output": "", "error": str(exc)}
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
        claims.finish(occurrence, result)
        if job.notify_channel and should_notify(job.policy, result, previous):
            output = result.get("output") or result.get("error") or result["status"]
            if not result.get("verified"):
                output = f"{result['status']}: {output}"
            await _send_to_channel(job.notify_channel, job.name, output)
    except (asyncio.CancelledError, Exception) as exc:
        cancelled = isinstance(exc, asyncio.CancelledError)
        if decision == "claimed":
            stopped = {"status": ("interrupted" if started else "cancelled") if cancelled else "failed",
                       "success": False, "verified": False, "not_started": not started, "execution_unknown": started,
                       "output": result.get("output", ""), "error": "Task cancelled" if cancelled else str(exc),
                       "workspace": job.policy.workspace, "routine_goal": job.goal, "routine_command": job.command}
            if measured is not None:
                stopped["timings"] = timing_snapshot(measured)
            claims.finish(occurrence, stopped)
        if cancelled:
            raise
        log.warning("cron_execution_failed", job_id=job.id, error=str(exc))
    finally:
        claims.close()


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

    result = await run_background(BackgroundTask(
        goal=job.goal, source="cron", workspace=job.policy.workspace,
        verification=job.policy.verification, max_steps=job.policy.max_steps,
        timeout_seconds=job.policy.timeout_seconds, token_budget=job.policy.token_budget,
        continuation=job.continuation,
        approved_action=job.approved_action,
        file_changes=job.file_changes,
        routing=job.routing,
    ))
    log.info("cron_goal_done", name=job.name, status=result["status"])
    return result


async def _send_to_channel(channel_name: str, job_name: str, text: str) -> None:
    """Route a job result by channel and priority, with a local notification fallback."""
    try:
        from rune.daemon.gateway import GatewayNotification, get_gateway

        gateway = get_gateway()
        if gateway is None:
            log.warning("cron_gateway_not_available")
            return

        notification = GatewayNotification(
            title=f"🔔 [{job_name}]",
            body=text,
            priority="high" if channel_name else "medium",
            source="cron",
            channel=channel_name,
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
    """Register cron capabilities."""
    registry.register(CapabilityDefinition(
        name="cron_create",
        description=(
            "Create a scheduled cron job. Use 'goal' for agent tasks "
            "(e.g., 'find Steam deals') and 'notify_channel' to send results "
            "to Telegram/Discord/Slack. Use 'command' for bash commands."
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
