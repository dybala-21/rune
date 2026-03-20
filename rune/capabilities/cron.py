"""Cron capabilities for RUNE.

Ported from src/capabilities/cron.ts - create, list, update, and delete
scheduled tasks using the HeartbeatScheduler.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)


# CronJob - DB-backed via MemoryStore.  The ``command`` column stores a
# JSON payload ``{"bash": "...", "goal": "...", "notify_channel": "...",
# "description": "..."}`` so goal-based jobs work without schema migration.

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

    try:
        payload = _json.loads(raw_cmd)
        if isinstance(payload, dict):
            bash_cmd = payload.get("bash", "")
            goal = payload.get("goal", "")
            notify = payload.get("notify_channel", "")
            desc = payload.get("description", "")
    except (ValueError, TypeError):
        pass  # plain bash command string

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
    )


def _pack_command(command: str, goal: str, notify_channel: str, description: str) -> str:
    """Pack goal/notify/description into the command column as JSON."""
    if goal or notify_channel:
        return _json.dumps({
            "bash": command,
            "goal": goal,
            "notify_channel": notify_channel,
            "description": description,
        }, ensure_ascii=False)
    return command


# Parameter schemas

class CronCreateParams(BaseModel):
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

async def cron_create(params: CronCreateParams) -> CapabilityResult:
    """Create a new scheduled cron job.

    Supports two execution modes:
    - ``command``: runs a bash command (legacy).
    - ``goal``: runs a full agent loop with the given goal, optionally
      sending the result to a channel (``notify_channel``).
    """
    log.debug("cron_create", name=params.name, schedule=params.schedule)

    if not params.command and not params.goal:
        return CapabilityResult(
            success=False,
            error="Either 'command' (bash) or 'goal' (agent task) must be provided.",
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
        )
        job_id = store.create_cron_job(
            name=params.name,
            schedule=params.schedule,
            command=packed,
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

        if params.schedule is not None:
            error = _validate_cron_expr(params.schedule)
            if error:
                return CapabilityResult(success=False, error=error)
            store.update_cron_job(job_id, schedule=params.schedule)

        if params.enabled is not None:
            store.update_cron_job(job_id, enabled=params.enabled)

        # Update command payload (goal/notify_channel/command)
        if params.command is not None or params.goal is not None or params.notify_channel is not None:
            current = _row_to_cronjob(existing)
            new_cmd = params.command if params.command is not None else current.command
            new_goal = params.goal if params.goal is not None else current.goal
            new_notify = params.notify_channel if params.notify_channel is not None else current.notify_channel
            new_desc = current.description
            packed = _pack_command(new_cmd, new_goal, new_notify, new_desc)
            store.update_cron_job(job_id, command=packed)

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

async def execute_cron_job(job: CronJob) -> None:
    """Execute a single cron job (bash command or agent goal)."""
    if job.status != "active":
        return

    if job.goal:
        await _execute_goal_job(job)
    elif job.command:
        await _execute_bash_job(job)

    # Record execution in DB
    try:
        store = _get_store()
        store.record_cron_run(job.id)
    except Exception:
        pass


async def _execute_bash_job(job: CronJob) -> None:
    """Execute a cron job as a bash command."""
    try:
        from rune.capabilities.bash import BashParams, bash_execute
        result = await bash_execute(BashParams(command=job.command))
        if not result.success:
            log.warning("cron_bash_failed", name=job.name, error=result.error)
    except Exception as exc:
        log.warning("cron_bash_error", name=job.name, error=str(exc))


async def _execute_goal_job(job: CronJob) -> None:
    """Execute a cron job as an agent goal, optionally sending results to a channel."""
    import asyncio

    log.info("cron_goal_start", name=job.name, goal=job.goal[:100])

    try:
        # Run agent loop (reuse the same factory pattern as proactive bridge)
        from rune.agent.loop import NativeAgentLoop
        from rune.types import AgentConfig

        cfg = AgentConfig(max_iterations=30, timeout_seconds=120)
        loop = NativeAgentLoop(config=cfg)
        result = await asyncio.wait_for(loop.run(job.goal), timeout=120)

        output = getattr(result, "answer", None) or getattr(result, "reason", str(result))
        success = getattr(result, "reason", "") in ("completed", "verified")
        log.info("cron_goal_done", name=job.name, success=success)

        # Send result to channel if configured
        if job.notify_channel and output:
            await _send_to_channel(job.notify_channel, job.name, output)

    except TimeoutError:
        log.warning("cron_goal_timeout", name=job.name)
        if job.notify_channel:
            await _send_to_channel(
                job.notify_channel, job.name, f"⏱ Timed out: {job.goal[:100]}"
            )
    except Exception as exc:
        log.warning("cron_goal_error", name=job.name, error=str(exc))
        if job.notify_channel:
            await _send_to_channel(
                job.notify_channel, job.name, f"❌ Failed: {str(exc)[:200]}"
            )


async def _send_to_channel(channel_name: str, job_name: str, text: str) -> None:
    """Send cron job result via the gateway notification router.

    Routes through ``ChannelGateway.route_notification`` which handles:
    - Named channel delivery (telegram/discord/slack)
    - Priority-based routing rules
    - TUI fallback when no external channel is available
    """
    try:
        from rune.channels.types import Priority
        from rune.daemon.gateway import GatewayNotification, get_gateway

        gateway = get_gateway()
        if gateway is None:
            log.warning("cron_gateway_not_available")
            return

        notification = GatewayNotification(
            title=f"🔔 [{job_name}]",
            body=text,
            priority=Priority.HIGH if channel_name else Priority.NORMAL,
            source="cron",
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
        return [_row_to_cronjob(r) for r in rows]
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
