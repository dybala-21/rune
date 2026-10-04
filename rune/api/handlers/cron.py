"""Manage scheduled jobs through the MemoryStore-backed API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from rune.api.auth import TokenAuthDependency
from rune.memory.store import get_memory_store
from rune.proactive.routine import RoutinePolicy
from rune.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/cron", tags=["cron"])
auth = TokenAuthDependency()


# Models


class CronJobInfo(BaseModel):
    id: str
    name: str
    schedule: str
    command: str
    goal: str = ""
    notify_channel: str = Field("", alias="notifyChannel")
    policy: RoutinePolicy = Field(default_factory=RoutinePolicy)
    recent_runs: list[dict[str, Any]] = Field(default_factory=list, alias="recentRuns")
    enabled: bool = True
    created_at: str = Field(alias="createdAt")
    last_run_at: str | None = Field(None, alias="lastRunAt")
    run_count: int = Field(0, alias="runCount")
    max_runs: int | None = Field(None, alias="maxRuns")

    model_config = ConfigDict(populate_by_name=True)


class CronListResponse(BaseModel):
    jobs: list[CronJobInfo]
    builtin_tasks: list[dict[str, Any]] = Field(default_factory=list, alias="builtinTasks")
    heartbeat_active: bool = Field(False, alias="heartbeatActive")

    model_config = ConfigDict(populate_by_name=True)


class CronCreateRequest(BaseModel):
    name: str
    schedule: str
    command: str = ""
    goal: str = ""
    notify_channel: str = Field("", alias="notifyChannel")
    policy: RoutinePolicy = Field(default_factory=RoutinePolicy)
    enabled: bool = True
    max_runs: int | None = Field(None, alias="maxRuns", ge=1)

    model_config = ConfigDict(populate_by_name=True)


class CronCreateResponse(BaseModel):
    job: CronJobInfo


class CronUpdateRequest(BaseModel):
    name: str | None = None
    schedule: str | None = None
    command: str | None = None
    goal: str | None = None
    notify_channel: str | None = Field(None, alias="notifyChannel")
    policy: RoutinePolicy | None = None
    enabled: bool | None = None
    max_runs: int | None = Field(None, alias="maxRuns", ge=1)

    model_config = ConfigDict(populate_by_name=True)


class CronUpdateResponse(BaseModel):
    job: CronJobInfo


class CronDeleteResponse(BaseModel):
    deleted: bool


class CronToggleResponse(BaseModel):
    id: str
    enabled: bool


class ReconcileRequest(BaseModel):
    operation_id: str = Field(alias="operationId")
    note: str = Field(min_length=1, max_length=2000)


# Helpers


def _job_dict_to_info(j: dict[str, Any]) -> CronJobInfo:
    from rune.capabilities.cron import _row_to_cronjob
    from rune.proactive.execution_store import ExecutionStore
    from rune.utils.paths import rune_data

    job = _row_to_cronjob(j)
    records = ExecutionStore(rune_data() / "routine-executions.db")
    try:
        runs = records.recent(f"cron:{job.id}:")
    finally:
        records.close()
    return CronJobInfo(
        id=j["id"],
        name=j["name"],
        schedule=j["schedule"],
        command=job.command,
        goal=job.goal,
        notifyChannel=job.notify_channel,
        policy=job.policy,
        recentRuns=runs,
        enabled=j.get("enabled", True),
        createdAt=j.get("created_at", ""),
        lastRunAt=j.get("last_run_at"),
        runCount=j.get("run_count", 0),
        maxRuns=j.get("max_runs"),
    )


def _validate_cron_schedule(schedule: str) -> None:
    from rune.capabilities.cron import _validate_cron_expr
    error = _validate_cron_expr(schedule)
    if error:
        raise HTTPException(status_code=400, detail=error)


# Routes


@router.get("", response_model=CronListResponse, dependencies=[Depends(auth)])
async def list_cron_jobs() -> CronListResponse:
    """List all user-defined cron jobs."""
    store = get_memory_store()
    rows = store.list_cron_jobs()
    jobs = [_job_dict_to_info(j) for j in rows]
    return CronListResponse(jobs=jobs)


@router.post("", response_model=CronCreateResponse, dependencies=[Depends(auth)])
async def create_cron_job(req: CronCreateRequest) -> CronCreateResponse:
    """Create a job with a five-field cron schedule."""
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    if not req.schedule.strip():
        raise HTTPException(status_code=400, detail="Schedule is required")
    if bool(req.command.strip()) == bool(req.goal.strip()):
        raise HTTPException(status_code=400, detail="Provide one goal or shell command")

    _validate_cron_schedule(req.schedule)

    from rune.capabilities.cron import _pack_command
    store = get_memory_store()
    job_id = store.create_cron_job(
        name=req.name.strip(),
        schedule=req.schedule.strip(),
        command=_pack_command(req.command.strip(), req.goal.strip(), req.notify_channel.strip(), "", req.policy.bind_workspace()),
        enabled=req.enabled,
        max_runs=req.max_runs,
    )

    job = store.get_cron_job(job_id)
    assert job is not None

    log.info("cron_job_created", job_id=job_id, name=req.name)
    return CronCreateResponse(job=_job_dict_to_info(job))


@router.get("/{job_id}", response_model=CronJobInfo, dependencies=[Depends(auth)])
async def get_cron_job(job_id: str) -> CronJobInfo:
    """Get a cron job by ID."""
    store = get_memory_store()
    job = store.get_cron_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Cron job not found: {job_id}")

    return _job_dict_to_info(job)


@router.patch("/{job_id}", response_model=CronUpdateResponse, dependencies=[Depends(auth)])
async def update_cron_job(job_id: str, req: CronUpdateRequest) -> CronUpdateResponse:
    """Update a cron job."""
    store = get_memory_store()

    existing = store.get_cron_job(job_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Cron job not found: {job_id}")

    if req.schedule is not None:
        _validate_cron_schedule(req.schedule)

    from rune.capabilities.cron import _pack_command, _row_to_cronjob
    current = _row_to_cronjob(existing)

    kwargs: dict[str, Any] = {}
    if req.name is not None:
        kwargs["name"] = req.name.strip()
    if req.schedule is not None:
        kwargs["schedule"] = req.schedule.strip()
    if any(value is not None for value in (req.command, req.goal, req.policy, req.notify_channel)):
        command = req.command.strip() if req.command is not None else current.command
        goal = req.goal.strip() if req.goal is not None else current.goal
        if bool(command) == bool(goal):
            raise HTTPException(status_code=400, detail="Provide one goal or shell command")
        channel = current.notify_channel if req.notify_channel is None else req.notify_channel.strip()
        kwargs["command"] = _pack_command(command, goal, channel,
                                           current.description, req.policy.bind_workspace() if req.policy else current.policy)
    if req.enabled is not None:
        kwargs["enabled"] = req.enabled
    if "max_runs" in req.model_fields_set:
        kwargs["max_runs"] = req.max_runs

    if kwargs:
        store.update_cron_job(job_id, **kwargs)

    job = store.get_cron_job(job_id)
    assert job is not None

    log.info("cron_job_updated", job_id=job_id)
    return CronUpdateResponse(job=_job_dict_to_info(job))


@router.post("/{job_id}/toggle", response_model=CronToggleResponse, dependencies=[Depends(auth)])
async def toggle_cron_job(job_id: str) -> CronToggleResponse:
    """Toggle the enabled state of a cron job."""
    store = get_memory_store()
    new_state = store.toggle_cron_job(job_id)

    if new_state is None:
        raise HTTPException(status_code=404, detail=f"Cron job not found: {job_id}")

    log.info("cron_job_toggled", job_id=job_id, enabled=new_state)
    return CronToggleResponse(id=job_id, enabled=new_state)


@router.post("/{job_id}/reconcile", dependencies=[Depends(auth)])
async def reconcile_cron_job(job_id: str, req: ReconcileRequest) -> dict[str, bool]:
    from rune.proactive.execution_store import ExecutionStore
    from rune.utils.paths import rune_data

    job = get_memory_store().get_cron_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if job["enabled"]:
        raise HTTPException(status_code=409, detail="Pause the task before reviewing its interrupted run")
    if not req.note.strip():
        raise HTTPException(status_code=400, detail="Describe the external effects you checked")
    records = ExecutionStore(rune_data() / "routine-executions.db")
    try:
        reconciled = records.reconcile(f"cron:{job_id}", req.operation_id, req.note.strip())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        records.close()
    if not reconciled:
        raise HTTPException(status_code=409, detail="This run is no longer awaiting review")
    return {"reconciled": True}


@router.delete("/{job_id}", response_model=CronDeleteResponse, dependencies=[Depends(auth)])
async def delete_cron_job(job_id: str) -> CronDeleteResponse:
    """Delete a cron job by ID."""
    store = get_memory_store()

    if not store.delete_cron_job(job_id):
        raise HTTPException(status_code=404, detail=f"Cron job not found: {job_id}")

    log.info("cron_job_deleted", job_id=job_id)
    return CronDeleteResponse(deleted=True)
