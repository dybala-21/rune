"""Cron handler - GET /cron, POST /cron, DELETE /cron/{id}, PATCH /cron/{id}.

Ported from src/api/handlers/cron.ts - CRUD API for scheduled
cron jobs backed by the MemoryStore (SQLite).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from rune.api.auth import TokenAuthDependency
from rune.memory.store import get_memory_store
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
    command: str
    enabled: bool = True
    max_runs: int | None = Field(None, alias="maxRuns")

    model_config = ConfigDict(populate_by_name=True)


class CronCreateResponse(BaseModel):
    job: CronJobInfo


class CronUpdateRequest(BaseModel):
    name: str | None = None
    schedule: str | None = None
    command: str | None = None
    enabled: bool | None = None
    max_runs: int | None = Field(None, alias="maxRuns")

    model_config = ConfigDict(populate_by_name=True)


class CronUpdateResponse(BaseModel):
    job: CronJobInfo


class CronDeleteResponse(BaseModel):
    deleted: bool


class CronToggleResponse(BaseModel):
    id: str
    enabled: bool


# Helpers


def _job_dict_to_info(j: dict[str, Any]) -> CronJobInfo:
    return CronJobInfo(
        id=j["id"],
        name=j["name"],
        schedule=j["schedule"],
        command=j["command"],
        enabled=j.get("enabled", True),
        createdAt=j.get("created_at", ""),
        lastRunAt=j.get("last_run_at"),
        runCount=j.get("run_count", 0),
        maxRuns=j.get("max_runs"),
    )


def _validate_cron_schedule(schedule: str) -> None:
    """Validate that a cron schedule has 5 fields."""
    parts = schedule.strip().split()
    if len(parts) != 5:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid cron schedule: expected 5 fields, got {len(parts)}",
        )


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
    """Create a new cron job.

    The ``schedule`` field should be a valid cron expression
    (5 fields: minute hour day-of-month month day-of-week).
    """
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    if not req.schedule.strip():
        raise HTTPException(status_code=400, detail="Schedule is required")
    if not req.command.strip():
        raise HTTPException(status_code=400, detail="Command is required")

    _validate_cron_schedule(req.schedule)

    store = get_memory_store()
    job_id = store.create_cron_job(
        name=req.name.strip(),
        schedule=req.schedule.strip(),
        command=req.command.strip(),
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

    kwargs: dict[str, Any] = {}
    if req.name is not None:
        kwargs["name"] = req.name.strip()
    if req.schedule is not None:
        kwargs["schedule"] = req.schedule.strip()
    if req.command is not None:
        kwargs["command"] = req.command.strip()
    if req.enabled is not None:
        kwargs["enabled"] = req.enabled
    if req.max_runs is not None:
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


@router.delete("/{job_id}", response_model=CronDeleteResponse, dependencies=[Depends(auth)])
async def delete_cron_job(job_id: str) -> CronDeleteResponse:
    """Delete a cron job by ID."""
    store = get_memory_store()

    if not store.delete_cron_job(job_id):
        raise HTTPException(status_code=404, detail=f"Cron job not found: {job_id}")

    log.info("cron_job_deleted", job_id=job_id)
    return CronDeleteResponse(deleted=True)
