"""Runs handler - GET /runs, GET /runs/{id}, GET /runs/{id}/events (SSE).

Ported from src/api/handlers/runs.ts - persistent run tracking and
event log retrieval.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from rune.api.auth import TokenAuthDependency
from rune.api.event_logger import read_events
from rune.api.handlers.agent import get_tracker
from rune.utils.fast_serde import json_encode
from rune.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])
auth = TokenAuthDependency()


# Models


class RunInfoResponse(BaseModel):
    run_id: str = Field(alias="runId")
    session_id: str = Field(alias="sessionId")
    client_id: str = Field(alias="clientId")
    status: str
    goal: str
    started_at: float = Field(alias="startedAt")
    completed_at: float | None = Field(None, alias="completedAt")
    error: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class RunsListResponse(BaseModel):
    runs: list[RunInfoResponse]
    total: int


class EventEntry(BaseModel):
    event: str
    data: Any
    timestamp: str


# Routes


@router.get("", response_model=RunsListResponse, dependencies=[Depends(auth)])
async def list_runs(
    client_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> RunsListResponse:
    """List agent execution runs with optional filters."""
    tracker = get_tracker()
    raw_runs = await tracker.query_runs(
        client_id=client_id,
        status=status,  # type: ignore[arg-type]
        limit=limit,
        offset=offset,
    )

    runs = [
        RunInfoResponse(
            runId=r["runId"],
            sessionId=r["sessionId"],
            clientId=r["clientId"],
            status=r["status"],
            goal=r["goal"],
            startedAt=r["startedAt"],
            completedAt=r.get("completedAt"),
            error=r.get("error"),
        )
        for r in raw_runs
    ]

    return RunsListResponse(runs=runs, total=len(runs))


@router.get("/{run_id}", response_model=RunInfoResponse, dependencies=[Depends(auth)])
async def get_run(run_id: str) -> RunInfoResponse:
    """Get details for a specific run."""
    tracker = get_tracker()
    run = tracker.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    return RunInfoResponse(
        runId=run.run_id,
        sessionId=run.session_id,
        clientId=run.client_id,
        status=run.status,
        goal=run.goal,
        startedAt=run.started_at,
        completedAt=run.completed_at,
        error=run.error,
    )


@router.get("/{run_id}/events", dependencies=[Depends(auth)])
async def get_run_events(run_id: str) -> StreamingResponse:
    """Stream events for a run via SSE.

    Returns ``text/event-stream`` content. Each event is a JSON
    ``data:`` line followed by a blank line.
    """
    tracker = get_tracker()
    run = tracker.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    conversation_id = run.conversation_id or run.session_id

    async def _generate() -> AsyncGenerator[str]:
        events = await read_events(
            conversation_id,
            run_id=run_id,
            include_tools=True,
            include_thinking=True,
        )
        for evt in events:
            yield f"data: {json_encode(evt)}\n\n"
        yield f"data: {json_encode({'event': 'done'})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
