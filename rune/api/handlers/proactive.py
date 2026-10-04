"""Expose proactive suggestions, execution status and user feedback."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from rune.api.auth import TokenAuthDependency
from rune.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/proactive", tags=["proactive"])
auth = TokenAuthDependency()


# Models


class SuggestionAction(BaseModel):
    command: str
    auto_executable: bool = Field(False, alias="autoExecutable")

    model_config = ConfigDict(populate_by_name=True)


class PendingSuggestion(BaseModel):
    id: str
    type: str
    priority: str
    title: str
    description: str
    confidence: float
    created_at: str = Field(alias="createdAt")
    action: SuggestionAction | None = None

    model_config = ConfigDict(populate_by_name=True)


class EngineStats(BaseModel):
    running: bool = False
    evaluation_count: int = Field(0, alias="evaluationCount")
    accept_rate: float = Field(0.0, alias="acceptRate")
    pending_count: int = Field(0, alias="pendingCount")
    interaction_count: int = Field(0, alias="interactionCount")

    model_config = ConfigDict(populate_by_name=True)


class ProactiveDashboardResponse(BaseModel):
    stats: dict[str, Any] = Field(default_factory=dict)
    patterns: list[dict[str, Any]] = Field(default_factory=list)
    recent_executions: list[dict[str, Any]] = Field(default_factory=list, alias="recentExecutions")
    engine: EngineStats = Field(default_factory=EngineStats)
    pending_suggestions: list[PendingSuggestion] = Field(default_factory=list, alias="pendingSuggestions")
    governance: dict[str, Any] | None = None
    policy: dict[str, Any] | None = None

    model_config = ConfigDict(populate_by_name=True)


class FeedbackRequest(BaseModel):
    suggestion_id: str = Field(alias="suggestionId")
    response: str  # "accept", "reject", "dismiss"

    model_config = ConfigDict(populate_by_name=True)


class FeedbackResponse(BaseModel):
    acknowledged: bool
    execution_status: str = Field("not_started", alias="executionStatus")

    model_config = ConfigDict(populate_by_name=True)


def _engine():
    from rune.memory.store import get_memory_store
    from rune.proactive.engine import get_proactive_engine

    engine = get_proactive_engine()
    engine.load_persisted_suggestions(get_memory_store())
    return engine


# Routes


@router.get("/feed", dependencies=[Depends(auth)])
async def get_proactive_feed(limit: int = Query(20, ge=1, le=100)) -> dict:
    """Restore recent proposals and outcomes without starting any work."""
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    suggestions = sorted(_engine().list_suggestions(), key=lambda s: (s.created_at, s.id))[-limit:]
    items = []
    for suggestion in suggestions:
        response = suggestion.status
        if response in ("accepted", "dismissed") and suggestion.response_source is None:
            response = "unconfirmed"
        if response == "pending" and suggestion.expires_at and suggestion.expires_at <= now:
            response = "expired"
        items.append({
            "id": suggestion.id, "title": suggestion.title, "description": suggestion.description,
            "confidence": suggestion.confidence, "createdAt": suggestion.created_at.isoformat(),
            "response": response, "executionStatus": suggestion.execution_status,
            "result": {key: suggestion.execution_result[key] for key in ("output", "error")
                       if key in suggestion.execution_result},
        })
    return {"suggestions": items}


@router.get(
    "/suggestions",
    response_model=ProactiveDashboardResponse,
    dependencies=[Depends(auth)],
)
async def get_proactive_suggestions(limit: int = Query(20, ge=1, le=100)) -> ProactiveDashboardResponse:
    """Return execution, suggestion, pattern and governance status."""
    from datetime import UTC, datetime

    from rune.proactive.bridge import get_proactive_bridge

    engine = _engine()
    suggestions = engine.list_suggestions()
    stats = engine.get_stats()
    bridge = get_proactive_bridge()
    completed = [s for s in suggestions if s.execution_status is not None]
    return ProactiveDashboardResponse(
        stats={
            "totalExecutions": len(completed),
            "verifiedExecutions": sum(s.execution_status == "success" for s in completed),
        },
        engine=EngineStats(
            running=bool(bridge and bridge.is_running),
            evaluation_count=stats["evaluation_count"],
            accept_rate=stats["acceptance_rate"],
            pending_count=stats["pending_count"],
            interaction_count=stats["interaction_count"],
        ),
        pending_suggestions=[PendingSuggestion(
            id=s.id, type=s.type, priority="normal", title=s.title,
            description=s.description, confidence=s.confidence,
            created_at=s.created_at.isoformat(),
        ) for s in suggestions if s.status == "pending" and s.execution_status is None
            and (s.expires_at is None or s.expires_at > datetime.now(UTC))][:limit],
        recent_executions=[{
            "suggestionId": s.id, "title": s.title,
            "status": s.execution_status, "response": s.status,
            "result": s.execution_result,
        } for s in completed[-limit:]],
    )


@router.get("/suggestions/{suggestion_id}", dependencies=[Depends(auth)])
async def get_suggestion(suggestion_id: str) -> dict:
    suggestion = _engine().get_suggestion(suggestion_id)
    if suggestion is None:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    response = suggestion.status
    if response in ("accepted", "dismissed") and suggestion.response_source is None:
        response = "unconfirmed"
    return {"id": suggestion.id, "response": response,
            "executionStatus": suggestion.execution_status,
            "result": suggestion.execution_result}


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    dependencies=[Depends(auth)],
)
async def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Record an accept, reject or dismiss response to a suggestion."""
    if req.response not in ("accept", "reject", "dismiss"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response: {req.response}. Use accept/reject/dismiss.",
        )

    engine = _engine()
    try:
        found = engine.handle_response(req.suggestion_id, req.response == "accept")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not found:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    suggestion = engine.get_suggestion(req.suggestion_id)
    if req.response == "accept" and suggestion is not None:
        from rune.proactive.bridge import get_proactive_bridge
        bridge = get_proactive_bridge()
        if bridge is not None:
            bridge.queue_accepted(suggestion)
    log.info(
        "proactive_feedback",
        suggestion_id=req.suggestion_id,
        response=req.response,
    )

    return FeedbackResponse(
        acknowledged=True,
        execution_status=(suggestion.execution_status if suggestion else None)
        or ("queued" if req.response == "accept" else "not_started"),
    )
