"""Proactive handler - GET /proactive/suggestions, POST /proactive/feedback.

Ported from src/api/handlers/proactive.ts - autonomous execution
dashboard, pending suggestions, and user feedback.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
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


# Routes


@router.get(
    "/suggestions",
    response_model=ProactiveDashboardResponse,
    dependencies=[Depends(auth)],
)
async def get_proactive_suggestions(limit: int = 20) -> ProactiveDashboardResponse:
    """Get the proactive engine dashboard.

    Includes autonomous execution stats, pending suggestions,
    pattern statistics, and governance information.
    """
    # Placeholder - in production, this queries the ProactiveEngine
    # and AutonomousExecutor subsystems.
    return ProactiveDashboardResponse(
        stats={
            "totalExecutions": 0,
            "successRate": 0.0,
            "autonomousCount": 0,
        },
        engine=EngineStats(running=False),
    )


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    dependencies=[Depends(auth)],
)
async def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Submit user feedback for a proactive suggestion.

    Accepted ``response`` values: ``accept``, ``reject``, ``dismiss``.
    """
    if req.response not in ("accept", "reject", "dismiss"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response: {req.response}. Use accept/reject/dismiss.",
        )

    log.info(
        "proactive_feedback",
        suggestion_id=req.suggestion_id,
        response=req.response,
    )

    return FeedbackResponse(acknowledged=True)
