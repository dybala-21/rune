"""Skills handler - GET /skills, GET /skills/{id}, POST /skills/match.

Ported from src/api/handlers/skills.ts - skill registry CRUD and
matching API.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from rune.api.auth import TokenAuthDependency
from rune.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/skills", tags=["skills"])
auth = TokenAuthDependency()


# Models


class SkillInfoResponse(BaseModel):
    name: str
    description: str
    scope: str
    lifecycle: str = "stable"
    author: str | None = None
    version: str | None = None
    category: str | None = None
    tags: list[str] | None = None
    user_invocable: bool | None = Field(None, alias="userInvocable")
    created_at: str | None = Field(None, alias="createdAt")
    file_path: str | None = Field(None, alias="filePath")

    model_config = ConfigDict(populate_by_name=True)


class SkillListResponse(BaseModel):
    skills: list[SkillInfoResponse]
    project_path: str = Field("", alias="projectPath")
    user_path: str = Field("", alias="userPath")

    model_config = ConfigDict(populate_by_name=True)


class SkillDetailResponse(SkillInfoResponse):
    body: str = ""
    frontmatter_raw: str = Field("", alias="frontmatterRaw")


class SkillMatchRequest(BaseModel):
    query: str
    limit: int = 5


class SkillMatchResponse(BaseModel):
    matches: list[SkillInfoResponse]


# Routes


@router.get("", response_model=SkillListResponse, dependencies=[Depends(auth)])
async def list_skills(scope: str | None = None) -> SkillListResponse:
    """List all registered skills.

    Optionally filter by ``scope`` (``user``, ``project``, ``builtin``).
    """
    # Placeholder - in production, queries the SkillRegistry
    return SkillListResponse(
        skills=[],
        projectPath="",
        userPath="",
    )


@router.get("/{skill_name}", response_model=SkillDetailResponse, dependencies=[Depends(auth)])
async def get_skill(skill_name: str) -> SkillDetailResponse:
    """Get detailed information about a skill including its body."""
    # Placeholder - in production, queries the SkillRegistry
    raise HTTPException(status_code=404, detail=f"Skill not found: {skill_name}")


@router.post("/match", response_model=SkillMatchResponse, dependencies=[Depends(auth)])
async def match_skills(req: SkillMatchRequest) -> SkillMatchResponse:
    """Find skills matching a natural language query.

    Uses keyword/semantic matching to find the most relevant skills
    for the given query string.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    log.info("skill_match", query=req.query[:80], limit=req.limit)

    # Placeholder - in production, uses the SkillRegistry's match method
    return SkillMatchResponse(matches=[])
