"""Skills handler - GET /skills, GET /skills/{id}, POST /skills/match.

Ported from src/api/handlers/skills.ts - skill registry CRUD and
matching API.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from rune.api.auth import TokenAuthDependency
from rune.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/skills", tags=["skills"])
auth = TokenAuthDependency()

_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")


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

    model_config = ConfigDict(populate_by_name=True)


class SkillMatchRequest(BaseModel):
    query: str
    limit: int = 5


class SkillMatchResponse(BaseModel):
    matches: list[SkillInfoResponse]


# Routes


class SkillCreateRequest(BaseModel):
    name: str
    description: str = ""
    body: str = ""
    scope: Literal["user", "project"] = "user"


class SkillUpdateRequest(BaseModel):
    description: str | None = None
    body: str | None = None


class SkillDeleteResponse(BaseModel):
    name: str
    deleted: bool


def _skills_root(scope: str) -> Path:
    """Directory a skill of this scope is stored in."""
    if scope == "project":
        return Path.cwd() / ".rune" / "skills"
    from rune.utils.paths import rune_home

    return rune_home() / "skills"


def _scope_of(path: Path) -> str | None:
    """Which skills directory a file lives under, or ``None`` if neither.

    Frontmatter carries a ``scope`` field, but it is just text in a file the
    skill itself provides. Trusting it let a skill claim ``builtin`` to become
    undeletable, or claim the wrong scope so the containment check pointed at a
    directory it was never in. Location is the only trustworthy answer.
    """
    try:
        resolved = path.resolve()
    except OSError:
        return None
    for scope in ("project", "user"):
        try:
            if resolved.is_relative_to(_skills_root(scope).resolve()):
                return scope
        except OSError:
            continue
    return None


def _editable_file(skill: Any) -> tuple[Path, str]:
    """The file to rewrite for this skill, and the scope it actually has.

    Raises if the skill has no file, or its file sits outside both skills
    directories — the two cases where a write would land somewhere the user
    never asked for.
    """
    if not skill.file_path:
        raise HTTPException(
            status_code=409, detail=f"Skill has no file on disk: {skill.name}"
        )

    path = Path(skill.file_path)
    if not path.is_file():
        raise HTTPException(
            status_code=409, detail=f"Skill file is missing: {skill.file_path}"
        )

    scope = _scope_of(path)
    if scope is None:
        raise HTTPException(
            status_code=409,
            detail=f"Skill file is outside the skills directories: {skill.file_path}",
        )
    return path.resolve(), scope


def _to_info(skill: Any) -> SkillInfoResponse:
    meta = skill.metadata or {}
    location = _scope_of(Path(skill.file_path)) if skill.file_path else None
    return SkillInfoResponse(
        name=skill.name,
        description=skill.description,
        # Where the file is, falling back to what it says only when it is not
        # in a skills directory at all (a programmatically registered skill).
        scope=location or skill.scope,
        lifecycle=meta.get("lifecycle", "stable"),
        author=skill.author or None,
        version=meta.get("version"),
        category=meta.get("category"),
        tags=[t.strip() for t in meta["tags"].split(",")] if meta.get("tags") else None,
        userInvocable=None,
        createdAt=meta.get("created_at") or meta.get("createdAt"),
        filePath=skill.file_path or None,
    )


async def _registry() -> Any:
    """The skill registry, built off the event loop the first time.

    Building it walks both skill directories with rglob. That is tens of
    milliseconds on a cold call, and doing it inline stalls everything else the
    server is serving — SSE heartbeats and a run's streaming output included.
    """
    from rune.skills import registry as registry_module

    if registry_module._registry is not None:  # noqa: SLF001
        return registry_module._registry  # noqa: SLF001
    return await asyncio.to_thread(registry_module.get_skill_registry)


async def _reload_registry_async() -> Any:
    """Re-scan off the event loop; see :func:`_reload_registry`."""
    return await asyncio.to_thread(_reload_registry)


def _reload_registry() -> Any:
    """Re-read both skill directories so a write is visible immediately.

    Skills registered in memory rather than loaded from a file — the ones the
    self-improving loop distills — are carried across. Dropping the registry
    outright discarded them on every save from the settings panel.
    """
    from rune.skills import registry as registry_module

    existing = registry_module._registry  # noqa: SLF001
    in_memory = [s for s in existing.list() if not s.file_path] if existing else []

    registry_module._registry = None  # noqa: SLF001
    fresh = registry_module.get_skill_registry()

    for skill in in_memory:
        # A file on disk wins: it is what the caller just wrote.
        if fresh.get(skill.name) is None:
            fresh.register(skill)
    return fresh


def _one_line(value: str) -> str:
    """Collapse a value to a single line so it cannot forge frontmatter.

    The registry reads frontmatter line by line, so a newline inside a value
    would start a new key: a description ending in "\\nscope: builtin" would
    write a skill that comes back undeletable, under a name nobody asked for.
    """
    return " ".join(value.split())


def _write_skill(name: str, description: str, body: str, scope: str) -> Path:
    """Write a new SKILL.md at the conventional path, creating its directory."""
    return _write_skill_at(_skills_root(scope) / name / "SKILL.md", name, description, body, scope)


def _write_skill_at(path: Path, name: str, description: str, body: str, scope: str) -> Path:
    """Write skill content to an exact path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    front = "\n".join(
        [
            "---",
            f"name: {name}",
            f"description: {_one_line(description)}",
            f"scope: {scope}",
            "---",
            "",
        ]
    )
    path.write_text(front + body.rstrip() + "\n", encoding="utf-8")
    return path


def _discard(path: Path) -> None:
    """Remove a skill file (and its folder) that failed to load back."""
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)
        if path.parent.is_dir() and not any(path.parent.iterdir()):
            path.parent.rmdir()


@router.get("", response_model=SkillListResponse, dependencies=[Depends(auth)])
async def list_skills(scope: str | None = None) -> SkillListResponse:
    """List all registered skills, optionally filtered by ``scope``."""
    skills = (await _registry()).list()
    if scope:
        skills = [s for s in skills if s.scope == scope]

    return SkillListResponse(
        skills=[_to_info(s) for s in sorted(skills, key=lambda s: s.name)],
        projectPath=str(_skills_root("project")),
        userPath=str(_skills_root("user")),
    )


@router.get("/{skill_name}", response_model=SkillDetailResponse, dependencies=[Depends(auth)])
async def get_skill(skill_name: str) -> SkillDetailResponse:
    """Get a skill including its body."""
    skill = (await _registry()).get(skill_name)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Skill not found: {skill_name}")

    info = _to_info(skill)
    return SkillDetailResponse(**info.model_dump(by_alias=True), body=skill.body)


@router.post("", response_model=SkillDetailResponse, dependencies=[Depends(auth)])
async def create_skill(req: SkillCreateRequest) -> SkillDetailResponse:
    """Create a skill as a SKILL.md file and register it."""
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Skill name is required")
    if not _SKILL_NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail="Skill name must be kebab-case (e.g. 'daily-briefing')",
        )

    if (await _registry()).get(name) is not None:
        raise HTTPException(status_code=409, detail=f"Skill already exists: {name}")

    target = _skills_root(req.scope) / name / "SKILL.md"
    if target.exists():
        # A file can be there under a name the registry does not know, because
        # the frontmatter names it something else. Overwriting it would destroy
        # a skill the user never mentioned.
        raise HTTPException(
            status_code=409, detail=f"A file already exists at {target}"
        )

    path = _write_skill(name, req.description, req.body, req.scope)
    log.info("skill_created", name=name, scope=req.scope)

    skill = (await _reload_registry_async()).get(name)
    if skill is None:
        # Written but it did not load back under its own name. Take the file
        # away rather than leaving it to be picked up by the next scan.
        _discard(path)
        await _reload_registry_async()
        raise HTTPException(status_code=500, detail=f"Skill written but not loadable: {name}")
    return SkillDetailResponse(**_to_info(skill).model_dump(by_alias=True), body=skill.body)


@router.patch("/{skill_name}", response_model=SkillDetailResponse, dependencies=[Depends(auth)])
async def update_skill(skill_name: str, req: SkillUpdateRequest) -> SkillDetailResponse:
    """Update a skill's description or body."""
    skill = (await _registry()).get(skill_name)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Skill not found: {skill_name}")

    path, scope = _editable_file(skill)
    _write_skill_at(
        path,
        skill_name,
        req.description if req.description is not None else skill.description,
        req.body if req.body is not None else skill.body,
        scope,
    )
    log.info("skill_updated", name=skill_name, scope=scope)

    updated = (await _reload_registry_async()).get(skill_name)
    if updated is None:
        raise HTTPException(status_code=500, detail=f"Skill written but not loadable: {skill_name}")
    return SkillDetailResponse(**_to_info(updated).model_dump(by_alias=True), body=updated.body)


@router.delete("/{skill_name}", response_model=SkillDeleteResponse, dependencies=[Depends(auth)])
async def delete_skill(skill_name: str) -> SkillDeleteResponse:
    """Delete a skill's SKILL.md and drop it from the registry."""
    skill = (await _registry()).get(skill_name)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Skill not found: {skill_name}")

    # _editable_file already refuses anything outside the skills directories.
    path, scope = _editable_file(skill)
    root = _skills_root(scope).resolve()

    path.unlink()
    if path.parent != root and not any(path.parent.iterdir()):
        path.parent.rmdir()

    await _reload_registry_async()
    log.info("skill_deleted", name=skill_name, scope=scope)
    return SkillDeleteResponse(name=skill_name, deleted=True)


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
