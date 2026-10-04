"""Expose summaries first; load instructions for the current step only."""

from __future__ import annotations

import json

from rune.skills.executor import build_skill_context, validate_skill_requirements
from rune.skills.lifecycle import is_injectable
from rune.skills.registry import get_skill_registry


def candidates(goal: str, workspace: str) -> list[dict]:
    registry = get_skill_registry(workspace=workspace)
    eligible = [skill for skill in registry.list() if is_injectable(skill)]
    # Small catalogs need no embedding call.
    if len(eligible) > 12:
        eligible = [m.skill for m in registry.search(goal) if is_injectable(m.skill)][:12]
    return [{"name": s.name, "description": s.description[:320]} for s in eligible]


def catalog(goal: str, workspace: str) -> str | None:
    skills = candidates(goal, workspace)
    if not skills:
        return None
    return (
        "Available skills (summaries, not active instructions):\n"
        + json.dumps(skills, ensure_ascii=False)
        + "\nUse skill_load only when its description fits the current step. You may load different "
        "skills for later steps. Use none for unrelated work. Use skill_search to find other skills. "
        "Load these tools with tool_search if needed. Skills do not grant tool or approval permissions."
    )


def load(name: str, workspace: str) -> tuple[str, str]:
    skill = get_skill_registry(workspace=workspace).get(name)
    if skill is None or not is_injectable(skill):
        raise ValueError("Skill is missing or not active")
    if missing := validate_skill_requirements(skill):
        raise ValueError("; ".join(missing))
    if len(skill.body) > 32_000:
        raise ValueError("Skill instructions exceed 32,000 characters; split the skill before loading")
    return build_skill_context(skill).formatted_context, skill.name
