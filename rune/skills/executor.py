"""Skill execution for RUNE.

Parses a skill body into executable steps, builds execution contexts
for agent prompt injection, and validates skill requirements.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rune.skills.types import Skill, SkillMatch
from rune.utils.logger import get_logger

if TYPE_CHECKING:
    from rune.skills.registry import SkillRegistry

log = get_logger(__name__)


# SkillExecutionContext

@dataclass(slots=True)
class SkillExecutionContext:
    """Context object injected into the agent prompt for skill execution."""

    active_skill_name: str
    instructions: str
    metadata: dict[str, Any] = field(default_factory=dict)
    formatted_context: str = ""


# Context building

def build_skill_context(
    skill: Skill,
    *,
    include_body: bool = True,
    max_body_chars: int = 0,
) -> SkillExecutionContext:
    """Convert a skill into an agent-injectable execution context.

    Parameters
    ----------
    skill:
        The loaded skill to convert.
    include_body:
        Whether to include the full skill body (instructions) in the context.
    max_body_chars:
        If > 0, truncate the body to this many characters.
    """
    body = skill.body
    if not include_body:
        body_for_prompt = ""
    elif max_body_chars > 0 and len(body) > max_body_chars:
        body_for_prompt = body[:max_body_chars] + "\n...(truncated)"
    else:
        body_for_prompt = body

    sections: list[str] = []

    # Skill header
    sections.append(f"## Active Skill: {skill.name}")
    sections.append(f"Description: {skill.description}")

    # Compatibility
    compatibility = skill.metadata.get("compatibility")
    if compatibility:
        sections.append(f"Compatibility: {compatibility}")

    # Requirements
    requires = skill.metadata.get("requires") or {}
    req_lines: list[str] = []
    env_vars: list[str] = requires.get("env") or []
    bins: list[str] = requires.get("bins") or []
    mcp_services: list[str] = requires.get("mcp") or []

    if env_vars:
        req_lines.append(f"- Environment variables: {', '.join(env_vars)}")
    if bins:
        req_lines.append(f"- Required binaries: {', '.join(bins)}")
    if mcp_services:
        req_lines.append(f"- MCP services: {', '.join(mcp_services)}")
    if req_lines:
        sections.append(f"Requirements:\n{chr(10).join(req_lines)}")

    # Script paths
    scripts: list[str] = skill.metadata.get("scripts") or []
    base_path = str(Path(skill.file_path).parent) if skill.file_path else ""
    if scripts:
        sections.append("\n### Executable Scripts")
        sections.append(
            "**Important: Use bash capability to execute the scripts below.**"
        )
        if base_path:
            sections.append(f"Skill path: {base_path}")
        for script in scripts:
            script_path = f"{base_path}/scripts/{script}" if base_path else script
            sections.append(f"- `{script_path}`")

    # Skill instructions (body)
    if include_body:
        sections.append(f"\n### Skill Instructions\n{body_for_prompt}")
    else:
        sections.append(
            "\n### Skill Instructions\n(summary mode) Detailed body not injected."
        )

    formatted = (
        f"\n--- Skill Context ---\n"
        f"{chr(10).join(sections)}\n"
        f"--- End Skill Context ---\n"
    )

    meta: dict[str, Any] = {
        "description": skill.description,
    }
    if compatibility:
        meta["compatibility"] = compatibility
    if requires:
        meta["requirements"] = requires

    return SkillExecutionContext(
        active_skill_name=skill.name,
        instructions=skill.body,
        metadata=meta,
        formatted_context=formatted,
    )


def build_skill_context_for_goal(
    goal: str,
    registry: SkillRegistry | None = None,
) -> SkillExecutionContext | None:
    """Find the best-matching skill for a goal and return its context.

    Uses the skill registry's keyword/fuzzy search to rank skills,
    then returns the context for the highest-scoring match.
    """
    from rune.skills.registry import get_skill_registry

    reg = registry or get_skill_registry()

    try:
        matches: list[SkillMatch] = reg.search(goal)
    except Exception:
        log.warning("skill_search_failed", goal=goal[:50])
        return None

    if not matches:
        log.debug("no_matching_skills", goal=goal[:50])
        return None

    best = matches[0]
    log.info(
        "skill_matched_for_goal",
        skill=best.skill.name,
        score=best.score,
        goal=goal[:50],
    )
    return build_skill_context(best.skill)


def merge_skill_contexts(contexts: list[SkillExecutionContext]) -> str:
    """Merge multiple skill contexts into a single prompt string."""
    if not contexts:
        return ""
    return "\n".join(ctx.formatted_context for ctx in contexts)


# Requirement validation

def validate_skill_requirements(skill: Skill) -> list[str]:
    """Validate that a skill's requirements are satisfied.

    Returns a list of human-readable strings describing missing requirements.
    An empty list means all requirements are met.
    """
    requires = skill.metadata.get("requires") or {}
    missing: list[str] = []

    # Check environment variables
    for var in requires.get("env") or []:
        if not os.environ.get(var):
            missing.append(f"Missing environment variable: {var}")

    # Check binaries on PATH
    for binary in requires.get("bins") or []:
        if shutil.which(binary) is None:
            missing.append(f"Missing binary on PATH: {binary}")

    # Check MCP services (currently cannot verify connectivity - report as unchecked)
    for svc in requires.get("mcp") or []:
        missing.append(f"MCP service not verifiable: {svc}")

    return missing


# Full execution context (memory + skills)

async def build_full_execution_context(
    goal: str,
    *,
    include_skills: bool = True,
    include_memory: bool = True,
    memory_context: str | None = None,
    registry: SkillRegistry | None = None,
) -> str:
    """Build the combined context string for agent execution.

    Merges optional memory context with skill context looked up by goal.
    """
    sections: list[str] = []

    if include_memory and memory_context:
        sections.append(memory_context)

    if include_skills:
        skill_ctx = build_skill_context_for_goal(goal, registry)
        if skill_ctx:
            sections.append(skill_ctx.formatted_context)

    return "\n".join(sections)

# Matches numbered list items (1. ..., 2. ...) or bullet items (- ..., * ...)
_STEP_RE = re.compile(r"^\s*(?:\d+\.\s+|[-*]\s+)(.+)$", re.MULTILINE)


def _parse_skill_body(body: str) -> list[str]:
    """Parse a skill body (Markdown) into a list of instruction steps.

    Extracts ordered/unordered list items as individual steps.
    If no list items are found, treats each non-empty paragraph as a step.
    """
    steps: list[str] = []

    for m in _STEP_RE.finditer(body):
        step_text = m.group(1).strip()
        if step_text:
            steps.append(step_text)

    if not steps:
        # Fallback: split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
        steps = paragraphs

    return steps


async def execute_skill(skill: Skill, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a skill given an optional context.

    Parameters
    ----------
    skill:
        The skill to execute.
    context:
        Arbitrary context dict (may include "agent", "session", "args", etc.).

    Returns
    -------
    dict with keys:
        - success (bool)
        - steps_executed (int)
        - total_steps (int)
        - results (list[str]): per-step result descriptions
        - error (str|None)
    """
    ctx = context or {}
    steps = _parse_skill_body(skill.body)

    log.info("skill_execute_start", name=skill.name, steps=len(steps))

    results: list[str] = []
    steps_executed = 0

    for i, step_text in enumerate(steps, 1):
        try:
            # If an agent is provided in context, delegate execution
            agent = ctx.get("agent")
            if agent is not None and hasattr(agent, "execute_step"):
                result = await agent.execute_step(step_text, context=ctx)
                results.append(str(result))
            else:
                # No agent - just record the step as "pending"
                results.append(f"[pending] {step_text}")

            steps_executed = i
            log.debug(
                "skill_step_done",
                name=skill.name,
                step=i,
                total=len(steps),
            )

        except Exception as exc:
            log.error(
                "skill_step_error",
                name=skill.name,
                step=i,
                error=str(exc),
            )
            return {
                "success": False,
                "steps_executed": steps_executed,
                "total_steps": len(steps),
                "results": results,
                "error": f"Step {i} failed: {exc}",
            }

    log.info("skill_execute_done", name=skill.name, steps_executed=steps_executed)

    return {
        "success": True,
        "steps_executed": steps_executed,
        "total_steps": len(steps),
        "results": results,
        "error": None,
    }
