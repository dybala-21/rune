"""Create, register and promote reusable SKILL.md instructions."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.safety.execution_environment import execution_workspace
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Kebab-case regex (mirrors TS SkillCreateParamsSchema name validator)
_KEBAB_RE = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")


# Parameter schemas

class SkillCreateParams(BaseModel):
    """Parameters for skill.create (autonomous skill creation)."""

    name: str = Field(
        description='Skill name (kebab-case, e.g. "daily-briefing")',
    )
    description: str = Field(
        max_length=1024,
        description="What the skill does + when it activates + trigger phrases",
    )
    body: str = Field(
        description="Skill body (Markdown) — execution steps, caveats, examples",
    )
    scope: str = Field(
        default="user",
        description="Storage scope: user (~/.rune/skills/) or project (.rune/skills/)",
    )
    author: str = Field(
        default="rune-agent",
        max_length=128,
        description="Author identifier (for trust policy verification)",
    )
    signature: str | None = Field(
        default=None,
        max_length=256,
        description="Skill signature (hmac-sha256:<hex> or hex)",
    )

    @field_validator("name")
    @classmethod
    def _validate_kebab_case(cls, v: str) -> str:
        if not _KEBAB_RE.match(v):
            raise ValueError(
                f"Skill name must be kebab-case (e.g. 'daily-briefing'), got: {v!r}"
            )
        return v


class SkillPromoteParams(BaseModel):
    """Parameters for skill.promote to promote candidate/shadow skill to active."""

    name: str = Field(min_length=1, description="Skill name (frontmatter.name)")
    force: bool = Field(
        default=False,
        description="Force-promote even a retired skill to active",
    )


# Helpers

def _skills_dir(scope: str) -> Path:
    """Resolve the skills directory based on scope."""
    if scope == "project":
        return Path(execution_workspace()) / ".rune" / "skills"
    from rune.utils.paths import rune_home
    return rune_home() / "skills"


def _format_skill_md(
    name: str,
    description: str,
    body: str,
    author: str = "rune-agent",
    signature: str | None = None,
) -> str:
    """Build SKILL.md with validated YAML frontmatter."""
    lines = [
        "---",
        f"name: {name}",
        f"description: {json.dumps(description, ensure_ascii=False)}",
        f"author: {json.dumps(author, ensure_ascii=False)}",
    ]
    if signature:
        lines.append(f"signature: {json.dumps(signature)}")
    lines.append("metadata:")
    lines.append("  lifecycle: active")
    lines.append("---")
    lines.append("")
    lines.append(f"# {name}")
    lines.append("")
    lines.append(description)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(body)
    lines.append("")
    return "\n".join(lines)


# skill.create implementation

async def skill_create(params: SkillCreateParams) -> CapabilityResult:
    """Validate, write and register a kebab-case skill with optional signing."""
    log.info("skill_create", name=params.name, scope=params.scope, author=params.author)

    skills_dir = _skills_dir(params.scope)
    skill_dir = skills_dir / params.name
    skill_md_path = skill_dir / "SKILL.md"

    # Check existing skill in registry first
    try:
        from rune.skills.registry import get_skill_registry

        registry = get_skill_registry(workspace=execution_workspace())
        existing = registry.get(params.name)
        if existing:
            return CapabilityResult(
                success=False,
                error=(
                    f'Skill "{params.name}" already exists '
                    f"(source: {getattr(existing, 'source', 'unknown')}). "
                    "Choose a different name."
                ),
            )
    except ImportError:
        pass  # registry not available, proceed

    # Write to disk
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = _format_skill_md(
        name=params.name,
        description=params.description,
        body=params.body,
        author=params.author,
        signature=params.signature,
    )
    skill_md_path.write_text(content)

    # Hot-reload into registry
    loaded = False
    try:
        from rune.skills.registry import get_skill_registry

        registry = get_skill_registry()
        # Load the new directory through the registry's SKILL.md scanner.
        loaded = registry.load_skills(skill_dir) > 0
    except Exception as exc:
        log.warning("skill_hot_reload_failed", name=params.name, error=str(exc))

    if not loaded:
        log.warning(
            "skill_written_but_not_loaded",
            name=params.name,
            path=str(skill_md_path),
        )

    return CapabilityResult(
        success=True,
        output="\n".join([
            f'Skill "{params.name}" created successfully.',
            f"File: {skill_md_path}",
            f"Scope: {params.scope}",
            f"Status: {'Loaded and active in registry' if loaded else 'Written to disk (registry load pending)'}",
            "",
            f'The skill will activate when the user\'s request matches: "{params.description.split(chr(10))[0]}"',
        ]),
        metadata={
            "skill_name": params.name,
            "skill_path": str(skill_dir),
            "scope": params.scope,
            "author": params.author,
            "signed": params.signature is not None,
            "loaded": loaded,
        },
    )


# skill.promote implementation

async def skill_promote(params: SkillPromoteParams) -> CapabilityResult:
    """Promote an eligible skill to active."""
    log.info("skill_promote", name=params.name, force=params.force)

    try:
        from rune.skills.lifecycle import SkillState, get_state, set_state
        from rune.skills.persistence import persist_skill_state
        from rune.skills.registry import get_skill_registry

        registry = get_skill_registry(workspace=execution_workspace())
        # Apply promotion through the lifecycle state transition.
        skill = registry.get(params.name)
        if skill is None:
            return CapabilityResult(
                success=False,
                error=f'Skill "{params.name}" not found.',
            )

        prev_lifecycle = get_state(skill)
        current_lifecycle = SkillState.ACTIVE
        changed = prev_lifecycle != SkillState.ACTIVE
        if changed:
            set_state(skill, SkillState.ACTIVE)
            if skill.file_path and not persist_skill_state(skill):
                set_state(skill, prev_lifecycle)
                return CapabilityResult(success=False, error="The skill's new state could not be saved.")

        return CapabilityResult(
            success=True,
            output="\n".join([
                f'Skill "{params.name}" promoted successfully.',
                f"Previous lifecycle: {prev_lifecycle}",
                f"Current lifecycle: {current_lifecycle}",
                "Status: active and model-invocable" if changed else "Status: already active",
            ]),
            metadata={
                "skill_name": params.name,
                "previous_lifecycle": prev_lifecycle,
                "lifecycle": current_lifecycle,
                "changed": changed,
            },
        )

    except ImportError:
        return CapabilityResult(
            success=False,
            error="Skill registry not available",
        )
    except Exception as exc:
        log.error("skill_promote_failed", name=params.name, error=str(exc))
        return CapabilityResult(
            success=False,
            error=f"Failed to promote skill: {exc}",
        )


# Registration

class SkillLookupParams(BaseModel):
    name: str = Field(min_length=1, max_length=200, description="Exact name from the skill catalog")


class SkillSearchParams(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="The current step needing expertise")


async def skill_load(params: SkillLookupParams) -> CapabilityResult:
    from rune.skills.discovery import load

    try:
        text, name = await asyncio.to_thread(load, params.name, execution_workspace())
        return CapabilityResult(success=True, output=text, metadata={"loaded_skill": name})
    except Exception as exc:
        log.debug("skill_load_failed", name=params.name, error=str(exc))
        return CapabilityResult(success=False, error=str(exc))


async def skill_search(params: SkillSearchParams) -> CapabilityResult:
    from rune.skills.discovery import candidates

    try:
        found = await asyncio.to_thread(candidates, params.query, execution_workspace())
        return CapabilityResult(success=True, output=json.dumps(found, ensure_ascii=False))
    except Exception as exc:
        log.debug("skill_search_failed", error=str(exc))
        return CapabilityResult(success=False, error=str(exc))


def register_skill_ops_capabilities(registry: CapabilityRegistry) -> None:
    """Register skill discovery, loading and lifecycle operations."""
    registry.register(CapabilityDefinition(
        name="skill_search", description="Find active skill summaries for the current step. Does not load instructions.",
        domain=Domain.FILE, risk_level=RiskLevel.LOW, group="read",
        parameters_model=SkillSearchParams, execute=skill_search,
    ))
    registry.register(CapabilityDefinition(
        name="skill_load", description="Read an active skill's instructions by exact name. Load only skills relevant to the current step; different steps may use different skills. This grants no execution permission.",
        domain=Domain.FILE, risk_level=RiskLevel.LOW, group="read",
        parameters_model=SkillLookupParams, execute=skill_load,
    ))
    registry.register(CapabilityDefinition(
        name="skill_create",
        description=(
            "Create a new skill (SKILL.md) and register it immediately. "
            "Use this to teach yourself new routines, workflows, or automations "
            "that the user can trigger later."
        ),
        domain=Domain.FILE,
        risk_level=RiskLevel.MEDIUM,
        group="write",
        parameters_model=SkillCreateParams,
        execute=skill_create,
    ))
    registry.register(CapabilityDefinition(
        name="skill_promote",
        description=(
            "Promote an existing skill lifecycle to active so it becomes "
            "model-invocable. Use this after validating a candidate/shadow skill."
        ),
        domain=Domain.FILE,
        risk_level=RiskLevel.MEDIUM,
        group="write",
        parameters_model=SkillPromoteParams,
        execute=skill_promote,
    ))
