"""Track generated skills through evaluation; legacy authored skills remain active."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from rune.skills.types import Skill


class SkillState:
    """Canonical lifecycle states (stored as a string in skill metadata)."""

    CANDIDATE: Final = "candidate"   # just distilled; not yet evaluated
    SHADOW: Final = "shadow"         # under A/B evaluation
    ACTIVE: Final = "active"         # measured to help; injected live
    DEPRECATED: Final = "deprecated"  # regressed/ineffective; kept for audit
    RETIRED: Final = "retired"       # hard-removed from the registry

    ALL: Final = frozenset({CANDIDATE, SHADOW, ACTIVE, DEPRECATED, RETIRED})


# Metadata key under which the state is persisted in SKILL.md frontmatter.
STATE_KEY: Final = "state"

# Treat legacy skills without lifecycle metadata as active for compatibility.
LEGACY_DEFAULT_STATE: Final = SkillState.ACTIVE


def get_state(skill: Skill) -> str:
    """Return a skill's lifecycle state, defaulting legacy skills to active."""
    raw = skill.metadata.get(STATE_KEY)
    if isinstance(raw, str) and raw in SkillState.ALL:
        return raw
    if raw is not None or skill.author == "auto" or skill.metadata.get("source") == "auto_distill":
        return SkillState.CANDIDATE
    return LEGACY_DEFAULT_STATE


def set_state(skill: Skill, state: str) -> None:
    """Set a skill's lifecycle state in its metadata (in-memory)."""
    if state not in SkillState.ALL:
        raise ValueError(f"unknown skill state: {state!r}")
    skill.metadata[STATE_KEY] = state


def is_injectable(skill: Skill, *, gated: bool = False) -> bool:
    """Only active skills are reusable, even when background evaluation is off."""
    return get_state(skill) == SkillState.ACTIVE


# Decision -> state transitions (driven by the evaluator).

def next_state(current: str, action: str) -> str:
    """Apply promote/reject/hold transitions; leave unknown or terminal states unchanged."""
    from rune.skills.evaluation import HOLD, PROMOTE, REJECT

    if current not in (SkillState.CANDIDATE, SkillState.SHADOW, SkillState.ACTIVE):
        return current
    if action == REJECT:
        return SkillState.DEPRECATED
    if action == PROMOTE:
        return SkillState.ACTIVE
    if action == HOLD:
        return SkillState.SHADOW if current == SkillState.CANDIDATE else current
    return current
