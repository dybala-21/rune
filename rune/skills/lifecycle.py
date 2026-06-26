"""Skill lifecycle states for Gated Skill Learning (T1-1).

A distilled skill moves through an explicit lifecycle so its real effect on
task success can be measured before it is trusted:

    candidate -> shadow -> active -> deprecated -> retired

State is stored on a skill's frontmatter ``metadata``. Gating is opt-in: with
``is_injectable(skill)`` (default) every non-retired skill is injectable,
preserving pre-gating behaviour; with ``is_injectable(skill, gated=True)`` only
ACTIVE skills — those the evaluator has measured to help — are injected.
Transitions are driven by the evaluator via :func:`next_state`.
"""

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

# Default state for a skill that carries no explicit ``state`` field. Legacy
# skills (authored before lifecycle tracking) must keep behaving as before, so
# they are treated as ACTIVE rather than as unevaluated candidates.
LEGACY_DEFAULT_STATE: Final = SkillState.ACTIVE


def get_state(skill: Skill) -> str:
    """Return a skill's lifecycle state, defaulting legacy skills to active."""
    raw = skill.metadata.get(STATE_KEY)
    if isinstance(raw, str) and raw in SkillState.ALL:
        return raw
    return LEGACY_DEFAULT_STATE


def set_state(skill: Skill, state: str) -> None:
    """Set a skill's lifecycle state in its metadata (in-memory)."""
    if state not in SkillState.ALL:
        raise ValueError(f"unknown skill state: {state!r}")
    skill.metadata[STATE_KEY] = state


def is_injectable(skill: Skill, *, gated: bool = False) -> bool:
    """Whether a skill may be injected into a live run.

    ``gated=False`` (default): permissive — everything except a retired skill is
    injectable, preserving pre-gating behaviour (matched skills are injected
    regardless of state).

    ``gated=True`` (config ``skills.gated_learning`` on): only ACTIVE skills —
    those measured to raise the verified rate — are injected. Candidates and
    skills under evaluation stay out of live runs until promoted.
    """
    if gated:
        return get_state(skill) == SkillState.ACTIVE
    return get_state(skill) != SkillState.RETIRED


# Decision -> state transitions (driven by the evaluator).

def next_state(current: str, action: str) -> str:
    """Map an evaluator action (promote/reject/hold) to the next state.

    - PROMOTE: candidate/shadow → active (active stays active).
    - REJECT:  any evaluable state → deprecated (covers live regression of an
      already-active skill).
    - HOLD:    a candidate enters shadow (evaluation under way); otherwise the
      state is unchanged.
    Unknown/terminal states (deprecated, retired) are returned unchanged.
    """
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
