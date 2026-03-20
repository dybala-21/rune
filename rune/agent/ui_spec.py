"""UI spec - agent-to-UI communication types and labels.

Ported from src/agent/ui-spec.ts (36 lines) - single source of truth
for TUI/Gateway progress phase labels and app status labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Types

ProgressPhase = Literal["thinking", "researching", "analyzing", "executing", "composing"]


@dataclass(slots=True)
class ProgressPhaseSpec:
    """Specification for a single progress phase."""

    emoji: str
    label: str


# Phase specifications

PROGRESS_PHASE_SPEC: dict[ProgressPhase, ProgressPhaseSpec] = {
    "thinking": ProgressPhaseSpec(emoji="\U0001f9e0", label="Analyzing request"),
    "researching": ProgressPhaseSpec(emoji="\U0001f50d", label="Gathering information"),
    "analyzing": ProgressPhaseSpec(emoji="\U0001f4ca", label="Analyzing"),
    "executing": ProgressPhaseSpec(emoji="\U0001f6e0\ufe0f", label="Executing task"),
    "composing": ProgressPhaseSpec(emoji="\u270d\ufe0f", label="Composing response"),
}


# Label helpers

AgentPhase = Literal["thinking", "streaming", "executing", "idle"]


def get_default_running_label(agent_phase: AgentPhase) -> str:
    """Get the default running label for a given agent phase."""
    if agent_phase == "thinking":
        return f"{PROGRESS_PHASE_SPEC['thinking'].label}..."
    if agent_phase == "streaming":
        return f"{PROGRESS_PHASE_SPEC['composing'].label}..."
    if agent_phase == "executing":
        return f"{PROGRESS_PHASE_SPEC['executing'].label}..."
    return "Working through the request..."


APP_STATUS_LABEL_SPEC: dict[str, str] = {
    "idle": "Ready",
    "thinking": PROGRESS_PHASE_SPEC["thinking"].label,
    "planning": PROGRESS_PHASE_SPEC["analyzing"].label,
    "awaiting_approval": "Awaiting approval",
    "running": PROGRESS_PHASE_SPEC["executing"].label,
    "error": "Error",
}
