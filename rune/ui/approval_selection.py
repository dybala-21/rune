"""Approval UI selection logic for RUNE TUI.

Ported from src/ui/approval-selection.ts - approval/deny/modify option
cycling, keyboard shortcuts, and input resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class ApprovalDecision(StrEnum):
    APPROVE_ONCE = "approve_once"
    APPROVE_ALWAYS = "approve_always"
    DENY = "deny"


APPROVAL_OPTIONS: list[ApprovalDecision] = [
    ApprovalDecision.APPROVE_ONCE,
    ApprovalDecision.APPROVE_ALWAYS,
    ApprovalDecision.DENY,
]

DEFAULT_APPROVAL_SELECTION_INDEX = 2  # deny
APPROVAL_PROMPT_SELECTION_INDEX = 3   # "deny with instructions"


@dataclass(slots=True, frozen=True)
class ApprovalInputDecision:
    kind: Literal["decision"] = "decision"
    decision: ApprovalDecision = ApprovalDecision.DENY


@dataclass(slots=True, frozen=True)
class ApprovalInputNeedsPrompt:
    kind: Literal["needs_prompt"] = "needs_prompt"


ApprovalInputIntent = ApprovalInputDecision | ApprovalInputNeedsPrompt


def move_approval_selection(
    current_index: int,
    direction: Literal["left", "right", "up", "down"],
) -> int:
    """Cycle through approval options, wrapping around at edges."""
    max_index = APPROVAL_PROMPT_SELECTION_INDEX
    normalized = max(0, min(current_index, max_index))

    if direction in ("left", "up"):
        return max_index if normalized <= 0 else normalized - 1
    return 0 if normalized >= max_index else normalized + 1


def selected_approval_decision(index: int) -> ApprovalDecision:
    """Map a selection index to the corresponding decision."""
    if index <= 0:
        return ApprovalDecision.APPROVE_ONCE
    if index == 1:
        return ApprovalDecision.APPROVE_ALWAYS
    return ApprovalDecision.DENY


def is_approval_prompt_selection(index: int) -> bool:
    return index == APPROVAL_PROMPT_SELECTION_INDEX


def resolve_approval_from_input(
    text: str,
    selected_index: int,
) -> ApprovalInputIntent | None:
    """Resolve user input to an approval intent.

    Returns *None* when the input cannot be mapped to any valid option.
    """
    trimmed = text.strip()

    if not trimmed:
        if is_approval_prompt_selection(selected_index):
            return ApprovalInputNeedsPrompt()
        return ApprovalInputDecision(decision=selected_approval_decision(selected_index))

    if trimmed.isdigit():
        num = int(trimmed)
        if 1 <= num <= len(APPROVAL_OPTIONS):
            return ApprovalInputDecision(decision=APPROVAL_OPTIONS[num - 1])
        if num == APPROVAL_PROMPT_SELECTION_INDEX + 1:
            return ApprovalInputNeedsPrompt()
        return None

    shortcut_map: dict[str, ApprovalInputIntent] = {
        "y": ApprovalInputDecision(decision=ApprovalDecision.APPROVE_ONCE),
        "a": ApprovalInputDecision(decision=ApprovalDecision.APPROVE_ALWAYS),
        "n": ApprovalInputDecision(decision=ApprovalDecision.DENY),
    }
    return shortcut_map.get(trimmed.lower())
