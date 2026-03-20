"""Track pending UI focus targets for RUNE TUI.

Ported from src/ui/pending-focus.ts - determines which UI area
needs user attention (setup, approval, question, credential)
and provides display metadata for the focus banner.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class PendingFocusMode(StrEnum):
    IDLE = "idle"
    SETUP = "setup"
    APPROVAL = "approval"
    QUESTION = "question"
    CREDENTIAL = "credential"


@dataclass(slots=True, frozen=True)
class PendingFocusState:
    active: bool
    mode: PendingFocusMode
    color: str
    title: str
    headline: str
    prompt: str
    action_label: str
    action_hint: str
    detail: str
    status_text: str


@dataclass(slots=True)
class PendingFocusInput:
    setup_phase: Literal["select-provider", "enter-key"] | None = None
    has_approval: bool = False
    has_question: bool = False
    has_credential: bool = False
    approval_command: str = ""
    question: str = ""
    question_option_count: int = 0
    question_custom_selected: bool = False


_IDLE_STATE = PendingFocusState(
    active=False,
    mode=PendingFocusMode.IDLE,
    color="",
    title="",
    headline="",
    prompt="",
    action_label="",
    action_hint="",
    detail="",
    status_text="",
)


def _preview_text(text: str, max_len: int = 84) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 1] + "\u2026"


def get_pending_focus_state(inp: PendingFocusInput) -> PendingFocusState:
    """Derive the current pending-focus UI state from application signals."""

    if inp.setup_phase:
        is_provider = inp.setup_phase == "select-provider"
        return PendingFocusState(
            active=True,
            mode=PendingFocusMode.SETUP,
            color="cyan",
            title="Setup Required",
            headline="Initial setup is required to start using RUNE.",
            prompt=(
                "Select your AI provider." if is_provider else "Enter your API key."
            ),
            action_label="Action needed",
            action_hint=(
                "Type 1 or 2, then Enter"
                if is_provider
                else "Paste your API key and press Enter (empty to go back)"
            ),
            detail=(
                "Select your AI provider. (1: OpenAI, 2: Anthropic)"
                if is_provider
                else "Enter your API key and press Enter to save. Empty input goes back."
            ),
            status_text="INPUT REQUIRED",
        )

    if inp.has_approval:
        cmd_preview = (
            f"Command: {_preview_text(inp.approval_command, 140)}"
            if inp.approval_command
            else "Review the requested command."
        )
        return PendingFocusState(
            active=True,
            mode=PendingFocusMode.APPROVAL,
            color="yellow",
            title="Waiting for input",
            headline="Approval required before executing a risky command.",
            prompt=cmd_preview,
            action_label="Actions",
            action_hint="Use arrow keys then Enter, or type 1-4",
            detail="Use arrow keys or 1-4 to select, then Enter. Option 4 lets you deny with instructions.",
            status_text="INPUT REQUIRED",
        )

    if inp.has_question:
        question_preview = _preview_text(inp.question, 180) if inp.question else ""
        if inp.question_option_count > 0:
            hint = (
                "Type your custom answer below"
                if inp.question_custom_selected
                else "Use arrow keys to select or type a response"
            )
        else:
            hint = "Type your answer and press Enter"
        return PendingFocusState(
            active=True,
            mode=PendingFocusMode.QUESTION,
            color="yellow",
            title="Waiting for input",
            headline="The agent has a question that needs your answer.",
            prompt=question_preview,
            action_label="Action needed",
            action_hint=hint,
            detail=hint,
            status_text="INPUT REQUIRED",
        )

    if inp.has_credential:
        return PendingFocusState(
            active=True,
            mode=PendingFocusMode.CREDENTIAL,
            color="cyan",
            title="Credential Required",
            headline="A credential is needed to proceed.",
            prompt="Enter the requested credential.",
            action_label="Action needed",
            action_hint="Type the credential and press Enter",
            detail="Provide the requested credential to continue.",
            status_text="INPUT REQUIRED",
        )

    return _IDLE_STATE
