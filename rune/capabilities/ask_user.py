"""Ask-user capability for RUNE.

Ported from src/capabilities/ask-user.ts - allows the agent to ask the
user a question during task execution, with session call limits and
non-interactive mode handling.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Parameter schema

class AskUserOption(BaseModel):
    """A single selectable option."""
    label: str = Field(description="Option label (1-5 words)")
    description: str | None = Field(default=None, description="Optional description")


class AskUserParams(BaseModel):
    """Parameters for the ask_user capability."""
    question: str = Field(description="Question to ask the user")
    options: list[AskUserOption] | None = Field(
        default=None,
        min_length=2,
        max_length=4,
        description="Structured choices. Omit for free-text input.",
    )
    reason: str = Field(description="Why this question is needed (internal logging)")
    urgency: Literal["blocking", "clarifying", "confirming"] = Field(
        default="clarifying",
        description=(
            "blocking: answer required; "
            "clarifying: resolve ambiguity; "
            "confirming: confirm before proceeding"
        ),
    )


# User response types

@dataclass(slots=True)
class UserResponse:
    """Response received from the user."""
    selected_index: int  # -1 for free text
    answer: str
    raw_input: str | None = None
    free_text: bool = False


AskUserCallback = Callable[[AskUserParams], Awaitable[UserResponse]]

# Session state

_ask_count: int = 0
_DEFAULT_MAX_ASK: int = 2
_max_ask_limit: int = _DEFAULT_MAX_ASK
_response_callback: AskUserCallback | None = None


def set_ask_user_callback(callback: AskUserCallback | None) -> None:
    """Set the callback that delivers the question to the user (TUI/CLI)."""
    global _response_callback
    _response_callback = callback


def set_ask_user_limit(limit: int) -> None:
    """Adjust the per-session ask limit (clamped to 1..6)."""
    global _max_ask_limit
    _max_ask_limit = max(1, min(limit, 6))


def reset_ask_user_count() -> None:
    """Reset counters at the start of a new session."""
    global _ask_count, _max_ask_limit
    _ask_count = 0
    _max_ask_limit = _DEFAULT_MAX_ASK


def get_ask_user_count() -> int:
    """Return the current session ask count (useful for tests)."""
    return _ask_count


# Capability implementation

async def ask_user(params: AskUserParams) -> CapabilityResult:
    """Ask the user a question."""
    global _ask_count

    log.debug("ask_user", urgency=params.urgency, question=params.question, reason=params.reason)

    # Session limit
    if _ask_count >= _max_ask_limit:
        return CapabilityResult(
            success=False,
            error=(
                f"Maximum ask_user calls ({_max_ask_limit}) reached for this session. "
                "Make a decision based on available information."
            ),
            suggestions=[
                "Use file.read, file.search, or file.list to find the answer",
                "Pick the most reasonable default and proceed",
            ],
        )

    # Non-interactive mode
    if _response_callback is None:
        log.info("ask_user_blocked_non_interactive", question=params.question)
        return CapabilityResult(
            success=False,
            error=(
                "Cannot ask user in non-interactive mode. You MUST find the answer "
                "yourself using available tools, or make a reasonable default decision "
                "and proceed autonomously."
            ),
            suggestions=[
                "Use available tools (file.read, file.search, web.search) to find the answer",
                "Make a reasonable default choice and proceed",
                "Try an alternative approach that does not require user input",
            ],
        )

    # Interactive: ask the user
    _ask_count += 1

    try:
        response = await _response_callback(params)
        raw_input = response.raw_input if response.raw_input is not None else response.answer

        # Empty response - user skipped or dismissed the question
        if not (raw_input or "").strip() and not (response.answer or "").strip():
            log.info("ask_user_empty_response")
            return CapabilityResult(
                success=True,
                output=(
                    "User did not provide an answer (skipped). "
                    "Proceed autonomously with the most reasonable default. "
                    "Do NOT ask the same question again."
                ),
            )

        answer_text = (
            f'User typed: "{raw_input}"'
            if response.free_text
            else f'User selected: "{response.answer}"'
        )
        log.info("ask_user_response", answer=answer_text)

        return CapabilityResult(
            success=True,
            output=f'User responded: "{response.answer}"',
            metadata={
                "selectedIndex": response.selected_index,
                "freeText": response.free_text,
                "rawInput": raw_input,
                "urgency": params.urgency,
                "askCount": _ask_count,
            },
        )
    except Exception as exc:
        err_msg = str(exc)
        log.error("ask_user_callback_failed", error=err_msg)
        return CapabilityResult(
            success=False,
            error=f"Failed to get user response: {err_msg}",
            suggestions=["Proceed with the most reasonable default"],
        )


# Registration

def register_ask_user_capability(registry: CapabilityRegistry) -> None:
    """Register the ask_user capability."""
    registry.register(CapabilityDefinition(
        name="ask_user",
        description=(
            "Ask the user a question when ambiguity cannot be resolved "
            "through other tools. Use sparingly."
        ),
        domain=Domain.GENERAL,
        risk_level=RiskLevel.LOW,
        group="safe",
        parameters_model=AskUserParams,
        execute=ask_user,
    ))
