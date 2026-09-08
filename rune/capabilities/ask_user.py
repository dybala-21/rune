"""Ask-user capability for RUNE.

Ported from src/capabilities/ask-user.ts - allows the agent to ask the
user a question during task execution, with session call limits and
non-interactive mode handling.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
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

_DEFAULT_MAX_ASK: int = 2


@dataclass
class _AskSession:
    callback: AskUserCallback | None = None
    count: int = 0
    limit: int = _DEFAULT_MAX_ASK
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_session: ContextVar[_AskSession | None] = ContextVar("rune_ask_user_session", default=None)


def _current_session() -> _AskSession:
    state = _session.get()
    if state is None:
        state = _AskSession()
        _session.set(state)
    return state


def set_ask_user_callback(callback: AskUserCallback | None) -> None:
    """Set the callback that delivers the question to the user (TUI/CLI)."""
    _session.set(_AskSession(callback=callback))


def set_ask_user_limit(limit: int) -> None:
    """Adjust the per-session ask limit (clamped to 1..6)."""
    _session.set(replace(_current_session(), limit=max(1, min(limit, 6))))


def reset_ask_user_count() -> None:
    """Reset counters at the start of a new session."""
    _session.set(_AskSession(callback=_current_session().callback))


def get_ask_user_count() -> int:
    """Return the current session ask count (useful for tests)."""
    return _current_session().count


def user_response(params: AskUserParams, answer: str, selected_index: int | None = None) -> UserResponse:
    """Normalize channel input while preserving selection versus free text."""
    if selected_index is None or selected_index == -1:
        return UserResponse(selected_index=-1, answer=answer, raw_input=answer, free_text=True)
    if (isinstance(selected_index, bool) or not isinstance(selected_index, int)
            or not params.options or not 0 <= selected_index < len(params.options)):
        raise ValueError("Selected option is not available for this question")
    return UserResponse(selected_index=selected_index, answer=params.options[selected_index].label,
                        raw_input=answer, free_text=False)


# Capability implementation

async def ask_user(params: AskUserParams) -> CapabilityResult:
    """Ask the user a question."""
    state = _current_session()
    async with state.lock:
        return await _ask_user(params, state)


async def _ask_user(params: AskUserParams, state: _AskSession) -> CapabilityResult:

    log.debug("ask_user", urgency=params.urgency, question=params.question, reason=params.reason)

    # Session limit
    if state.count >= state.limit:
        return CapabilityResult(
            success=False,
            error=(
                f"Maximum ask_user calls ({state.limit}) reached for this session. "
                "Make a decision based on available information."
            ),
            suggestions=[
                "Use file.read, file.search, or file.list to find the answer",
                "Pick the most reasonable default and proceed",
            ],
        )

    # Non-interactive mode
    if state.callback is None:
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
    state.count += 1

    try:
        response = await state.callback(params)
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
                "askCount": state.count,
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
