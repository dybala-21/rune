"""Finish a run by reporting that the task cannot be done as stated.

Every other ending is shaped like success, so an agent facing an
unsatisfiable request — a test that contradicts the spec it guards, a
premise that does not hold, an input that is not there — still produces
the nearest thing to an answer: it bends the code until the wrong test
passes, or complies and breaks a documented contract.

This gives refusing its own ending. Calling it stops the run with a
non-zero status and a statement of what conflicts with what.

Offering the tool is not enough by itself; how readily a model reaches for
it varies. The adapter names it at the points where a conflict has already
been detected mechanically — a blocked attempt to rewrite a test, an input
that could not be found.
"""

from __future__ import annotations

from contextvars import ContextVar

from pydantic import BaseModel, Field

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)

BLOCKED_MARKER = "[TASK_BLOCKED]"

# Set by the tool, read once by the loop when it finishes. A ContextVar so
# concurrent sessions in one process cannot see each other's outcome.
_BLOCKED: ContextVar[str] = ContextVar("rune_task_blocked", default="")


def blocked_reason() -> str:
    """Peek at the abstain reason without clearing it."""
    return _BLOCKED.get()


def consume_block() -> str:
    """Return and clear the abstain reason recorded during this run."""
    value = _BLOCKED.get()
    if value:
        _BLOCKED.set("")
    return value


class TaskBlockedParams(BaseModel):
    reason: str = Field(
        description=(
            "What makes the task impossible as stated. Name both sides of "
            "the conflict concretely, e.g. which test expects which value "
            "and which document requires a different one."
        )
    )
    evidence: str = Field(
        default="",
        description="File paths, quoted lines or command output showing it.",
    )


async def task_blocked(params: TaskBlockedParams) -> CapabilityResult:
    """End the run: the task cannot be completed correctly as stated."""
    reason = (params.reason or "").strip()
    if not reason:
        return CapabilityResult(
            success=False,
            error="A reason is required — say what conflicts with what.",
        )
    log.info("task_blocked", reason=reason[:200])
    _BLOCKED.set(reason)
    body = reason if not params.evidence else f"{reason}\n\n{params.evidence}"
    return CapabilityResult(
        success=True,
        output=f"{BLOCKED_MARKER} {body}",
        metadata={"blocked": True, "reason": reason},
    )


def register_blocked_capability(registry: CapabilityRegistry) -> None:
    registry.register(
        CapabilityDefinition(
            name="task_blocked",
            description=(
                "Report that the task cannot be completed correctly as "
                "stated and stop. Use when the instructions conflict with "
                "something authoritative in the project (a spec, a "
                "documented contract, an existing test), when a required "
                "input does not exist, or when following the request "
                "literally would break behaviour the project relies on. "
                "Reporting a conflict is a successful outcome; quietly "
                "working around it is not."
            ),
            domain=Domain.GENERAL,
            risk_level=RiskLevel.LOW,
            group="safe",
            parameters_model=TaskBlockedParams,
            execute=task_blocked,
        )
    )
