"""Completion failure message - user-friendly failure messages.

Ported from src/agent/completion-failure-message.ts (28 lines) - generates
user-facing messages when the completion gate detects hard failures.

buildGateAwareAnswer(): Wraps raw answer with gate-aware failure handling.
"""

from __future__ import annotations

from rune.agent.completion_gate import (
    REQUIREMENT_IDS,
    CompletionGateResult,
)
from rune.utils.logger import get_logger

log = get_logger(__name__)

MAX_HARD_FAILURE_LINES = 3


def build_gate_aware_answer(
    raw_answer: str,
    gate_result: CompletionGateResult,
) -> str:
    """Build a gate-aware answer from raw agent output.

    When the gate is blocked with hard failures, logs them internally
    but returns the raw answer to the user - execution evidence and
    gate metadata are not exposed to the user.
    """
    if gate_result.outcome != "blocked":
        return raw_answer

    has_hard_failure = (
        len(gate_result.hard_failures) > 0
        or REQUIREMENT_IDS["NO_HARD_FAILURES"] in gate_result.missing_requirement_ids
    )

    if has_hard_failure:
        hard_failure_lines = gate_result.hard_failures[-MAX_HARD_FAILURE_LINES:]
        log.warning(
            "completion_gate_hard_failures",
            outcome=gate_result.outcome,
            hard_failures=hard_failure_lines,
            missing_requirement_ids=gate_result.missing_requirement_ids,
        )

    return raw_answer
