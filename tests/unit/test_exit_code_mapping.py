"""Exit code = the claim. Only a completed run may exit 0.

Measured on the harsh bench: a run that abstained (task_blocked) and one
that burned its budget (token_budget_exhausted) both exited 0, so the
harness — and any script — read an honest inside-give-up as a success.
"""

from __future__ import annotations

import pytest

from rune.cli.main import _exit_code_for_reason


def test_completed_is_the_only_zero_reason():
    assert _exit_code_for_reason("completed") == 0


@pytest.mark.parametrize("reason,code", [
    ("checks_failed", 1),
    ("completed_gate_warnings", 1),  # delivered but "treat as unverified"
    ("task_blocked", 3),
    ("cancelled", 130),
])
def test_named_endings_keep_their_codes(reason, code):
    assert _exit_code_for_reason(reason) == code


@pytest.mark.parametrize("reason", [
    "token_budget_exhausted", "stalled", "no_progress", "max_gate_blocked",
    "max_iterations", "advisor_abort", "no_pydantic_ai",
    "some_future_reason",
])
def test_every_other_way_of_stopping_short_is_nonzero(reason):
    # 4, not 2 — click already uses 2 for usage errors, and a script must
    # be able to tell "you called it wrong" from "it stopped short".
    assert _exit_code_for_reason(reason) == 4


def test_traceless_paths_are_not_judged():
    assert _exit_code_for_reason("") == 0
