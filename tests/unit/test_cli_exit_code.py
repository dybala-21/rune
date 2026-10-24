"""A single --message run exits with the verdict the trust card shows."""

from types import SimpleNamespace

import pytest

from rune.cli.main import EXIT_UNCONFIRMED, run_exit_code


@pytest.mark.parametrize("trace,code", [
    (SimpleNamespace(reason="completed"), 0),
    (SimpleNamespace(reason="completed", requirement_acceptance={"required": True, "status": "acknowledged"}), 0),
    (SimpleNamespace(reason="checks_failed"), 1),
    (SimpleNamespace(reason="error: provider down"), 1),
    (SimpleNamespace(reason="completed", verification={"required": True, "status": "fail"}), 1),
    (SimpleNamespace(reason="completed_gate_warnings"), EXIT_UNCONFIRMED),
    (SimpleNamespace(reason="task_blocked"), EXIT_UNCONFIRMED),
    (SimpleNamespace(reason="request_budget_exhausted"), EXIT_UNCONFIRMED),
    (SimpleNamespace(reason="verification_inconclusive"), EXIT_UNCONFIRMED),
    (None, 1),
])
def test_exit_code_follows_the_run_verdict(trace, code):
    assert run_exit_code(trace) == code
