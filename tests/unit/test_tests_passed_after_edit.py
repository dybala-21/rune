"""Fresh checks drive both completion and the UI/learning signals."""

from types import SimpleNamespace

import pytest

from rune.agent.completion_gate import (
    CompletionGateInput,
    ExecutionEvidenceSnapshot,
    evaluate_completion_gate,
)
from rune.agent.memory_bridge import compute_auto_skill_quality_score
from rune.agent.verification_state import VerificationState, verified_outcome
from rune.api.server import build_trust_payload


def test_test_edit_echo_does_not_verify_new_revision():
    state = VerificationState()
    state.observe_command("pytest -q", True, "3 passed")
    state.changed()
    state.observe_command("echo done", True, "done")
    assert state.pending
    assert state.tests_passed_after_edit is False
    result = evaluate_completion_gate(CompletionGateInput(
        intent_resolved=True, requires_code_verification=True,
        evidence=ExecutionEvidenceSnapshot(verifications=1),
        verify_freshness_enabled=True, last_code_write_step=state.last_write,
        last_verify_step=state.last_pass, verification_passed=state.passed,
    ))
    assert result.outcome == "blocked"
    trace = SimpleNamespace(
        reason="completed", verification=state.snapshot(),
        tests_passed_after_edit=state.tests_passed_after_edit,
        mech_check="pass", evidence_score=1.0, final_step=3,
    )
    assert build_trust_payload(trace)["testsPassedAfterEdit"] is False
    assert build_trust_payload(trace)["verified"] is False
    assert verified_outcome(trace) is False
    assert compute_auto_skill_quality_score(trace) == 0.0


@pytest.mark.parametrize("command,output", [
    ("uv run pytest -q", "3 passed"),
    ("npm test", "Tests: 5 passed, 5 total"),
    ("node --test tests/*.test.mjs", "# tests 5\n# pass 5\n# fail 0"),
    ("npx vitest run", "Tests: 5 passed, 5 total"),
])
def test_fresh_tests_propagate_to_ui_and_learning(command, output):
    state = VerificationState()
    state.changed()
    assert state.observe_command(command, True, output)
    trace = SimpleNamespace(
        reason="completed", verification=state.snapshot(),
        tests_passed_after_edit=state.tests_passed_after_edit,
    )
    assert not state.pending
    assert build_trust_payload(trace)["testsPassedAfterEdit"] is True
    assert verified_outcome(trace) is True


@pytest.mark.parametrize("success,output", [
    (False, "1 failed"), (True, "no tests ran"), (True, "0 passed"),
])
def test_failed_or_empty_check_invalidates_previous_pass(success, output):
    state = VerificationState()
    state.changed()
    state.observe_command("pytest", True, "3 passed")
    state.observe_command("pytest", success, output)
    state.observe_command("ls", True, "main.py")
    assert state.pending
    assert state.tests_passed_after_edit is False


def test_quiet_lint_is_a_check_but_not_a_test_claim():
    state = VerificationState()
    state.changed()
    state.observe_command("ruff check .", True, "")
    assert state.passed
    assert state.tests_passed_after_edit is False


def test_no_check_is_unknown_not_a_learning_win():
    state = VerificationState()
    trace = SimpleNamespace(reason="completed", verification=state.snapshot())
    assert state.tests_passed_after_edit is None
    assert verified_outcome(trace) is None
    assert build_trust_payload(trace)["testsPassedAfterEdit"] is None
    assert build_trust_payload(trace)["verified"] is False


def test_task_gate_failure_is_not_overridden_by_a_fresh_test_pass():
    state = VerificationState()
    state.changed()
    state.observe_command("pytest", True, "3 passed")
    trace = SimpleNamespace(reason="completed", verification=state.snapshot(),
                            evidence_gate={"last_verdict": "fail"})
    assert verified_outcome(trace) is False
    assert build_trust_payload(trace)["verified"] is False


@pytest.mark.parametrize("later", ["ruff check .", "pytest tests/unit/test_one.py"])
def test_another_check_cannot_clear_a_failed_suite(later):
    state = VerificationState()
    state.changed()
    state.observe_command("pytest", True, "4 passed")
    state.observe_command("pytest", False, "1 failed, 3 passed")
    state.observe_command(later, True, "1 passed" if later.startswith("pytest") else "")
    assert state.pending
    assert state.tests_passed_after_edit is False
    assert state.snapshot()["status"] == "fail"
    state.observe_command("pytest", True, "4 passed")
    assert not state.pending
    assert state.tests_passed_after_edit is True


@pytest.mark.parametrize("command", [
    "pytest || true", "pytest; echo done", "pytest | tee result.txt", "! pytest",
])
def test_shell_success_cannot_certify_a_masked_check(command):
    state = VerificationState()
    state.changed()
    assert not state.observe_command(command, True, "1 failed, 3 passed")
    assert state.pending
    assert not state.tests_passed_after_edit
    state.observe_command("pytest", True, "4 passed")
    assert not state.pending


def test_pipeline_without_a_runner_verdict_is_inconclusive():
    state = VerificationState()
    state.changed()
    state.observe_command("pytest | tee result.txt", True, "")
    assert state.snapshot()["status"] == "inconclusive"
    assert state.pending
    assert verified_outcome({"verification": state.snapshot(), "mech_check": "pass"}) is False
    assert build_trust_payload(SimpleNamespace(reason="completed", verification=state.snapshot()))["verificationStatus"] == "inconclusive"


def test_check_scope_includes_directory_and_previous_revision():
    state = VerificationState()
    state.changed()
    state.observe_command("cd api && pytest", False, "1 failed", cwd="/project")
    state.observe_command("pytest", True, "2 passed", cwd="/project/web")
    assert state.pending
    state.observe_command("pytest", True, "3 passed", cwd="/project/api")
    assert not state.pending
    state.changed()
    state.observe_command("pytest", True, "3 passed", cwd="/project/api")
    assert state.pending
    assert not state.tests_passed_after_edit
