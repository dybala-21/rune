"""Completion and check results stay distinct in the UI contract."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("status,reason", [("unavailable", "verification_unavailable"),
                                          ("inconclusive", "verification_inconclusive")])
def test_browser_verification_error_is_not_execution_failure(status, reason):
    from rune.agent.run_outcome import run_outcome

    trace = SimpleNamespace(reason=reason, requirement_acceptance={"required": True, "status": status},
                            completion_check={"name": "Task outcome", "detail": "Readback unavailable"})
    outcome = run_outcome(trace)
    payload = build_trust_payload(trace)
    assert outcome.completion == "unknown" and outcome.status == "unverified" and not outcome.success
    assert payload["verificationStatus"] == status
    assert payload["completionCheck"] == trace.completion_check
    assert not payload["canEscalate"]


@pytest.mark.parametrize("review", ["pass", "unavailable", "fail"])
def test_browser_checks_and_output_review_cannot_overwrite_each_other(review):
    from rune.agent.run_outcome import merge_requirement_checks, run_outcome

    browser = {"required": True, "status": "pass", "method": "page_readback"}
    explanation = {"required": True, "status": review, "method": "model_review"}
    merged = merge_requirement_checks(browser, explanation)
    assert merged["status"] == review
    assert merged["components"] == [browser, explanation]
    assert run_outcome(SimpleNamespace(reason="completed", requirement_acceptance=merged)).success is (review == "pass")

from rune.api.server import build_trust_payload


@pytest.mark.parametrize("reason,fields,completion,verification,required,escalate", [
    ("completed", {}, "completed", "not_checked", False, False),
    ("completed", {"verification": {"required": False, "status": "unverified"}},
     "completed", "not_checked", False, False),
    ("completed", {"evidence_gate": {"has_check": True, "last_verdict": "pass"}},
     "completed", "passed", False, False),
    ("completed", {"evidence_gate": {"has_check": True, "last_verdict": "skip"}},
     "completed", "inconclusive", False, False),
    ("completed", {"evidence_gate": {"has_check": False, "last_verdict": "skip"}},
     "completed", "not_checked", False, False),
    ("completed", {"verification": {"required": True, "status": "unverified"},
                   "mech_check": "pass", "tests_passed_after_edit": False},
     "completed", "not_checked", True, False),
    ("completed", {"verification": {"required": True, "status": "fail"}},
     "completed", "failed", True, False),
    ("completed", {"verification": {"required": True, "status": "pass"},
                   "evidence_gate": {"last_verdict": "fail"}},
     "completed", "failed", True, False),
    ("checks_failed", {"mech_check": "fail"}, "incomplete", "failed", False, False),
    ("completed", {"tool_budget_exhausted": True, "mech_check": "pass"},
     "incomplete", "passed", False, False),
    ("token_budget_exhausted", {}, "incomplete", "not_checked", False, False),
    ("max_gate_blocked", {}, "incomplete", "not_checked", False, True),
    ("completed_gate_warnings", {}, "incomplete", "not_checked", False, False),
    ("completed_gate_warnings", {"evidence_gate": {"has_check": True, "last_verdict": "fail"}},
     "incomplete", "failed", False, False),
    ("advisor_abort", {}, "incomplete", "not_checked", False, True),
    ("task_blocked", {}, "incomplete", "not_checked", False, False),
    ("error: connection lost", {}, "failed", "not_checked", False, False),
    ("cancelled", {"mech_check": "pass"}, "cancelled", "passed", False, False),
    ("", {}, "unknown", "not_checked", False, False),
])
def test_completion_and_verification_are_independent(
    reason, fields, completion, verification, required, escalate,
):
    payload = build_trust_payload(SimpleNamespace(reason=reason, **fields))
    assert payload["completionStatus"] == completion
    assert payload["verificationStatus"] == verification
    assert payload["verificationRequired"] is required
    assert payload["canEscalate"] is escalate
    assert payload["verified"] is (completion == "completed" and verification == "passed")


def test_web_lookup_does_not_get_a_failure_note_or_retry():
    payload = build_trust_payload(SimpleNamespace(reason="completed"))
    assert payload["honestNote"] == ""
    assert payload["escalationHint"] == ""
    assert payload["canEscalate"] is False


@pytest.mark.parametrize("limit", ["budget", "time", "tool_rounds"])
def test_answer_reservation_does_not_claim_an_exhausted_budget_or_a_verified_answer(limit):
    payload = build_trust_payload(SimpleNamespace(reason="completed", answer_reserve_reason=limit))
    assert payload["answerReserveReason"] == limit
    assert payload["completionStatus"] == "completed" and not payload["budgetExhausted"]
    assert not payload["verified"] and not payload["canEscalate"]


def test_document_receipts_do_not_verify_the_entire_task():
    receipts = [{"kind": "document_bundle", "checks": {"native_content": "pass"}}]
    payload = build_trust_payload(SimpleNamespace(reason="completed", artifact_receipts=receipts))
    assert payload["artifactReceipts"] == receipts
    assert payload["verificationStatus"] == "not_checked"
    assert payload["verified"] is False


def test_warning_reports_the_blocking_check_without_claiming_a_file_exists():
    check = {"name": "Output requirements", "detail": "Comparison table is missing."}
    payload = build_trust_payload(SimpleNamespace(reason="completed_gate_warnings", completion_check=check))
    assert payload["completionCheck"] == check
    assert "artifact exists" not in payload["honestNote"]
    assert payload["verified"] is False


def test_completed_execution_keeps_its_unchecked_explanation_warning():
    from rune.agent.run_outcome import explanation_warning

    trace = SimpleNamespace(reason="completed", verification={"status": "pass", "required": True},
                            explanation_review={"status": "unavailable"})
    payload = build_trust_payload(trace)
    assert payload["completionStatus"] == "completed" and payload["verificationStatus"] == "passed"
    assert payload["explanationReview"] == trace.explanation_review
    assert payload["honestNote"] == explanation_warning(trace)
    assert "could not be verified" in payload["honestNote"]


@pytest.mark.parametrize("status,expected", [("fail", "failed"), ("unavailable", "unavailable"),
                                            ("inconclusive", "inconclusive"), ("unverified", "not_checked")])
def test_required_report_state_reaches_ui_and_learning(status, expected):
    from rune.agent.memory_bridge import compute_auto_skill_quality_score
    from rune.agent.run_outcome import run_outcome
    from rune.agent.verification_state import verified_outcome

    trace = {"reason": "completed", "success": True, "mech_check": "pass",
             "table_acceptance": {"required": True, "status": status}}
    assert run_outcome(trace).verification == expected
    assert not run_outcome(trace).success
    assert build_trust_payload(SimpleNamespace(**trace))["verificationStatus"] == expected
    assert verified_outcome(trace) is False
    assert compute_auto_skill_quality_score(trace) == 0


def test_interruption_does_not_change_passed_check_evidence():
    from rune.agent.run_outcome import run_outcome
    from rune.agent.verification_state import verification_status, verified_outcome

    trace = {"reason": "request_budget_exhausted", "verification": {"required": True, "status": "pass"}}
    assert verification_status(trace) == run_outcome(trace).verification == "passed"
    assert verified_outcome(trace) is False
    assert not run_outcome(trace).success


@pytest.mark.parametrize("reason", ["verification_inconclusive", "request_budget_exhausted"])
def test_scoped_readback_survives_without_attesting_the_whole_request(reason):
    from rune.agent.run_outcome import merge_requirement_checks, run_outcome

    readback = {"status": "pass", "scope": "observed_page_state"}
    browser = {"required": True, "status": "inconclusive", "method": "page_readback", "readback": readback}
    combined = merge_requirement_checks(browser, {"required": True, "status": "pass", "method": "model_review"})
    trace = SimpleNamespace(reason=reason, requirement_acceptance=combined)
    payload = build_trust_payload(trace)
    assert payload["requirementAcceptance"]["components"][0]["readback"] == readback
    assert not payload["verified"] and not payload["canEscalate"]
    assert not run_outcome(trace).success


@pytest.mark.parametrize("reason", ["completed", "cancelled", "error: timeout"])
def test_old_completion_block_is_not_attached_to_other_outcomes(reason):
    payload = build_trust_payload(SimpleNamespace(
        reason=reason, completion_check={"name": "Output requirements", "detail": "Old failure"},
    ))
    assert payload["completionCheck"] is None
