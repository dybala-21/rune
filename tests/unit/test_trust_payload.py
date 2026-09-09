"""Completion and check results stay distinct in the UI contract."""

from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize("reason", ["completed", "cancelled", "error: timeout"])
def test_old_completion_block_is_not_attached_to_other_outcomes(reason):
    payload = build_trust_payload(SimpleNamespace(
        reason=reason, completion_check={"name": "Output requirements", "detail": "Old failure"},
    ))
    assert payload["completionCheck"] is None
