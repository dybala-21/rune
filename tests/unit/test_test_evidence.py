"""Historical test claims must agree with the actual runner results."""

import json
import subprocess
import sys
from unittest.mock import AsyncMock

import pytest

from rune.agent.test_claims import TestClaimGate as ClaimGate
from rune.agent.test_claims import check_claims, comparison_evidence
from rune.agent.verification_state import VerificationState


@pytest.fixture
def state(tmp_path):
    source = tmp_path / "test_example.py"
    source.write_text("import unittest\nclass Cases(unittest.TestCase):\n"
                      "    def test_passing(self): self.assertTrue(True)\n"
                      "    def test_values(self):\n"
                      "        for value in [1, 2, 3]:\n"
                      "            with self.subTest(value=value): self.assertEqual(value, 0)\n")
    args = [sys.executable, "-B", "-m", "unittest", "discover", "-v"]
    before = subprocess.run(args, cwd=tmp_path, text=True, capture_output=True)
    result = VerificationState()
    command = "python3 -B -m unittest discover -v"
    result.observe_command(command, False, before.stdout + before.stderr, str(tmp_path))
    result.changed()
    source.write_text(source.read_text().replace("self.assertEqual(value, 0)", "self.assertGreater(value, 0)"))
    after = subprocess.run(args, cwd=tmp_path, text=True, capture_output=True)
    assert after.returncode == 0
    result.observe_command(command, True, after.stdout + after.stderr, str(tmp_path))
    return result


def test_before_after_and_subtests_survive_latest_result_replacement(state):
    assert len(state.checks) == 1 and len(state.history) == 2
    first = state.history[0].report
    assert first.tests_run == 2 and first.failure_events == 3 and first.complete
    assert len([c for c in first.cases if c.status == "fail"]) == 1
    assert state.history[-1].report.failure_events == 0 and state.passed
    context = state.model_context()
    assert "test_passing" in context and '"failure_events": 3' in context


def test_false_historical_failure_and_subtest_count_are_rejected(state):
    evidence = comparison_evidence(state)
    identity = next(c["identity"] for c in evidence[0]["before"]["cases"] if c["status"] == "pass")
    claim = {"check_id": evidence[0]["check_id"], "phase": "before", "metric": "status",
             "test_id": identity, "value": "fail", "source_line": 1}
    assert check_claims("passing failed before", [claim], evidence)
    claim.update(metric="failed_tests", test_id="*", value="3")
    assert check_claims("passing failed before", [claim], evidence)
    claim.update(metric="failure_events")
    assert not check_claims("passing failed before", [claim], evidence)


def test_partial_output_does_not_invent_missing_passes(state):
    from rune.agent.test_evidence import parse_test_report
    report = parse_test_report("FAIL: test_values (test_example.Cases) (value=1)\nRan 2 tests in 0.001s\nFAILED (failures=1)\n")
    assert not report.complete and len(report.cases) == 1


def test_suite_failure_does_not_mean_every_case_failed(state):
    evidence = comparison_evidence(state)
    claim = {"check_id": evidence[0]["check_id"], "phase": "before", "metric": "status",
             "test_id": "*", "value": "fail", "source_line": 1}
    assert not check_claims("suite failed", [claim], evidence)
    claim["metric"] = "all_tests_status"
    assert check_claims("suite failed", [claim], evidence)


def test_individual_failure_counts_distinguish_methods_and_subtests(state):
    evidence = comparison_evidence(state)
    identity = next(c["identity"] for c in evidence[0]["before"]["cases"] if c["status"] == "fail")
    claim = {"check_id": evidence[0]["check_id"], "phase": "before", "metric": "failure_events",
             "test_id": identity, "value": "3", "source_line": 1}
    assert not check_claims("three subtests failed", [claim], evidence)
    claim.update(metric="failed_tests", value="1")
    assert not check_claims("three subtests failed", [claim], evidence)


def test_check_history_retention_is_bounded_and_disclosed(state):
    for _ in range(40):
        state.observe_command("python3 -m unittest -v", True, "Ran 1 test in 0s\nOK", "/fixture")
    snapshot = state.snapshot()
    assert len(snapshot["history"]) == 32 and snapshot["dropped_checks"] == 10
    assert state.evidence_context()["history_incomplete"]


def test_incremental_evidence_preserves_baseline_without_resending_it(state):
    baseline = state.history[0]
    delta = state.evidence_context(since_sequence=baseline.sequence)
    assert len(delta["checks"]) == 1
    assert delta["checks"][0]["status"] == "pass"
    assert state.model_context(since_sequence=state.sequence) == ""
    sequence = state.sequence
    state.changed()
    assert '"pending": true' in state.model_context(since_sequence=sequence)
    assert len(state.history) == 2 and state.history[0] == baseline


async def test_claim_review_is_cached_and_does_not_run_without_historical_failure(monkeypatch, state):
    client = AsyncMock()
    client.completion.return_value = {"choices": [{"message": {"content": json.dumps({"claims": []})}}]}
    monkeypatch.setattr("rune.llm.client.get_llm_client", lambda: client)
    gate = ClaimGate()
    assert await gate.review(VerificationState(), "Summary") is None
    assert client.completion.await_count == 0
    assert await gate.review(state, "Updated the implementation.") is None
    assert await gate.review(state, "Updated the implementation.") is None
    assert client.completion.await_count == 1
