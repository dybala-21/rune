"""Historical test claims must agree with the actual runner results."""

import json
import subprocess
import sys
from unittest.mock import AsyncMock

import pytest

from rune.agent.test_claims import TestClaimGate as ClaimGate
from rune.agent.test_claims import (
    check_claims,
    check_explanations,
    claim_runs,
    code_observations,
    comparison_evidence,
    expand_claims,
)
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


def test_compact_claims_resolve_the_exact_run_and_test_before_comparison(state):
    evidence = comparison_evidence(state)
    runs = claim_runs(evidence)
    case = next(case for case in runs[0]["cases"] if case["status"] == "pass")
    def wire(claim):
        return ["/".join(str(claim[key]) for key in ("run", "test", "metric", "value", "line"))]

    claim = {"run": 0, "test": case["id"], "metric": "status", "value": "fail", "line": 1}
    assert check_claims("test_passing failed", expand_claims(wire(claim), evidence), evidence)
    claim["value"] = "pass"
    assert not check_claims("test_passing passed", expand_claims(wire(claim), evidence), evidence)
    claim.update(run=1, test=-1, metric="tests_run", value="2")
    assert not check_claims("two tests ran after the edit", expand_claims(wire(claim), evidence), evidence)
    claim.update(test=-2, metric="status", value="unknown")
    assert not check_claims("unobserved", expand_claims(wire(claim), evidence), evidence)
    claim["value"] = "pass"
    assert check_claims("unrecorded test passed", expand_claims(wire(claim), evidence), evidence)
    for changes in ({"run": 2}, {"run": True}, {"test": 999}):
        with pytest.raises(ValueError):
            expand_claims(wire({**claim, **changes}), evidence)
    for invalid in ([claim], ["0/0/status/pass/1 extra"], ["0/0/unknown/pass/1"]):
        with pytest.raises(ValueError):
            expand_claims(invalid, evidence)


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
    client.completion.return_value = {"choices": [{"message": {"content": json.dumps({"claims": [], "explanation_issues": []})}}]}
    monkeypatch.setattr("rune.llm.client.get_llm_client", lambda: client)
    gate = ClaimGate()
    assert await gate.review(VerificationState(), "Summary") is None
    assert client.completion.await_count == 0
    assert await gate.review(state, "Updated the implementation.") is None
    assert await gate.review(state, "Updated the implementation.") is None
    assert client.completion.await_count == 1


async def test_review_deadline_is_shared_and_an_outage_does_not_trigger_a_rewrite(monkeypatch, state):
    client = AsyncMock()
    client.completion.side_effect = TimeoutError
    monkeypatch.setattr("rune.llm.client.get_llm_client", lambda: client)
    gate = ClaimGate(_review_seconds=51.0)
    note = await gate.review(state, "Changed the implementation.")
    assert "could not be checked" in note and gate.attempts == 2
    assert client.completion.call_args.kwargs["timeout"] == 9.0
    assert client.completion.call_args.kwargs["max_retries"] == 0
    await gate.review(state, "Rephrased answer.")
    assert client.completion.await_count == 1
    client.completion.side_effect = None
    client.completion.return_value = {"choices": [{"message": {"content": json.dumps({"claims": [], "explanation_issues": []})}}]}
    assert await ClaimGate(attempts=1, _review_seconds=17.4).review(state, "Corrected answer.") is None
    assert client.completion.call_args.kwargs["timeout"] == 30.0


async def test_explanation_contradiction_uses_existing_review_and_observed_code(monkeypatch, state):
    client = AsyncMock()
    messages = [{"role": "assistant", "tool_calls": [{"id": "read", "function": {
        "name": "file_read", "arguments": '{"path":"stats.py"}'}}]},
        {"role": "tool", "tool_call_id": "read", "content": "return sum(values) / (len(values) + 1)"}]
    issue = {"source_line": 1, "quote": "always smaller", "evidence_id": 1,
             "evidence_quote": "return sum(values) / (len(values) + 1)",
             "reason": "A zero sum is unchanged, and a negative sum gives a larger value.",
             "needs_correction": True}
    client.completion.return_value = {"choices": [{"message": {"content": json.dumps({
        "claims": [], "explanation_issues": [issue]})}}]}
    monkeypatch.setattr("rune.llm.client.get_llm_client", lambda: client)
    gate = ClaimGate()
    note = await gate.review(state, "The result was always smaller.", messages)
    assert "zero sum" in note and "Do not edit files or rerun" in note
    assert await gate.review(state, "The result was always smaller.", messages) == note
    assert client.completion.await_count == 1
    payload = json.loads(client.completion.call_args.kwargs["messages"][-1]["content"])
    assert "sum(values)" in payload["code_observations"][0]["text"]
    issue["evidence_quote"] = "Invented code"
    with pytest.raises(ValueError, match="did not identify"):
        check_explanations("The result was always smaller.", [issue], code_observations(messages))
    client.completion.return_value = {"choices": [{"message": {"content": json.dumps({
        "claims": [], "explanation_issues": [issue]})}}]}
    invalid = ClaimGate()
    note = await invalid.review(state, "The result was always smaller.", messages)
    assert "could not be checked" in note and invalid.attempts == 2
    await invalid.review(state, "Rephrased answer.", messages)
    assert client.completion.await_count == 2


def test_review_does_not_block_a_claim_it_concludes_is_supported():
    issue = {"source_line": 1, "quote": "returned 0.0", "evidence_id": 1,
             "evidence_quote": "return sum(values) / (len(values) + 1)",
             "reason": "For an empty input the result is 0 / 1 = 0.0, which matches the claim.",
             "needs_correction": False}
    observations = [{"id": 1, "text": issue["evidence_quote"]}]
    assert not check_explanations("The empty input returned 0.0.", [issue], observations)
    issue["needs_correction"] = "false"
    with pytest.raises(ValueError, match="verdict"):
        check_explanations("The empty input returned 0.0.", [issue], observations)


async def test_gemini_review_reserves_tokens_for_its_result(monkeypatch, state):
    from rune.llm.model_selection import ActiveModelSelection
    from rune.types import Provider

    client = AsyncMock()
    client.completion.return_value = {"choices": [{"message": {"content": json.dumps({"claims": [], "explanation_issues": []})}}]}
    monkeypatch.setattr("rune.llm.client.get_llm_client", lambda: client)
    monkeypatch.setattr("rune.llm.model_selection.get_effective_model_selection",
                        lambda: ActiveModelSelection(provider=Provider.GEMINI, model="gemini-2.5-flash"))
    assert await ClaimGate().review(state, "Changed the implementation.") is None
    from rune.llm.reasoning import apply_reasoning_control
    params = {"model": "vertex_ai/gemini-2.5-flash", "reasoning_effort": client.completion.call_args.kwargs["reasoning_effort"]}
    apply_reasoning_control(params)
    assert params["thinkingConfig"] == {"thinkingBudget": 2048}


async def test_correction_keeps_tool_evidence_after_transcript_compaction(monkeypatch, state):
    from rune.types import CapabilityResult

    client = AsyncMock()
    client.completion.return_value = {"choices": [{"message": {"content": json.dumps({"claims": [], "explanation_issues": []})}}]}
    monkeypatch.setattr("rune.llm.client.get_llm_client", lambda: client)
    gate = ClaimGate()
    gate.observe("file_read", {"path": "stats.py"}, CapabilityResult(success=True, output="return sum(values) / (len(values) + 1)"))
    await gate.review(state, "Initial explanation.", [])
    first = json.loads(client.completion.call_args.kwargs["messages"][-1]["content"])["code_observations"]
    gate.observe("file_read", {"path": "stats.py"}, CapabilityResult(success=True, output="Read stats.py", metadata={"cached": True}))
    gate.observe("file_read", {"path": "test_stats.py"}, CapabilityResult(success=True, output="self.assertEqual(average([-2, 2]), 0)"))
    await gate.review(state, "Corrected explanation.", [{"role": "tool", "content": "Earlier observations summarized"}])
    second = json.loads(client.completion.call_args.kwargs["messages"][-1]["content"])["code_observations"]
    assert second[0] == first[0] and len(second) == 2
    assert "average([-2, 2])" in second[1]["text"]
    for i in range(30):
        gate.observe("file_edit", {"path": "stats.py", "replace": "x" * 10000}, CapabilityResult(success=False, error=f"failed revision {i}"))
    assert len(gate._observations) == 6 and sum(len(r["text"]) for r in gate._observations) <= 10800
    assert gate._observations[0] == first[0] and "failed revision 29" in gate._observations[-1]["text"]


def test_code_observations_preserve_baseline_and_latest_revision_with_bounded_size():
    messages = []
    for i in range(30):
        messages.extend([{"role": "assistant", "tool_calls": [{"id": str(i), "function": {
            "name": "file_read", "arguments": '{"path":"code.py"}'}}]},
            {"role": "tool", "tool_call_id": str(i), "content": f"revision {i}\n" + "x" * 10000}])
    observations = code_observations(messages)
    assert len(observations) == 6 and sum(len(r["text"]) for r in observations) <= 10800
    assert "revision 0" in observations[0]["text"] and "revision 29" in observations[-1]["text"]


def test_unique_short_test_name_matches_complete_record_but_not_wrong_status(state):
    evidence = comparison_evidence(state)
    claim = {"check_id": evidence[0]["check_id"], "phase": "before", "metric": "status",
             "test_id": "test_passing", "value": "pass", "source_line": 1}
    assert not check_claims("test_passing passed before", [claim], evidence)
    claim["value"] = "fail"
    assert check_claims("test_passing failed before", [claim], evidence)


def test_short_test_name_cannot_choose_between_owners_or_incomplete_logs(state):
    from copy import deepcopy

    evidence = comparison_evidence(state)
    claim = {"check_id": evidence[0]["check_id"], "phase": "before", "metric": "status",
             "test_id": "test_passing", "value": "pass", "source_line": 1}
    incomplete = deepcopy(evidence)
    incomplete[0]["before"]["complete"] = False
    assert check_claims("test_passing passed before", [claim], incomplete)
    report = evidence[0]["before"]
    duplicate = {"identity": "other.Cases.test_passing", "status": "fail", "subtest_failures": ()}
    report["cases"] = (*report["cases"], duplicate)
    assert check_claims("test_passing passed before", [claim], evidence)
    claim["test_id"] = "test_example.Cases.test_passing"
    assert not check_claims("test_example.Cases.test_passing passed before", [claim], evidence)


@pytest.mark.parametrize('owner', ['test_example.Cases', 'test_example.Cases.test_values'])
def test_unittest_versions_produce_the_same_identity(owner):
    from rune.agent.test_evidence import parse_test_report

    report = parse_test_report(f'test_values ({owner}) ... FAIL\n'
        f'FAIL: test_values ({owner})\nRan 1 test in 0.001s\nFAILED (failures=1)\n')
    assert report.complete and len(report.cases) == 1
    assert report.cases[0].identity == 'test_example.Cases.test_values'
