"""Unit tests for the Behavioral Compliance Tracker (BCT)."""

from __future__ import annotations

from rune.agent.advisor.compliance import (
    build_pending,
    check_compliance,
)
from rune.agent.advisor.protocol import AdvisorDecision


def _decision(action: str = "retry_tool", target: str | None = "web_fetch") -> AdvisorDecision:
    return AdvisorDecision(
        action=action,
        plan_steps=["step 1"],
        raw_text="NEXT: retry_tool:web_fetch\n1. step 1",
        target_tool=target,
        trigger="stuck",
    )


class TestCheckCompliance:
    def test_returns_none_during_grace(self):
        pa = build_pending(_decision(), step=5, evidence_total=10, hard_failure_count=0)
        v = check_compliance(pa, 6, 10, 0, ["file_read"])
        assert v is None  # step 6, grace=2, need step >= 7

    def test_tool_matched(self):
        pa = build_pending(_decision(), step=5, evidence_total=10, hard_failure_count=0)
        check_compliance(pa, 6, 10, 0, ["file_read"])  # grace
        v = check_compliance(pa, 7, 11, 0, ["web_fetch"])
        assert v is not None
        assert v.followed is True
        assert v.reason == "tool_matched"

    def test_progress_without_tool_match(self):
        pa = build_pending(_decision(), step=5, evidence_total=10, hard_failure_count=0)
        check_compliance(pa, 6, 11, 0, ["bash_execute"])
        v = check_compliance(pa, 7, 13, 0, ["bash_execute"])
        assert v is not None
        assert v.followed is True
        assert v.reason == "progress_made"

    def test_ignored_no_progress(self):
        pa = build_pending(_decision(), step=5, evidence_total=10, hard_failure_count=0)
        check_compliance(pa, 6, 10, 0, [])
        v = check_compliance(pa, 7, 10, 0, [])
        assert v is not None
        assert v.followed is False
        assert v.reason == "ignored"

    def test_worse_with_new_failures(self):
        pa = build_pending(_decision(), step=5, evidence_total=10, hard_failure_count=1)
        check_compliance(pa, 6, 10, 2, [])
        v = check_compliance(pa, 7, 10, 3, [])
        assert v is not None
        assert v.followed is False
        assert v.reason == "worse"

    def test_tool_observed_across_grace_steps(self):
        """Tool used in step 6 (grace) should still count at step 7."""
        pa = build_pending(_decision(), step=5, evidence_total=10, hard_failure_count=0)
        check_compliance(pa, 6, 10, 0, ["web_fetch"])  # grace, but tool used
        v = check_compliance(pa, 7, 10, 0, [])
        assert v is not None
        assert v.followed is True
        assert v.reason == "tool_matched"

    def test_no_target_tool_falls_to_progress(self):
        pa = build_pending(
            _decision(target=None), step=5, evidence_total=10, hard_failure_count=0,
        )
        check_compliance(pa, 6, 12, 0, ["bash_execute"])
        v = check_compliance(pa, 7, 14, 0, [])
        assert v is not None
        assert v.followed is True
        assert v.reason == "progress_made"


class TestBuildPending:
    def test_captures_baseline(self):
        d = _decision()
        pa = build_pending(d, step=10, evidence_total=25, hard_failure_count=2)
        assert pa.injected_at_step == 10
        assert pa.baseline_evidence == 25
        assert pa.baseline_hard_failures == 2
        assert pa.expected_tool == "web_fetch"
        assert pa.expected_action == "retry_tool"
