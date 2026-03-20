"""Tests for evaluation module."""

from __future__ import annotations

from typing import Any

from rune.evaluation.cost.tracker import CostTracker
from rune.evaluation.grading.soft_failure import classify_failure
from rune.evaluation.probes.runner import ProbeRunner
from rune.evaluation.probes.types import Probe
from rune.evaluation.types import EvaluationRun, ProbeResult


def test_probe_result_fields():
    """ProbeResult has expected fields and defaults."""
    result = ProbeResult()
    assert result.probe_name == ""
    assert result.success is False
    assert result.output == ""
    assert result.expected == ""
    assert result.duration_ms == 0.0
    assert result.score == 0.0


def test_evaluation_run_fields():
    """EvaluationRun has expected fields."""
    run = EvaluationRun()
    assert run.id  # auto-generated
    assert run.probe_results == []
    assert run.total_score == 0.0
    assert run.timestamp is not None


def test_probe_runner_register():
    """Register and list probes."""

    class DummyProbe(Probe):
        @property
        def name(self) -> str:
            return "dummy"

        async def run(self, context: dict[str, Any]) -> ProbeResult:
            return ProbeResult(probe_name="dummy", success=True, score=1.0)

    runner = ProbeRunner()
    probe = DummyProbe()
    runner.register(probe)
    assert "dummy" in runner._probes


def test_classify_failure_soft():
    """'timeout' -> soft."""
    result = classify_failure("Connection timed out after 30s")
    assert result == "soft"

    result2 = classify_failure("timeout waiting for response")
    assert result2 == "soft"


def test_classify_failure_hard():
    """'auth' -> hard."""
    result = classify_failure("authentication failed: invalid token")
    assert result == "hard"

    result2 = classify_failure("403 Forbidden")
    assert result2 == "hard"


def test_cost_tracker():
    """Record and get total cost."""
    tracker = CostTracker()
    tracker.record("gpt-4o", input_tokens=10000, output_tokens=5000)
    total = tracker.get_total_cost()
    assert isinstance(total, float)
    assert total > 0

    # Second recording increases total
    tracker.record("gpt-4o", input_tokens=10000, output_tokens=5000)
    total2 = tracker.get_total_cost()
    assert total2 > total
