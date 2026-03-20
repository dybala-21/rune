"""Tests for the autonomous executor module."""

from __future__ import annotations

from rune.agent.autonomous import (
    AutonomousExecutor,
    AutonomyLevel,
)


class TestAutonomousExecutor:
    def test_default_decision_suggest(self):
        executor = AutonomousExecutor()
        decision = executor.decide("some unknown command xyz", risk_score=0.8)
        assert decision.level == AutonomyLevel.SUGGEST
        assert decision.domain == "unknown"

    def test_low_risk_inform(self):
        """Notify domain has JUST_DO base level; low risk should keep it or above INFORM_DO."""
        executor = AutonomousExecutor()
        decision = executor.decide("notify user about completion", risk_score=0.1)
        assert decision.domain == "notify"
        assert decision.level >= AutonomyLevel.INFORM_DO

    def test_kill_switch(self):
        executor = AutonomousExecutor()
        executor.kill_switch = True
        decision = executor.decide("git push origin main", risk_score=0.1)
        assert decision.level == AutonomyLevel.SUGGEST
        assert "kill_switch" in decision.reason

    def test_classify_domain(self):
        executor = AutonomousExecutor()
        assert executor._classify_domain("git push origin main") == "git"
        assert executor._classify_domain("npm install express") == "build"
        assert executor._classify_domain("ls -la") == "unknown"

    def test_record_feedback_promotion(self):
        executor = AutonomousExecutor()
        pattern_key = "git:git status"
        # Record enough approvals to potentially promote
        for _ in range(10):
            executor.record_feedback(pattern_key, "approved")

        stats = executor.pattern_stats
        assert pattern_key in stats
        assert stats[pattern_key].approved == 10
        assert stats[pattern_key].total_executions == 10

    def test_demotion_window(self):
        executor = AutonomousExecutor()
        pattern_key = "file:cp src"

        # Fill history with some approved, then many reverts
        for _ in range(5):
            executor.record_feedback(pattern_key, "approved")
        for _ in range(5):
            executor.record_feedback(pattern_key, "full_revert")

        # Now decide — demotion should trigger because >2 reverts in window
        decision = executor.decide("cp src dest", risk_score=0.3)
        assert decision.level == AutonomyLevel.SUGGEST

    def test_serialize_restore(self):
        executor = AutonomousExecutor()
        executor.record_feedback("git:git commit", "approved")
        executor.record_feedback("git:git commit", "approved")
        executor.shadow_mode = True

        data = executor.serialize()
        restored = AutonomousExecutor.restore(data)

        assert restored.shadow_mode is True
        stats = restored.pattern_stats
        assert "git:git commit" in stats
        assert stats["git:git commit"].approved == 2
        assert len(restored.feedback_history) == 2
