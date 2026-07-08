"""Tests for proactive modules: engagement tracker, pattern learner, reflexion."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from rune.proactive.engagement_tracker import EngagementTracker
from rune.proactive.patterns import PatternLearner
from rune.proactive.reflexion import ReflexionLearner


class TestEngagementTracker:
    def test_engagement_tracker(self):
        tracker = EngagementTracker()
        tracker.record_shown("s1")
        tracker.record_shown("s2")
        tracker.record_accepted("s1")
        tracker.record_dismissed("s2")

        metrics = tracker.get_metrics()
        assert metrics.suggestions_shown == 2
        assert metrics.suggestions_accepted == 1
        assert metrics.suggestions_dismissed == 1
        assert metrics.acceptance_rate == 0.5

    def test_engagement_suppress(self):
        tracker = EngagementTracker()
        # Show 12 suggestions, dismiss all of them
        for i in range(12):
            sid = f"s{i}"
            tracker.record_shown(sid)
            tracker.record_dismissed(sid)

        # Acceptance rate is 0%, well below 10% threshold
        assert tracker.should_suppress() is True

    def test_no_suppress_with_few_records(self):
        tracker = EngagementTracker()
        tracker.record_shown("s1")
        tracker.record_dismissed("s1")
        # Not enough data to suppress (min 10 resolved)
        assert tracker.should_suppress() is False


class TestPatternLearner:
    def test_pattern_learner_time_slot(self):
        from rune.proactive.patterns import _TIME_SLOTS

        # Verify that time slots cover all 24 hours
        all_hours: set[int] = set()
        for hours in _TIME_SLOTS.values():
            all_hours.update(hours)
        assert all_hours == set(range(24))

    def test_record_and_predict(self):
        learner = PatternLearner()

        # Patch get_current_time_slot and get_current_day_type for determinism
        with patch("rune.proactive.patterns.get_current_time_slot", return_value="morning"), \
             patch("rune.proactive.patterns.get_current_day_type", return_value="weekday"):
            learner.record_activity("coding", {"command": "vim"})
            learner.record_activity("coding", {"command": "vim"})
            learner.record_activity("testing", {"command": "pytest"})

        with patch("rune.proactive.patterns.get_current_time_slot", return_value="morning"), \
             patch("rune.proactive.patterns.get_current_day_type", return_value="weekday"):
            prediction = learner.predict_current_activity()
            assert prediction["likely_activity"] == "coding"
            assert prediction["confidence"] > 0


class TestReflexionLearner:
    def test_reflexion_learner(self):
        learner = ReflexionLearner()

        # Record a failed outcome — should produce a lesson
        learner.record_task_outcome({
            "domain": "git",
            "success": False,
            "goal": "push to protected branch",
            "error": "Permission denied",
            "steps_taken": 3,
        })

        lessons = learner.get_domain_lessons("git")
        assert len(lessons) == 1
        assert "Permission denied" in lessons[0]

    def test_efficient_success_lesson(self):
        learner = ReflexionLearner()
        learner.record_task_outcome({
            "domain": "file",
            "success": True,
            "goal": "create config file",
            "steps_taken": 1,
        })

        lessons = learner.get_domain_lessons("file")
        assert len(lessons) == 1
        assert "Efficiently" in lessons[0]

    async def test_analyze_with_llm_uses_llm_path(self):
        """The LLM path must actually run and its parsed result be returned.

        Regression guard: a wrong ModelTier import once made this silently fall
        through to the rule-based path. The mocked reply carries values the
        rule-based analyzer never produces (confidence 0.9, reason user_busy),
        and the rejection text triggers no rule-based keyword, so a fall-through
        would return unknown/0.4 and fail this assertion.
        """
        message = SimpleNamespace(
            content='{"reason": "user_busy", "lesson": "Wait for a lull.", '
            '"confidence": 0.9}'
        )
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )

        async def fake_completion(*args, **kwargs):
            return fake_response

        fake_client = SimpleNamespace(completion=fake_completion)

        learner = ReflexionLearner()
        with patch("rune.llm.client.get_llm_client", return_value=fake_client):
            result = await learner.analyze_with_llm(
                "the phrasing here matches no rule keyword",
                {"domain": "calendar", "event_type": "suggestion", "score": -1.0},
            )

        assert result["reason"] == "user_busy"
        assert result["lesson"] == "Wait for a lull."
        assert result["confidence"] == 0.9

    async def test_analyze_with_llm_falls_back_on_error(self):
        """A broken client falls back to rule-based analysis, not an exception."""
        learner = ReflexionLearner()
        with patch(
            "rune.llm.client.get_llm_client", side_effect=RuntimeError("no client")
        ):
            result = await learner.analyze_with_llm(
                "I am busy right now, later",
                {"domain": "general", "event_type": "suggestion"},
            )

        # Rule-based branch for "busy"/"later" text.
        assert result["reason"] == "bad_timing"
        assert result["confidence"] == 0.4
