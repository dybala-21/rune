"""Tests for proactive modules."""

from __future__ import annotations

from rune.proactive.activity_mode import ActivityModeDetector
from rune.proactive.context import AwarenessContext
from rune.proactive.feedback import FeedbackEntry, FeedbackLearner
from rune.proactive.intelligence import TimingScore, compute_timing_score
from rune.proactive.prediction.behavior_predictor import BehaviorPredictor
from rune.proactive.prediction.frustration_detector import FrustrationDetector
from rune.proactive.prediction.need_inferer import NeedInferer
from rune.proactive.prediction.temporal_context import TemporalContextAnalyzer
from rune.proactive.types import EngagementMetrics, Suggestion

# --- Activity mode ---

def test_activity_mode_acceleration():
    """edit+test pattern -> acceleration."""
    detector = ActivityModeDetector()
    tools = ["file.edit", "bash.execute", "file.edit"]
    mode = detector.detect(tools, error_count=0, step=5)
    assert mode == "acceleration"


def test_activity_mode_exploration():
    """Many reads -> exploration."""
    detector = ActivityModeDetector()
    tools = ["file.read"] * 6
    mode = detector.detect(tools, error_count=0, step=1)
    assert mode == "exploration"


def test_activity_mode_debug():
    """Errors -> debug."""
    detector = ActivityModeDetector()
    tools = ["bash.execute", "bash.execute"]
    mode = detector.detect(tools, error_count=3, step=5)
    assert mode == "debug"


# --- Awareness context ---

def test_awareness_context_fields():
    """AwarenessContext has expected fields."""
    ctx = AwarenessContext()
    assert ctx.workspace_root == ""
    assert ctx.recent_files == []
    assert ctx.git_status == ""
    assert ctx.running_processes == []
    assert ctx.time_context == {}
    assert ctx.user_activity_mode == "exploration"


# --- Evaluator ---

def test_feedback_learner():
    """Record feedback and check suppression."""
    learner = FeedbackLearner()

    # Record enough rejected entries of one type to trigger suppression
    for i in range(10):
        entry = FeedbackEntry(
            suggestion_id=f"s{i}",
            suggestion_type="nagging",
            response="dismissed",
        )
        learner.record(entry, suggestion_type="nagging")

    assert learner.should_suppress_type("nagging") is True
    assert learner.should_suppress_type("unknown_type") is False


def test_feedback_learner_record_feedback():
    """record_feedback with full Suggestion + multi-state response."""
    learner = FeedbackLearner()

    suggestion = Suggestion(type="reminder", title="Test", confidence=0.7)
    learner.record_feedback(suggestion, "accepted")
    learner.record_feedback(suggestion, "annoyed")

    stats = learner.get_stats()
    assert stats["total_feedback"] == 2


# --- Timing score ---

def test_timing_score_components():
    """All 4 components present in TimingScore."""
    suggestion = Suggestion(type="warning", title="Alert", confidence=0.8)
    context = AwarenessContext()
    engagement = EngagementMetrics()

    score = compute_timing_score(suggestion, context, engagement)
    assert isinstance(score, TimingScore)
    assert hasattr(score, "urgency")
    assert hasattr(score, "relevance")
    assert hasattr(score, "receptivity")
    assert hasattr(score, "value")
    assert hasattr(score, "total")
    assert 0.0 <= score.total <= 1.0


# --- Message composer ---

def test_frustration_none():
    """No errors -> none."""
    detector = FrustrationDetector()
    signal = detector.analyze(recent_actions=[], error_count=0, repeated_commands=0)
    assert signal.level == "none"


def test_frustration_high():
    """Many errors -> high or moderate."""
    detector = FrustrationDetector()
    actions = [{"type": "execute", "success": False}] * 5
    signal = detector.analyze(recent_actions=actions, error_count=5, repeated_commands=3)
    assert signal.level in ("high", "moderate")
    assert len(signal.indicators) > 0


# --- Behavior predictor ---

def test_behavior_predictor():
    """Record calls, predict next."""
    predictor = BehaviorPredictor()
    sequence = ["file.read", "file.edit", "bash.execute"] * 5
    for tool in sequence:
        predictor.record_tool_call(tool)

    predictions = predictor.predict_next(n=3)
    assert isinstance(predictions, list)
    # With enough history there should be predictions
    assert len(predictions) > 0
    # Each prediction is (tool_name, probability)
    for tool_name, prob in predictions:
        assert isinstance(tool_name, str)
        assert 0.0 <= prob <= 1.0


# --- Temporal context ---

def test_temporal_context():
    """Record slots, check stats."""
    analyzer = TemporalContextAnalyzer()

    # Record enough to pass min_samples threshold (3)
    for _ in range(5):
        analyzer.record("morning", accepted=True)
    for _ in range(5):
        analyzer.record("evening", accepted=False)

    stats = analyzer.get_best_slots()
    assert len(stats) >= 1
    # Morning should have higher rate
    assert stats[0].slot == "morning"
    assert stats[0].rate > 0


# --- Initiative contract ---

def test_need_inferer():
    """Infer needs from patterns."""
    inferer = NeedInferer()
    context = AwarenessContext()

    # Lots of reads without writes -> documentation need
    actions = [{"tool": "file_read"}] * 8
    needs = inferer.infer(context, actions)
    assert len(needs) >= 1
    assert any(n.need_type == "documentation" for n in needs)


def test_testing_need_uses_completed_code_changes_and_fresh_checks():
    inferer = NeedInferer()
    edit = {"tool": "file_edit", "success": True, "code_changed": True, "session_id": "a", "workspace": "/project"}
    check = {"tool": "bash_execute", "success": True, "tests_passed": True, "session_id": "a", "workspace": "/project"}
    assert inferer._check_testing_need([edit] * 4) is not None
    assert inferer._check_testing_need([edit] * 4 + [check]) is None
    assert inferer._check_testing_need([check] + [edit] * 4) is not None
    assert inferer._check_testing_need([{**edit, "success": False}] * 4) is None
    assert inferer._check_testing_need([{**edit, "code_changed": False}] * 4) is None
    assert inferer._check_testing_need([{"tool": "bash_execute", "success": True}] * 4) is None
    assert inferer._check_testing_need([edit] * 4 + [{**check, "tests_passed": False}]) is not None


def test_testing_need_does_not_mix_sessions_or_workspaces():
    inferer = NeedInferer()
    edit = {"tool": "file_edit", "success": True, "code_changed": True, "session_id": "a", "workspace": "/project"}
    other = {**edit, "session_id": "b"}
    assert inferer._check_testing_need([edit] * 3 + [other]) is None
    assert inferer._check_testing_need([edit] * 3 + [{**edit, "workspace": "/other"}]) is None
