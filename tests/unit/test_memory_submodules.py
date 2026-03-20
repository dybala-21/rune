"""Tests for memory submodules."""

from __future__ import annotations

from rune.memory.lessons import LessonEntry, LessonExtractor
from rune.memory.project_memory import PROJECT_MEMORY_FILENAME
from rune.memory.rollout_manager import RolloutManager
from rune.memory.tiered_memory import (
    DailyMemoryEntry,
    DurableMemoryEntry,
    MemoryTier,
)
from rune.memory.tuning import (
    MEMORY_TUNING_DEFAULTS,
    PRESETS,
    get_tuning_config,
)
from rune.memory.user_model import UserModel, WorkProfile


def test_work_profile_defaults():
    """WorkProfile defaults to empty collections."""
    wp = WorkProfile()
    assert wp.preferred_languages == []
    assert wp.preferred_tools == {}
    assert wp.active_hours == {}
    assert wp.workspaces == {}
    assert wp.language_stats == {}


def test_user_model_defaults():
    """UserModel defaults to empty profile."""
    model = UserModel()
    assert model.user_id == ""
    assert model.communication_style == "concise"
    assert model.goals == []
    assert isinstance(model.work_profile, WorkProfile)


def test_memory_tier_enum():
    """MemoryTier has the expected values."""
    assert MemoryTier.SESSION.value == "session"
    assert MemoryTier.DAILY.value == "daily"
    assert MemoryTier.DURABLE.value == "durable"
    assert len(MemoryTier) == 3


def test_daily_memory_entry():
    """DailyMemoryEntry initialises with correct defaults."""
    entry = DailyMemoryEntry(date="2025-12-01")
    assert entry.date == "2025-12-01"
    assert entry.goal_summaries == []
    assert entry.key_decisions == []
    assert entry.patterns_learned == []
    assert entry.total_tasks == 0
    assert entry.successful_tasks == 0


def test_durable_memory_entry():
    """DurableMemoryEntry initialises with correct defaults."""
    entry = DurableMemoryEntry()
    assert entry.category == ""
    assert entry.key == ""
    assert entry.value == ""
    assert entry.confidence == 0.0
    assert entry.last_verified == ""
    assert entry.source == ""


def test_tuning_defaults_exist():
    """MEMORY_TUNING_DEFAULTS contains all expected keys."""
    expected_keys = {
        "semantic_limit",
        "semantic_min_score",
        "uncertain_semantic_limit",
        "uncertain_semantic_min_score",
        "max_episodes",
        "context_max_chars",
    }
    assert set(MEMORY_TUNING_DEFAULTS.keys()) == expected_keys


def test_tuning_presets():
    """All 4 presets exist."""
    assert "minimal" in PRESETS
    assert "balanced" in PRESETS
    assert "aggressive" in PRESETS
    assert "research" in PRESETS
    assert len(PRESETS) == 4


def test_get_tuning_config_balanced():
    """get_tuning_config('balanced') returns merged config with all keys."""
    config = get_tuning_config("balanced")
    assert isinstance(config, dict)
    assert "semantic_limit" in config
    assert "semantic_min_score" in config
    assert "max_episodes" in config
    assert config["semantic_limit"] == 5
    assert config["semantic_min_score"] == 0.3


def test_lesson_entry_fields():
    """LessonEntry has expected fields and defaults."""
    entry = LessonEntry()
    assert entry.domain == ""
    assert entry.lesson == ""
    assert entry.confidence == 0.5
    assert entry.source_goal == ""
    assert entry.timestamp  # non-empty


def test_lesson_extractor_failure():
    """Extracts a failure lesson from a failed result."""
    extractor = LessonExtractor()
    lessons = extractor.extract_from_result(
        goal="deploy service",
        result={"success": False, "error": "Permission denied"},
    )
    assert len(lessons) >= 1
    failure_lesson = lessons[0]
    assert failure_lesson.domain == "failure"
    assert "permission" in failure_lesson.lesson.lower() or "Permission" in failure_lesson.lesson
    assert failure_lesson.confidence > 0


def test_lesson_extractor_success():
    """Extracts a success lesson from a successful result."""
    extractor = LessonExtractor()
    lessons = extractor.extract_from_result(
        goal="build project",
        result={"success": True, "iterations": 1},
    )
    assert len(lessons) >= 1
    success_lesson = lessons[0]
    assert success_lesson.domain == "efficiency"
    assert "first attempt" in success_lesson.lesson.lower() or "succeeded" in success_lesson.lesson.lower()


def test_rollout_manager_mode_default(tmp_path):
    """RolloutManager defaults to 'balanced' mode."""
    manager = RolloutManager(config_path=tmp_path / "rollout.json")
    assert manager.get_mode() == "balanced"


def test_project_memory_filename():
    """PROJECT_MEMORY_FILENAME equals 'MEMORY.md'."""
    assert PROJECT_MEMORY_FILENAME == "MEMORY.md"
