"""Tests for MemoryStore — proactive feedback and sequence patterns."""

from __future__ import annotations

import pytest

from rune.memory.store import MemoryStore


@pytest.fixture
def store(tmp_dir):
    """Create a temporary MemoryStore for each test."""
    s = MemoryStore(db_path=tmp_dir / "test_memory.db")
    yield s
    s.close()


class TestSaveProactiveFeedback:
    def test_save_and_retrieve(self, store):
        fb_id = store.save_proactive_feedback(
            suggestion_type="reminder", response="accepted",
        )
        results = store.get_proactive_feedback()
        assert len(results) == 1
        assert results[0]["id"] == fb_id
        assert results[0]["response"] == "accepted"
        assert results[0]["timestamp"]  # non-empty

    def test_save_multiple(self, store):
        store.save_proactive_feedback(response="accepted")
        store.save_proactive_feedback(response="dismissed")
        results = store.get_proactive_feedback()
        assert len(results) == 2

    def test_save_different_types(self, store):
        store.save_proactive_feedback(
            suggestion_type="reminder", response="accepted",
        )
        store.save_proactive_feedback(
            suggestion_type="optimization", response="dismissed",
        )
        store.save_proactive_feedback(
            suggestion_type="warning", response="accepted",
        )

        all_feedback = store.get_proactive_feedback()
        assert len(all_feedback) == 3

    def test_text_id_uniqueness(self, store):
        store.save_proactive_feedback(response="accepted")
        store.save_proactive_feedback(response="dismissed")
        all_feedback = store.get_proactive_feedback()
        ids = {f["id"] for f in all_feedback}
        assert len(ids) == 2  # unique TEXT IDs

    def test_explicit_feedback_id(self, store):
        fb_id = store.save_proactive_feedback(
            feedback_id="custom_id_123", response="accepted",
        )
        assert fb_id == "custom_id_123"
        results = store.get_proactive_feedback()
        assert results[0]["id"] == "custom_id_123"

    def test_confidence_stored(self, store):
        store.save_proactive_feedback(
            response="accepted", confidence=0.85,
        )
        results = store.get_proactive_feedback()
        assert results[0]["confidence"] == 0.85

    def test_context_summary_stored(self, store):
        store.save_proactive_feedback(
            response="accepted", context_summary="user editing main.py",
        )
        results = store.get_proactive_feedback()
        assert results[0]["context_summary"] == "user editing main.py"


class TestGetProactiveFeedback:
    def test_get_all(self, store):
        store.save_proactive_feedback(response="accepted")
        store.save_proactive_feedback(response="dismissed")

        results = store.get_proactive_feedback()
        assert len(results) == 2

    def test_respects_limit(self, store):
        for _i in range(10):
            store.save_proactive_feedback(response="accepted")

        results = store.get_proactive_feedback(limit=3)
        assert len(results) == 3

    def test_ordered_by_timestamp_desc(self, store):
        id1 = store.save_proactive_feedback(
            suggestion_type="first", response="accepted",
        )
        id2 = store.save_proactive_feedback(
            suggestion_type="second", response="dismissed",
        )

        results = store.get_proactive_feedback()
        # Most recent first
        assert results[0]["id"] == id2
        assert results[1]["id"] == id1

    def test_empty_results(self, store):
        results = store.get_proactive_feedback()
        assert results == []

    def test_result_has_expected_keys(self, store):
        store.save_proactive_feedback(
            suggestion_type="reminder",
            context_summary="test context",
            response="accepted",
            confidence=0.7,
        )
        results = store.get_proactive_feedback()
        r = results[0]
        assert "id" in r
        assert "timestamp" in r
        assert "suggestion_type" in r
        assert "context_summary" in r
        assert "response" in r
        assert "confidence" in r
        # These old columns should NOT be present
        assert "suggestion_id" not in r
        assert "feedback" not in r
        assert "created_at" not in r


class TestSaveSequencePattern:
    def test_save_new_pattern(self, store):
        store.save_sequence_pattern("read", "edit")
        patterns = store.get_sequence_patterns()
        assert len(patterns) == 1
        assert patterns[0]["from_activity"] == "read"
        assert patterns[0]["to_activity"] == "edit"
        assert patterns[0]["count"] == 1

    def test_increment_existing_pattern(self, store):
        store.save_sequence_pattern("read", "edit")
        store.save_sequence_pattern("read", "edit")
        store.save_sequence_pattern("read", "edit")

        patterns = store.get_sequence_patterns()
        assert len(patterns) == 1
        assert patterns[0]["count"] == 3

    def test_increment_with_custom_count(self, store):
        store.save_sequence_pattern("read", "edit", count=5)
        store.save_sequence_pattern("read", "edit", count=3)

        patterns = store.get_sequence_patterns()
        assert len(patterns) == 1
        assert patterns[0]["count"] == 8

    def test_avg_transition_minutes(self, store):
        store.save_sequence_pattern("deploy", "verify", avg_transition_minutes=10.0)
        patterns = store.get_sequence_patterns()
        assert len(patterns) == 1
        assert patterns[0]["avg_transition_minutes"] == 10.0

    def test_avg_transition_minutes_weighted_on_increment(self, store):
        store.save_sequence_pattern("deploy", "verify", count=2, avg_transition_minutes=10.0)
        store.save_sequence_pattern("deploy", "verify", count=2, avg_transition_minutes=20.0)

        patterns = store.get_sequence_patterns()
        # (10*2 + 20*2) / 4 = 15.0
        assert patterns[0]["avg_transition_minutes"] == 15.0

    def test_multiple_distinct_patterns(self, store):
        store.save_sequence_pattern("a", "b")
        store.save_sequence_pattern("c", "d")
        store.save_sequence_pattern("e", "f")

        patterns = store.get_sequence_patterns()
        assert len(patterns) == 3

    def test_id_is_text_uuid(self, store):
        store.save_sequence_pattern("x", "y")
        patterns = store.get_sequence_patterns()
        pid = patterns[0]["id"]
        assert isinstance(pid, str)
        assert len(pid) == 36  # UUID format


class TestGetSequencePatterns:
    def test_min_count_filter(self, store):
        store.save_sequence_pattern("rare", "thing")
        store.save_sequence_pattern("common", "thing")
        store.save_sequence_pattern("common", "thing")
        store.save_sequence_pattern("common", "thing")

        patterns = store.get_sequence_patterns(min_count=3)
        assert len(patterns) == 1
        assert patterns[0]["from_activity"] == "common"

    def test_ordered_by_count_desc(self, store):
        store.save_sequence_pattern("low", "x", count=1)
        store.save_sequence_pattern("high", "x", count=10)
        store.save_sequence_pattern("mid", "x", count=5)

        patterns = store.get_sequence_patterns()
        assert patterns[0]["from_activity"] == "high"
        assert patterns[1]["from_activity"] == "mid"
        assert patterns[2]["from_activity"] == "low"

    def test_respects_limit(self, store):
        for i in range(10):
            store.save_sequence_pattern(f"from_{i}", f"to_{i}")

        patterns = store.get_sequence_patterns(limit=3)
        assert len(patterns) == 3

    def test_empty_results(self, store):
        patterns = store.get_sequence_patterns(min_count=100)
        assert patterns == []

    def test_updated_at_updated(self, store):
        store.save_sequence_pattern("test", "pattern")
        patterns1 = store.get_sequence_patterns()
        first_seen = patterns1[0]["updated_at"]

        store.save_sequence_pattern("test", "pattern")
        patterns2 = store.get_sequence_patterns()
        second_seen = patterns2[0]["updated_at"]

        # updated_at should be updated (or at least not earlier)
        assert second_seen >= first_seen

    def test_result_has_expected_keys(self, store):
        store.save_sequence_pattern("a", "b")
        patterns = store.get_sequence_patterns()
        p = patterns[0]
        assert "id" in p
        assert "from_activity" in p
        assert "to_activity" in p
        assert "count" in p
        assert "avg_transition_minutes" in p
        assert "updated_at" in p
