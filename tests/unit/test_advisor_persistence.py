"""Tests for Tier 2 advisor persistence layer.

Covers:
- advisor_events table creation via migration
- Episode advisor counter fields (round-trip)
- log_advisor_event write path
- update_advisor_outcome sets outcome on all session rows
- get_advisor_stats aggregates correctly
- AdvisorBudget.call_history captures structured metadata
- Plan-injected semantics: only non-error decisions with plan_steps count
"""

from __future__ import annotations

import pytest

from rune.agent.advisor.protocol import (
    AdvisorBudget,
    AdvisorDecision,
)
from rune.memory.store import MemoryStore
from rune.memory.types import Episode


@pytest.fixture
def store(tmp_dir):
    s = MemoryStore(db_path=tmp_dir / "advisor.db")
    yield s
    s.close()


class TestSchemaMigration:
    def test_advisor_events_table_exists(self, store):
        rows = list(store.conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='advisor_events'"
        ))
        assert len(rows) == 1

    def test_advisor_events_columns(self, store):
        cols = {
            r[1] for r in store.conn.execute(
                "PRAGMA table_info(advisor_events)"
            )
        }
        expected = {
            "id", "session_id", "trigger", "action", "provider", "model",
            "output_tokens", "latency_ms", "episode_outcome",
            "plan_injected", "stuck_reason", "created_at",
        }
        assert expected.issubset(cols)

    def test_episodes_has_advisor_columns(self, store):
        cols = {
            r[1] for r in store.conn.execute(
                "PRAGMA table_info(episodes)"
            )
        }
        assert "advisor_calls" in cols
        assert "advisor_followed_count" in cols
        assert "advisor_output_tokens" in cols


class TestEpisodeRoundTrip:
    def test_save_and_load_advisor_fields(self, store):
        ep = Episode(
            id="e1",
            timestamp="2026-04-11T00:00:00Z",
            task_summary="test task",
            advisor_calls=3,
            advisor_followed_count=2,
            advisor_output_tokens=450,
        )
        store.save_episode(ep)
        loaded = store.get_recent_episodes(limit=1)
        assert len(loaded) == 1
        assert loaded[0].advisor_calls == 3
        assert loaded[0].advisor_followed_count == 2
        assert loaded[0].advisor_output_tokens == 450

    def test_backward_compat_episode_without_advisor_fields(self, store):
        """Episodes constructed without the new fields default to 0."""
        ep = Episode(id="e_legacy", task_summary="legacy")
        store.save_episode(ep)
        loaded = store.get_recent_episodes(limit=1)
        assert loaded[0].advisor_calls == 0
        assert loaded[0].advisor_followed_count == 0
        assert loaded[0].advisor_output_tokens == 0


class TestLogAdvisorEvent:
    def test_insert_one(self, store):
        row_id = store.log_advisor_event(
            session_id="sess-1",
            trigger="stuck",
            action="retry_tool",
            provider="openai",
            model="gpt-5.4",
            output_tokens=200,
            latency_ms=1500,
            plan_injected=True,
        )
        assert row_id > 0
        rows = list(store.conn.execute(
            "SELECT trigger, action, provider, model, "
            "output_tokens, latency_ms, plan_injected "
            "FROM advisor_events WHERE id=?",
            (row_id,),
        ))
        assert len(rows) == 1
        trig, act, prov, mod, tok, lat, inj = rows[0]
        assert trig == "stuck"
        assert act == "retry_tool"
        assert prov == "openai"
        assert mod == "gpt-5.4"
        assert tok == 200
        assert lat == 1500
        assert inj == 1

    def test_insert_multiple_same_session(self, store):
        for trig in ("early", "pre_done", "stuck"):
            store.log_advisor_event(
                session_id="sess-batch",
                trigger=trig,
                action="continue",
                provider="anthropic",
                model="claude-opus-4-6",
            )
        (cnt,) = store.conn.execute(
            "SELECT COUNT(*) FROM advisor_events WHERE session_id='sess-batch'"
        ).fetchone()
        assert cnt == 3

    def test_default_episode_outcome_is_empty(self, store):
        store.log_advisor_event(
            session_id="sess-2",
            trigger="early",
            action="continue",
            provider="openai",
            model="gpt-4o-mini",
        )
        (outcome,) = store.conn.execute(
            "SELECT episode_outcome FROM advisor_events WHERE session_id='sess-2'"
        ).fetchone()
        assert outcome == ""


class TestUpdateAdvisorOutcome:
    def test_updates_all_rows_in_session(self, store):
        for _ in range(3):
            store.log_advisor_event(
                session_id="sess-out",
                trigger="stuck",
                action="continue",
                provider="openai",
                model="o1",
            )
        updated = store.update_advisor_outcome(
            session_id="sess-out",
            outcome="completed",
        )
        assert updated == 3
        rows = list(store.conn.execute(
            "SELECT episode_outcome FROM advisor_events WHERE session_id='sess-out'"
        ))
        assert all(r[0] == "completed" for r in rows)

    def test_only_updates_empty_outcomes(self, store):
        store.log_advisor_event(
            session_id="sess-x", trigger="early", action="continue",
            provider="openai", model="o1",
        )
        store.update_advisor_outcome(session_id="sess-x", outcome="completed")
        # Second call should not re-touch already-set rows
        touched = store.update_advisor_outcome(
            session_id="sess-x", outcome="failed",
        )
        assert touched == 0

    def test_returns_zero_for_unknown_session(self, store):
        touched = store.update_advisor_outcome(
            session_id="no-such-session", outcome="completed",
        )
        assert touched == 0


class TestGetAdvisorStats:
    def test_empty_store(self, store):
        stats = store.get_advisor_stats()
        assert stats["total_calls"] == 0
        assert stats["by_trigger"] == {}
        assert stats["by_outcome"] == {}

    def test_aggregates_by_trigger(self, store):
        for trig in ("early", "early", "stuck"):
            store.log_advisor_event(
                session_id=f"s-{trig}",
                trigger=trig,
                action="continue",
                provider="openai",
                model="gpt-5.4",
                output_tokens=100,
            )
        stats = store.get_advisor_stats()
        assert stats["total_calls"] == 3
        assert stats["by_trigger"]["early"] == 2
        assert stats["by_trigger"]["stuck"] == 1

    def test_completion_rate_by_trigger(self, store):
        # 2 completed early calls
        store.log_advisor_event(
            session_id="s1", trigger="early", action="continue",
            provider="openai", model="gpt-5.4",
        )
        store.log_advisor_event(
            session_id="s2", trigger="early", action="continue",
            provider="openai", model="gpt-5.4",
        )
        store.update_advisor_outcome(session_id="s1", outcome="completed")
        store.update_advisor_outcome(session_id="s2", outcome="completed")
        # 1 failed stuck call
        store.log_advisor_event(
            session_id="s3", trigger="stuck", action="continue",
            provider="openai", model="gpt-5.4",
        )
        store.update_advisor_outcome(
            session_id="s3", outcome="max_gate_blocked",
        )
        stats = store.get_advisor_stats()
        assert stats["completion_rate_by_trigger"]["early"] == 1.0
        assert stats["completion_rate_by_trigger"]["stuck"] == 0.0

    def test_avg_tokens_and_latency(self, store):
        for tok, lat in [(100, 1000), (200, 2000), (300, 3000)]:
            store.log_advisor_event(
                session_id=f"s-{tok}",
                trigger="stuck",
                action="continue",
                provider="openai",
                model="o1",
                output_tokens=tok,
                latency_ms=lat,
            )
        stats = store.get_advisor_stats()
        assert stats["avg_output_tokens"] == 200.0
        assert stats["avg_latency_ms"] == 2000.0


class TestAdvisorBudgetCallHistory:
    def _decision(self, **overrides) -> AdvisorDecision:
        defaults = dict(
            action="continue",
            plan_steps=["step 1", "step 2"],
            raw_text="NEXT: continue\n1. step 1\n2. step 2",
            trigger="stuck",
            provider="openai",
            model="gpt-5.4",
            input_tokens=400,
            output_tokens=120,
            latency_ms=1500,
        )
        defaults.update(overrides)
        return AdvisorDecision(**defaults)  # type: ignore[arg-type]

    def test_record_populates_history_entry(self):
        budget = AdvisorBudget(max_calls=3)
        budget.record(self._decision())
        assert len(budget.call_history) == 1
        entry = budget.call_history[0]
        assert entry["trigger"] == "stuck"
        assert entry["action"] == "continue"
        assert entry["provider"] == "openai"
        assert entry["model"] == "gpt-5.4"
        assert entry["output_tokens"] == 120
        assert entry["latency_ms"] == 1500
        assert entry["plan_injected"] is True

    def test_plan_injected_false_on_empty_plan(self):
        budget = AdvisorBudget(max_calls=3)
        budget.record(self._decision(plan_steps=[]))
        assert budget.call_history[0]["plan_injected"] is False

    def test_plan_injected_false_on_error(self):
        budget = AdvisorBudget(max_calls=3)
        budget.record(
            AdvisorDecision.noop("timeout", trigger="stuck"),
        )
        assert budget.call_history[0]["plan_injected"] is False
        assert budget.call_history[0]["error_code"] == "timeout"

    def test_multiple_records_append_in_order(self):
        budget = AdvisorBudget(max_calls=5)
        budget.record(self._decision(trigger="early"))
        budget.record(self._decision(trigger="pre_done"))
        budget.record(self._decision(trigger="stuck"))
        triggers = [e["trigger"] for e in budget.call_history]
        assert triggers == ["early", "pre_done", "stuck"]
        assert budget.calls_used == 3
        assert budget.tokens_used == 360  # 3 × 120


class TestStuckReasonPersistence:
    """2b-3: stuck_reason column round-trip + stats aggregation."""

    def test_log_event_with_stuck_reason(self, store):
        store.log_advisor_event(
            session_id="sess-stuck",
            trigger="stuck",
            action="retry_tool",
            provider="anthropic",
            model="claude-opus-4-6",
            stuck_reason="gate_blocked",
        )
        (reason,) = store.conn.execute(
            "SELECT stuck_reason FROM advisor_events "
            "WHERE session_id='sess-stuck'"
        ).fetchone()
        assert reason == "gate_blocked"

    def test_stuck_reason_default_empty(self, store):
        store.log_advisor_event(
            session_id="sess-nostuck",
            trigger="early",
            action="continue",
            provider="openai",
            model="gpt-5.4",
        )
        (reason,) = store.conn.execute(
            "SELECT stuck_reason FROM advisor_events "
            "WHERE session_id='sess-nostuck'"
        ).fetchone()
        assert reason == ""

    def test_stats_aggregates_by_stuck_reason(self, store):
        for sr in ("gate_blocked", "gate_blocked", "wind_down", "stall"):
            store.log_advisor_event(
                session_id=f"s-{sr}-{_uniq()}",
                trigger="stuck",
                action="retry_tool",
                provider="openai",
                model="o1",
                stuck_reason=sr,
            )
        stats = store.get_advisor_stats()
        assert stats["by_stuck_reason"]["gate_blocked"] == 2
        assert stats["by_stuck_reason"]["wind_down"] == 1
        assert stats["by_stuck_reason"]["stall"] == 1

    def test_stats_excludes_non_stuck_from_reason_breakdown(self, store):
        """EARLY/PRE_DONE events should not appear in by_stuck_reason."""
        store.log_advisor_event(
            session_id="s-early",
            trigger="early",
            action="continue",
            provider="openai",
            model="o1",
        )
        stats = store.get_advisor_stats()
        assert stats["by_stuck_reason"] == {}


# Local helper to avoid id collision within a test
_counter = [0]
def _uniq() -> int:
    _counter[0] += 1
    return _counter[0]
