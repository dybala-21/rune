"""Tests for `rune advisor stats` CLI subcommand (2b-2)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from rune.cli.advisor_cmd import advisor_app
from rune.memory.store import MemoryStore

runner = CliRunner()


@pytest.fixture
def store(tmp_dir):
    s = MemoryStore(db_path=tmp_dir / "cli_advisor.db")
    yield s
    s.close()


def _patch_store(store: MemoryStore):
    """Redirect get_memory_store() to the per-test temp store."""
    return patch("rune.memory.store.get_memory_store", return_value=store)


class TestAdvisorStatsEmpty:
    def test_empty_table_output(self, store):
        with _patch_store(store):
            result = runner.invoke(advisor_app, ["stats"])
        assert result.exit_code == 0
        assert "No advisor events recorded yet" in result.stdout
        assert "Total calls:" in result.stdout

    def test_empty_json_output(self, store):
        with _patch_store(store):
            result = runner.invoke(advisor_app, ["stats", "--format", "json"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["total_calls"] == 0
        assert payload["by_trigger"] == {}
        assert payload["by_stuck_reason"] == {}


class TestAdvisorStatsPopulated:
    def test_table_with_triggers(self, store):
        store.log_advisor_event(
            session_id="s-early",
            trigger="early",
            action="continue",
            provider="openai",
            model="gpt-5.4",
            output_tokens=100,
            latency_ms=1000,
        )
        store.log_advisor_event(
            session_id="s-stuck-1",
            trigger="stuck",
            action="retry_tool",
            provider="anthropic",
            model="claude-opus-4-6",
            output_tokens=150,
            latency_ms=2000,
            stuck_reason="gate_blocked",
        )
        store.log_advisor_event(
            session_id="s-stuck-2",
            trigger="stuck",
            action="retry_tool",
            provider="anthropic",
            model="claude-opus-4-6",
            output_tokens=140,
            latency_ms=1800,
            stuck_reason="wind_down",
        )
        store.update_advisor_outcome(
            session_id="s-early", outcome="completed",
        )
        store.update_advisor_outcome(
            session_id="s-stuck-1", outcome="completed",
        )
        store.update_advisor_outcome(
            session_id="s-stuck-2", outcome="max_gate_blocked",
        )

        with _patch_store(store):
            result = runner.invoke(advisor_app, ["stats"])
        assert result.exit_code == 0
        # Header
        assert "Total calls:" in result.stdout
        assert "3" in result.stdout
        # Trigger breakdown
        assert "early" in result.stdout
        assert "stuck" in result.stdout
        # Stuck sub-reasons
        assert "gate_blocked" in result.stdout
        assert "wind_down" in result.stdout
        # Outcomes
        assert "completed" in result.stdout
        assert "max_gate_blocked" in result.stdout

    def test_json_round_trip(self, store):
        store.log_advisor_event(
            session_id="s-json",
            trigger="pre_done",
            action="continue",
            provider="openai",
            model="gpt-5.4",
            output_tokens=200,
            latency_ms=1500,
        )
        with _patch_store(store):
            result = runner.invoke(
                advisor_app, ["stats", "--format", "json"],
            )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["total_calls"] == 1
        assert payload["by_trigger"] == {"pre_done": 1}
        assert payload["avg_output_tokens"] == 200.0

    def test_days_flag_passes_through(self, store):
        with _patch_store(store):
            result = runner.invoke(
                advisor_app, ["stats", "--days", "7", "--format", "json"],
            )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["since_days"] == 7

    def test_days_short_flag(self, store):
        with _patch_store(store):
            result = runner.invoke(
                advisor_app, ["stats", "-d", "14", "-f", "json"],
            )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["since_days"] == 14
