"""Tests for rune.daemon.progress_reporter — phase detection, throttling, and broadcasting."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from rune.daemon.progress_reporter import (
    Phase,
    ProgressReporter,
    ProgressReporterConfig,
    ProgressUpdate,
    detect_phase,
)

# ---------------------------------------------------------------------------
# detect_phase
# ---------------------------------------------------------------------------


class TestDetectPhase:
    def test_file_edit_returns_executing(self):
        assert detect_phase("file.edit") == Phase.EXECUTING

    def test_bash_returns_executing(self):
        assert detect_phase("bash") == Phase.EXECUTING

    def test_web_search_returns_researching(self):
        assert detect_phase("web.search") == Phase.RESEARCHING

    def test_file_read_returns_researching(self):
        assert detect_phase("file.read") == Phase.RESEARCHING

    def test_code_analysis_returns_analyzing(self):
        assert detect_phase("code.lint") == Phase.ANALYZING

    def test_think_early_returns_thinking(self):
        assert detect_phase("think") == Phase.THINKING

    def test_think_late_returns_composing(self):
        assert detect_phase("think", is_late_stage=True) == Phase.COMPOSING

    def test_delegate_returns_analyzing(self):
        assert detect_phase("delegate") == Phase.ANALYZING

    def test_unknown_tool_returns_thinking(self):
        assert detect_phase("some_unknown_tool") == Phase.THINKING

    def test_underscore_normalized_to_dot(self):
        assert detect_phase("file_write") == Phase.EXECUTING


# ---------------------------------------------------------------------------
# ProgressUpdate
# ---------------------------------------------------------------------------


class TestProgressUpdate:
    def test_to_dict(self):
        update = ProgressUpdate(
            run_id="run-1",
            phase=Phase.RESEARCHING,
            action="Reading file.py",
            step_number=3,
            total_steps=10,
            tool_name="file.read",
        )
        d = update.to_dict()
        assert d["run_id"] == "run-1"
        assert d["phase"] == "researching"
        assert d["action"] == "Reading file.py"
        assert d["step_number"] == 3


# ---------------------------------------------------------------------------
# ProgressReporter — throttling
# ---------------------------------------------------------------------------


class TestProgressReporterThrottle:
    @pytest.mark.asyncio
    async def test_throttles_rapid_updates(self):
        config = ProgressReporterConfig(min_edit_interval_ms=1000)
        reporter = ProgressReporter(config)

        writer = MagicMock()
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        reporter.add_client(writer)

        update = ProgressUpdate(
            run_id="r1", phase=Phase.THINKING, action="Thinking",
        )

        # First send should go through
        await reporter.send_update(update)
        # Second send immediately should be throttled
        await reporter.send_update(update)

        # Only one write call expected (throttled second)
        assert writer.write.call_count == 1


# ---------------------------------------------------------------------------
# ProgressReporter — client management
# ---------------------------------------------------------------------------


class TestProgressReporterClients:
    def test_add_and_remove_client(self):
        reporter = ProgressReporter()
        writer = MagicMock()
        reporter.add_client(writer)
        assert reporter.client_count == 1

        reporter.remove_client(writer)
        assert reporter.client_count == 0

    def test_remove_nonexistent_client_does_not_raise(self):
        reporter = ProgressReporter()
        writer = MagicMock()
        reporter.remove_client(writer)  # should not raise


# ---------------------------------------------------------------------------
# ProgressReporter — broadcast
# ---------------------------------------------------------------------------


class TestProgressReporterBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all_clients(self):
        reporter = ProgressReporter()

        w1 = MagicMock()
        w1.write = MagicMock()
        w1.drain = AsyncMock()

        w2 = MagicMock()
        w2.write = MagicMock()
        w2.drain = AsyncMock()

        reporter.add_client(w1)
        reporter.add_client(w2)

        await reporter.broadcast("test_event", {"key": "value"})

        assert w1.write.call_count == 1
        assert w2.write.call_count == 1
