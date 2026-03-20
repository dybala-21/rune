"""Tests for rune.api.event_logger — ported from event-logger.test.ts."""

from datetime import datetime

import pytest

import rune.api.event_logger as event_logger_mod
from rune.api.event_logger import (
    append_event,
    flush_events,
    list_runs,
    read_events,
)


@pytest.fixture(autouse=True)
def _tmp_events_dir(tmp_path):
    """Redirect EVENTS_DIR to a temp directory for every test."""
    original = event_logger_mod.EVENTS_DIR
    event_logger_mod.EVENTS_DIR = tmp_path / "events"
    # Clear module-level state
    event_logger_mod._write_buffer.clear()
    event_logger_mod._ensured_dirs.clear()
    yield tmp_path / "events"
    event_logger_mod.EVENTS_DIR = original
    event_logger_mod._write_buffer.clear()
    event_logger_mod._ensured_dirs.clear()


class TestEventLogger:
    """Tests for the event logger module."""

    @pytest.mark.asyncio
    async def test_reads_events_with_filtering(self):
        now = datetime.now().isoformat()
        append_event("c1", "run_1", {"event": "tool_call", "data": {"name": "x"}, "timestamp": now})
        append_event("c1", "run_1", {"event": "thinking", "data": {"text": "hmm"}, "timestamp": now})
        append_event("c1", "run_1", {"event": "text_delta", "data": {"text": "ok"}, "timestamp": now})

        all_events = await read_events("c1", include_tools=True, include_thinking=True)
        assert [e["event"] for e in all_events] == ["tool_call", "thinking", "text_delta"]

        filtered = await read_events("c1", include_tools=False, include_thinking=False)
        assert [e["event"] for e in filtered] == ["text_delta"]

    @pytest.mark.asyncio
    async def test_flushes_pending_buffer_before_queries(self):
        now = datetime.now().isoformat()
        append_event("c2", "run_a", {"event": "agent_start", "data": {"goal": "a"}, "timestamp": now})
        append_event("c2", "run_b", {"event": "agent_start", "data": {"goal": "b"}, "timestamp": now})

        await flush_events()
        runs = await list_runs("c2")
        assert runs == ["run_a", "run_b"]

        events = await read_events("c2")
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_stores_event_logs_with_owner_only_permissions(self):
        now = datetime.now().isoformat()
        append_event("cperm", "run_perm", {"event": "agent_start", "data": {"goal": "perm"}, "timestamp": now})
        await flush_events()

        conv_dir = event_logger_mod.EVENTS_DIR / "cperm"
        log_file = conv_dir / "run_perm.jsonl"

        dir_mode = conv_dir.stat().st_mode & 0o777
        file_mode = log_file.stat().st_mode & 0o777

        assert dir_mode == 0o700
        assert file_mode == 0o600

    @pytest.mark.asyncio
    async def test_returns_empty_for_missing_conversation(self):
        events = await read_events("nonexistent")
        assert events == []

    @pytest.mark.asyncio
    async def test_list_runs_returns_empty_for_missing_conversation(self):
        runs = await list_runs("nonexistent")
        assert runs == []

    @pytest.mark.asyncio
    async def test_handles_large_number_of_events(self):
        now = datetime.now().isoformat()
        total = 100
        for i in range(total):
            append_event("clarge", "run_large", {
                "event": "text_delta",
                "data": {"index": i},
                "timestamp": now,
            })
        events = await read_events("clarge")
        assert len(events) == total
