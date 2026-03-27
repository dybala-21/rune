"""Tests for heartbeat_checker — HEARTBEAT.md parsing and execution."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from rune.daemon.heartbeat_checker import (
    CheckItem,
    parse_heartbeat_md,
    should_run,
)


@pytest.fixture
def heartbeat_file(tmp_dir):
    """Create a temporary HEARTBEAT.md for testing."""
    path = tmp_dir / "HEARTBEAT.md"
    return path


class TestParseHeartbeatMd:

    def test_empty_file(self, heartbeat_file):
        heartbeat_file.write_text("")
        assert parse_heartbeat_md(heartbeat_file) == []

    def test_no_file(self, tmp_dir):
        assert parse_heartbeat_md(tmp_dir / "nonexistent.md") == []

    def test_basic_items(self, heartbeat_file):
        heartbeat_file.write_text(
            "## Every 30 minutes\n"
            "- [ ] `git status --short` — changes\n"
            "- [ ] `df -h /` — disk\n"
        )
        items = parse_heartbeat_md(heartbeat_file)
        assert len(items) == 2
        assert items[0].command == "git status --short"
        assert items[0].interval_minutes == 30
        assert items[1].command == "df -h /"

    def test_hourly_schedule(self, heartbeat_file):
        heartbeat_file.write_text(
            "## Every 2 hours\n"
            "- [ ] `uptime` — check\n"
        )
        items = parse_heartbeat_md(heartbeat_file)
        assert len(items) == 1
        assert items[0].interval_minutes == 120

    def test_daily_schedule(self, heartbeat_file):
        heartbeat_file.write_text(
            "## Daily at 09:00\n"
            "- [ ] `git log --oneline -5` — recent\n"
        )
        items = parse_heartbeat_md(heartbeat_file)
        assert len(items) == 1
        assert items[0].interval_minutes == 0
        assert items[0].daily_hour == 9
        assert items[0].daily_minute == 0

    def test_no_backtick_skips(self, heartbeat_file):
        heartbeat_file.write_text(
            "## Every 30 minutes\n"
            "- [ ] no backtick here — ignored\n"
            "- [ ] `valid command` — kept\n"
        )
        items = parse_heartbeat_md(heartbeat_file)
        assert len(items) == 2
        assert items[0].command == ""  # no backtick
        assert items[1].command == "valid command"

    def test_comments_ignored(self, heartbeat_file):
        heartbeat_file.write_text(
            "# This is a comment\n"
            "## Every 30 minutes\n"
            "- [ ] `echo hi` — test\n"
            "# Another comment\n"
        )
        items = parse_heartbeat_md(heartbeat_file)
        assert len(items) == 1

    def test_mixed_schedules(self, heartbeat_file):
        heartbeat_file.write_text(
            "## Every 10 minutes\n"
            "- [ ] `cmd1` — fast\n"
            "## Every 1 hour\n"
            "- [ ] `cmd2` — slow\n"
        )
        items = parse_heartbeat_md(heartbeat_file)
        assert items[0].interval_minutes == 10
        assert items[1].interval_minutes == 60

    def test_korean_schedule(self, heartbeat_file):
        heartbeat_file.write_text(
            "## 매 30 분\n"
            "- [ ] `echo test` — 테스트\n"
        )
        items = parse_heartbeat_md(heartbeat_file)
        assert len(items) == 1
        assert items[0].interval_minutes == 30


class TestShouldRun:

    def test_first_run_always(self):
        item = CheckItem(text="test", command="echo", interval_minutes=30)
        assert should_run(item, datetime.now()) is True

    def test_interval_not_elapsed(self):
        item = CheckItem(
            text="test", command="echo", interval_minutes=30,
            last_run=datetime.now() - timedelta(minutes=10),
        )
        assert should_run(item, datetime.now()) is False

    def test_interval_elapsed(self):
        item = CheckItem(
            text="test", command="echo", interval_minutes=30,
            last_run=datetime.now() - timedelta(minutes=31),
        )
        assert should_run(item, datetime.now()) is True

    def test_daily_wrong_hour(self):
        item = CheckItem(
            text="test", command="echo", interval_minutes=0,
            daily_hour=9, daily_minute=0,
        )
        now = datetime(2026, 3, 25, 14, 0, 0)  # 2PM
        assert should_run(item, now) is False

    def test_daily_correct_hour(self):
        item = CheckItem(
            text="test", command="echo", interval_minutes=0,
            daily_hour=9, daily_minute=0,
        )
        now = datetime(2026, 3, 25, 9, 0, 0)
        assert should_run(item, now) is True

    def test_daily_already_ran_today(self):
        item = CheckItem(
            text="test", command="echo", interval_minutes=0,
            daily_hour=9, daily_minute=0,
            last_run=datetime(2026, 3, 25, 9, 0, 0),
        )
        now = datetime(2026, 3, 25, 9, 0, 30)  # same day
        assert should_run(item, now) is False
