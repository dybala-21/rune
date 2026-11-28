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


class TestTheTemplate:
    """The file RUNE writes on first start explains the format and runs nothing the owner did not add."""

    @pytest.fixture
    def template(self, tmp_dir, monkeypatch):
        from rune.daemon import heartbeat_checker

        monkeypatch.setattr(heartbeat_checker, "_HEARTBEAT_PATH", tmp_dir / "HEARTBEAT.md")
        return heartbeat_checker.create_default_heartbeat()

    def test_it_schedules_no_command(self, template):
        assert parse_heartbeat_md(template) == []

    def test_each_example_once_uncommented_is_a_check_with_its_command(self, template):
        uncommented = "".join(line.removeprefix("# ") + "\n" for line in template.read_text().splitlines()
                              if line.startswith(("# ## ", "# - [ ]")))
        template.write_text(uncommented)
        assert [(item.command, item.interval_minutes) for item in parse_heartbeat_md(template)] == [
            ("git -C ~/project diff --quiet", 30),
            ("df -P ~ | awk 'NR==2 {exit ($5+0 > 90)}'", 0),
        ]


async def test_a_check_that_needs_attention_is_kept_where_the_owner_sees_it(tmp_path, monkeypatch):
    """It went to gateway.broadcast_text, which the gateway never had: every alert was lost."""
    from rune.api import conversation_wiring
    from rune.channels.registry import ChannelRegistry
    from rune.daemon import gateway as gateway_module
    from rune.daemon.main import RuneDaemon

    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    monkeypatch.setattr(conversation_wiring, "_conv_manager", None)
    monkeypatch.setattr(conversation_wiring, "_conv_manager_failed", False)
    monkeypatch.setattr(gateway_module, "_gateway_instance", gateway_module.ChannelGateway(ChannelRegistry()))
    await RuneDaemon.__new__(RuneDaemon)._tell_owner("`df -P ~` — the disk is over 90% full", "Capacity 93%")
    inbox = await conversation_wiring.get_conv_manager()._store.load("rune:inbox")
    assert inbox.turns[-1].content == "`df -P ~` — the disk is over 90% full\nCapacity 93%"


async def test_a_check_every_30_minutes_runs_once_across_wakes_a_minute_apart(tmp_dir, monkeypatch):
    """The file is read again at every wake; its items used to forget their last run and ran each minute."""
    from rune.daemon import heartbeat_checker

    path = tmp_dir / "HEARTBEAT.md"
    path.write_text(f"## Every 30 minutes\n- [ ] `echo ran >> {tmp_dir / 'runs'}` — count\n")
    monkeypatch.setattr(heartbeat_checker, "_HEARTBEAT_PATH", path)
    monkeypatch.setattr(heartbeat_checker, "_last_runs", {})
    assert len(await heartbeat_checker.run_heartbeat_checks()) == 1
    assert await heartbeat_checker.run_heartbeat_checks() == []
    assert (tmp_dir / "runs").read_text() == "ran\n"


def test_a_failing_check_is_told_once_and_again_after_it_passed():
    from rune.daemon.heartbeat_checker import to_tell

    failing: set[str] = set()

    def run(attention: bool) -> int:
        return len(to_tell([{"item": "disk", "command": "df", "output": "", "needs_attention": attention}], failing))

    assert [run(True), run(True), run(True), run(False), run(True)] == [1, 0, 0, 0, 1]
