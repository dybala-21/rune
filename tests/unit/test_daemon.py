"""Tests for daemon types and heartbeat scheduler."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from rune.daemon.heartbeat import HeartbeatScheduler, _matches_cron_field
from rune.daemon.types import (
    CommandType,
    DaemonCommand,
    DaemonResponse,
    DaemonStatus,
)


class TestCommandType:
    def test_enum_values(self):
        assert CommandType.EXECUTE == "execute"
        assert CommandType.STATUS == "status"
        assert CommandType.CANCEL == "cancel"
        assert CommandType.SHUTDOWN == "shutdown"


class TestDaemonCommand:
    def test_defaults(self):
        cmd = DaemonCommand(type=CommandType.STATUS)
        assert cmd.type == CommandType.STATUS
        assert cmd.payload == {}
        assert len(cmd.request_id) == 16

    def test_custom_payload(self):
        cmd = DaemonCommand(
            type=CommandType.EXECUTE,
            payload={"goal": "deploy"},
            request_id="abc123",
        )
        assert cmd.payload["goal"] == "deploy"
        assert cmd.request_id == "abc123"


class TestDaemonResponse:
    def test_success_response(self):
        resp = DaemonResponse(request_id="r1", success=True, data={"result": "ok"})
        assert resp.success is True
        assert resp.data["result"] == "ok"
        assert resp.error is None

    def test_error_response(self):
        resp = DaemonResponse(request_id="r2", success=False, error="not found")
        assert resp.success is False
        assert resp.error == "not found"


class TestDaemonStatus:
    def test_status_fields(self):
        status = DaemonStatus(
            running=True,
            uptime_seconds=120.5,
            active_tasks=2,
            queued_tasks=5,
            channels=["telegram", "discord"],
        )
        assert status.running is True
        assert status.uptime_seconds == 120.5
        assert status.active_tasks == 2
        assert len(status.channels) == 2

    def test_status_defaults(self):
        status = DaemonStatus(
            running=False,
            uptime_seconds=0.0,
            active_tasks=0,
            queued_tasks=0,
        )
        assert status.channels == []


class TestCronFieldMatching:
    def test_wildcard(self):
        assert _matches_cron_field("*", 42) is True

    def test_exact_match(self):
        assert _matches_cron_field("5", 5) is True
        assert _matches_cron_field("5", 6) is False

    def test_step(self):
        assert _matches_cron_field("*/5", 0) is True
        assert _matches_cron_field("*/5", 10) is True
        assert _matches_cron_field("*/5", 3) is False

    def test_range(self):
        assert _matches_cron_field("1-5", 3) is True
        assert _matches_cron_field("1-5", 6) is False
        assert _matches_cron_field("1-5", 1) is True
        assert _matches_cron_field("1-5", 5) is True

    def test_list(self):
        assert _matches_cron_field("1,3,5", 3) is True
        assert _matches_cron_field("1,3,5", 4) is False

    def test_invalid_field(self):
        assert _matches_cron_field("abc", 0) is False


class TestHeartbeatSchedulerCronMatch:
    def test_matches_every_minute(self):
        dt = datetime(2026, 3, 10, 14, 30, 0)
        assert HeartbeatScheduler._matches_cron("* * * * *", dt) is True

    def test_matches_specific_minute(self):
        dt = datetime(2026, 3, 10, 14, 30, 0)
        assert HeartbeatScheduler._matches_cron("30 * * * *", dt) is True
        assert HeartbeatScheduler._matches_cron("0 * * * *", dt) is False

    def test_matches_specific_hour_and_minute(self):
        dt = datetime(2026, 3, 10, 9, 0, 0)
        assert HeartbeatScheduler._matches_cron("0 9 * * *", dt) is True
        assert HeartbeatScheduler._matches_cron("0 10 * * *", dt) is False

    def test_invalid_cron_expression(self):
        dt = datetime(2026, 3, 10, 14, 30, 0)
        # Only 3 fields instead of 5
        assert HeartbeatScheduler._matches_cron("* * *", dt) is False


class TestHeartbeatSchedulerTasks:
    def test_add_and_remove_task(self):
        sched = HeartbeatScheduler(interval_seconds=60.0)
        callback = AsyncMock()
        sched.add_task("test-task", "* * * * *", callback)
        assert "test-task" in sched._tasks
        sched.remove_task("test-task")
        assert "test-task" not in sched._tasks

    def test_remove_nonexistent_no_error(self):
        sched = HeartbeatScheduler()
        # Should not raise
        sched.remove_task("nope")

    @pytest.mark.asyncio
    async def test_start_stop(self):
        sched = HeartbeatScheduler(interval_seconds=0.05)
        await sched.start()
        assert sched._running is True
        assert sched._loop_task is not None
        await sched.stop()
        assert sched._running is False
        assert sched._loop_task is None

    @pytest.mark.asyncio
    async def test_double_start_idempotent(self):
        sched = HeartbeatScheduler(interval_seconds=60.0)
        await sched.start()
        first_task = sched._loop_task
        await sched.start()  # second start should be a no-op
        assert sched._loop_task is first_task
        await sched.stop()
