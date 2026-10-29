"""Test scheduler failures, shutdown cleanup, and storage limits in isolation."""

import asyncio
import os
import plistlib
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rune.config.loader import get_config
from rune.daemon.heartbeat import HeartbeatScheduler
from rune.daemon.main import RuneDaemon, _default_config
from rune.daemon.process_lock import acquire_lock, release_lock


def test_service_managers_run_the_foreground_process(monkeypatch, tmp_path):
    from rune.daemon.service import launchd_config, systemd_config

    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    monkeypatch.setenv("RUNE_WEB_STATIC_DIR", '/tmp/a & b/<static>')
    launchd = plistlib.loads(plistlib.dumps(launchd_config()))
    assert launchd["ProgramArguments"][3:5] == ["daemon", "run"]
    assert launchd["EnvironmentVariables"]["RUNE_WEB_STATIC_DIR"] == str(Path('/tmp/a & b/<static>').resolve())
    unit = systemd_config()
    assert "Type=exec" in unit and "daemon run" in unit
    assert "daemon start" not in unit and "TimeoutStopSec=35" in unit


def test_failed_lock_contender_cannot_erase_owner(tmp_path):
    path = tmp_path / "daemon.lock"
    acquire_lock(path)
    try:
        before = path.read_bytes()
        with pytest.raises(RuntimeError):
            acquire_lock(path)
        assert path.read_bytes() == before
    finally:
        release_lock(path)


async def test_failed_start_releases_owned_state(monkeypatch):
    daemon = RuneDaemon(install_signal_handlers=False)
    monkeypatch.setattr(daemon, "_load_env", lambda: None)
    monkeypatch.setattr(daemon, "_init_subsystems", AsyncMock(side_effect=RuntimeError("startup fault")))
    cleanup = AsyncMock()
    monkeypatch.setattr(daemon, "_shutdown_subsystems", cleanup)
    with pytest.raises(RuntimeError, match="startup fault"):
        await daemon.serve_forever()
    cleanup.assert_awaited_once()
    assert daemon._lock_path is None
    assert not os.path.exists(daemon._config["pid_file"])


async def test_rejected_start_does_not_remove_another_daemons_pid(monkeypatch):
    from pathlib import Path

    config = _default_config()
    Path(config["pid_file"]).write_text("1234")
    owner = acquire_lock()
    daemon = RuneDaemon(config, install_signal_handlers=False)
    monkeypatch.setattr(daemon, "_load_env", lambda: None)
    try:
        with pytest.raises(RuntimeError):
            await daemon.serve_forever()
        assert Path(config["pid_file"]).read_text() == "1234"
    finally:
        release_lock(owner)


async def test_concurrent_stop_waits_for_cleanup(monkeypatch):
    daemon = RuneDaemon(install_signal_handlers=False)
    await daemon._acquire_process_lock()
    entered, finish = asyncio.Event(), asyncio.Event()

    async def cleanup():
        entered.set()
        await finish.wait()

    monkeypatch.setattr(daemon, "_shutdown_subsystems", cleanup)
    first = asyncio.create_task(daemon.stop())
    await entered.wait()
    second = asyncio.create_task(daemon.stop())
    await asyncio.sleep(0)
    assert not first.done() and not second.done()
    finish.set()
    await asyncio.gather(first, second)
    assert daemon._lock_path is None


async def test_critical_task_failure_exits_supervision(monkeypatch):
    daemon = RuneDaemon(install_signal_handlers=False)

    async def failed():
        raise RuntimeError("API died")

    async def start():
        daemon._api_server_task = asyncio.create_task(failed())
        await asyncio.sleep(0)

    monkeypatch.setattr(daemon, "start", start)
    with pytest.raises(RuntimeError, match="API died"):
        await daemon.serve_forever()


async def test_slow_heartbeat_task_does_not_block_or_overlap_other_tasks():
    scheduler = HeartbeatScheduler()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def held():
        calls.append("slow")
        entered.set()
        await release.wait()

    fast = AsyncMock()
    scheduler.add_task("slow", "* * * * *", held)
    scheduler.add_task("fast", "* * * * *", fast)
    now = datetime(2026, 10, 25, 8, 59)
    scheduler.tick(now)
    await entered.wait()
    await asyncio.sleep(0)
    fast.assert_awaited_once()
    scheduler.tick(now + timedelta(minutes=1))
    await asyncio.sleep(0)
    assert calls == ["slow"] and fast.await_count == 2
    await scheduler.stop()


async def test_heartbeat_coalesces_missed_minute_and_ignores_clock_rewind():
    scheduler = HeartbeatScheduler()
    callback = AsyncMock()
    scheduler.add_task("daily", "0 9 * * *", callback)
    scheduler.tick(datetime(2026, 10, 25, 8, 59))
    scheduler.tick(datetime(2026, 10, 25, 9, 1))
    await asyncio.sleep(0)
    scheduler.tick(datetime(2026, 10, 25, 9, 0))
    await asyncio.sleep(0)
    callback.assert_awaited_once()
    await scheduler.stop()


@pytest.fixture
def scheduled_job(monkeypatch):
    from rune.capabilities import cron
    from rune.memory.store import MemoryStore

    store = MemoryStore()
    monkeypatch.setattr(cron, "_get_store", lambda: store)
    job_id = store.create_cron_job(name="daily", schedule="0 9 * * *", command="true")
    yield cron._row_to_cronjob(store.get_cron_job(job_id)), store
    store.close()


async def test_schedule_survives_restart_and_recovers_one_missed_occurrence(scheduled_job, tmp_path, monkeypatch):
    from rune.capabilities import cron
    from rune.daemon.cron_dispatcher import CronDispatcher

    clock = [datetime(2026, 10, 25, 8, 59).timestamp()]
    monkeypatch.setattr("rune.daemon.cron_dispatcher.time.time", lambda: clock[0])
    execute = AsyncMock()
    monkeypatch.setattr(cron, "execute_cron_job", execute)
    dispatcher = CronDispatcher(tmp_path / "schedules.db")
    await dispatcher.tick()
    await dispatcher.stop()
    clock[0] += 120
    dispatcher = CronDispatcher(tmp_path / "schedules.db")
    await dispatcher.tick()
    await asyncio.sleep(0)
    await dispatcher.tick()
    await asyncio.sleep(0)
    execute.assert_awaited_once()
    assert execute.call_args.kwargs["scheduled_minute"] == int((clock[0] - 60) // 60)
    await dispatcher.stop()


async def test_expired_schedule_is_not_replayed(scheduled_job, tmp_path, monkeypatch):
    from rune.capabilities import cron
    from rune.daemon.cron_dispatcher import CronDispatcher

    clock = [datetime(2026, 10, 25, 8, 59).timestamp()]
    monkeypatch.setattr("rune.daemon.cron_dispatcher.time.time", lambda: clock[0])
    execute = AsyncMock()
    monkeypatch.setattr(cron, "execute_cron_job", execute)
    dispatcher = CronDispatcher(tmp_path / "schedules.db")
    await dispatcher.tick()
    clock[0] += 3600
    await dispatcher.tick()
    await asyncio.sleep(0)
    execute.assert_not_awaited()
    await dispatcher.stop()


async def test_all_cron_entry_points_share_concurrency_limit(monkeypatch):
    from rune.capabilities import cron

    config = get_config().cron_execution
    config.max_concurrent = 2
    release, ready = asyncio.Event(), asyncio.Event()
    active = peak = completed = 0

    async def worker(*args, **kwargs):
        nonlocal active, peak, completed
        active += 1
        peak = max(peak, active)
        if active == 2:
            ready.set()
        await release.wait()
        active -= 1
        completed += 1

    monkeypatch.setattr(cron, "_execute_cron_job", worker)
    tasks = [asyncio.create_task(cron.execute_cron_job(SimpleNamespace())) for _ in range(5)]
    await asyncio.wait_for(ready.wait(), 1)
    await asyncio.sleep(0)
    assert peak == 2
    release.set()
    await asyncio.gather(*tasks)
    assert completed == 5 and peak == 2
    config.enabled = False
    await cron.execute_cron_job(SimpleNamespace())
    assert completed == 5


def test_storage_guard_rejects_low_disk_and_recovers(monkeypatch):
    from rune.daemon.maintenance import check_storage

    monkeypatch.setattr("shutil.disk_usage", lambda _: SimpleNamespace(free=100))
    with pytest.raises(RuntimeError, match="disk space"):
        check_storage()
    monkeypatch.setattr("shutil.disk_usage", lambda _: SimpleNamespace(free=1024 ** 3))
    check_storage()


def test_trash_retention_uses_deletion_date_and_keeps_unrecognized_files(tmp_path, monkeypatch):
    import time

    from rune.tools.file import cleanup_trash

    monkeypatch.setattr("rune.tools.file._trash_dir", lambda: tmp_path)
    recent = tmp_path / f"{int(time.time() * 1000)}_old-source.txt"
    expired = tmp_path / f"{int((time.time() - 8 * 86400) * 1000)}_expired.txt"
    unknown = tmp_path / "user-file.txt"
    for path in (recent, expired, unknown):
        path.write_text("keep unless retention expired")
        os.utime(path, (1, 1))
    assert cleanup_trash() == 1
    assert recent.exists() and unknown.exists() and not expired.exists()


async def test_removed_callback_is_drained_before_shutdown():
    scheduler = HeartbeatScheduler()
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def task():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.set()

    scheduler.add_task("callback", "* * * * *", task)
    scheduler.tick()
    await entered.wait()
    scheduler.remove_task("callback")
    await scheduler.stop()
    assert cleaned.is_set() and not scheduler._inflight


async def test_pending_occurrence_survives_cancellation_before_execution(scheduled_job, tmp_path, monkeypatch):
    from rune.capabilities import cron
    from rune.daemon.cron_dispatcher import CronDispatcher

    monkeypatch.setattr("rune.daemon.cron_dispatcher.time.time",
                        lambda: datetime(2026, 10, 25, 9).timestamp())
    entered = asyncio.Event()

    async def waiting(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(cron, "execute_cron_job", waiting)
    dispatcher = CronDispatcher(tmp_path / "schedule.db")
    await dispatcher.tick()
    await entered.wait()
    await dispatcher.stop()
    execute = AsyncMock()
    monkeypatch.setattr(cron, "execute_cron_job", execute)
    reopened = CronDispatcher(tmp_path / "schedule.db")
    await reopened.tick()
    await asyncio.sleep(0)
    execute.assert_awaited_once()
    await reopened.stop()


async def test_failed_dispatch_does_not_spin_or_block_other_jobs(scheduled_job, tmp_path, monkeypatch):
    from rune.capabilities import cron
    from rune.daemon.cron_dispatcher import CronDispatcher

    monkeypatch.setattr("rune.daemon.cron_dispatcher.time.time",
                        lambda: datetime(2026, 10, 25, 9).timestamp())
    first, store = scheduled_job
    store.create_cron_job(name="second", schedule="0 9 * * *", command="true")
    get_config().cron_execution.max_concurrent = 1
    calls = []

    async def fail_one(job, **kwargs):
        calls.append(job.id)
        if job.id == first.id:
            raise OSError("disk unavailable")

    monkeypatch.setattr(cron, "execute_cron_job", fail_one)
    dispatcher = CronDispatcher(tmp_path / "schedule.db")
    await dispatcher.tick()
    for _ in range(5):
        await asyncio.sleep(0)
    assert len(calls) == 2 and len(set(calls)) == 2
    await dispatcher.stop()


async def test_three_simulated_days_with_restarts_do_not_duplicate_or_grow_queue(scheduled_job, tmp_path, monkeypatch):
    from rune.capabilities import cron
    from rune.daemon.cron_dispatcher import CronDispatcher

    _, store = scheduled_job
    hourly = store.create_cron_job(name="hourly", schedule="0 * * * *", command="true")
    clock = [datetime(2026, 10, 25).timestamp()]
    monkeypatch.setattr("rune.daemon.cron_dispatcher.time.time", lambda: clock[0])
    seen = set()

    async def execute(job, *, scheduled_minute):
        key = job.id, scheduled_minute
        assert key not in seen
        seen.add(key)

    monkeypatch.setattr(cron, "execute_cron_job", execute)
    path = tmp_path / "schedules.db"
    dispatcher = CronDispatcher(path)
    try:
        for minute in range(3 * 24 * 60):
            await dispatcher.tick()
            await asyncio.sleep(0)
            assert len(dispatcher._active) <= 2
            if minute % 137 == 0:
                await dispatcher.stop()
                dispatcher = CronDispatcher(path)
                await dispatcher.tick()
                await asyncio.sleep(0)
            clock[0] += 60
        assert sum(job == hourly for job, _ in seen) == 72
        assert len(seen) == 75
        assert dispatcher._db.execute("SELECT count(*) FROM schedules").fetchone()[0] == 2
    finally:
        await dispatcher.stop()


def test_daemon_logs_rotate_without_an_unbounded_stderr_copy(tmp_path):
    import subprocess
    import sys

    output = tmp_path / "daemon.log"
    source = """
import logging
from rune.utils.logger import get_logger
log = get_logger('rotation-test')
handlers = logging.getLogger().handlers
assert len(handlers) == 1
assert handlers[0].maxBytes == 10 * 1024 * 1024
assert handlers[0].backupCount == 3
handlers[0].maxBytes = 1024
for i in range(100):
    log.error('record', value='x' * 100)
"""
    subprocess.run([sys.executable, "-c", source],
                   env={**os.environ, "RUNE_LOG_FILE": str(output)}, check=True,
                   capture_output=True, timeout=10)
    files = list(tmp_path.glob("daemon.log*"))
    assert len(files) == 4 and sum(path.stat().st_size for path in files) <= 4096


async def test_explicit_routine_timeout_is_not_clamped_to_the_legacy_default(scheduled_job, monkeypatch):
    from rune.capabilities import cron
    from rune.proactive.routine import RoutinePolicy

    _, store = scheduled_job
    job_id = store.create_cron_job(name="long-report", schedule="* * * * *", command=cron._pack_command(
        "", "report", "", "", RoutinePolicy(timeout_seconds=600)))
    timeout = asyncio.timeout
    limits = []

    def observe(seconds):
        limits.append(seconds)
        return timeout(seconds)

    monkeypatch.setattr(asyncio, "timeout", observe)
    monkeypatch.setattr(cron, "_execute_goal_job", AsyncMock(return_value={"status": "verified", "verified": True}))
    await cron.execute_cron_job(cron._row_to_cronjob(store.get_cron_job(job_id)))
    assert limits == [600]
    get_config().cron_execution.default_timeout_seconds = 20
    legacy, _ = scheduled_job
    assert cron._row_to_cronjob(store.get_cron_job(legacy.id)).policy.timeout_seconds == 20


def test_invalid_runtime_field_preserves_valid_spending_limits(tmp_path):
    from rune.config.loader import _validate_salvaging_valid_sections

    config = _validate_salvaging_valid_sections({"runtime": {
        "dailyCostUsd": 1, "monthlyCostUsd": 10, "minFreeDiskMb": -1,
    }}, tmp_path / "config.yaml")
    assert config.runtime.daily_cost_usd == 1
    assert config.runtime.monthly_cost_usd == 10
    assert config.runtime.min_free_disk_mb == 256


@pytest.mark.parametrize("value", ["invalid", -1, float("nan")])
def test_invalid_spending_limit_blocks_paid_requests(value, tmp_path):
    from rune.config.loader import _validate_salvaging_valid_sections

    config = _validate_salvaging_valid_sections({"runtime": {
        "dailyCostUsd": value, "monthlyCostUsd": 10,
    }}, tmp_path / "config.yaml")
    assert config.runtime.daily_cost_usd == 0
    assert config.runtime.monthly_cost_usd == 10


def test_unreadable_config_cannot_reset_limits_to_unlimited(tmp_path, monkeypatch):
    from rune.config import loader

    path = tmp_path / "config.yaml"
    monkeypatch.setattr(loader, "_find_config_file", lambda: path)
    path.write_text("runtime: [broken")
    with pytest.raises(ValueError, match="Could not load"):
        loader.load_config(force=True)
    path.write_text("runtime:\n  dailyCostUsd: 1\n")
    assert loader.load_config(force=True).runtime.daily_cost_usd == 1
    path.write_text("runtime: [broken")
    assert loader.load_config(force=True).runtime.daily_cost_usd == 1


def test_stale_pid_cannot_trigger_a_stop_signal(tmp_path, monkeypatch):
    from unittest.mock import Mock

    import typer

    from rune.cli import main
    from rune.utils.paths import rune_home

    (rune_home() / "daemon.pid").write_text(str(os.getpid()))
    signal = Mock()
    monkeypatch.setattr(os, "kill", signal)
    with pytest.raises(typer.Exit) as stopped:
        main.daemon_stop()
    assert stopped.value.exit_code == 0
    signal.assert_not_called()


def test_stop_uses_control_socket_and_leaves_cleanup_to_owner(monkeypatch, tmp_path):
    from unittest.mock import Mock

    from rune.cli import main
    from rune.daemon import control

    owner = iter([123, None])
    monkeypatch.setattr(main, "_daemon_pid", lambda: next(owner))
    pid = tmp_path / "rune.pid"
    pid.write_text("456")
    monkeypatch.setattr(main, "_daemon_paths", lambda: (str(tmp_path / "rune.sock"), str(pid)))
    request = Mock(return_value={"success": True, "pid": 123})
    monkeypatch.setattr(control, "request", request)
    kill = Mock()
    monkeypatch.setattr(os, "kill", kill)
    main.daemon_stop()
    request.assert_called_once_with(str(tmp_path / "rune.sock"), "stop", expected_pid=123)
    kill.assert_not_called()
    assert pid.read_text() == "456"


async def test_stop_request_rejects_a_different_instance():
    daemon = RuneDaemon(install_signal_handlers=False)
    result = await daemon._dispatch({"command": "stop", "params": {"expected_pid": -1}})
    assert result["success"] is False
    assert not daemon._stop_requested.is_set()


async def test_stop_during_startup_cancels_initialization(monkeypatch):
    daemon = RuneDaemon(install_signal_handlers=False)
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def stalled_start():
        await daemon._acquire_process_lock()
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    cleanup = AsyncMock()
    monkeypatch.setattr(daemon, "start", stalled_start)
    monkeypatch.setattr(daemon, "_shutdown_subsystems", cleanup)
    task = asyncio.create_task(daemon.serve_forever())
    await entered.wait()
    daemon._stop_requested.set()
    await asyncio.wait_for(task, 1)
    assert cancelled.is_set() and daemon._lock_path is None
    cleanup.assert_awaited_once()


async def test_startup_timeout_releases_ownership(monkeypatch):
    from rune.daemon import lifecycle

    daemon = RuneDaemon(install_signal_handlers=False)
    monkeypatch.setattr(lifecycle, "STARTUP_TIMEOUT", .01)
    monkeypatch.setattr(daemon, "_load_env", lambda: None)
    monkeypatch.setattr(daemon, "_init_subsystems", asyncio.Event().wait)
    monkeypatch.setattr(daemon, "_shutdown_subsystems", AsyncMock())
    with pytest.raises(TimeoutError, match="startup"):
        await daemon.serve_forever()
    assert daemon._lock_path is None
    assert not Path(daemon._config["pid_file"]).exists()


async def test_cleanup_failure_does_not_skip_api_or_storage(monkeypatch):
    from unittest.mock import Mock

    daemon = RuneDaemon(install_signal_handlers=False)
    daemon._heartbeat_scheduler = SimpleNamespace(stop=AsyncMock(side_effect=OSError("scheduler failure")))
    daemon._api_server = SimpleNamespace(should_exit=False)
    daemon._cron_dispatcher = SimpleNamespace(stop=AsyncMock())
    daemon._memory_store = SimpleNamespace(close=Mock())
    monkeypatch.setattr(daemon, "_shutdown_browser_subsystems", AsyncMock())
    monkeypatch.setattr(daemon, "_shutdown_mcp_bridge", AsyncMock())
    monkeypatch.setattr("rune.daemon.gateway.get_gateway", lambda: None)
    monkeypatch.setattr("rune.memory.manager._manager", None)
    with pytest.raises(RuntimeError, match="scheduler"):
        await daemon._shutdown_subsystems()
    assert daemon._api_server.should_exit is True
    daemon._cron_dispatcher.stop.assert_awaited_once()
    daemon._memory_store.close.assert_called_once()


async def test_cleanup_timeout_and_cancelled_task_allow_later_steps():
    from rune.daemon.lifecycle import Cleanup

    cleanup = Cleanup(seconds=1)
    await cleanup.run("stalled", asyncio.Event().wait, timeout=.01)
    task = asyncio.create_task(asyncio.sleep(100))
    task.cancel()
    await cleanup.run("cancelled", lambda: task)
    finished = []
    await cleanup.run("remaining", lambda: finished.append(True))
    assert finished == [True]
    assert cleanup.errors == ["stalled", "cancelled"]


async def test_schedule_failure_and_recovery_are_visible_in_health():
    daemon = RuneDaemon({"api_enabled": False}, install_signal_handlers=False)
    daemon._running = True
    daemon._cron_dispatcher = object()
    scheduler = HeartbeatScheduler(interval_seconds=999)
    daemon._heartbeat_scheduler = scheduler
    callback = AsyncMock(side_effect=OSError("schedule DB unavailable"))
    scheduler.add_task("cron_goal_tick", "* * * * *", callback)
    await scheduler.start()
    try:
        await asyncio.sleep(0)
        await scheduler._tasks["cron_goal_tick"].running
        assert scheduler.is_running
        assert daemon.health()["checks"]["schedules"] is False
        callback.side_effect = None
        await scheduler._invoke(scheduler._tasks["cron_goal_tick"])
        assert daemon.health()["checks"]["schedules"] is True
        scheduler._tasks["cron_goal_tick"].last_success -= 3000
        assert daemon.health()["checks"]["schedules"] is False
    finally:
        await scheduler.stop()


async def test_failed_cleanup_keeps_lock_until_process_exit(monkeypatch):
    from rune.daemon.process_lock import get_lock_owner

    daemon = RuneDaemon(install_signal_handlers=False)
    await daemon._acquire_process_lock()
    monkeypatch.setattr(daemon, "_shutdown_subsystems", AsyncMock(side_effect=RuntimeError("cleanup failed")))
    try:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await daemon.stop()
        assert get_lock_owner() == os.getpid()
        with pytest.raises(RuntimeError):
            acquire_lock()
    finally:
        await daemon._release_process_lock()
