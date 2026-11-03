"""An external scheduler fires occurrences idempotently and the local clock stays silent."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from rune.capabilities import cron
from rune.daemon import external_schedule as ext
from rune.memory.store import MemoryStore


@pytest.fixture
def job(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    store = MemoryStore(tmp_path / "memory.db")
    monkeypatch.setattr(cron, "_get_store", lambda: store)
    job_id = store.create_cron_job(name="digest", schedule="0 9 * * *", command=cron._pack_command("", "Summarise", "", ""))
    ext._inflight.clear()
    ext._not_run.clear()
    monkeypatch.setattr(ext, "_stopping", False)
    yield job_id, store
    store.close()


def _minute(hour, minute=0):
    return int(datetime(2026, 10, 25, hour, minute).timestamp() // 60)


def _clock(monkeypatch, minute):
    # cron.py imports time inside functions, so pin the module-level clock everywhere.
    monkeypatch.setattr("time.time", lambda: minute * 60 + 5)


async def _drain():
    while ext._inflight:
        await asyncio.gather(*ext._inflight.values(), return_exceptions=True)


async def test_fire_runs_once_and_reports_the_verified_result(job, monkeypatch):
    job_id, store = job
    due = _minute(9)
    _clock(monkeypatch, due)
    worker = AsyncMock(return_value={"status": "verified", "success": True, "verified": True, "output": "Sent"})
    monkeypatch.setattr(cron, "_execute_goal_job", worker)

    first = await ext.fire(job_id, due)
    assert first["state"] == "accepted" and first["occurrenceId"] == f"cron:{job_id}:{due}"
    again = await ext.fire(job_id, due)  # at-least-once delivery while the first is running
    await _drain()
    repeat = await ext.fire(job_id, due)
    assert repeat["replayed"] and repeat["state"] == "verified"
    assert worker.await_count == 1 and store.get_cron_job(job_id)["run_count"] == 1
    done = ext.occurrence(job_id, due)
    assert done["result"]["verified"] is True and done["result"]["output"] == "Sent"
    assert again["state"] in {"accepted", "verified"}


@pytest.mark.parametrize(("minute", "message"), [
    (lambda due: due + 1, "future"),
    (lambda due: due - 600, "misfire grace"),
    (lambda due: due - 1, "not on this job's schedule"),
])
async def test_fire_refuses_occurrences_that_should_not_run(job, monkeypatch, minute, message):
    job_id, _ = job
    due = _minute(9)
    _clock(monkeypatch, due)
    with pytest.raises(ValueError, match=message):
        await ext.fire(job_id, minute(due))
    assert not ext._inflight


@pytest.mark.parametrize(("job_id", "minute"), [("", 1), ("x", -1), ("x", "12"), ("x", True), (None, 1)])
def test_bad_identifiers_are_rejected(job_id, minute):
    with pytest.raises(ValueError):
        ext.occurrence(job_id, minute)


async def test_unknown_paused_and_exhausted_jobs_start_nothing(job, monkeypatch):
    job_id, store = job
    due = _minute(9)
    _clock(monkeypatch, due)
    with pytest.raises(ValueError, match="Unknown cron job"):
        await ext.fire("missing", due)
    store.update_cron_job(job_id, enabled=False)
    with pytest.raises(ValueError, match="not active"):
        await ext.fire(job_id, due)


@pytest.mark.parametrize(("reason", "retryable"), [("busy", True), ("rate_limited", True),
                                                  ("expired", False), ("changed", False)])
async def test_skipped_occurrences_report_the_real_reason(job, monkeypatch, reason, retryable):
    job_id, _ = job
    due = _minute(9)
    _clock(monkeypatch, due)
    execute = AsyncMock(return_value=reason)
    monkeypatch.setattr(cron, "execute_cron_job", execute)
    await ext.fire(job_id, due)
    await _drain()
    state = ext.occurrence(job_id, due)
    assert state["state"] == "not_run" and state["reason"] == reason and state["retryable"] is retryable
    again = await ext.fire(job_id, due)
    await _drain()
    assert again.get("replayed") is (not retryable)
    assert execute.await_count == (2 if retryable else 1)


async def test_busy_is_reported_when_the_previous_run_still_holds_the_job(job, monkeypatch):
    from rune.proactive.execution_store import ExecutionStore
    from rune.utils.paths import rune_data

    job_id, _ = job
    due = _minute(9)
    _clock(monkeypatch, due)
    holder = ExecutionStore(rune_data() / "routine-executions.db")
    assert holder.claim(f"cron:{job_id}:{due - 1440}", "x", 1000, resource=f"cron:{job_id}") == "claimed"
    holder.close()
    monkeypatch.setattr(cron, "_execute_goal_job", AsyncMock())
    await ext.fire(job_id, due)
    await _drain()
    assert ext.occurrence(job_id, due)["reason"] == "busy"


async def test_failure_before_claim_stays_refireable(job, monkeypatch):
    job_id, _ = job
    due = _minute(9)
    _clock(monkeypatch, due)
    execute = AsyncMock(side_effect=[OSError("disk"), None])
    monkeypatch.setattr(cron, "execute_cron_job", execute)
    await ext.fire(job_id, due)
    await _drain()
    assert ext.occurrence(job_id, due)["reason"] == "failed_to_start"
    await ext.fire(job_id, due)
    await _drain()
    assert execute.await_count == 2


async def test_shutdown_cancels_in_flight_and_refuses_new_fires(job, monkeypatch):
    job_id, _ = job
    due = _minute(9)
    _clock(monkeypatch, due)
    started, cancelled = asyncio.Event(), []

    async def slow(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    monkeypatch.setattr(cron, "execute_cron_job", slow)
    await ext.fire(job_id, due)
    await started.wait()
    await ext.shutdown()
    assert cancelled == [True] and not ext._inflight
    with pytest.raises(ValueError, match="shutting down"):
        await ext.fire(job_id, due)


async def test_external_mode_silences_the_local_dispatcher(job, tmp_path, monkeypatch):
    from rune.daemon.cron_dispatcher import CronDispatcher

    clock = [datetime(2026, 10, 25, 8, 59).timestamp()]
    monkeypatch.setattr("rune.daemon.cron_dispatcher.time.time", lambda: clock[0])
    execute = AsyncMock()
    monkeypatch.setattr(cron, "execute_cron_job", execute)
    monkeypatch.setenv("RUNE_EXTERNAL_SCHEDULER", "1")
    dispatcher = CronDispatcher(tmp_path / "schedules.db")
    for _ in range(3):
        await dispatcher.tick()
        clock[0] += 60
        await asyncio.sleep(0)
    execute.assert_not_awaited()
    # Switching back does not replay the minutes the external scheduler owned.
    monkeypatch.delenv("RUNE_EXTERNAL_SCHEDULER")
    await dispatcher.tick()
    await asyncio.sleep(0)
    execute.assert_not_awaited()
    await dispatcher.stop()


async def test_cron_list_exposes_a_definition_fingerprint(job, monkeypatch):
    from rune.api.handlers import cron as api
    from rune.proactive.routine import task_fingerprint

    job_id, store = job
    monkeypatch.setattr(api, "get_memory_store", lambda: store)
    listed = await api.list_cron_jobs()
    info = next(j for j in listed.jobs if j.id == job_id)
    assert info.fingerprint == task_fingerprint(cron._row_to_cronjob(store.get_cron_job(job_id)))
