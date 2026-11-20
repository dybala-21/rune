"""A job run by code (the briefing) keeps running after one run of it stopped part way."""

import asyncio
import subprocess
import sys

import pytest

from rune.capabilities import cron
from rune.memory.store import MemoryStore
from rune.proactive import execution_store
from rune.proactive.execution_store import ExecutionStore
from rune.proactive.routine import RoutinePolicy

DONE = {"status": "completed", "success": True, "verified": True, "output": "Good morning."}


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    clock = [1_000_000.0]
    monkeypatch.setattr("rune.proactive.routine.time.time", lambda: clock[0])
    store = MemoryStore(tmp_path / "memory.db")
    monkeypatch.setattr(cron, "_get_store", lambda: store)

    class Jobs:
        def make(self, kind="briefing"):
            policy = RoutinePolicy.model_validate({"notify": "always"}).bind_workspace()
            command = cron._pack_command("", "" if kind else "write the report", "", "", policy, kind=kind)
            return store.create_cron_job(name="Morning briefing", schedule="0 7 * * *", command=command)

        async def run(self, job_id):
            clock[0] += 60  # the next occurrence
            return await cron.execute_cron_job(cron._row_to_cronjob(store.get_cron_job(job_id)))

        def last(self, job_id):
            records = ExecutionStore(tmp_path / "data" / "routine-executions.db")
            try:
                return records.recent(f"cron:{job_id}:")[0]
            finally:
                records.close()

        store_ = store

    yield Jobs()
    store.close()


def _briefing(monkeypatch, *outcomes):
    """The briefing's run, answering with each outcome in turn; returns the calls made."""
    calls, left = [], list(outcomes)

    async def run(job=None):
        calls.append(job)
        outcome = left.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return await outcome(job)
        return dict(outcome)

    monkeypatch.setattr("rune.briefing.job.run_briefing", run)
    return calls


async def test_a_code_job_that_ran_out_of_time_runs_again_at_its_next_occurrence(jobs, monkeypatch):
    calls = _briefing(monkeypatch, TimeoutError(), DONE)
    job_id = jobs.make()
    await jobs.run(job_id)
    stopped = jobs.last(job_id)
    assert not stopped["blocked"] and stopped["result"]["status"] == "failed"
    assert "execution_unknown" not in stopped["result"]
    # Nothing for the owner to inspect: it only reads and reports.
    assert "nspect" not in stopped["result"]["error"]
    assert await jobs.run(job_id) is None
    assert len(calls) == 2 and jobs.last(job_id)["result"]["status"] == "completed"


async def test_a_code_job_that_raised_says_only_what_kind_of_error_and_runs_again(jobs, monkeypatch):
    calls = _briefing(monkeypatch, RuntimeError("Subject: wire the money to account 12345"), DONE)
    job_id = jobs.make()
    await jobs.run(job_id)
    stopped = jobs.last(job_id)
    # What a mail or a page said must not reach the owner's chat through an error's wording.
    assert stopped["result"]["error"] == "RuntimeError" and "wire the money" not in str(stopped["result"])
    assert not stopped["blocked"]
    await jobs.run(job_id)
    assert len(calls) == 2


async def test_a_code_job_whose_settings_changed_while_it_ran_runs_again(jobs, monkeypatch):
    job_id = jobs.make()

    async def moved(job):
        # The owner moves it to another channel while it is running.
        policy = RoutinePolicy.model_validate({"notify": "always"}).bind_workspace()
        jobs.store_.update_cron_job(job.id, command=cron._pack_command("", "", "slack", "", policy, kind="briefing"))
        return dict(DONE)

    calls = _briefing(monkeypatch, moved, DONE)
    await jobs.run(job_id)
    assert not jobs.last(job_id)["blocked"]
    await jobs.run(job_id)
    assert len(calls) == 2 and jobs.last(job_id)["result"]["status"] == "completed"


async def test_a_code_job_stopped_with_the_machine_runs_again_after_it(jobs, monkeypatch):
    started = asyncio.Event()

    async def waits(job):
        started.set()
        await asyncio.Event().wait()

    calls = _briefing(monkeypatch, waits, DONE)
    job_id = jobs.make()
    running = asyncio.create_task(jobs.run(job_id))
    await asyncio.wait_for(started.wait(), 2)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert not jobs.last(job_id)["blocked"]
    await jobs.run(job_id)
    assert len(calls) == 2


@pytest.mark.parametrize("error", [TimeoutError(), RuntimeError("disk full at /var/reports")])
async def test_an_agent_job_that_stopped_part_way_still_waits_for_the_owner(jobs, monkeypatch, error):
    async def stops(job):
        raise error

    monkeypatch.setattr(cron, "_execute_goal_job", stops)
    job_id = jobs.make(kind="")
    await jobs.run(job_id)
    stopped = jobs.last(job_id)
    # It may have changed files or sent something: the owner looks before it runs again.
    assert stopped["blocked"] and stopped["result"]["execution_unknown"]
    if isinstance(error, RuntimeError):
        assert stopped["result"]["error"] == "disk full at /var/reports"  # its own error is said in full, as before
    assert await jobs.run(job_id) == "busy"


def _ended_pid() -> int:
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    return done.pid


@pytest.mark.parametrize("kind", ["briefing", ""])
async def test_a_job_whose_process_died_mid_run_runs_again_only_if_code_runs_it(jobs, monkeypatch, tmp_path, kind):
    calls = _briefing(monkeypatch, DONE)

    async def agent(job):
        calls.append(job)
        return dict(DONE)

    monkeypatch.setattr(cron, "_execute_goal_job", agent)
    job_id = jobs.make(kind=kind)
    # Yesterday's run: claimed, and then the machine lost power. No result was written.
    records = ExecutionStore(tmp_path / "data" / "routine-executions.db")
    assert records.claim(f"cron:{job_id}:1", "a", 10, resource=f"cron:{job_id}") == "claimed"
    records._db.execute("UPDATE operations SET owner_pid = ? WHERE id = ?", (_ended_pid(), f"cron:{job_id}:1"))
    records.close()
    outcome = await jobs.run(job_id)
    assert (outcome, len(calls)) == ((None, 1) if kind else ("busy", 0))


def test_a_claim_left_by_a_process_that_ended_is_taken_over_for_a_code_job_only(tmp_path):
    path = tmp_path / "runs.db"
    records = ExecutionStore(path)
    assert records.claim("one", "a", 10, resource="briefing") == "claimed"
    # The process died mid-run: no result was ever written.
    records._db.execute("UPDATE operations SET owner_pid = ? WHERE id = 'one'", (_ended_pid(),))
    assert records.claim("two", "a", 10, resource="briefing") == "busy"
    assert records.claim("two", "a", 10, resource="briefing", rerunnable=True) == "claimed"
    left = records.get("one")[1]
    assert left["status"] == "failed" and not left["success"]
    records.finish("two", {"status": "completed"})
    assert "two" not in execution_store._claimed_here  # nothing is kept for a run that is over
    records.close()


def test_a_claim_held_by_another_process_that_is_still_running_is_left_alone(tmp_path):
    records = ExecutionStore(tmp_path / "runs.db")
    assert records.claim("one", "a", 10, resource="briefing") == "claimed"
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        records._db.execute("UPDATE operations SET owner_pid = ? WHERE id = 'one'", (other.pid,))
        assert records.claim("two", "a", 10, resource="briefing", rerunnable=True) == "busy"
        assert records.get("one")[1] is None
    finally:
        other.kill()
        other.wait()
        records.close()


def test_a_claim_from_before_a_restart_into_the_same_pid_is_taken_over(tmp_path, monkeypatch):
    records = ExecutionStore(tmp_path / "runs.db")
    assert records.claim("one", "a", 10, resource="briefing") == "claimed"
    # Still running here: nothing is taken from it.
    assert records.claim("two", "a", 10, resource="briefing", rerunnable=True) == "busy"
    # A machine that boots into the same pid: the claim is this pid's, but this process never made it.
    monkeypatch.setattr(execution_store, "_claimed_here", set())
    assert records.claim("two", "a", 10, resource="briefing", rerunnable=True) == "claimed"
    records.close()


def test_a_code_job_held_by_an_earlier_stopped_run_is_released_but_not_one_awaiting_approval(tmp_path):
    records = ExecutionStore(tmp_path / "runs.db")
    assert records.claim("one", "a", 10, resource="briefing") == "claimed"
    # How a stopped run was recorded before: the hold stayed.
    records.finish("one", {"status": "interrupted", "execution_unknown": True})
    assert records.claim("two", "a", 10, resource="briefing", rerunnable=True) == "claimed"
    assert records.get("one")[1]["status"] == "interrupted"  # its record is kept as it was
    records.finish("two", {"status": "needs_approval"})
    assert records.claim("three", "a", 10, resource="briefing", rerunnable=True) == "busy"
    records.close()


def test_a_claim_whose_owner_is_unknown_is_left_alone(tmp_path):
    records = ExecutionStore(tmp_path / "runs.db")
    assert records.claim("one", "a", 10, resource="briefing") == "claimed"
    records._db.execute("UPDATE operations SET owner_pid = NULL WHERE id = 'one'")
    assert records.claim("two", "a", 10, resource="briefing", rerunnable=True) == "busy"
    records.close()
