"""A long task runs as steps parked in between, picks up where it left off, and tells the owner
only when it matters."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rune.capabilities import cron, task_tools
from rune.daemon import external_schedule as ext
from rune.memory.store import MemoryStore
from rune.proactive import long_task


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    memory = MemoryStore(tmp_path / "memory.db")
    monkeypatch.setattr(cron, "_get_store", lambda: memory)
    ext._inflight.clear()
    ext._not_run.clear()
    monkeypatch.setattr(ext, "_stopping", False)
    yield memory
    memory.close()


@pytest.fixture
def told(monkeypatch):
    sent = AsyncMock()
    monkeypatch.setattr(cron, "_send_to_channel", sent)
    return sent


def now_minute():
    return int(time.time() // 60)


async def start(store, **values):
    answer = await task_tools.task_start(task_tools.TaskStartParams(name="research", goal="Read every review and sum them up",
                                                        **values))
    assert answer.success, answer.error
    return answer.metadata["id"]


def state(job):
    """The task's state without the record of which step it follows from."""
    return {k: v for k, v in (job.task or {}).items() if k != "step"}


def job_of(store, job_id):
    return cron._row_to_cronjob(store.get_cron_job(job_id))


async def drain():
    while ext._inflight:
        await asyncio.gather(*ext._inflight.values(), return_exceptions=True)


async def step(store, job_id, monkeypatch, result):
    """Fire the task's next step, as the host does, with the step's run ending in `result`."""
    worker = AsyncMock(return_value=result)
    monkeypatch.setattr(cron, "_execute_goal_job", worker)
    wake = job_of(store, job_id).wake_at
    fired = await ext.fire(job_id, wake)
    assert fired["state"] == "accepted", fired
    await drain()
    return worker


def parked(note, minutes=30, notify=False):
    return {"status": "parked", "success": True, "verified": False, "output": note,
            "park": {"resume_at": now_minute() + minutes, "notify": notify}}


async def test_a_task_starts_at_once_with_this_machine_s_limits(store, monkeypatch):
    monkeypatch.setenv("RUNE_TASK_STEP_LIMIT", "10")
    monkeypatch.setenv("RUNE_TASK_TOKEN_LIMIT", "300000")
    job_id = await start(store, steps=40, total_tokens=5_000_000, step_minutes=20)
    row = store.get_cron_job(job_id)
    job = job_of(store, job_id)
    assert row["schedule"] == long_task.SCHEDULE and row["max_runs"] == 10, "capped by the machine"
    assert job.policy.total_tokens == 300_000 and job.policy.timeout_seconds == 1200
    assert job.wake_at == now_minute(), "its first step is due now"
    assert (await task_tools.task_start(task_tools.TaskStartParams(name="research", goal="again"))).success is False


async def test_steps_park_quietly_pick_up_the_note_and_end_with_the_answer(store, told, monkeypatch):
    job_id = await start(store)
    first = await step(store, job_id, monkeypatch, parked("Read 1-100. Next: 101-200."))
    assert told.await_count == 0, "a quiet step tells no one"
    job = job_of(store, job_id)
    assert job.status == "active" and job.wake_at == now_minute() + 30
    assert "step 1 of at most 20" in first.await_args.args[0].preamble

    later = (now_minute() + 30) * 60 + 5
    monkeypatch.setattr("time.time", lambda: later)
    second = await step(store, job_id, monkeypatch, {"status": "completed", "success": True, "verified": False,
                                                     "task_done": True, "output": "Most reviews praise the battery."})
    told_step = second.await_args.args[0]
    assert "step 2" in told_step.preamble and "Read 1-100. Next: 101-200." in told_step.continuation, \
        "the next step reads the note"
    job = job_of(store, job_id)
    assert job.status == "paused" and state(job) == {"wake_at": None, "ended": "done"}
    assert told.await_args.args[2] == "Most reviews praise the battery."


async def test_a_step_that_finds_something_tells_the_owner(store, told, monkeypatch):
    job_id = await start(store)
    await step(store, job_id, monkeypatch, parked("Half way: the price dropped 20%.", notify=True))
    assert told.await_args.args[2] == "Half way: the price dropped 20%."
    assert job_of(store, job_id).status == "active"


async def test_the_last_allowed_step_ends_the_task_and_says_where_it_got(store, told, monkeypatch):
    job_id = await start(store, steps=1)
    await step(store, job_id, monkeypatch, parked("Read 1-100."))
    job = job_of(store, job_id)
    assert job.status == "paused" and job.task["ended"] == "step_limit"
    assert "Stopped after 1 steps" in told.await_args.args[2] and "Read 1-100." in told.await_args.args[2]


@pytest.mark.parametrize(("result", "ended", "says"), [
    ({"status": "failed", "success": False, "verified": False, "output": "", "error": "site down"}, "failed", "site down"),
    ({"status": "budget_exhausted", "success": False, "verified": False, "output": "", "error": "no tokens left"},
     "budget_exhausted", "budget"),
])
async def test_a_step_that_fails_stops_the_task_and_tells_the_owner(store, told, monkeypatch, result, ended, says):
    job_id = await start(store)
    await step(store, job_id, monkeypatch, result)
    job = job_of(store, job_id)
    assert job.status == "paused" and job.task["ended"] == ended and job.wake_at is None
    assert says in told.await_args.args[2]


async def test_a_step_waiting_for_approval_holds_the_task_until_it_is_decided(store, told, monkeypatch):
    job_id = await start(store)
    await step(store, job_id, monkeypatch, {"status": "needs_approval", "success": False, "verified": False,
                                            "output": "", "error": "approval", "pending_approval": {"action": {}}})
    job = job_of(store, job_id)
    assert job.status == "active" and job.wake_at is None, "nothing fires it but the approval"
    assert "approval" in told.await_args.args[2]


async def test_the_host_fires_a_task_s_next_step_late_but_only_that_minute(store, monkeypatch):
    job_id = await start(store)
    await step(store, job_id, monkeypatch, parked("next", minutes=5))
    wake = job_of(store, job_id).wake_at
    monkeypatch.setattr("time.time", lambda: (wake + 600) * 60)  # the VM was down for ten hours
    with pytest.raises(ValueError, match="next step"):
        await ext.fire(job_id, wake + 1)
    monkeypatch.setattr(cron, "_execute_goal_job", AsyncMock(return_value=parked("again")))
    assert (await ext.fire(job_id, wake))["state"] == "accepted", "no misfire grace for a task's step"
    await drain()
    monkeypatch.setattr("time.time", lambda: (wake - 1) * 60)
    with pytest.raises(ValueError, match="future"):
        await ext.fire(job_id, job_of(store, job_id).wake_at)


async def test_cron_list_reports_the_next_step_to_the_host(store, monkeypatch):
    from rune.api.handlers.cron import list_cron_jobs
    monkeypatch.setattr("rune.api.handlers.cron.get_memory_store", lambda: store)
    job_id = await start(store)
    listed = (await list_cron_jobs()).model_dump(by_alias=True)["jobs"]
    [info] = [j for j in listed if j["id"] == job_id]
    assert info["schedule"] == "@task" and info["wakeAt"] == now_minute()


async def test_the_local_dispatcher_fires_a_due_step(store, monkeypatch, tmp_path):
    from rune.daemon.cron_dispatcher import CronDispatcher
    monkeypatch.delenv("RUNE_EXTERNAL_SCHEDULER", raising=False)
    job_id = await start(store)
    run = AsyncMock(return_value=None)
    monkeypatch.setattr("rune.capabilities.cron.execute_cron_job", run)
    dispatcher = CronDispatcher(tmp_path / "schedule.db")
    await dispatcher.tick()
    await asyncio.gather(*dispatcher._active.values(), return_exceptions=True)
    assert run.await_args.kwargs["scheduled_minute"] == now_minute()
    assert run.await_args.args[0].id == job_id


def test_parking_outside_a_task_s_step_is_refused():
    with pytest.raises(ValueError, match="only inside a long task"):
        long_task.park("x", 5, False)


async def test_a_step_that_parks_ends_as_parked_with_its_note(tmp_path, monkeypatch):
    from rune.agent.background import BackgroundTask, run_background
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))

    class Loop:
        def __init__(self, config):
            pass

        def set_approval_callback(self, callback):
            pass

        async def run(self, goal, **kwargs):
            answer = await task_tools.task_park(task_tools.TaskParkParams(progress="Did A. Next: B.", resume_in_minutes=45,
                                                              notify=True))
            assert answer.success, answer.error
            return SimpleNamespace(reason="cancelled", final_step=3, answer="", error=None)

    result = await run_background(BackgroundTask(goal="g", source="task", workspace=str(tmp_path)), loop_factory=Loop)
    assert result["status"] == "parked" and result["output"] == "Did A. Next: B."
    assert result["park"] == {"resume_at": now_minute() + 45, "notify": True}

    class Scheduled(Loop):
        async def run(self, goal, **kwargs):
            refused = await task_tools.task_park(task_tools.TaskParkParams(progress="x"))
            assert not refused.success, "a cron job's run can't park"
            return SimpleNamespace(reason="completed", final_step=1, answer="done", error=None)

    plain = await run_background(BackgroundTask(goal="g", source="cron", workspace=str(tmp_path)), loop_factory=Scheduled)
    assert plain["status"] != "parked"


def test_a_park_is_bounded_to_a_week_and_only_the_step_itself_parks():
    from rune.agent.run_control import RunControl, control_scope

    step_control, subagent = RunControl("step"), RunControl("subagent")
    slot, token = long_task.park_slot(step_control)
    try:
        with control_scope(subagent), pytest.raises(ValueError, match="Only the step itself"):
            long_task.park("x", 5, False)
        with control_scope(step_control):
            assert long_task.park("x", 10**9, False) == now_minute() + long_task.MAX_PARK_MINUTES
        assert step_control.state == "stopped"
    finally:
        long_task.release_slot(token)


async def test_changing_a_parked_task_keeps_its_next_step(store, monkeypatch):
    job_id = await start(store)
    await step(store, job_id, monkeypatch, parked("next", minutes=20))
    wake = job_of(store, job_id).wake_at
    changed = await cron.cron_update(cron.CronUpdateParams(job_id=job_id, notify_channel="telegram",
                                                           policy=cron.RoutinePolicy(total_tokens=900_000)))
    assert changed.success, changed.error
    job = job_of(store, job_id)
    assert job.wake_at == wake and job.notify_channel == "telegram" and job.policy.total_tokens == 900_000


async def test_turning_a_task_off_mid_step_holds_it_and_turning_it_on_goes_on(store, told, monkeypatch):
    job_id = await start(store)

    async def turned_off(job):
        store.update_cron_job(job_id, enabled=False)
        return {"status": "interrupted", "success": False, "verified": False, "execution_unknown": True,
                "output": "", "error": "The task changed, stopped or expired."}

    monkeypatch.setattr(cron, "_execute_goal_job", turned_off)
    await ext.fire(job_id, job_of(store, job_id).wake_at)
    await drain()
    job = job_of(store, job_id)
    assert state(job) == {"wake_at": None, "paused": True} and told.await_count == 0, "the owner did it: not over"
    refused = await cron.cron_update(cron.CronUpdateParams(job_id=job_id, enabled=True))
    assert not refused.success and "reconcile" in refused.error, "its step may have done something: check first"

    from rune.proactive.execution_store import ExecutionStore
    from rune.utils.paths import rune_data
    records = ExecutionStore(rune_data() / "routine-executions.db")
    [last] = records.recent(f"cron:{job_id}:", limit=1)
    assert records.reconcile(f"cron:{job_id}", last["id"], "checked")
    records.close()
    assert (await cron.cron_update(cron.CronUpdateParams(job_id=job_id, enabled=True))).success
    job = job_of(store, job_id)
    assert job.status == "active" and job.wake_at == now_minute() + 1, "after the minute its last step has a record for"


async def test_a_finished_task_with_steps_left_can_be_turned_on_again(store, told, monkeypatch):
    job_id = await start(store, steps=3)
    await step(store, job_id, monkeypatch, {"status": "failed", "success": False, "verified": False,
                                            "output": "", "error": "site down"})
    assert job_of(store, job_id).status == "paused"
    assert (await cron.cron_update(cron.CronUpdateParams(job_id=job_id, enabled=True))).success
    assert job_of(store, job_id).wake_at == now_minute() + 1
    over = await cron.cron_update(cron.CronUpdateParams(job_id=job_id, max_runs=10**4))
    assert not over.success and "at most" in over.error, "the machine's step limit can't be raised past"


async def test_the_machine_s_step_limit_holds_at_every_step(store, told, monkeypatch):
    job_id = await start(store, steps=10)
    monkeypatch.setenv("RUNE_TASK_STEP_LIMIT", "1")  # lowered, say for a smaller plan, after it started
    await step(store, job_id, monkeypatch, parked("Read 1-100."))
    assert job_of(store, job_id).task["ended"] == "step_limit"


async def test_a_step_cut_off_by_a_restart_tells_the_owner_once_the_machine_is_back(store, told, monkeypatch):
    job_id = await start(store)

    async def cut_off(job):
        raise asyncio.CancelledError

    monkeypatch.setattr(cron, "_execute_goal_job", cut_off)
    await ext.fire(job_id, job_of(store, job_id).wake_at)
    await drain()
    job = job_of(store, job_id)
    assert job.task["ended"] == "interrupted" and "machine stopped" in job.task["notice"]
    assert told.await_count == 0, "not while the machine is stopping"
    monkeypatch.setattr("rune.daemon.gateway.get_gateway", lambda: object())
    await cron.tell_task_notices()
    assert "machine stopped" in told.await_args.args[2]
    assert "notice" not in job_of(store, job_id).task
    await cron.tell_task_notices()
    assert told.await_count == 1, "once"


async def test_a_failure_after_the_step_is_recorded_never_records_it_twice(store, told, monkeypatch):
    job_id = await start(store)
    real = store.update_cron_job

    def locked(*args, **kwargs):
        raise RuntimeError("locked")

    monkeypatch.setattr(store, "update_cron_job", locked)
    await step(store, job_id, monkeypatch, parked("note"))
    monkeypatch.setattr(store, "update_cron_job", real)
    state = ext.occurrence(job_id, now_minute())
    assert state["state"] == "parked" and state["result"]["output"] == "note"


async def test_the_local_dispatcher_does_not_queue_a_step_that_already_ran(store, monkeypatch, tmp_path):
    from rune.daemon.cron_dispatcher import CronDispatcher
    from rune.proactive.execution_store import ExecutionStore
    from rune.utils.paths import rune_data
    monkeypatch.delenv("RUNE_EXTERNAL_SCHEDULER", raising=False)
    job_id = await start(store)
    records = ExecutionStore(rune_data() / "routine-executions.db")
    records.claim(f"cron:{job_id}:{now_minute()}", "f", 1000)
    records.close()
    run = AsyncMock(return_value=None)
    monkeypatch.setattr("rune.capabilities.cron.execute_cron_job", run)
    await CronDispatcher(tmp_path / "schedule.db").tick()
    assert run.await_count == 0


async def test_the_app_s_switch_turns_a_task_on_where_it_left_off(store, told, monkeypatch):
    from rune.api.handlers.cron import toggle_cron_job
    monkeypatch.setattr("rune.api.handlers.cron.get_memory_store", lambda: store)
    job_id = await start(store, steps=3)
    await step(store, job_id, monkeypatch, {"status": "failed", "success": False, "verified": False,
                                            "output": "", "error": "site down"})
    assert (await toggle_cron_job(job_id)).enabled is True
    assert job_of(store, job_id).wake_at == now_minute() + 1


async def test_a_lowered_step_limit_never_blocks_turning_a_task_off(store, monkeypatch):
    job_id = await start(store, steps=40)
    monkeypatch.setenv("RUNE_TASK_STEP_LIMIT", "10")
    assert (await cron.cron_update(cron.CronUpdateParams(job_id=job_id, enabled=False))).success
    assert (await cron.cron_update(cron.CronUpdateParams(job_id=job_id, goal="A narrower goal"))).success


async def test_kept_messages_go_once_however_often_they_are_looked_for(store, told, monkeypatch):
    job_ids = []
    for name in ("one", "two"):
        answer = await task_tools.task_start(task_tools.TaskStartParams(name=name, goal="g"))
        job_ids.append(answer.metadata["id"])

    async def cut_off(job):
        raise asyncio.CancelledError

    monkeypatch.setattr(cron, "_execute_goal_job", cut_off)
    for job_id in job_ids:
        await ext.fire(job_id, job_of(store, job_id).wake_at)
        await drain()
    monkeypatch.setattr("rune.daemon.gateway.get_gateway", lambda: object())

    async def slow(*args):
        await asyncio.sleep(0.01)

    told.side_effect = slow
    await asyncio.gather(cron.tell_task_notices(), cron.tell_task_notices(), cron.tell_task_notices())
    await cron.tell_task_notices()
    assert told.await_count == 2, "one message per task"


async def test_a_step_whose_settling_failed_is_settled_later(store, told, monkeypatch):
    job_id = await start(store)
    real = store.update_cron_job

    def locked(*args, **kwargs):
        raise RuntimeError("locked")

    monkeypatch.setattr(store, "update_cron_job", locked)
    await step(store, job_id, monkeypatch, parked("note", minutes=40))
    monkeypatch.setattr(store, "update_cron_job", real)
    assert job_of(store, job_id).wake_at == now_minute(), "stuck at the step that ran"
    await cron.tell_task_notices()
    assert job_of(store, job_id).wake_at == now_minute() + 40, "settled once the store took it"


async def test_the_host_is_offered_only_a_step_that_could_start(store, monkeypatch):
    from rune.api.handlers.cron import list_cron_jobs
    monkeypatch.setattr("rune.api.handlers.cron.get_memory_store", lambda: store)
    job_id = await start(store, steps=2)
    await step(store, job_id, monkeypatch, parked("next"))

    async def wake_at():
        listed = (await list_cron_jobs()).model_dump(by_alias=True)["jobs"]
        return next(j for j in listed if j["id"] == job_id)["wakeAt"]

    assert await wake_at() == now_minute() + 30
    store.update_cron_job(job_id, max_runs=1)  # lowered while parked: it can't take another step
    assert await wake_at() is None, "not worth waking the machine for"
    store.update_cron_job(job_id, max_runs=2, enabled=False)
    assert await wake_at() is None


def test_a_subagent_inside_a_step_is_outside_it():
    from rune.agent.run_control import RunControl
    _, token = long_task.park_slot(RunControl("step"))
    try:
        assert long_task.in_step()
        with long_task.outside_step():
            assert not long_task.in_step()
            with pytest.raises(ValueError, match="only inside"):
                long_task.park("x", 5, False)
        assert long_task.in_step()
    finally:
        long_task.release_slot(token)


async def test_a_step_that_just_answers_is_taken_as_unfinished_and_the_task_goes_on(store, told, monkeypatch):
    job_id = await start(store)
    await step(store, job_id, monkeypatch, {"status": "completed", "success": True, "verified": False,
                                            "output": "I'll continue next time."})
    job = job_of(store, job_id)
    assert job.status == "active" and job.wake_at == now_minute() + long_task.DEFAULT_RESUME_MINUTES
    assert told.await_count == 0, "not 'Done.'"


async def test_a_task_turned_on_again_stays_on(store, told, monkeypatch):
    job_id = await start(store, steps=3)
    await step(store, job_id, monkeypatch, {"status": "failed", "success": False, "verified": False,
                                            "output": "", "error": "site down"})
    assert told.await_count == 1
    assert (await cron.cron_update(cron.CronUpdateParams(job_id=job_id, enabled=True))).success
    for _ in range(3):
        await cron.tell_task_notices()
    job = job_of(store, job_id)
    assert job.status == "active" and job.wake_at == now_minute() + 1, "its last step isn't settled again"
    assert told.await_count == 1, "and the owner isn't told about it twice"


async def test_this_machine_s_token_limit_holds_whatever_the_task_is_changed_to(store, monkeypatch):
    monkeypatch.setenv("RUNE_TASK_TOKEN_LIMIT", "300000")
    job_id = await start(store)
    assert (await cron.cron_update(cron.CronUpdateParams(job_id=job_id,
                                                         policy=cron.RoutinePolicy(total_tokens=10**9)))).success
    seen = {}

    def owner(policy, usage):
        seen["total_tokens"] = policy.total_tokens
        raise __import__("rune.agent.request_budget", fromlist=["BudgetExceeded"]).BudgetExceeded("over")

    monkeypatch.setattr("rune.proactive.routine_budget.budget_owner", owner)
    monkeypatch.setattr(cron, "_execute_goal_job", AsyncMock())
    await ext.fire(job_id, job_of(store, job_id).wake_at)
    await drain()
    assert seen["total_tokens"] == 300_000


async def test_a_step_the_machine_died_in_is_settled_and_told_once_it_is_back(store, told, monkeypatch):
    from rune.proactive.execution_store import ExecutionStore
    from rune.utils.paths import rune_data
    job_id = await start(store)
    records = ExecutionStore(rune_data() / "routine-executions.db")
    records.claim(f"cron:{job_id}:{now_minute()}", "f", 1000, resource=f"cron:{job_id}")
    records._db.execute("UPDATE operations SET owner_pid = 999999999")  # a process long gone
    records.close()
    await cron.tell_task_notices()
    job = job_of(store, job_id)
    assert job.task["ended"] == "interrupted" and "machine stopped" in told.await_args.args[2]


async def test_a_late_step_still_runs_and_only_the_host_s_giving_up_ends_the_task(store, told, monkeypatch):
    job_id = await start(store)
    wake = job_of(store, job_id).wake_at
    later = (wake + 24 * 60) * 60  # the laptop slept a day
    monkeypatch.setattr("time.time", lambda: later)
    await cron.tell_task_notices()
    assert job_of(store, job_id).status == "active" and told.await_count == 0, "late isn't given up"
    assert await cron.task_stalled(job_id, wake + 1) == {"state": "moved_on"}, "only the step it waits on"
    assert await cron.task_stalled(job_id, wake) == {"state": "ended"}
    job = job_of(store, job_id)
    assert job.status == "paused" and job.task["ended"] == "failed" and "couldn't start" in told.await_args.args[2]


async def test_a_hosted_task_out_of_steps_under_a_lowered_limit_ends_and_says_so(store, told, monkeypatch):
    job_id = await start(store, steps=10)
    await step(store, job_id, monkeypatch, parked("Read 1-100."))
    monkeypatch.setenv("RUNE_TASK_STEP_LIMIT", "1")
    await cron.tell_task_notices()
    assert job_of(store, job_id).task["ended"] == "step_limit" and "Stopped after 1 steps" in told.await_args.args[2]


async def test_a_step_that_already_ran_is_not_offered_to_the_host_again(store, monkeypatch):
    from rune.api.handlers.cron import list_cron_jobs
    from rune.proactive.execution_store import ExecutionStore
    from rune.utils.paths import rune_data
    monkeypatch.setattr("rune.api.handlers.cron.get_memory_store", lambda: store)
    job_id = await start(store)
    records = ExecutionStore(rune_data() / "routine-executions.db")
    records.claim(f"cron:{job_id}:{now_minute()}", "f", 1000)
    records.close()
    listed = (await list_cron_jobs()).model_dump(by_alias=True)["jobs"]
    assert next(j for j in listed if j["id"] == job_id)["wakeAt"] is None


async def test_a_task_with_a_step_running_isnt_turned_on_twice(store, monkeypatch):
    from rune.proactive.execution_store import ExecutionStore
    from rune.utils.paths import rune_data
    job_id = await start(store)
    store.update_cron_job(job_id, enabled=False)
    records = ExecutionStore(rune_data() / "routine-executions.db")
    records.claim(f"cron:{job_id}:{now_minute()}", "f", 1000, resource=f"cron:{job_id}")  # this process: running
    records.close()
    refused = await cron.cron_update(cron.CronUpdateParams(job_id=job_id, enabled=True))
    assert not refused.success and "running" in refused.error


async def test_a_lowered_step_limit_starts_no_further_step(store, told, monkeypatch):
    job_id = await start(store, steps=10)
    await step(store, job_id, monkeypatch, parked("Read 1-100.", minutes=1))
    monkeypatch.setenv("RUNE_TASK_STEP_LIMIT", "1")
    later = (now_minute() + 1) * 60 + 5
    monkeypatch.setattr("time.time", lambda: later)
    worker = AsyncMock(return_value=parked("never"))
    monkeypatch.setattr(cron, "_execute_goal_job", worker)
    await ext.fire(job_id, job_of(store, job_id).wake_at)
    await drain()
    assert worker.await_count == 0 and job_of(store, job_id).task["ended"] == "step_limit"


async def test_the_step_continued_after_an_approval_is_told_it_is_a_step(tmp_path, monkeypatch):
    from rune.agent.background import BackgroundTask, run_background
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    seen = {}

    class Loop:
        def __init__(self, config):
            pass

        def set_approval_callback(self, callback):
            pass

        async def run(self, goal, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(reason="completed", final_step=1, answer="ok", error=None)

    await run_background(BackgroundTask(goal="g", source="task", workspace=str(tmp_path), preamble="STEP BRIEF",
                                        continuation="note"), loop_factory=Loop)
    assert seen["extra_system_context"].startswith("STEP BRIEF\nnote")
