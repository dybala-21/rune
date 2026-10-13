"""Repeated tasks retain progress, enforce total limits and inspect changed inputs."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rune.agent.timing import timed_completion
from rune.capabilities import cron
from rune.memory.store import MemoryStore
from rune.proactive.execution_store import ExecutionStore
from rune.proactive.routine import RoutinePolicy, should_notify
from rune.proactive.routine_budget import budget_state
from rune.proactive.routine_observation import changes, observe


@pytest.fixture
def routine(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr("rune.agent.request_budget.input_estimate", lambda _: 10)
    clock = [1000.0]
    monkeypatch.setattr("rune.proactive.routine.time.time", lambda: clock[0])
    state = SimpleNamespace(store=MemoryStore(tmp_path / "tasks.db"), clock=clock, root=tmp_path)
    monkeypatch.setattr(cron, "_get_store", lambda: state.store)
    yield state
    state.store.close()


def create(state, **policy):
    return state.store.create_cron_job(name="Report", schedule="* * * * *",
        command=cron._pack_command("", "Update the report", "", "",
                                  RoutinePolicy(workspace=str(state.root), **policy)))


async def tick(state, job_id):
    state.clock[0] += 60
    await cron.execute_cron_job(cron._row_to_cronjob(state.store.get_cron_job(job_id)))


def records(state):
    return ExecutionStore(state.root / "state" / "data" / "routine-executions.db")


async def call(request):
    return await timed_completion(request, {"model": "xai/grok-4.6", "messages": [], "max_tokens": 100})


@pytest.mark.parametrize("limit,expected", [
    ({"total_model_requests": 2}, 2), ({"total_tokens": 220}, 2), ({"total_cost_usd": .000001}, 1),
    ({"total_model_requests": 0}, 0), ({"total_cost_usd": 0}, 0),
])
async def test_total_budget_survives_restart_pause_and_edit(routine, monkeypatch, limit, expected):
    request = AsyncMock(return_value={"usage": {"prompt_tokens": 100, "completion_tokens": 10}})
    async def work(job):
        await call(request)
        return {"status": "verified", "success": True, "verified": True, "output": "Saved"}
    monkeypatch.setattr(cron, "_execute_goal_job", work)
    job_id = create(routine, **limit)
    await tick(routine, job_id)
    routine.store.update_cron_job(job_id, enabled=False)
    routine.store.close()
    routine.store = MemoryStore(routine.root / "tasks.db")
    routine.store.update_cron_job(job_id, enabled=True, name="Renamed")
    await tick(routine, job_id)
    await tick(routine, job_id)
    assert request.await_count == expected
    saved = records(routine)
    try:
        usage = saved.usage(f"cron:{job_id}:")
        assert usage["modelRequests"] == expected and usage["tokens"] == 110 * expected
        assert saved.latest_result(f"cron:{job_id}:")["status"] == "budget_exhausted"
    finally:
        saved.close()


async def test_multiple_calls_in_one_run_share_the_remaining_total(routine, monkeypatch):
    request = AsyncMock(return_value={"usage": {"prompt_tokens": 100, "completion_tokens": 10}})
    async def work(job):
        for _ in range(3):
            await call(request)
    monkeypatch.setattr(cron, "_execute_goal_job", work)
    job_id = create(routine, total_model_requests=2)
    await tick(routine, job_id)
    await tick(routine, job_id)
    request_count = request.await_count
    assert request_count == 2
    saved = records(routine)
    try:
        first = saved.recent(f"cron:{job_id}:")[-1]
        assert first["result"]["status"] == "budget_exhausted" and not first["blocked"]
        assert saved.usage(f"cron:{job_id}:")["modelRequests"] == 2
    finally:
        saved.close()


async def test_directory_changes_continue_from_saved_result_without_rebilling_reuse(routine, monkeypatch):
    inbox = routine.root / "inbox"
    inbox.mkdir()
    (inbox / "notes.md").write_text("Initial notes")
    output = inbox / "report.md"
    contexts = []
    request = AsyncMock(return_value={"usage": {"prompt_tokens": 100, "completion_tokens": 10}})
    async def work(job):
        contexts.append(job.continuation)
        await call(request)
        output.write_text("Updated report")
        return {"status": "verified", "success": True, "verified": True, "output": "Saved inbox/report.md"}
    monkeypatch.setattr(cron, "_execute_goal_job", work)
    job_id = create(routine, input_paths=["inbox"], output_paths=["inbox/report.md"], total_model_requests=2)
    await tick(routine, job_id)
    routine.store.close()
    routine.store = MemoryStore(routine.root / "tasks.db")
    await tick(routine, job_id)
    (inbox / "new.pdf").write_bytes(b"new source")
    (inbox / "notes.md").unlink()
    await tick(routine, job_id)
    await tick(routine, job_id)
    assert request.await_count == 2 and contexts[0] == ""
    context = json.loads(contexts[1].split("\n", 1)[1])
    assert context["previous_output"] == "Saved inbox/report.md"
    assert context["changes"]["inputs"]["added"] == ["inbox/new.pdf"]
    assert context["changes"]["inputs"]["removed"] == ["inbox/notes.md"]
    assert "inbox/report.md" not in str(context["changes"]["inputs"])
    saved = records(routine)
    try:
        assert saved.usage(f"cron:{job_id}:")["modelRequests"] == 2
        assert saved.latest_result(f"cron:{job_id}:")["status"] == "unchanged"
    finally:
        saved.close()


async def test_output_parent_creation_does_not_rerun_but_other_inputs_do(routine, monkeypatch):
    inbox = routine.root / "inbox"
    inbox.mkdir()
    (inbox / "source.md").write_text("unchanged source")
    folder = inbox / "reports" / "daily"

    async def work(job):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "summary.md").write_text("report")
        return {"status": "verified", "success": True, "verified": True, "output": "Saved"}

    worker = AsyncMock(side_effect=work)
    monkeypatch.setattr(cron, "_execute_goal_job", worker)
    job_id = create(routine, input_paths=["inbox"], output_paths=["inbox/reports/daily/summary.md"])
    await tick(routine, job_id)
    saved = records(routine)
    try:
        assert saved.latest_result(f"cron:{job_id}:")["status"] == "verified"
        await tick(routine, job_id)
        assert worker.await_count == 1
        source = folder / "new-input.md"
        source.write_text("one")
        for edit in (None, "two", "delete"):
            if edit == "delete":
                source.unlink()
            elif edit:
                source.write_text(edit)
            await tick(routine, job_id)
        assert worker.await_count == 4
        await tick(routine, job_id)
        assert worker.await_count == 4
    finally:
        saved.close()


@pytest.mark.parametrize("stop", ["cancel", "error"])
async def test_final_observation_failure_keeps_usage_and_partial_result(routine, monkeypatch, stop):
    (routine.root / "input.txt").write_text("source")
    request = AsyncMock(return_value={"usage": {"prompt_tokens": 100, "completion_tokens": 10}})
    async def work(job):
        await call(request)
        return {"status": "verified", "success": True, "verified": True, "output": "Draft saved"}
    monkeypatch.setattr(cron, "_execute_goal_job", work)
    job_id = create(routine, input_paths=["input.txt"], total_model_requests=3)
    original = asyncio.to_thread
    entered = asyncio.Event()
    observations = 0
    async def inspect(fn, *args):
        nonlocal observations
        observations += 1
        if observations == 2:
            entered.set()
            if stop == "error":
                raise OSError("Cannot inspect output")
            await asyncio.Event().wait()
        return await original(fn, *args)
    monkeypatch.setattr(asyncio, "to_thread", inspect)
    task = asyncio.create_task(tick(routine, job_id))
    await asyncio.wait_for(entered.wait(), 2)
    if stop == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task
    saved = records(routine)
    try:
        usage = saved.usage(f"cron:{job_id}:")
        assert usage["modelRequests"] == 1 and usage["tokens"] == 110 and usage["costUsd"] > 0
        assert usage["requestsKnown"] and usage["tokensKnown"] and usage["costKnown"]
        run = saved.recent(f"cron:{job_id}:")[0]
        assert run["blocked"] and run["result"]["execution_unknown"]
        assert not run["result"]["verified"] and run["result"]["output"] == "Draft saved"
        assert run["result"]["routine_goal"] == "Update the report"
        saved.reconcile(f"cron:{job_id}", run["id"], "Checked the saved draft")
        assert not budget_state(RoutinePolicy(total_model_requests=3), saved.usage(f"cron:{job_id}:"))["blocked"]
    finally:
        saved.close()


def test_reviewed_progress_survives_restart_without_granting_permission(tmp_path):
    from rune.proactive.routine_context import continuation
    from rune.safety.approval_context import was_approved

    path = tmp_path / "runs.db"
    saved = ExecutionStore(path)
    job = SimpleNamespace(goal="Update report", command="", policy=RoutinePolicy(workspace=str(tmp_path)))
    saved.claim("cron:job:1", "one", 10, resource="cron:job")
    saved.finish("cron:job:1", {"status": "interrupted", "execution_unknown": True,
        "routine_goal": job.goal, "routine_command": "", "workspace": str(tmp_path),
        "output": "Draft saved; verification unfinished."})
    assert continuation(job, None, saved.latest_result("cron:job:")) == ""
    saved.reconcile("cron:job", "cron:job:1", "Checked report.md; only section 3 is unfinished.")
    saved.close()
    saved = ExecutionStore(path)
    try:
        previous = saved.latest_result("cron:job:", executed_only=True)
        context = continuation(job, None, previous)
        data = json.loads(context.split("\n", 1)[1])
        assert data["user_review"] == "Checked report.md; only section 3 is unfinished."
        assert data["previous_output"] == "Draft saved; verification unfinished."
        assert data["previous_status"] == "interrupted"
        assert "does not grant new permissions" in context and not was_approved()
        assert continuation(job, None, {**previous, "reviewed_by": "agent"}) == ""
        job.goal = "Different task"
        assert continuation(job, None, previous) == ""
    finally:
        saved.close()


def test_incomplete_usage_is_not_erased_by_reconciliation(tmp_path):
    saved = ExecutionStore(tmp_path / "runs.db")
    saved.claim("cron:job:one", "a", 10, resource="cron:job")
    saved.finish("cron:job:one", {"status": "interrupted"})
    saved.reconcile("cron:job", "cron:job:one", "Checked files")
    saved.close()
    saved = ExecutionStore(tmp_path / "runs.db")
    try:
        usage = saved.usage("cron:job:")
        assert not usage["requestsKnown"] and not usage["costKnown"]
        assert budget_state(RoutinePolicy(total_model_requests=10), usage)["blocked"]
    finally:
        saved.close()


def test_file_hashes_detect_same_size_edits_and_directory_escape(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "state"))
    folder = tmp_path / "inputs"
    folder.mkdir()
    source = folder / "data.xlsx"
    source.write_bytes(b"before")
    stamp = source.stat()
    policy = RoutinePolicy(workspace=str(tmp_path), input_paths=["inputs"])
    before = observe(policy)
    source.write_bytes(b"after!")
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert changes(observe(policy), {"observations": before})["inputs"]["modified"] == ["inputs/data.xlsx"]
    (folder / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    assert not observe(policy)["complete"]


@pytest.mark.parametrize("previous_completed", [False, True])
async def test_task_changed_during_observation_does_not_run_replacement(routine, monkeypatch, previous_completed):
    (routine.root / "input.txt").write_text("source")
    job_id = create(routine, input_paths=["input.txt"])
    worker = AsyncMock(return_value={"status": "verified", "success": True, "verified": True, "output": "Saved"})
    monkeypatch.setattr(cron, "_execute_goal_job", worker)
    if previous_completed:
        await tick(routine, job_id)
        worker.reset_mock()
    original = asyncio.to_thread
    async def changed(fn, *args):
        result = await original(fn, *args)
        routine.store.update_cron_job(job_id, command=cron._pack_command("", "Different goal", "", ""))
        return result
    monkeypatch.setattr(asyncio, "to_thread", changed)
    await tick(routine, job_id)
    worker.assert_not_called()
    assert routine.store.get_cron_job(job_id)["run_count"] == int(previous_completed)
    saved = records(routine)
    try:
        assert saved.latest_result(f"cron:{job_id}:")["status"] == "cancelled"
        assert saved.latest_result(f"cron:{job_id}:")["not_started"]
    finally:
        saved.close()


def test_failure_notifications_only_repeat_when_the_problem_changes():
    policy = RoutinePolicy(notify="failures")
    result = {"status": "budget_exhausted", "output": "", "error": "Total limit reached"}
    assert should_notify(policy, result, None)
    assert not should_notify(policy, result, result)
    assert should_notify(policy, {**result, "error": "Usage unavailable"}, result)


async def test_unreadable_tracking_scope_does_not_spend_tokens(routine, monkeypatch):
    job_id = create(routine, input_paths=["../outside"])
    worker = AsyncMock()
    monkeypatch.setattr(cron, "_execute_goal_job", worker)
    await tick(routine, job_id)
    await tick(routine, job_id)
    worker.assert_not_called()
    assert routine.store.get_cron_job(job_id)["run_count"] == 0
    saved = records(routine)
    try:
        assert saved.usage(f"cron:{job_id}:")["modelRequests"] == 0
        assert saved.latest_result(f"cron:{job_id}:")["not_started"]
    finally:
        saved.close()


async def test_background_waits_after_approval_without_granting_permission(routine, monkeypatch):
    from rune.safety.approval_context import was_approved
    from rune.types import CompletionTrace

    seen = []
    class Loop:
        def __init__(self, config):
            self._last_answer_text = "Saved result"
        def set_approval_callback(self, callback):
            self.approve = callback
        async def run(self, goal, **kwargs):
            assert not was_approved()
            assert not await self.approve("send_mail", "Not authorized")
            seen.append(kwargs.get("extra_system_context", ""))
            change = await cron.cron_update(cron.CronUpdateParams(job_id=job_id, policy=RoutinePolicy(total_model_requests=None)))
            assert not change.success and change.metadata["requires_approval"]
            return CompletionTrace(reason="completed")
    monkeypatch.setattr("rune.agent.loop.NativeAgentLoop", Loop)
    job_id = create(routine, total_model_requests=10)
    await tick(routine, job_id)
    await tick(routine, job_id)
    assert seen == [""]
    saved = records(routine)
    try:
        result = saved.latest_result(f"cron:{job_id}:")
        assert result["status"] == "needs_approval" and result["output"] == "Saved result"
    finally:
        saved.close()
    assert cron._row_to_cronjob(routine.store.get_cron_job(job_id)).policy.total_model_requests == 10


async def test_proactive_retries_share_one_model_budget(monkeypatch):
    from rune.config import get_config
    from rune.proactive.bridge import BridgeConfig, ProactiveAgentBridge
    from rune.proactive.engine import ProactiveEngine
    from rune.proactive.types import Suggestion

    monkeypatch.setattr(get_config().llm, "model_request_limit", 1)
    monkeypatch.setattr("rune.agent.request_budget.input_estimate", lambda _: 10)
    request = AsyncMock(return_value={"usage": {"prompt_tokens": 100, "completion_tokens": 10}})
    async def factory(goal):
        await call(request)
        return {"success": False, "error": "Retryable failure"}
    bridge = ProactiveAgentBridge(ProactiveEngine({"min_confidence": 0}), factory,
        BridgeConfig(auto_execute=True, retry_safe=True, max_retries=3, backoff_base_seconds=0))
    result = await bridge.execute_suggestion(Suggestion(title="Update report", description="Test"))
    request.assert_awaited_once()
    assert result.result["status"] == "budget_exhausted"
    assert result.result["timings"]["usage"]["calls"] == 1


async def test_api_exposes_total_limits_and_saved_usage(routine, monkeypatch):
    from pydantic import ValidationError

    from rune.api.handlers import cron as api
    monkeypatch.setattr(api, "get_memory_store", lambda: routine.store)
    created = await api.create_cron_job(api.CronCreateRequest(name="Budget", schedule="* * * * *", goal="Report",
        policy=RoutinePolicy(total_tokens=1000, total_model_requests=2, total_cost_usd=.1)))
    request = AsyncMock(return_value={"usage": {"prompt_tokens": 100, "completion_tokens": 10}})
    async def work(job):
        await call(request)
        return {"status": "verified", "success": True, "verified": True, "output": "Saved"}
    monkeypatch.setattr(cron, "_execute_goal_job", work)
    await tick(routine, created.job.id)
    updated = await api.update_cron_job(created.job.id, api.CronUpdateRequest(
        policy=RoutinePolicy(total_model_requests=1)))
    assert updated.job.policy.total_tokens == 1000 and updated.job.policy.total_cost_usd == .1
    assert updated.job.budget["used"]["modelRequests"] == 1
    assert updated.job.budget["blocked"]
    for invalid in ({"total_tokens": True}, {"total_model_requests": 1.5}, {"total_cost_usd": float("nan")}):
        with pytest.raises(ValidationError):
            RoutinePolicy(**invalid)


def test_changing_limits_does_not_invalidate_unchanged_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "state"))
    (tmp_path / "input.txt").write_text("source")
    policy = RoutinePolicy(workspace=str(tmp_path), input_paths=["input.txt"])
    patched = policy.patched(RoutinePolicy(total_model_requests=3, notify="failures"))
    assert observe(policy) == observe(patched)
    assert patched.workspace == str(tmp_path) and patched.input_paths == ["input.txt"]
