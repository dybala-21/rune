"""Durable routine limits, duplicate ticks and interrupted execution."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from rune.capabilities import cron
from rune.memory.store import MemoryStore
from rune.proactive.execution_store import ExecutionStore
from rune.proactive.routine import RoutinePolicy, should_notify


def test_blank_form_lines_do_not_become_tracked_paths_or_commands():
    policy = RoutinePolicy(input_paths=["source.txt", "", "  "], output_paths=["result.docx", ""], verification=[""])
    assert policy.input_paths == ["source.txt"] and policy.output_paths == ["result.docx"]
    assert not policy.verification


async def test_updated_routine_binds_the_same_workspace_used_to_observe_files(tmp_path, monkeypatch):
    from rune.api.handlers import cron as api

    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "state"))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr("rune.utils.paths.user_workspace", lambda: work)
    store = MemoryStore(tmp_path / "tasks.db")
    monkeypatch.setattr(api, "get_memory_store", lambda: store)
    try:
        created = await api.create_cron_job(api.CronCreateRequest(name="Report", schedule="* * * * *", goal="Read source.txt"))
        updated = await api.update_cron_job(created.job.id, api.CronUpdateRequest(policy=RoutinePolicy(input_paths=["source.txt", ""])))
        assert updated.job.policy.workspace == str(work)
        assert updated.job.policy.input_paths == ["source.txt"]
        restored = await api.get_cron_job(created.job.id)
        assert restored.policy == updated.job.policy
    finally:
        store.close()


def test_notification_ignores_wording_when_tracked_result_is_unchanged():
    observation = {"complete": True, "outputs": {"report.md": {"state": "present", "sha256": "a"}}}
    previous = {"status": "verified", "output": "Done", "observations": observation}
    current = {**previous, "output": "All finished"}
    assert not should_notify(RoutinePolicy(), current, previous)
    current["observations"] = {"complete": True, "outputs": {"report.md": {"state": "present", "sha256": "b"}}}
    assert should_notify(RoutinePolicy(), current, previous)
    assert should_notify(RoutinePolicy(), {**current, "status": "needs_approval"}, previous)


@pytest.mark.asyncio
async def test_unchanged_inputs_reuse_result_after_restart_and_changed_files_rerun(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    clock = [1000.0]
    monkeypatch.setattr("rune.proactive.routine.time.time", lambda: clock[0])
    source, output = tmp_path / "source.txt", tmp_path / "report.md"
    source.write_text("one")
    policy = RoutinePolicy(workspace=str(tmp_path), input_paths=["source.txt"], output_paths=["report.md"])
    store = MemoryStore(tmp_path / "memory.db")
    monkeypatch.setattr(cron, "_get_store", lambda: store)
    job_id = store.create_cron_job(name="report", schedule="* * * * *", command=cron._pack_command("", "report", "", "", policy))

    async def work(job):
        output.write_text(source.read_text())
        return {"status": "verified", "success": True, "verified": True, "output": "Saved"}

    worker = AsyncMock(side_effect=work)
    monkeypatch.setattr(cron, "_execute_goal_job", worker)
    job = cron._row_to_cronjob(store.get_cron_job(job_id))
    await cron.execute_cron_job(job)
    clock[0] += 60
    await cron.execute_cron_job(job)
    assert worker.await_count == 1 and store.get_cron_job(job_id)["run_count"] == 1
    clock[0] += 60
    source.write_text("two")
    await cron.execute_cron_job(job)
    assert worker.await_count == 2 and output.read_text() == "two"
    clock[0] += 60
    output.unlink()
    await cron.execute_cron_job(job)
    assert worker.await_count == 3 and output.exists()
    store.close()


def test_unknown_or_outside_observation_cannot_skip_execution(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from rune.proactive.routine_observation import observe, reusable

    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    policy = RoutinePolicy(workspace=str(tmp_path), input_paths=["../private.txt"])
    current = observe(policy)
    assert not current["complete"]
    job = SimpleNamespace(policy=policy, goal="report", command="")
    previous = {"success": True, "status": "verified", "observations": current,
                "routine_goal": "report", "routine_command": ""}
    assert not reusable(job, current, previous)


@pytest.mark.asyncio
@pytest.mark.parametrize("change_source", [False, True])
async def test_unstable_or_missing_artifacts_cannot_be_reported_verified(tmp_path, monkeypatch, change_source):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "state"))
    source = tmp_path / "source.txt"
    source.write_text("before")
    policy = RoutinePolicy(workspace=str(tmp_path), input_paths=["source.txt"], output_paths=["report.md"])
    store = MemoryStore(tmp_path / "tasks.db")
    monkeypatch.setattr(cron, "_get_store", lambda: store)
    job_id = store.create_cron_job(name="report", schedule="* * * * *", command=cron._pack_command("", "report", "", "", policy))

    async def worker(job):
        if change_source:
            source.write_text("after")
            (tmp_path / "report.md").write_text("stale")
        return {"success": True, "verified": True, "status": "verified", "output": "Done"}

    monkeypatch.setattr(cron, "_execute_goal_job", worker)
    await cron.execute_cron_job(cron._row_to_cronjob(store.get_cron_job(job_id)))
    records = ExecutionStore(tmp_path / "state" / "data" / "routine-executions.db")
    result = records.latest_result(f"cron:{job_id}:")
    assert result["status"] == "unverified" and not result["success"] and not result["verified"]
    records.close()
    store.close()


@pytest.mark.asyncio
async def test_duplicate_ticks_claim_once_and_respect_max_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    store = MemoryStore(tmp_path / "memory.db")
    monkeypatch.setattr(cron, "_get_store", lambda: store)
    job_id = store.create_cron_job(name="report", schedule="* * * * *", command=cron._pack_command("", "report", "", ""), max_runs=1)
    job = cron._row_to_cronjob(store.get_cron_job(job_id))
    run = AsyncMock(return_value={"status": "unverified", "output": "report", "verified": False})
    monkeypatch.setattr(cron, "_execute_goal_job", run)
    await asyncio.gather(cron.execute_cron_job(job), cron.execute_cron_job(job))
    await cron.execute_cron_job(job)
    assert run.await_count == 1
    assert store.get_cron_job(job_id)["run_count"] == 1
    store.close()


def test_unknown_effects_keep_the_resource_claim_after_restart(tmp_path):
    path = tmp_path / "runs.db"
    first = ExecutionStore(path)
    assert first.claim("one", "a", 10, resource="report") == "claimed"
    first.finish("one", {"status": "unverified", "execution_unknown": True})
    first.close()
    second = ExecutionStore(path)
    assert second.claim("two", "b", 10, resource="report") == "busy"
    second.close()


def test_notification_uses_observed_results_not_no_change_claims():
    previous = {"status": "verified", "output": "3 issues"}
    policy = RoutinePolicy()
    assert not should_notify(policy, previous, previous)
    assert should_notify(policy, {"status": "verified", "output": "4 issues"}, previous)
    assert should_notify(policy, {"status": "needs_approval", "output": ""}, previous)
    assert not should_notify(RoutinePolicy(notify="failures"), previous, None)


def test_recovery_never_overwrites_unknown_effects_or_releases_live_worker(tmp_path):
    records = ExecutionStore(tmp_path / "runs.db")
    assert records.claim("one", "a", 10, resource="report") == "claimed"
    with pytest.raises(ValueError, match="still running"):
        records.reconcile("report", "one", "Checked files")
    records.finish("one", {"status": "interrupted"})
    records.finish("one", {"status": "verified"})
    assert records.claim("two", "b", 10, resource="report") == "busy"
    assert records.reconcile("report", "one", "Checked files; no external changes")
    assert not records.reconcile("report", "one", "Repeated click")
    assert records.claim("two", "b", 10, resource="report") == "claimed"
    receipt = records.get("one")[1]
    assert receipt["previous_result"] == {"status": "interrupted"}
    assert not receipt["verified"]
    records.close()


@pytest.mark.asyncio
async def test_pausing_active_task_stops_worker_and_requires_review(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    store = MemoryStore(tmp_path / "memory.db")
    monkeypatch.setattr(cron, "_get_store", lambda: store)
    job_id = store.create_cron_job(name="report", schedule="* * * * *", command=cron._pack_command("", "report", "", ""))
    started, stopped = asyncio.Event(), asyncio.Event()

    async def worker(job):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(cron, "_execute_goal_job", worker)
    pending = asyncio.create_task(cron.execute_cron_job(cron._row_to_cronjob(store.get_cron_job(job_id))))
    await asyncio.wait_for(started.wait(), 2)
    store.update_cron_job(job_id, enabled=False)
    await asyncio.wait_for(pending, 3)
    assert stopped.is_set()
    records = ExecutionStore(tmp_path / "data" / "routine-executions.db")
    run = records.recent(f"cron:{job_id}:")[0]
    assert run["blocked"] and run["result"]["execution_unknown"]
    records.close()
    store.close()


@pytest.mark.parametrize("change", ["pause", "delete", "goal"])
async def test_change_during_fast_completion_keeps_recovery_claim(tmp_path, monkeypatch, change):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    store = MemoryStore(tmp_path / "memory.db")
    monkeypatch.setattr(cron, "_get_store", lambda: store)
    job_id = store.create_cron_job(name="report", schedule="* * * * *",
                                  command=cron._pack_command("", "original", "", ""))

    async def worker(job):
        if change == "pause":
            store.update_cron_job(job.id, enabled=False)
        elif change == "delete":
            store.delete_cron_job(job.id)
        else:
            store.update_cron_job(job.id, command=cron._pack_command("", "replacement", "", ""))
        return {"status": "verified", "success": True, "verified": True, "output": "Saved"}

    monkeypatch.setattr(cron, "_execute_goal_job", worker)
    try:
        await cron.execute_cron_job(cron._row_to_cronjob(store.get_cron_job(job_id)))
        records = ExecutionStore(tmp_path / "data" / "routine-executions.db")
        try:
            run = records.recent(f"cron:{job_id}:")[0]
            assert run["blocked"] and run["result"]["execution_unknown"]
            assert not run["result"]["verified"]
        finally:
            records.close()
    finally:
        store.close()


@pytest.mark.asyncio
async def test_notification_never_falls_back_to_another_external_channel():
    from unittest.mock import Mock

    from rune.channels.registry import ChannelRegistry
    from rune.daemon.gateway import ChannelGateway, GatewayNotification

    registry = ChannelRegistry()
    telegram = Mock(name="telegram")
    telegram.name = "telegram"
    telegram.send_notification = AsyncMock(side_effect=RuntimeError("offline"))
    slack = Mock(name="slack")
    slack.name = "slack"
    slack.send_notification = AsyncMock()
    registry.register(telegram)
    registry.register(slack)
    gateway = ChannelGateway(registry)
    gateway._default_recipients.update(telegram="recipient", slack="other-recipient")
    await gateway.route_notification(GatewayNotification(title="Report", body="private result",
                                                         priority="high", channel="telegram"))
    telegram.send_notification.assert_awaited_once()
    slack.send_notification.assert_not_called()


@pytest.mark.asyncio
async def test_task_api_preserves_limits_and_paused_recovery(tmp_path, monkeypatch):
    from fastapi import HTTPException

    from rune.api.handlers import cron as api
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    store = MemoryStore(tmp_path / "memory.db")
    monkeypatch.setattr(api, "get_memory_store", lambda: store)
    created = await api.create_cron_job(api.CronCreateRequest(
        name="Daily report", schedule="0 9 * * 1-5", goal="Prepare report", maxRuns=3,
        policy=RoutinePolicy(workspace=str(tmp_path / "work"), timeout_seconds=60),
    ))
    job_id = created.job.id
    updated = await api.update_cron_job(job_id, api.CronUpdateRequest(name="Report", maxRuns=None))
    assert updated.job.goal == "Prepare report" and updated.job.policy.timeout_seconds == 60
    assert updated.job.max_runs is None
    records = ExecutionStore(tmp_path / "data" / "routine-executions.db")
    records.claim(f"cron:{job_id}:one", "a", 10, resource=f"cron:{job_id}")
    records.finish(f"cron:{job_id}:one", {"status": "interrupted"})
    req = api.ReconcileRequest(operationId=f"cron:{job_id}:one", note="Checked report")
    with pytest.raises(HTTPException) as exc:
        await api.reconcile_cron_job(job_id, req)
    assert exc.value.status_code == 409
    await api.update_cron_job(job_id, api.CronUpdateRequest(enabled=False))
    assert (await api.reconcile_cron_job(job_id, req))["reconciled"]
    assert not store.get_cron_job(job_id)["enabled"]
    records.close()
    store.close()
