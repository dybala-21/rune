"""Check routine approvals, replay protection and shared budgets."""

import asyncio
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from rune.agent.run_control import current_control
from rune.agent.timing import timed_completion
from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
from rune.api.handlers import cron as api
from rune.capabilities import cron
from rune.capabilities.file import register_file_capabilities
from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.memory.store import MemoryStore
from rune.proactive.execution_store import ExecutionStore
from rune.proactive.routine import RoutinePolicy
from rune.types import CapabilityResult, CompletionTrace


@pytest.fixture
def task(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("RUNE_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("RUNE_ISOLATION_ROOT", str(tmp_path))
    monkeypatch.setenv("RUNE_APPROVAL_MODE", "standard")
    monkeypatch.setenv("RUNE_HYBRID_API", "1")
    monkeypatch.setattr("rune.agent.request_budget.input_estimate", lambda _: 10)
    state = SimpleNamespace(root=tmp_path, store=MemoryStore(tmp_path / "tasks.db"), clock=[1000.0],
                            requests=AsyncMock(return_value={"usage": {"prompt_tokens": 100, "completion_tokens": 10}}),
                            send=AsyncMock(return_value=CapabilityResult(success=True, output="accepted")), contexts=[],
                            verification={"required": True, "status": "pass"})
    monkeypatch.setattr(cron, "_get_store", lambda: state.store)
    monkeypatch.setattr(api, "get_memory_store", lambda: state.store)
    monkeypatch.setattr("rune.proactive.routine.time", SimpleNamespace(time=lambda: state.clock[0]))
    registry = CapabilityRegistry()
    register_file_capabilities(registry)
    registry.register(CapabilityDefinition(name="web_fetch", description="Fixture API", execute=state.send))
    monkeypatch.setattr("rune.capabilities.registry.get_capability_registry", lambda: registry)
    (tmp_path / "source.txt").write_text("Source facts")

    class Loop:
        def __init__(self, config):
            self._last_answer_text = "Draft ready"

        def set_approval_callback(self, callback):
            self.approve = callback

        async def emit(self, *args):
            pass

        async def run(self, goal, **kwargs):
            from rune.agent.loop import StallState
            await timed_completion(state.requests, {"model": "xai/grok-4.6", "messages": [], "max_tokens": 100})
            state.contexts.append(kwargs)
            content = (tmp_path / "source.txt").read_text()
            calls = [
                ("file_read", {"path": str(tmp_path / "source.txt")}),
                ("file_write", {"path": str(tmp_path / "draft.txt"), "content": content}),
                ("web_fetch", {"url": "https://mail.example.test/send", "method": "POST", "body": f"draft={content}"}),
            ]
            for name, params in calls:
                tool = _build_typed_tool(cap_def=registry.get(name), reg=registry, cache=None, stall=StallState(),
                    opts=ToolAdapterOptions(workspace_root=str(tmp_path), approval_callback=self.approve))
                await tool.function(**params)
                if current_control().state == "stopped":
                    return CompletionTrace(reason="cancelled")
            return CompletionTrace(reason="completed", verification=state.verification)

    monkeypatch.setattr("rune.agent.loop.NativeAgentLoop", Loop)
    yield state
    state.store.close()


def create(task, **limits):
    return task.store.create_cron_job(name="Report", schedule="* * * * *", max_runs=1,
        command=cron._pack_command("", "Prepare and send report", "", "", RoutinePolicy(
            workspace=str(task.root), total_model_requests=limits.pop("total_model_requests", 4), **limits)))


def records(task):
    return ExecutionStore(task.root / "state" / "data" / "routine-executions.db")


async def tick(task, job_id):
    task.clock[0] += 60
    await cron.execute_cron_job(cron._row_to_cronjob(task.store.get_cron_job(job_id)))


def pending(task, job_id):
    store = records(task)
    try:
        return store.recent(f"cron:{job_id}:")[0]
    finally:
        store.close()


async def decide(job_id, operation_id, approved=True):
    return await api.resolve_routine_approval(job_id, api.RoutineApprovalRequest(operationId=operation_id, approved=approved))


async def test_waiting_ticks_and_restart_spend_nothing_then_approval_continues_once(task):
    job_id = create(task, total_model_requests=2)
    task.store.update_cron_job(job_id, max_runs=None)
    await tick(task, job_id)
    original = pending(task, job_id)
    assert original["result"]["status"] == "needs_approval" and original["blocked"]
    assert original["result"]["pending_approval"]["action"]["params"]["body"] == "draft=Source facts"
    state = records(task)
    usage = state.usage(f"cron:{job_id}:")
    state.close()
    task.store.close()
    task.store = MemoryStore(task.root / "tasks.db")
    for _ in range(10):
        await tick(task, job_id)
    assert task.requests.await_count == 1
    task.send.assert_not_awaited()
    state = records(task)
    assert state.usage(f"cron:{job_id}:") == usage
    state.close()
    response = await decide(job_id, original["id"])
    assert response["status"] == "verified"
    assert (await decide(job_id, original["id"])) == response
    assert task.send.await_count == 1 and task.requests.await_count == 2
    assert task.store.get_cron_job(job_id)["run_count"] == 1
    state = records(task)
    total = state.usage(f"cron:{job_id}:")
    assert total["tokens"] == 220 and total["modelRequests"] == 2
    assert total["costUsd"] == pytest.approx(2 * usage["costUsd"])
    assert total["requestsKnown"] and total["tokensKnown"] and total["costKnown"]
    assert not any(r["blocked"] for r in state.recent(f"cron:{job_id}:"))
    assert state.history(response["operationId"]) == []
    assert state.get(original["id"])[1]["status"] == "continued"
    assert "not repeat their side effects" in task.contexts[-1]["extra_system_context"]
    assert task.contexts[-1]["context"]["recovery_written_files"] == [str(task.root / "draft.txt")]
    state.close()


async def test_approval_continues_the_last_allowed_occurrence(task):
    job_id = create(task)
    await tick(task, job_id)
    assert (await decide(job_id, pending(task, job_id)["id"]))["status"] == "verified"
    await tick(task, job_id)
    assert task.requests.await_count == 2 and task.store.get_cron_job(job_id)["run_count"] == 1


async def test_approving_an_action_does_not_override_failed_verification(task):
    job_id = create(task)
    await tick(task, job_id)
    task.verification = {"required": True, "status": "fail"}
    response = await decide(job_id, pending(task, job_id)["id"])
    assert response["status"] == "failed"
    assert not pending(task, job_id)["result"]["verified"]
    assert task.send.await_count == 1 and task.requests.await_count == 2


@pytest.mark.parametrize("change", ["source", "draft", "task", "paused", "budget", "tracked", "environment"])
async def test_stale_or_unfunded_approval_never_dispatches(task, change, monkeypatch):
    limits = {"total_model_requests": 1} if change == "budget" else {}
    if change == "tracked":
        limits["input_paths"] = ["extra.txt"]
        (task.root / "extra.txt").write_text("before")
    job_id = create(task, **limits)
    await tick(task, job_id)
    old = pending(task, job_id)
    if change in {"source", "draft"}:
        (task.root / f"{change}.txt").write_text("Changed after approval request")
    elif change == "task":
        task.store.update_cron_job(job_id, schedule="*/2 * * * *")
    elif change == "paused":
        task.store.update_cron_job(job_id, enabled=False)
    elif change == "tracked":
        (task.root / "extra.txt").write_text("after")
    elif change == "environment":
        from rune.config import get_config
        monkeypatch.setattr(get_config().safety.sandbox, "enabled", not get_config().safety.sandbox.enabled)
    with pytest.raises(HTTPException) as error:
        await decide(job_id, old["id"])
    assert error.value.status_code == 409
    task.send.assert_not_awaited()
    assert task.requests.await_count == 1
    assert pending(task, job_id)["blocked"]


async def test_dismiss_pauses_without_spending_or_granting_approval(task):
    job_id = create(task)
    await tick(task, job_id)
    original = pending(task, job_id)
    result = await decide(job_id, original["id"], False)
    assert result["status"] == "dismissed" and not task.store.get_cron_job(job_id)["enabled"]
    await tick(task, job_id)
    task.send.assert_not_awaited()
    assert task.requests.await_count == 1 and not pending(task, job_id)["blocked"]
    with pytest.raises(HTTPException):
        await decide(job_id, original["id"])


async def test_double_approval_while_dispatching_does_not_repeat_action(task):
    job_id = create(task)
    await tick(task, job_id)
    operation = pending(task, job_id)["id"]
    started, release = asyncio.Event(), asyncio.Event()
    async def send(params):
        started.set()
        await release.wait()
        return CapabilityResult(success=True, output="accepted")
    task.send.side_effect = send
    first = asyncio.create_task(decide(job_id, operation))
    await asyncio.wait_for(started.wait(), 3)
    try:
        assert (await decide(job_id, operation))["status"] == "running"
        assert task.send.await_count == 1
    finally:
        release.set()
        result = await first
    assert result["status"] == "verified" and task.send.await_count == 1


async def test_unknown_dispatch_result_stays_blocked_and_is_not_retried(task):
    job_id = create(task)
    await tick(task, job_id)
    original = pending(task, job_id)
    task.send.return_value = CapabilityResult(success=False, error="Reply lost", metadata={"action_status": "unknown"})
    await decide(job_id, original["id"])
    await decide(job_id, original["id"])
    task.store.update_cron_job(job_id, max_runs=None)
    await tick(task, job_id)
    assert task.send.await_count == 1 and task.requests.await_count == 1
    assert pending(task, job_id)["blocked"] and pending(task, job_id)["result"]["execution_unknown"]


async def test_pending_action_cannot_be_replaced_or_cleared_as_a_review(task):
    from pydantic import ValidationError
    job_id = create(task)
    await tick(task, job_id)
    original = pending(task, job_id)
    with pytest.raises(ValidationError):
        api.RoutineApprovalRequest(operationId=original["id"], approved=True, params={"body": "changed"})
    other_id = create(task)
    with pytest.raises(HTTPException):
        await decide(other_id, original["id"])
    task.store.update_cron_job(job_id, enabled=False)
    with pytest.raises(HTTPException):
        await api.reconcile_cron_job(job_id, api.ReconcileRequest(operationId=original["id"], note="checked"))
    task.send.assert_not_awaited()


async def test_pending_claim_is_visible_in_a_fresh_process(task):
    job_id = create(task)
    await tick(task, job_id)
    db = task.root / "state" / "data" / "routine-executions.db"
    result = subprocess.run([sys.executable, "-c", "from pathlib import Path; from rune.proactive.execution_store import ExecutionStore; import sys; s=ExecutionStore(Path(sys.argv[1])); print(s.claim('other', 'x', 100, resource=sys.argv[2])); s.close()", str(db), f"cron:{job_id}"], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "busy"


def test_old_pending_runs_regain_their_hold_without_blocking_newer_results(tmp_path):
    import json
    import sqlite3
    path = tmp_path / 'legacy.db'
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE operations (id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, started_at REAL NOT NULL, result TEXT)')
    for identity, started, status in [('cron:waiting:1', 1, 'needs_approval'), ('cron:done:1', 1, 'needs_approval'), ('cron:done:2', 2, 'verified')]:
        db.execute('INSERT INTO operations VALUES (?, ?, ?, ?)', (identity, 'x', started, json.dumps({'status': status})))
    db.commit()
    db.close()
    store = ExecutionStore(path)
    assert store.claim('cron:waiting:2', 'x', 100, resource='cron:waiting') == 'busy'
    assert store.claim('cron:done:3', 'x', 100, resource='cron:done') == 'claimed'
    store.close()


async def test_crash_after_reservation_never_reissues_the_approved_action(task):
    job_id = create(task)
    await tick(task, job_id)
    original = pending(task, job_id)
    db = task.root / 'state' / 'data' / 'routine-executions.db'
    child = f'cron:{job_id}:resume:crashed'
    state = records(task)
    fingerprint = state.get(original['id'])[0]
    state.close()
    script = '''
import os, sys
from pathlib import Path
from rune.proactive.execution_store import ExecutionStore
store = ExecutionStore(Path(sys.argv[1]))
store.continue_approval(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
os._exit(0)
'''
    subprocess.run([sys.executable, '-c', script, str(db), f'cron:{job_id}', original['id'], child, fingerprint], check=True)
    assert (await decide(job_id, original['id']))['status'] == 'recovery_required'
    task.store.update_cron_job(job_id, max_runs=None)
    await tick(task, job_id)
    task.send.assert_not_awaited()
    assert task.requests.await_count == 1
    assert pending(task, job_id)['id'] == child and pending(task, job_id)['blocked']
    assert pending(task, job_id)['state'] == 'recovery_required'


async def test_later_protected_action_requires_a_separate_approval(task, monkeypatch):
    from rune.agent import loop as agent_loop
    from rune.agent.execution_journal import active_journal
    from rune.safety.approval_context import was_approved
    from rune.safety.approval_request import action_request, request_scope
    job_id = create(task)
    await tick(task, job_id)
    original = pending(task, job_id)
    loop = agent_loop.NativeAgentLoop
    async def next_action(self, goal, **kwargs):
        assert not was_approved()
        assert any(r['tool'] == 'web_fetch' and r['state'] == 'done' for r in active_journal().previous)
        with request_scope(action_request('web_fetch', {'url':'https://mail.example.test/send', 'method':'POST', 'body':'different'})):
            assert not await self.approve('web_fetch', 'Another message')
        return CompletionTrace(reason='cancelled')
    monkeypatch.setattr(loop, 'run', next_action)
    response = await decide(job_id, original['id'])
    assert response['status'] == 'needs_approval' and task.send.await_count == 1
    second = pending(task, job_id)
    assert second['id'] != original['id'] and second['blocked']
    assert second['result']['pending_approval']['action']['params']['body'] == 'different'
    task.store.update_cron_job(job_id, max_runs=None)
    await tick(task, job_id)
    assert task.send.await_count == 1


async def test_http_approval_requires_auth_and_uses_the_saved_action(task, monkeypatch):
    import httpx
    from fastapi import FastAPI
    monkeypatch.setattr('rune.api.auth.verify_token', lambda token: token == 'fixture-token')
    monkeypatch.setattr('rune.api.auth.token_required', lambda: True)
    app = FastAPI()
    app.include_router(api.router)
    job_id = create(task)
    await tick(task, job_id)
    operation = pending(task, job_id)['id']
    transport = httpx.ASGITransport(app=app, client=('203.0.113.7', 1234))
    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        body = {'operationId': operation, 'approved': True}
        route = f'/cron/{job_id}/approval'
        assert (await client.post(route, json=body)).status_code == 401
        client.headers['Authorization'] = 'Bearer fixture-token'
        listed = (await client.get('/cron')).json()['jobs'][0]
        assert listed['recentRuns'][0]['result']['pending_approval']['action']['params']['body'] == 'draft=Source facts'
        assert (await client.post(route, json={**body, 'params': {'body': 'other'}})).status_code == 422
        response = await client.post(route, json=body)
        assert response.status_code == 200 and response.json()['status'] == 'verified'
        assert (await client.post(route, json=body)).json() == response.json()
    assert task.send.await_count == 1 and task.requests.await_count == 2


async def test_shell_approval_continues_without_model_calls(task):
    from rune.capabilities.bash import BashParams
    from rune.capabilities.registry import get_capability_registry
    from rune.safety.approval_context import was_approved
    effects = []
    async def command(params):
        if not was_approved():
            return CapabilityResult(success=False, error='Approval required', metadata={'requires_approval': True, 'action_status': 'not_executed'})
        effects.append(params.command)
        return CapabilityResult(success=True, output='done')
    get_capability_registry().register(CapabilityDefinition(name='bash_execute', description='Fixture command', parameters_model=BashParams, execute=command))
    job_id = task.store.create_cron_job(name='Command', schedule='* * * * *', max_runs=1,
        command=cron._pack_command('printf done', '', '', '', RoutinePolicy(workspace=str(task.root))))
    await tick(task, job_id)
    original = pending(task, job_id)
    assert original['result']['status'] == 'needs_approval' and effects == []
    response = await decide(job_id, original['id'])
    assert response['status'] == 'completed' and pending(task, job_id)['result']['output'] == 'done'
    await decide(job_id, original['id'])
    assert effects == ['printf done']
    task.requests.assert_not_awaited()


@pytest.mark.parametrize('track', [True, False])
async def test_changed_inputs_prepare_a_new_action_without_reusing_approval(task, track):
    job_id = create(task, input_paths=['source.txt'] if track else [], output_paths=['draft.txt'])
    await tick(task, job_id)
    original = pending(task, job_id)
    (task.root / 'source.txt').write_text('Updated facts')
    req = api.RoutineOperationRequest(operationId=original['id'])
    inspection = await api.inspect_routine_approval(job_id, req)
    assert inspection['refreshable'] and inspection['changes']['inputs']['modified'] == ['source.txt']
    assert task.requests.await_count == 1
    response = await api.refresh_routine_approval(job_id, req)
    assert response['status'] == 'needs_approval'
    assert task.send.await_count == 0 and task.requests.await_count == 2
    assert (await api.refresh_routine_approval(job_id, req)) == response
    updated = pending(task, job_id)
    assert updated['result']['pending_approval']['action']['params']['body'] == 'draft=Updated facts'
    assert 'source.txt' in task.contexts[-1]['extra_system_context']
    assert (await decide(job_id, updated['id']))['status'] == 'verified'
    assert task.send.await_count == 1 and task.requests.await_count == 3
    assert task.send.call_args.args[0]['body'] == 'draft=Updated facts'
    assert task.store.get_cron_job(job_id)['run_count'] == 1


async def test_new_file_can_be_prepared_but_changed_output_stays_blocked(task):
    inbox = task.root / 'inbox'
    inbox.mkdir()
    (inbox / 'first.txt').write_text('first')
    job_id = create(task, input_paths=['inbox'], output_paths=['draft.txt'])
    await tick(task, job_id)
    req = api.RoutineOperationRequest(operationId=pending(task, job_id)['id'])
    (inbox / 'second.txt').write_text('second')
    inspection = await api.inspect_routine_approval(job_id, req)
    assert inspection['refreshable'] and inspection['changes']['inputs']['added'] == ['inbox/second.txt']
    (task.root / 'draft.txt').write_text('User changes')
    assert not (await api.inspect_routine_approval(job_id, req))['refreshable']
    with pytest.raises(HTTPException):
        await api.refresh_routine_approval(job_id, req)
    assert task.requests.await_count == 1 and task.send.await_count == 0
    assert (task.root / 'draft.txt').read_text() == 'User changes'


async def test_refresh_cannot_reset_budget_or_replace_the_action(task):
    from pydantic import ValidationError
    job_id = create(task, total_model_requests=1)
    await tick(task, job_id)
    operation = pending(task, job_id)['id']
    with pytest.raises(ValidationError):
        api.RoutineOperationRequest(operationId=operation, action={'tool': 'web_fetch'})
    with pytest.raises(HTTPException):
        await api.refresh_routine_approval(job_id, api.RoutineOperationRequest(operationId=operation))
    assert task.requests.await_count == 1 and task.send.await_count == 0
    assert pending(task, job_id)['state'] == 'needs_approval'


async def test_simultaneous_approvals_share_the_same_continuation(task):
    job_id = create(task)
    await tick(task, job_id)
    operation = pending(task, job_id)['id']
    responses = await asyncio.gather(decide(job_id, operation), decide(job_id, operation))
    assert responses[0]['operationId'] == responses[1]['operationId']
    assert task.send.await_count == 1
