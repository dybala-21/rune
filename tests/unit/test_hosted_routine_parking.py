"""A scheduled run on a hosted worker stops at a connector approval and continues once the owner decides."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rune.agent.run_control import current_control
from rune.agent.timing import timed_completion
from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
from rune.capabilities import cron
from rune.capabilities.connector import register_connector_capabilities
from rune.capabilities.registry import CapabilityRegistry
from rune.connectors.store import authority_key
from rune.daemon import external_schedule
from rune.memory.store import MemoryStore
from rune.proactive.execution_store import ExecutionStore
from rune.proactive.routine import RoutinePolicy
from rune.types import CompletionTrace

APPROVAL = "a" * 32
PARAMS = {"connector": "mail", "origin": "https://mail.test", "revision": "b" * 32, "method": "POST",
          "path": "/send", "body": "{\"to\":\"x\"}"}
PENDING = {"success": False, "error": "Waiting for the owner", "metadata": {
    "action_status": "not_executed", "approval_status": "pending", "approval_id": APPROVAL,
    "approval_url": "https://approve.example.com/"}}
RAN = {"success": True, "output": "sent", "metadata": {"status_code": 200}}


@pytest.fixture
def task(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("RUNE_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("RUNE_ISOLATION_ROOT", str(tmp_path))
    monkeypatch.setenv("RUNE_APPROVAL_MODE", "standard")
    monkeypatch.setenv("RUNE_BROKER_HOME", str(tmp_path / "broker"))
    monkeypatch.setenv("RUNE_CONNECTOR_URL", "http://10.207.0.1:4100")
    authority_key(tmp_path / "broker", create=True)
    monkeypatch.setattr("rune.agent.request_budget.input_estimate", lambda _: 10)
    # Token counts only size the recovery context here; keep the tokenizer's download out of it.
    monkeypatch.setattr("rune.utils.tokenizer._get_encoding", lambda model: SimpleNamespace(
        encode=lambda text, **_: text.split(), decode=lambda tokens: " ".join(tokens)))
    state = SimpleNamespace(root=tmp_path, store=MemoryStore(tmp_path / "tasks.db"), clock=[1000.0],
                            requests=AsyncMock(return_value={"usage": {"prompt_tokens": 100, "completion_tokens": 10}}),
                            answers=[PENDING], sent=[])

    async def broker(operation, payload):
        state.sent.append(payload)
        return state.answers.pop(0) if len(state.answers) > 1 else state.answers[0]

    monkeypatch.setattr("rune.capabilities.connector.call_broker", broker)
    monkeypatch.setattr(cron, "_get_store", lambda: state.store)
    monkeypatch.setattr("rune.proactive.routine.time", SimpleNamespace(time=lambda: state.clock[0]))
    registry = CapabilityRegistry()
    register_connector_capabilities(registry)
    monkeypatch.setattr("rune.capabilities.registry.get_capability_registry", lambda: registry)

    class Loop:
        def __init__(self, config):
            self._last_answer_text = "Mail sent"

        def set_approval_callback(self, callback):
            self.approve = callback

        async def emit(self, *args):
            pass

        async def run(self, goal, **kwargs):
            from rune.agent.loop import StallState
            await timed_completion(state.requests, {"model": "xai/grok-4.6", "messages": [], "max_tokens": 100})
            tool = _build_typed_tool(cap_def=registry.get("connector_request"), reg=registry, cache=None,
                                     stall=StallState(),
                                     opts=ToolAdapterOptions(workspace_root=str(tmp_path), approval_callback=self.approve))
            answer = await tool.function(**PARAMS)
            if current_control().state == "stopped":
                return CompletionTrace(reason="cancelled")
            self._last_answer_text = answer
            return CompletionTrace(reason="completed", verification={"required": True, "status": "pass"})

    monkeypatch.setattr("rune.agent.loop.NativeAgentLoop", Loop)
    yield state
    state.store.close()


def create(task):
    return task.store.create_cron_job(name="Mail", schedule="* * * * *", max_runs=None,
        command=cron._pack_command("", "Send the weekly mail", "", "", RoutinePolicy(
            workspace=str(task.root), total_model_requests=4)))


def runs(task, job_id):
    store = ExecutionStore(task.root / "state" / "data" / "routine-executions.db")
    try:
        return store.recent(f"cron:{job_id}:")
    finally:
        store.close()


async def tick(task, job_id):
    task.clock[0] += 60
    await cron.execute_cron_job(cron._row_to_cronjob(task.store.get_cron_job(job_id)))


async def decided(approval_id=APPROVAL, approved=True):
    """What the host does once the owner decides: continue, and wait for the continuation."""
    answer = await external_schedule.continue_decided(approval_id, approved)
    async with asyncio.timeout(10):
        while (running := external_schedule._inflight.get(f"approval:{approval_id}")) is not None:
            await asyncio.wait({running})
    return answer


async def test_the_run_stops_at_the_approval_and_continues_once_after_the_owner_approves(task):
    job_id = create(task)
    await tick(task, job_id)
    parked = runs(task, job_id)[0]
    assert parked["result"]["status"] == "needs_approval" and parked["blocked"]
    action = parked["result"]["pending_approval"]["action"]
    assert action["tool"] == "connector_request" and action["brokerApprovalId"] == APPROVAL
    assert action["params"]["body"] == PARAMS["body"]
    assert task.sent == [{**task.sent[0], "wait": "background"}], "asked once, as a background request"

    # Further ticks while it waits send nothing: the blocked task doesn't run again.
    for _ in range(3):
        await tick(task, job_id)
    assert len(task.sent) == 1 and task.requests.await_count == 1

    task.answers = [RAN]
    assert (await decided())["state"] == "accepted"
    assert [p["body"] for p in task.sent] == [PARAMS["body"]] * 2, "the same request, sent once more"
    assert task.sent[1]["wait"] == "background"
    after = await external_schedule.continue_decided(APPROVAL, True)
    assert after["state"] in {"verified", "completed"}, after
    assert len(task.sent) == 2, "asking again reports the continuation and sends nothing"
    assert not any(r["blocked"] for r in runs(task, job_id))


async def test_a_decline_dismisses_the_run_and_pauses_the_job_without_sending_anything(task):
    job_id = create(task)
    await tick(task, job_id)
    answer = await decided(approved=False)
    assert answer["state"] == "dismissed"
    assert len(task.sent) == 1, "nothing is sent again"
    assert not task.store.get_cron_job(job_id)["enabled"], "paused, as when declined in the app"
    assert not any(r["blocked"] for r in runs(task, job_id))
    assert (await decided(approved=False))["state"] == "dismissed"
    assert (await decided(approved=True))["state"] == "dismissed", "a decision can't be turned around"
    await tick(task, job_id)
    assert len(task.sent) == 1 and task.requests.await_count == 1


async def test_an_approval_no_run_waits_on_is_reported_unknown(task):
    create(task)
    assert (await external_schedule.continue_decided("c" * 32, True))["state"] == "unknown"
    for bad in ("", "../x", 7, None, "a" * 65):
        with pytest.raises(ValueError):
            await external_schedule.continue_decided(bad, True)
    with pytest.raises(ValueError):
        await external_schedule.continue_decided(APPROVAL, "yes")


async def test_a_run_someone_is_watching_still_waits_in_place(task):
    """Outside a scheduled run nothing is parked: the request goes out without the background flag."""
    registry = CapabilityRegistry()
    register_connector_capabilities(registry)
    tool = _build_typed_tool(cap_def=registry.get("connector_request"), reg=registry, cache=None, stall=None,
                             opts=ToolAdapterOptions(approval_callback=None, on_tool_end=AsyncMock()))
    assert "Waiting for the owner" in await tool.function(**PARAMS)
    assert "wait" not in task.sent[0]
