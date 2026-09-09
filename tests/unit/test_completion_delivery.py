"""Completion timing, conversation persistence, and deferred memory recovery."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from rune.agent import agent_context
from rune.api.run_maintenance import RunMaintenance
from rune.api.run_snapshot import RunSnapshots
from rune.api.run_store import RunStore
from rune.conversation.store import ConversationStore
from tests.unit.test_api_server_multiturn import (
    FakeLoop,
    _turn_rows,
    _wait_for,
)
from tests.unit.test_api_server_multiturn import client as client
from tests.unit.test_api_server_multiturn import isolated_wiring as isolated_wiring


@pytest.mark.parametrize("transport", ["web", "rest", "ndjson"])
@pytest.mark.parametrize("blocked_stage", ["learning", "embedding"])
def test_completed_turn_is_saved_and_followup_runs_while_maintenance_waits(
    client, monkeypatch, tmp_path, transport, blocked_stage,
):
    release = asyncio.Event()
    entered, completed, learned = [], [], []
    record = RunSnapshots.record

    def capture(self, event, data):
        result = record(self, event, data)
        if event == "agent_complete":
            assert _turn_rows(tmp_path / "conversations.db")[-1][0] == "assistant"
            completed.append(data)
        return result

    async def learn(inp):
        learned.append(inp.context.goal)
        if blocked_stage == "learning":
            entered.append(True)
            await release.wait()

    async def embed(self, turns):
        if blocked_stage == "embedding":
            entered.append(True)
            await release.wait()

    monkeypatch.setattr(RunSnapshots, "record", capture)
    monkeypatch.setattr(agent_context, "post_process_agent_result", learn)
    monkeypatch.setattr(ConversationStore, "_embed_new_turns", embed)

    def send(goal):
        if transport == "web":
            return client.post("/api/message", json={"text": goal, "sessionId": "followup"})
        return client.post("/api/v1/agent/execute", json={
            "goal": goal, "session_id": "followup", "stream": transport == "ndjson",
        })

    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            first = pool.submit(send, "remember heron")
            assert _wait_for(lambda: entered)
            assert first.result(timeout=1).status_code == 200
            assert _wait_for(lambda: len(completed) == 1, timeout=1)
            second = pool.submit(send, "what did I say?")
            assert second.result(timeout=1).status_code == 200
            assert _wait_for(lambda: len(completed) == 2, timeout=1)
            assert "answer to: remember heron" in str(FakeLoop.captured[-1]["history"])
            assert len(_turn_rows(tmp_path / "conversations.db")) == 4
            assert learned == ["remember heron"]
        finally:
            client.portal.call(release.set)
    assert _wait_for(lambda: len(learned) == 2)
    assert len(_turn_rows(tmp_path / "conversations.db")) == 4


@pytest.mark.parametrize("stream", [False, True])
def test_failed_conversation_save_never_reports_completion(client, monkeypatch, stream):
    save = ConversationStore.save

    async def fail_assistant(self, conversation, **kwargs):
        if conversation.turns[-1].role == "assistant":
            raise OSError("disk full")
        await save(self, conversation, **kwargs)

    monkeypatch.setattr(ConversationStore, "save", fail_assistant)
    response = client.post("/api/v1/agent/execute", json={
        "goal": "answer", "session_id": "failed-save", "stream": stream,
    })
    if stream:
        assert '"event":"agent_error"' in response.text
        assert '"event":"agent_complete"' not in response.text
    else:
        assert response.json()["status"] == "failed"


def test_text_does_not_complete_a_run_until_verification_returns(client, monkeypatch):
    import rune.agent.loop as loop_module

    release = asyncio.Event()
    emitted, completed = [], []
    record = RunSnapshots.record

    def capture(self, event, data):
        if event == "agent_complete":
            completed.append(data)
        return record(self, event, data)

    class CheckingLoop(FakeLoop):
        def on(self, event, cb):
            if event == "text_delta":
                self.on_text = cb

        async def run(self, *args, **kwargs):
            trace = await super().run(*args, **kwargs)
            await self.on_text("answer ready")
            emitted.append(True)
            await release.wait()
            trace.verification = {"status": "fail"}
            trace.reason = "verification_failed"
            return trace

    monkeypatch.setattr(loop_module, "NativeAgentLoop", CheckingLoop)
    monkeypatch.setattr(RunSnapshots, "record", capture)
    try:
        client.post("/api/message", json={"text": "verify the answer", "sessionId": "checking"})
        assert _wait_for(lambda: emitted)
        assert not completed
        assert client.get("/api/runs/snapshot", params={"sessionId": "checking"}).json()["run"]["status"] == "running"
    finally:
        client.portal.call(release.set)
    assert _wait_for(lambda: completed)
    assert completed[0]["success"] is False


def test_ndjson_wakes_when_a_quiet_run_finishes(client, monkeypatch):
    import rune.agent.loop as loop_module

    release = asyncio.Event()
    waiting = []

    class QuietLoop(FakeLoop):
        async def run(self, *args, **kwargs):
            trace = await super().run(*args, **kwargs)
            waiting.append(True)
            await release.wait()
            return trace

    monkeypatch.setattr(loop_module, "NativeAgentLoop", QuietLoop)
    with ThreadPoolExecutor(max_workers=1) as pool:
        request = pool.submit(client.post, "/api/v1/agent/execute", json={"goal": "answer", "stream": True})
        try:
            assert _wait_for(lambda: waiting)
            client.portal.call(release.set)
            assert '"event":"agent_complete"' in request.result(timeout=1).text
        finally:
            client.portal.call(release.set)


async def test_shutdown_retains_work_and_cancels_the_memory_worker(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "conversations.db")
    runs = RunSnapshots(store)
    runs.start("r1", "s1", "first")
    runs.start("r2", "s1", "second")
    entered = asyncio.Event()

    async def stalled(inp):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(agent_context, "post_process_agent_result", stalled)
    service = RunMaintenance(store)
    service.start()
    for run_id in ("r1", "r2"):
        service.enqueue(run_id, agent_context.AgentContext(goal=run_id),
                        SimpleNamespace(reason="completed"), "answer", 1)
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        await service.close(grace_seconds=0)
        assert service._worker.done()
        assert store.db.execute("SELECT stage FROM web_run_maintenance ORDER BY rowid").fetchall() == [
            ("learning",), ("pending",),
        ]
    finally:
        await service.close(grace_seconds=0)
        runs.close()


@pytest.mark.parametrize("interrupted_stage", ["pending", "learning", "indexing"])
async def test_restart_recovers_queued_work_without_repeating_learning(
    tmp_path, monkeypatch, isolated_wiring, interrupted_stage,
):
    path = tmp_path / "conversations.db"
    store = RunStore(path)
    snapshots = RunSnapshots(store)
    snapshots.start("r1", "s1", "first")
    snapshots.start("r2", "s1", "second")
    service = RunMaintenance(store)
    service.start()
    ctx = agent_context.AgentContext(goal="first", conversation_id="s1")
    trace = SimpleNamespace(reason="completed")
    service.enqueue("r1", ctx, trace, "answer", 10, classification_hint="chat")
    ctx.goal = "second"
    service.enqueue("r2", ctx, trace, "answer", 10, classification_hint="chat")
    store.db.execute("UPDATE web_run_maintenance SET stage = ? WHERE run_id = 'r1'", (interrupted_stage,))
    store.db.commit()
    await service.close(grace_seconds=0)
    snapshots.close()

    learned = []

    async def learn(inp):
        learned.append(inp.context.goal)

    monkeypatch.setattr(agent_context, "post_process_agent_result", learn)
    reopened = RunStore(path)
    reopened.open()
    restored = RunMaintenance(reopened)
    restored.start()
    try:
        await restored._worker
        assert learned == (["first", "second"] if interrupted_stage == "pending" else ["second"])
        rows = reopened.db.execute("SELECT stage FROM web_run_maintenance").fetchall()
        assert rows == ([("interrupted",)] if interrupted_stage == "learning" else [])
    finally:
        await restored.close()
        reopened.close()


async def test_failed_learning_does_not_block_the_next_job(tmp_path, monkeypatch, isolated_wiring):
    store = RunStore(tmp_path / "conversations.db")
    runs = RunSnapshots(store)
    service = RunMaintenance(store)
    seen = []

    async def learn(inp):
        seen.append(inp.context.goal)
        if inp.context.goal == "first":
            raise RuntimeError("memory unavailable")

    monkeypatch.setattr(agent_context, "post_process_agent_result", learn)
    runs.start("r1", "s1", "first")
    runs.start("r2", "s1", "second")
    service.start()
    for run_id, goal in [("r1", "first"), ("r2", "second")]:
        service.enqueue(run_id, agent_context.AgentContext(goal=goal),
                        SimpleNamespace(reason="completed"), "answer", 1)
    try:
        await service._worker
        assert seen == ["first", "second"]
        assert store.db.execute("SELECT run_id, stage FROM web_run_maintenance").fetchall() == [("r1", "failed")]
        await isolated_wiring.get_conv_manager()._store.delete("s1")
        assert store.db.execute("SELECT COUNT(*) FROM web_run_maintenance").fetchone()[0] == 0
    finally:
        await service.close()
        runs.close()
