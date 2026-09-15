"""Published checks survive real web and streaming cancellation paths."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from tests.unit.test_api_server_multiturn import (
    FakeLoop,
    _wait_for,
)
from tests.unit.test_api_server_multiturn import (
    client as client,
)
from tests.unit.test_api_server_multiturn import (
    isolated_wiring as isolated_wiring,
)

RECEIPT = {"kind": "document_bundle", "revision": "a" * 32}


@pytest.fixture
def events(monkeypatch):
    from rune.api.server import WsClientManager
    from rune.conversation.store import ConversationStore

    received = []

    async def capture(self, event, data):
        received.append({"event": event, "data": data})

    async def no_embeddings(self, turns):
        return None

    monkeypatch.setattr(WsClientManager, "broadcast", capture)
    monkeypatch.setattr(ConversationStore, "_embed_new_turns", no_embeddings)
    return received


@pytest.mark.parametrize("channel", ["web", "ndjson"])
def test_cancelled_trace_delivers_checks(client, monkeypatch, events, channel):
    async def run(self, *args, **kwargs):
        return SimpleNamespace(reason="cancelled", artifact_receipts=[RECEIPT])

    monkeypatch.setattr(FakeLoop, "run", run)
    if channel == "ndjson":
        response = client.post("/api/v1/agent/execute", json={"goal": "publish then stop", "stream": True})
        assert response.status_code == 200
        events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    else:
        assert client.post("/api/message", json={"text": "publish then stop"}).status_code == 200
        assert _wait_for(lambda: any(e["event"] == "agent_aborted" for e in events))
    aborted = next(e["data"] for e in events if e["event"] == "agent_aborted")
    assert aborted["trust"]["artifactReceipts"] == [RECEIPT]
    assert aborted["trust"]["completionStatus"] == "cancelled"
    assert not aborted["trust"]["verified"] and not aborted["trust"]["canEscalate"]
    assert not any(e["event"] == "agent_complete" for e in events)


@pytest.mark.parametrize("late_receipt", [False, True])
def test_stop_endpoint_preserves_checks_and_final_snapshot(client, monkeypatch, events, late_receipt):
    started = []
    later = {"kind": "document_bundle", "revision": "b" * 32}

    async def run(self, *args, **kwargs):
        self.artifact_receipts = [RECEIPT]
        started.append(True)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if not late_receipt:
                raise
            self.artifact_receipts.append(later)
            return SimpleNamespace(reason="cancelled", artifact_receipts=self.artifact_receipts)

    monkeypatch.setattr(FakeLoop, "run", run)
    response = client.post("/api/message", json={"text": "publish and keep working"})
    assert response.status_code == 200
    assert _wait_for(lambda: bool(started))
    run_id = next(e["data"]["runId"] for e in events if e["event"] == "agent_start")
    assert client.post("/api/abort", json={"runId": run_id}).json()["ok"]
    assert _wait_for(lambda: not client.post("/api/v1/rpc", json={"method": "runs.active", "params": {}})
                     .json()["data"]["runIds"])
    snapshots = [e["data"] for e in events if e["event"] == "agent_aborted"]
    assert snapshots[-1]["trust"]["artifactReceipts"] == ([RECEIPT, later] if late_receipt else [RECEIPT])
    assert len(snapshots) == 1
    assert not any(e["event"] in {"agent_complete", "agent_error"} for e in events)


def test_stop_keeps_the_journal_writable_until_the_dispatched_action_settles(client, monkeypatch, events):
    from rune.agent.execution_journal import active_journal

    started, attempts = [], []

    async def run(self, *args, **kwargs):
        journal = active_journal()

        async def write():
            started.append(True)
            await asyncio.Event().wait()

        try:
            await journal.execute("browser_act", {"action": "click", "selector": "save"}, write)
        finally:
            attempts.extend(journal.store.attempts(journal.run_id))

    monkeypatch.setattr(FakeLoop, "run", run)
    response = client.post("/api/message", json={"text": "save once"})
    assert _wait_for(lambda: bool(started))
    assert client.post("/api/abort", json={"runId": response.json()["runId"]}).json()["ok"]
    assert len(attempts) == 1 and attempts[0]["state"] == "unknown"
    assert any(event["event"] == "agent_aborted" for event in events)
    assert not any(event["event"] == "agent_error" for event in events)


def test_shutdown_records_unknown_effects_before_marking_the_run_interrupted(isolated_wiring, monkeypatch, tmp_path):
    import sqlite3

    from starlette.testclient import TestClient

    from rune.agent.execution_journal import active_journal
    from rune.api.server import create_app

    started, attempts = [], []

    async def run(self, *args, **kwargs):
        journal = active_journal()

        async def write():
            started.append(True)
            await asyncio.Event().wait()

        try:
            await journal.execute("browser_act", {"action": "click", "selector": "save"}, write)
        finally:
            attempts.extend(journal.store.attempts(journal.run_id))

    monkeypatch.setattr(FakeLoop, "run", run)
    monkeypatch.setattr("rune.agent.loop.NativeAgentLoop", FakeLoop)
    with TestClient(create_app(), client=("127.0.0.1", 50000)) as client:
        response = client.post("/api/message", json={"text": "save once", "sessionId": "shutdown"})
        assert _wait_for(lambda: bool(started))
        run_id = response.json()["runId"]
    assert len(attempts) == 1 and attempts[0]["state"] == "unknown"
    with sqlite3.connect(tmp_path / "conversations.db") as db:
        assert db.execute("SELECT status FROM web_runs WHERE run_id = ?", (run_id,)).fetchone()[0] == "interrupted"


@pytest.mark.parametrize("late_receipt", [False, True])
def test_stop_interrupts_ndjson_and_returns_final_receipts(client, monkeypatch, events, late_receipt):
    started = []
    later = {"kind": "document_bundle", "revision": "c" * 32}

    async def run(self, *args, **kwargs):
        self.artifact_receipts = [RECEIPT]
        started.append(True)
        try:
            await asyncio.wait_for(asyncio.Event().wait(), timeout=5)
        except asyncio.CancelledError:
            if not late_receipt:
                raise
            self.artifact_receipts.append(later)
            return SimpleNamespace(reason="cancelled", artifact_receipts=self.artifact_receipts)

    monkeypatch.setattr(FakeLoop, "run", run)
    with ThreadPoolExecutor(max_workers=1) as executor:
        request = executor.submit(client.post, "/api/v1/agent/execute",
                                  json={"goal": "publish then wait", "stream": True})
        assert _wait_for(lambda: bool(started))
        run_id, = client.post("/api/v1/rpc", json={"method": "runs.active", "params": {}}).json()["data"]["runIds"]
        assert client.post("/api/abort", json={"runId": run_id}).json()["ok"]
        response = request.result(timeout=5)
    assert response.status_code == 200
    stream = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    aborted = next(e["data"] for e in stream if e["event"] == "agent_aborted")
    expected = [RECEIPT, later] if late_receipt else [RECEIPT]
    assert aborted["trust"]["artifactReceipts"] == expected
    assert aborted["trust"]["completionStatus"] == "cancelled"
    assert not any(e["event"] in {"agent_complete", "agent_error"} for e in stream)
    assert [e for e in events if e["event"] == "agent_aborted"][-1]["data"]["trust"]["artifactReceipts"] == expected
    assert client.post("/api/abort", json={"runId": run_id}).json()["stopped"] is False
