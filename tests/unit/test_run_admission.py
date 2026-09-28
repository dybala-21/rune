"""Rejected submissions must not replace the conversation's running task."""

import asyncio

import pytest

from tests.unit.test_api_server_multiturn import FakeLoop, _wait_for
from tests.unit.test_api_server_multiturn import client as client
from tests.unit.test_api_server_multiturn import isolated_wiring as isolated_wiring


@pytest.fixture
def held_run(client, monkeypatch):
    started = []

    async def hold(self, *args, **kwargs):
        started.append(True)
        await asyncio.Event().wait()

    monkeypatch.setattr(FakeLoop, "run", hold)
    run_id = client.post("/api/message", json={"text": "first", "sessionId": "busy-chat"}).json()["runId"]
    assert _wait_for(lambda: bool(started))
    yield run_id
    client.post("/api/abort", json={"runId": run_id})


@pytest.mark.parametrize("path,payload", [
    ("/api/message", {"text": "second", "sessionId": "busy-chat"}),
    ("/api/v1/agent/execute", {"goal": "second", "session_id": "busy-chat"}),
    ("/api/v1/agent/execute", {"goal": "second", "session_id": "busy-chat", "stream": True}),
])
def test_busy_submission_preserves_the_active_snapshot(client, held_run, path, payload):
    result = client.post(path, json=payload)
    assert result.status_code == 409
    snapshot = client.get("/api/runs/snapshot", params={"sessionId": "busy-chat"}).json()["run"]
    assert snapshot["runId"] == held_run
    assert snapshot["status"] == "running"


def test_missing_abort_target_cannot_stop_another_task(client, held_run):
    assert client.post("/api/abort", json={"runId": ""}).status_code == 400
    active = client.post("/api/v1/rpc", json={"method": "runs.active", "params": {}}).json()
    assert held_run in active["data"]["runIds"]


def test_lost_acknowledgement_replays_pending_and_finished_run(client, monkeypatch):
    calls = []
    release = asyncio.Event()

    async def held(self, *args, **kwargs):
        calls.append(True)
        await release.wait()
        return await original(self, *args, **kwargs)

    original = FakeLoop.run
    monkeypatch.setattr(FakeLoop, "run", held)
    payload = {"text": "hello", "sessionId": "retry-chat", "requestId": "transport-1"}
    first = client.post("/api/message", json=payload).json()
    assert _wait_for(lambda: bool(calls))
    second = client.post("/api/message", json=payload).json()
    assert second["replayed"] and second["runId"] == first["runId"]
    assert len(calls) == 1
    assert client.post("/api/message", json={**payload, "text": "changed"}).status_code == 409
    client.portal.call(release.set)
    assert _wait_for(lambda: client.get("/api/runs/snapshot", params={"sessionId": "retry-chat"}).json()["run"]["status"] == "completed")
    assert client.post("/api/message", json=payload).json()["runId"] == first["runId"]
    assert len(calls) == 1
    repeat = client.post("/api/message", json={**payload, "requestId": "intentional-2"}).json()
    assert repeat["runId"] != first["runId"]
    assert _wait_for(lambda: len(calls) == 2)


def test_attachment_references_replay_and_restore_across_turns(client, tmp_path, monkeypatch):
    import base64
    import json
    import sqlite3

    contexts = []
    original = FakeLoop.run

    async def capture(self, *args, **kwargs):
        contexts.append(kwargs.get("context") or {})
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(FakeLoop, "run", capture)
    attachment = {"name": "매출.csv", "mimeType": "text/csv", "data": base64.b64encode(b"name,value\na,12\n").decode()}
    body = {"text": "read the table", "sessionId": "with-files", "requestId": "upload", "attachments": [attachment]}
    first = client.post("/api/message", json=body)
    assert first.status_code == 200
    refs = first.json()["attachments"]
    assert refs[0]["ref"] and "data" not in refs[0]
    assert _wait_for(lambda: client.get("/api/runs/snapshot", params={"sessionId": "with-files"}).json()["run"]["status"] == "completed")
    assert contexts[0]["attachments"][0]["data"] == attachment["data"]
    assert len(list(tmp_path.glob("rune-attachment-*.csv"))) == 1
    replay = client.post("/api/message", json={**body, "attachments": refs}).json()
    assert replay["replayed"] and replay["runId"] == first.json()["runId"]
    repeated = client.post("/api/message", json={**body, "requestId": "regenerate", "attachments": refs})
    assert repeated.status_code == 200
    assert _wait_for(lambda: len(contexts) == 2)
    assert len(list(tmp_path.glob("rune-attachment-*.csv"))) == 1
    assert client.post("/api/message", json={**body, "sessionId": "other", "attachments": refs}).status_code == 409
    assert _wait_for(lambda: client.get("/api/runs/snapshot", params={"sessionId": "with-files"}).json()["run"]["status"] == "completed")
    turns = client.post("/api/v1/rpc", json={"method": "sessions.turns", "params": {"sessionId": "with-files"}}).json()["data"]["turns"]
    user_turns = [t for t in turns if t["role"] == "user"]
    assert len(user_turns) == 2
    assert all(t["content"] == body["text"] and t["attachments"] == refs for t in user_turns)
    snapshot = client.get("/api/runs/snapshot", params={"sessionId": "with-files"}).json()["run"]
    assert all('data' not in a for a in snapshot["execution"]["attachments"])
    assert attachment["data"] not in json.dumps(snapshot)
    with sqlite3.connect(tmp_path / "conversations.db") as db:
        assert db.execute("SELECT COUNT(*) FROM web_attachments").fetchone()[0] == 1
