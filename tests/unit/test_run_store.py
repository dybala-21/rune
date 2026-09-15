import json
import sqlite3
import subprocess
import sys

import pytest
from filelock import Timeout

from rune.api.run_snapshot import RunSnapshots
from rune.api.run_store import RunStore


@pytest.mark.parametrize("boundary", ["question", "approval", "accepted", "written", "completed"])
def test_process_death_restores_committed_progress_without_repeating_work(tmp_path, boundary):
    script = r'''
import json, os, sys
from pathlib import Path
from rune.api.run_snapshot import RunSnapshots
from rune.api.run_store import RunStore
root, boundary = Path(sys.argv[1]), sys.argv[2]
runs = RunSnapshots(RunStore(root / 'conversations.db'))
runs.start('r1', 's1', '정산 보고서')
runs.record('run_context', {'runId': 'r1', 'workspace': str(root)})
for i in range(260):
    runs.record('text_delta', {'runId': 'r1', 'delta': f'{i}:🧾\n'})
kind = 'question' if boundary == 'question' else 'approval_request'
runs.record(kind, {'runId': 'r1', 'id': 'interaction1', 'question': '정산 기준?',
    'command': 'write report.csv', 'expiresAt': 9000000000000})
if boundary in {'accepted', 'written', 'completed'}:
    runs.accept('interaction1', 'response1', {'decision': 'approve', 'userGuidance': ''})
if boundary in {'written', 'completed'}:
    (root / 'report.csv').write_text('team,total\nA,165000\n')
    runs.record('tool_result', {'runId': 'r1', 'artifactReceipts': [{'revision': 'rev1', 'source_sha256': 'source1'}]})
if boundary == 'completed':
    runs.record('agent_complete', {'runId': 'r1', 'answer': '완료', 'success': True,
        'timings': {'totalMs': 100, 'spans': [{'kind': 'model', 'durationMs': 70}], 'deliveryMs': 2},
        'trust': {'completionStatus': 'completed', 'verified': True,
                  'artifactReceipts': [{'revision': 'rev1', 'source_sha256': 'source1'}]}})
(root / 'expected.json').write_text(json.dumps(runs.latest('s1')))
os._exit(23)
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), boundary], timeout=30)
    assert result.returncode == 23
    before = json.loads((tmp_path / "expected.json").read_text())
    restored = RunSnapshots(RunStore(tmp_path / "conversations.db"))
    try:
        run = restored.latest("s1")
        assert run["text"] == "".join(f"{i}:🧾\n" for i in range(260))
        assert run["workspace"] == str(tmp_path)
        assert run["startedAt"] == before["startedAt"]
        assert run["textStartedAt"] == before["textStartedAt"]
        assert run["status"] == ("completed" if boundary == "completed" else "interrupted")
        if boundary == "completed":
            assert run["timings"] == before["timings"]
        assert run["seq"] == before["seq"] + (boundary != "completed")
        assert run["question"] is None and run["approval"] is None
        assert run["interactions"][0]["request"]["expiresAt"] == 9000000000000
        accepted = boundary in {"accepted", "written", "completed"}
        assert restored.replay("interaction1", "response1", {"decision": "approve", "userGuidance": ""}) == accepted
        if accepted:
            with pytest.raises(ValueError):
                restored.replay("interaction1", "response1", {"decision": "deny"})
        assert (tmp_path / "report.csv").exists() == (boundary in {"written", "completed"})
        if boundary in {"written", "completed"}:
            assert run["trust"]["artifactReceipts"] == [{"revision": "rev1", "source_sha256": "source1"}]
            assert (tmp_path / "report.csv").read_text() == "team,total\nA,165000\n"
        assert restored.latest("unrelated") is None
        assert restored.record("agent_complete", {"runId": "r1", "success": True}) is None
    finally:
        restored.close()


def test_failed_acceptance_rolls_back_state_and_response_together(tmp_path):
    store = RunStore(tmp_path / "conversations.db")
    runs = RunSnapshots(store)
    runs.start("r1", "s1", "report")
    runs.record("approval_request", {"runId": "r1", "id": "a1"})
    before = runs.latest("s1")
    store.db.execute("CREATE TEMP TRIGGER full_disk BEFORE UPDATE ON web_run_interactions BEGIN SELECT RAISE(ABORT, 'disk full'); END")
    try:
        with pytest.raises(sqlite3.IntegrityError, match="disk full"):
            runs.accept("a1", "response1", {"decision": "approve"})
        assert runs.latest("s1") == before
        assert not runs.replay("a1", "response1", {"decision": "approve"})
        assert store.db.execute("SELECT seq FROM web_runs").fetchone()[0] == before["seq"]
    finally:
        runs.close()


def test_second_server_cannot_interrupt_the_current_owner(tmp_path):
    path = tmp_path / "conversations.db"
    owner, contender = RunSnapshots(RunStore(path)), RunSnapshots(RunStore(path))
    owner.start("r1", "s1", "report")
    try:
        with pytest.raises(Timeout):
            contender.open()
        assert owner.latest("s1")["status"] == "queued"
    finally:
        contender.close()
        owner.close()


async def test_deleting_a_conversation_removes_recovery_records(tmp_path):
    from rune.conversation.store import ConversationStore
    from rune.conversation.types import Conversation

    path = tmp_path / "conversations.db"
    conversations = ConversationStore(path)
    await conversations.save(Conversation(id="s1", user_id="web:session"), embed=False)
    store = RunStore(path)
    runs = RunSnapshots(store)
    runs.start("r1", "s1", "report")
    runs.record("question", {"runId": "r1", "id": "q1"})
    try:
        await conversations.delete("s1")
        assert runs.latest("s1") is None
        for table in ("web_runs", "web_run_events", "web_run_interactions"):
            assert store.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    finally:
        runs.close()
        conversations._conn.close()


@pytest.mark.parametrize("kind", ["question", "approval"])
def test_api_restart_replays_accepted_input_and_closes_unanswered_input(tmp_path, monkeypatch, kind):
    import asyncio
    import time

    from starlette.testclient import TestClient

    from rune.agent.agent_context import AgentContext
    from rune.api import conversation_wiring
    from rune.api.server import create_app
    from rune.capabilities.ask_user import AskUserParams

    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    conversation_wiring._reset_for_tests()
    writes = []

    async def prepare(options, **kwargs):
        return AgentContext(goal=options.goal, original_goal=options.goal, channel="web",
                            workspace_root=str(tmp_path), conversation_id=options.conversation_id)

    class WaitingLoop:
        def __init__(self, **kwargs):
            self._last_answer_text = ""

        def on(self, *args):
            pass

        def set_ask_user_callback(self, callback):
            self.ask = callback

        def set_approval_callback(self, callback):
            self.approve = callback

        async def run(self, goal, **kwargs):
            if kind == "question":
                await self.ask(AskUserParams(question="정산 기준?", reason="보고서 기준"))
            else:
                assert await self.approve("write report.csv", "보고서 저장")
            with sqlite3.connect(tmp_path / "conversations.db") as db:
                assert db.execute("SELECT count(*) FROM web_run_interactions WHERE response_id = 'response1'").fetchone()[0] == 1
            writes.append(goal)
            (tmp_path / "report.csv").write_text("confirmed,165000\n")
            await asyncio.Event().wait()

    monkeypatch.setattr("rune.agent.agent_context.prepare_agent_context", prepare)
    monkeypatch.setattr("rune.agent.loop.NativeAgentLoop", WaitingLoop)

    def pending(client, session):
        for _ in range(200):
            snapshot = client.get("/api/runs/snapshot", params={"sessionId": session}).json()["run"]
            if snapshot and snapshot[kind]:
                return snapshot
            time.sleep(0.01)
        pytest.fail("Interaction was not delivered")

    try:
        with TestClient(create_app(), client=("127.0.0.1", 50000)) as first:
            first.post("/api/message", json={"text": "accepted", "sessionId": "s1"})
            run = pending(first, "s1")
            payload = {"id": run[kind]["id"], "responseId": "response1"}
            payload.update({"answer": "확정 거래만 🧾"} if kind == "question" else {"decision": "approve"})
            assert first.post(f"/api/{kind}", json=payload).status_code == 200
            first.post("/api/message", json={"text": "unanswered", "sessionId": "s2"})
            unanswered = pending(first, "s2")
            assert writes == ["accepted"]
        conversation_wiring._reset_for_tests()
        with TestClient(create_app(), client=("127.0.0.1", 50000)) as second:
            restored = second.get("/api/runs/snapshot", params={"sessionId": "s1"}).json()["run"]
            assert restored["status"] == "interrupted"
            assert restored["seq"] > run["seq"]
            assert restored["history"][0]["content"] == "accepted"
            assert restored[kind] is None
            assert second.post(f"/api/{kind}", json=payload).status_code == 200
            assert second.post(f"/api/{kind}", json={**payload, "responseId": "different"}).status_code == 409
            assert second.post(f"/api/{kind}", json={**payload, "id": unanswered[kind]["id"]}).status_code == 410
            assert writes == ["accepted"]
            assert (tmp_path / "report.csv").read_text() == "confirmed,165000\n"
    finally:
        conversation_wiring._reset_for_tests()
