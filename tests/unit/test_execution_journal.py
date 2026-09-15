import asyncio
import subprocess
import sys

import pytest

from rune.agent.execution_journal import (
    ExecutionJournal,
    RecoveryBlocked,
    fingerprint,
    journal_scope,
)
from rune.api.run_recovery import RunRecovery
from rune.api.run_snapshot import RunSnapshots
from rune.api.run_store import RunStore
from rune.types import CapabilityResult


@pytest.fixture
def recovery(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    runs = RunSnapshots(store)
    runs.start("first", "session", "정산 보고서")
    runs.record("run_context", {"runId": "first", "workspace": str(tmp_path), "recoveryVersion": 1})
    yield store, runs, RunRecovery(runs, store)
    runs.close()


@pytest.mark.parametrize("boundary", ["before_write", "after_write", "after_receipt"])
async def test_process_death_does_not_repeat_a_file_effect(tmp_path, boundary):
    script = r'''
import asyncio, os, sys
from pathlib import Path
from rune.agent.execution_journal import ExecutionJournal
from rune.api.run_store import RunStore
from rune.api.run_snapshot import RunSnapshots
from rune.types import CapabilityResult
root, boundary = Path(sys.argv[1]), sys.argv[2]
store = RunStore(root / 'runs.db')
runs = RunSnapshots(store)
runs.start('first', 'session', '정산 보고서')
runs.record('run_context', {'runId': 'first', 'workspace': str(root), 'recoveryVersion': 1})
async def write():
    if boundary == 'before_write': os._exit(23)
    (root / 'report.csv').write_text('team,total\nA,165000\n')
    (root / 'calls').write_text('1')
    if boundary == 'after_write': os._exit(23)
    return CapabilityResult(success=True, output='written')
asyncio.run(ExecutionJournal(store, 'first', str(root)).execute('file_write',
    {'path': str(root / 'report.csv'), 'content': 'team,total\nA,165000\n'}, write))
os._exit(23)
'''
    process = subprocess.run([sys.executable, "-c", script, str(tmp_path), boundary], timeout=30)
    assert process.returncode == 23
    store = RunStore(tmp_path / "runs.db")
    runs = RunSnapshots(store)
    try:
        child, source, records = RunRecovery(runs, store).begin("first")
        assert source["status"] == "interrupted"
        invoked = []

        async def write():
            invoked.append(True)
            (tmp_path / "report.csv").write_text("team,total\nA,165000\n")
            (tmp_path / "calls").write_text("1")
            return CapabilityResult(success=True, output="written")

        result = await ExecutionJournal(store, child["runId"], str(tmp_path), previous=records).execute(
            "file_write", {"path": str(tmp_path / "report.csv"), "content": "team,total\nA,165000\n"}, write,
        )
        assert result.success
        assert len(invoked) == (boundary == "before_write")
        assert (tmp_path / "calls").read_text() == "1"
        assert (tmp_path / "report.csv").read_text() == "team,total\nA,165000\n"
    finally:
        runs.close()


async def test_unknown_external_effect_blocks_resumption(recovery, tmp_path):
    store, runs, service = recovery

    async def external_write():
        (tmp_path / "external-effect").write_text("sent")
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await ExecutionJournal(store, "first", str(tmp_path)).execute("send_mail", {"id": "invoice-1"}, external_write)
    runs.interrupt_active("server_shutdown")
    with pytest.raises(RecoveryBlocked, match="Cannot confirm whether send_mail finished"):
        service.begin("first")
    assert store.resumed_child("first") is None
    assert (tmp_path / "external-effect").read_text() == "sent"


async def test_failed_shell_that_already_wrote_cannot_repeat_without_review(recovery, tmp_path):
    from rune.capabilities.bash import BashParams, _execute_oneshot

    store, _, _ = recovery
    journal = ExecutionJournal(store, "first", str(tmp_path))
    params = BashParams(command="printf saved >> result.txt; exit 1", cwd=str(tmp_path))
    first = await journal.execute("bash_execute", params.model_dump(), lambda: _execute_oneshot(params))
    assert not first.success and first.metadata["exit_code"] == 1
    assert store.attempts("first")[0]["state"] == "unknown"
    denied = await journal.execute("bash_execute", params.model_dump(), lambda: _execute_oneshot(params))
    assert denied.metadata["action_status"] == "not_executed"
    assert (tmp_path / "result.txt").read_text() == "saved"

    async def read():
        return CapabilityResult(success=True, output=(tmp_path / "result.txt").read_text())

    assert (await journal.execute("file_read", {"path": str(tmp_path / "result.txt")}, read)).output == "saved"
    approvals = []

    async def approve(name, reason):
        approvals.append((name, reason))
        return len(approvals) > 1

    journal.approval = approve
    replacement = BashParams(command="printf repaired > result.txt", cwd=str(tmp_path))
    for expected in (False, True):
        result = await journal.execute("bash_execute", replacement.model_dump(), lambda: _execute_oneshot(replacement))
        assert result.success is expected
    assert "exit 1" in approvals[0][1] and "repaired" in approvals[0][1]
    assert (tmp_path / "result.txt").read_text() == "repaired"
    prior = store.attempts("first")[0]
    assert prior["state"] == "unknown" and prior["review"]["next_tool"] == "bash_execute"


async def test_shell_failure_before_dispatch_does_not_block_followup(recovery, tmp_path):
    from rune.capabilities.bash import BashParams, _execute_oneshot

    store, _, _ = recovery
    journal = ExecutionJournal(store, "first", str(tmp_path))
    params = BashParams(command="true", cwd=str(tmp_path / "missing"))
    result = await journal.execute("bash_execute", params.model_dump(), lambda: _execute_oneshot(params))
    assert result.metadata["action_status"] == "not_executed"
    valid = BashParams(command="printf ok", cwd=str(tmp_path))
    result = await journal.execute("bash_execute", valid.model_dump(), lambda: _execute_oneshot(valid))
    assert result.success and result.output == "ok"


async def test_cancelled_shell_stops_its_child_and_keeps_the_effect_unknown(recovery, tmp_path):
    import os
    import shlex

    from rune.capabilities.bash import BashParams, _execute_oneshot

    store, _, _ = recovery
    journal = ExecutionJournal(store, "first", str(tmp_path))
    script = "import os,time; from pathlib import Path; Path('started').write_text(str(os.getpid())); time.sleep(30)"
    params = BashParams(command=shlex.join([sys.executable, "-c", script]), cwd=str(tmp_path))
    task = asyncio.create_task(journal.execute("bash_execute", params.model_dump(), lambda: _execute_oneshot(params)))
    try:
        async with asyncio.timeout(3):
            while not (tmp_path / "started").exists():
                await asyncio.sleep(0.005)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
    pid = int((tmp_path / "started").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert store.attempts("first")[0]["state"] == "unknown"
    denied = await journal.execute("bash_execute", params.model_dump(), lambda: _execute_oneshot(params))
    assert denied.metadata["action_status"] == "not_executed"


@pytest.mark.parametrize("state", ["unknown", "dispatched"])
async def test_browser_effect_cannot_be_replayed_in_a_new_session(recovery, tmp_path, state):
    from unittest.mock import AsyncMock

    store, runs, service = recovery
    invoke = AsyncMock(return_value=CapabilityResult(success=state == "dispatched",
                                                   metadata={"action_status": state}))
    await ExecutionJournal(store, "first", str(tmp_path)).execute("browser_act", {"selector": "e1"}, invoke)
    assert store.attempts("first")[0]["state"] == ("done" if state == "dispatched" else "unknown")
    runs.interrupt_active("server_shutdown")
    with pytest.raises(RecoveryBlocked):
        service.begin("first")
    invoke.assert_awaited_once()


def test_resume_excludes_overlapping_workspaces(recovery, tmp_path):
    store, runs, service = recovery
    runs.interrupt_active("server_shutdown")
    runs.start("other", "another-session", "edit a subdirectory")
    runs.record("run_context", {"runId": "other", "workspace": str(tmp_path / "src")})
    with pytest.raises(RecoveryBlocked, match="Another execution"):
        service.begin("first")
    assert store.resumed_child("first") is None
    runs.record("agent_complete", {"runId": "other"})
    child, _, _ = service.begin("first")
    assert child["status"] == "queued" and child["workspace"] == str(tmp_path)
    with pytest.raises(RecoveryBlocked, match="Another execution"):
        service.workspace_available("new", str(tmp_path / "src"), resuming=False)
    with pytest.raises(RecoveryBlocked, match="Another execution"):
        service.workspace_available("new", str(tmp_path.parent), resuming=False)


async def test_recorded_effect_replays_but_new_external_effect_needs_new_approval(recovery, tmp_path):
    store, runs, service = recovery
    calls, approvals = [], []

    async def send():
        calls.append("sent")
        return CapabilityResult(success=True, output="receipt-123")

    async def approve(name, reason):
        approvals.append((name, reason))
        return False

    await ExecutionJournal(store, "first", str(tmp_path)).execute("send_mail", {"id": "invoice-1"}, send)
    runs.interrupt_active("server_shutdown")
    child, _, records = service.begin("first")
    journal = ExecutionJournal(store, child["runId"], str(tmp_path), previous=records, approval=approve)
    replay = await journal.execute("send_mail", {"id": "invoice-1"}, send)
    assert replay.metadata["replayed"] and replay.output == "receipt-123"
    assert calls == ["sent"] and not approvals
    denied = await journal.execute("send_mail", {"id": "invoice-2"}, send)
    assert not denied.success and len(approvals) == 1 and calls == ["sent"]
    # A lost response or repeated click resolves to the same continuation.
    repeated, source, _ = service.begin("first")
    assert repeated["runId"] == child["runId"] and source is None


@pytest.mark.parametrize("change", ["content", "symlink"])
async def test_changed_file_or_path_binding_prevents_resume(recovery, tmp_path, change):
    store, runs, service = recovery
    path = tmp_path / "report.txt"

    async def write():
        path.write_text("approved")
        return CapabilityResult(success=True)

    await ExecutionJournal(store, "first", str(tmp_path)).execute("file_write", {"path": str(path), "content": "approved"}, write)
    runs.interrupt_active("server_shutdown")
    if change == "content":
        path.write_text("user edit")
    else:
        other = tmp_path / "other.txt"
        other.write_text("approved")
        path.unlink()
        path.symlink_to(other)
    with pytest.raises(RecoveryBlocked, match="no longer matches"):
        service.begin("first")
    assert store.resumed_child("first") is None


async def test_disk_failure_after_effect_blocks_all_further_tools(recovery, tmp_path):
    store, _, _ = recovery
    journal = ExecutionJournal(store, "first", str(tmp_path))
    calls = []

    async def write():
        calls.append("executed")
        store.db.execute("CREATE TEMP TRIGGER broken BEFORE UPDATE ON web_tool_attempts BEGIN SELECT RAISE(ABORT, 'disk full'); END")
        return CapabilityResult(success=True)

    with pytest.raises(RecoveryBlocked, match="disk full"):
        await journal.execute("send_mail", {}, write)
    with pytest.raises(RecoveryBlocked, match="disk full"):
        await journal.execute("send_mail", {}, write)
    assert calls == ["executed"] and store.attempts("first")[0]["state"] == "started"


async def test_registry_records_validated_parameters_before_executing(recovery, tmp_path):
    from rune.capabilities.file import FileWriteParams
    from rune.capabilities.registry import CapabilityRegistry
    from rune.capabilities.types import CapabilityDefinition

    store, _, _ = recovery
    registry = CapabilityRegistry()

    async def execute(params):
        record = store.attempts("first")[0]
        assert record["state"] == "started" and record["params"]["encoding"] == "utf-8"
        assert record["effect"]["before"] == fingerprint(params.path)
        (tmp_path / "report.txt").write_text(params.content)
        return CapabilityResult(success=True)

    registry.register(CapabilityDefinition(name="file_write", description="write", parameters_model=FileWriteParams, execute=execute))
    with journal_scope(ExecutionJournal(store, "first", str(tmp_path))):
        result = await registry.execute("file_write", {"path": str(tmp_path / "report.txt"), "content": "hello"})
    assert result.success and store.attempts("first")[0]["state"] == "done"


async def test_reads_stay_parallel_but_writes_wait_for_them(recovery, tmp_path):
    store, _, _ = recovery
    journal = ExecutionJournal(store, "first", str(tmp_path))
    started, released = asyncio.Event(), asyncio.Event()
    reads, writes = [], []

    async def read():
        reads.append(True)
        if len(reads) == 2:
            started.set()
        await released.wait()
        return CapabilityResult(success=True)

    async def write():
        writes.append(True)
        return CapabilityResult(success=True)

    tasks = [asyncio.create_task(journal.execute("web_search", {"query": str(i)}, read)) for i in range(2)]
    try:
        await asyncio.wait_for(started.wait(), 2)
        writer = asyncio.create_task(journal.execute("send_mail", {}, write))
        tasks.append(writer)
        await asyncio.sleep(0)
        assert not writes
        released.set()
        await asyncio.gather(*tasks)
        assert writes == [True]
    finally:
        released.set()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_harness_checks_execute_again_after_resume(recovery, tmp_path):
    from rune.agent.execution_journal import record_check

    store, runs, service = recovery
    calls = []

    async def verify():
        calls.append(True)
        return ("pass", "2 passed") if len(calls) == 1 else ("fail", "dependency unavailable")

    params = {"command": "python -m unittest", "cwd": str(tmp_path)}
    with journal_scope(ExecutionJournal(store, "first", str(tmp_path))):
        assert (await record_check(params, verify))[0] == "pass"
    runs.interrupt_active("server_shutdown")
    child, _, records = service.begin("first")
    with journal_scope(ExecutionJournal(store, child["runId"], str(tmp_path), previous=records)):
        assert (await record_check(params, verify))[0] == "fail"
    assert len(calls) == 2


async def test_a_superseded_write_is_not_mistaken_for_a_duplicate(recovery, tmp_path):
    store, runs, service = recovery
    path = tmp_path / "report.txt"
    calls = []

    async def write(journal, content):
        async def invoke():
            calls.append(content)
            path.write_text(content)
            return CapabilityResult(success=True, output=content)
        return await journal.execute("file_write", {"path": str(path), "content": content}, invoke)

    original = ExecutionJournal(store, "first", str(tmp_path))
    await write(original, "A")
    await write(original, "B")
    runs.interrupt_active("server_shutdown")
    child, _, records = service.begin("first")
    resumed = ExecutionJournal(store, child["runId"], str(tmp_path), previous=records)
    assert (await write(resumed, "B")).metadata["replayed"]
    assert not (await write(resumed, "A")).metadata.get("replayed")
    assert path.read_text() == "A"
    runs.interrupt_active("server_shutdown")
    next_child, _, records = service.begin(child["runId"])
    again = ExecutionJournal(store, next_child["runId"], str(tmp_path), previous=records)
    assert not (await write(again, "B")).metadata.get("replayed")
    assert path.read_text() == "B" and calls == ["A", "B", "A", "B"]


@pytest.mark.parametrize("change", [False, True])
async def test_document_output_is_bound_to_its_saved_revision(recovery, tmp_path, change):
    store, runs, service = recovery
    path = tmp_path / "summary.csv"
    calls = []

    async def create():
        calls.append(True)
        path.write_text("team,total\nA,165000\n")
        return CapabilityResult(success=True, metadata={"path": str(path)})

    await ExecutionJournal(store, "first", str(tmp_path)).execute("document_create", {"path": str(path)}, create)
    runs.interrupt_active("server_shutdown")
    if change:
        path.write_text("user revision")
        with pytest.raises(RecoveryBlocked, match="no longer matches"):
            service.begin("first")
        assert store.resumed_child("first") is None
    else:
        child, _, records = service.begin("first")
        result = await ExecutionJournal(store, child["runId"], str(tmp_path), previous=records).execute(
            "document_create", {"path": str(path)}, create,
        )
        assert result.metadata["replayed"]
    assert len(calls) == 1


async def test_adapter_returns_recorded_receipt_without_reusing_an_approval(recovery, tmp_path):
    from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
    from rune.capabilities.registry import CapabilityRegistry
    from rune.capabilities.types import CapabilityDefinition

    store, runs, service = recovery
    calls, observed = [], []

    async def send(params):
        calls.append(params)
        return CapabilityResult(success=True, output="delivered")

    async def approve(*args):
        pytest.fail("Completed effects must not ask to execute again")

    async def on_end(name, result):
        observed.append(result)

    cap = CapabilityDefinition(name="mcp.mail.send", description="Send mail", execute=send)
    registry = CapabilityRegistry()
    registry.register(cap)
    with journal_scope(ExecutionJournal(store, "first", str(tmp_path))):
        await registry.execute(cap.name, {"target": "invoice-1"})
    runs.interrupt_active("server_shutdown")
    child, _, records = service.begin("first")
    wrapped = _build_typed_tool(cap_def=cap, opts=ToolAdapterOptions(
        workspace_root=str(tmp_path), approval_callback=approve, on_tool_end=on_end,
    ), reg=registry, cache=None, stall=None)
    with journal_scope(ExecutionJournal(store, child["runId"], str(tmp_path), previous=records)):
        output = await wrapped.function(target="invoice-1")
    assert "not a fresh verification" in output and len(calls) == 1
    assert observed[0].metadata["replayed"]


def test_api_resume_restores_accepted_question_and_creates_one_continuation(tmp_path, monkeypatch):
    import time

    from starlette.testclient import TestClient

    from rune.agent.agent_context import AgentContext
    from rune.agent.loop import _CALL_ID
    from rune.api import conversation_wiring
    from rune.api.server import create_app
    from rune.capabilities.ask_user import register_ask_user_capability, set_ask_user_callback
    from rune.capabilities.file import FileWriteParams
    from rune.capabilities.registry import CapabilityRegistry
    from rune.capabilities.types import CapabilityDefinition
    from rune.types import CompletionTrace
    from rune.utils.events import EventEmitter

    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    conversation_wiring._reset_for_tests()
    effects, prompts, started = [], [], []

    async def prepare(options, **kwargs):
        return AgentContext(goal=options.goal, original_goal=options.goal, channel="web",
                            workspace_root=str(tmp_path), conversation_id=options.conversation_id)

    async def post_process(value):
        pass

    class Loop(EventEmitter):
        def __init__(self, **kwargs):
            super().__init__()
            self._last_answer_text = ""

        def set_approval_callback(self, callback):
            self.approve = callback

        def set_ask_user_callback(self, callback):
            self.ask = callback

        async def run(self, goal, **kwargs):
            resumed = bool(kwargs.get("extra_system_context"))
            started.append(resumed)
            registry = CapabilityRegistry()

            async def write(params):
                effects.append(params.path)
                from pathlib import Path
                Path(params.path).write_text(params.content)
                return CapabilityResult(success=True, output="saved")

            async def ask(params):
                prompts.append(params.question)
                response = await self.ask(params)
                # The HTTP acknowledgement is durable, but the capability
                # has not returned when this first process shuts down.
                if not resumed:
                    await asyncio.Event().wait()
                return response

            registry.register(CapabilityDefinition(name="file_write", description="write",
                                                   parameters_model=FileWriteParams, execute=write))
            register_ask_user_capability(registry)
            set_ask_user_callback(ask)
            _CALL_ID.set("write-1")
            result = await registry.execute("file_write", {"path": str(tmp_path / "report.txt"), "content": "saved once"})
            assert result.success
            _CALL_ID.set("question-1")
            answer = await registry.execute("ask_user", {"question": "정산 기준?", "reason": "보고서 범위"})
            assert '확정 거래만' in answer.output
            self._last_answer_text = "saved work continued"
            return CompletionTrace(reason="completed")

    monkeypatch.setattr("rune.agent.agent_context.prepare_agent_context", prepare)
    monkeypatch.setattr("rune.agent.agent_context.post_process_agent_result", post_process)
    monkeypatch.setattr("rune.agent.loop.NativeAgentLoop", Loop)
    # The storage assertions do not need the embedding worker.
    async def skip_embeddings(self, turns):
        pass

    monkeypatch.setattr("rune.conversation.store.ConversationStore._embed_new_turns", skip_embeddings)

    def snapshot(client):
        return client.get("/api/runs/snapshot", params={"sessionId": "resume-session"}).json()["run"]

    try:
        with TestClient(create_app(), client=("127.0.0.1", 50000)) as first:
            parent = first.post("/api/message", json={"text": "정산 보고서", "sessionId": "resume-session"}).json()["runId"]
            for _ in range(200):
                run = snapshot(first)
                if run and run["question"]:
                    break
                time.sleep(0.01)
            assert run["question"]
            response = {"id": run["question"]["id"], "responseId": "answer-1", "answer": "확정 거래만"}
            assert first.post("/api/question", json=response).status_code == 200
        conversation_wiring._reset_for_tests()
        with TestClient(create_app(), client=("127.0.0.1", 50000)) as second:
            assert snapshot(second)["status"] == "interrupted"
            (tmp_path / "report.txt").write_text("user revision")
            blocked = second.post("/api/runs/resume", json={"runId": parent})
            assert blocked.status_code == 409 and "no longer matches" in blocked.json()["detail"]
            assert snapshot(second)["runId"] == parent
            (tmp_path / "report.txt").write_text("saved once")
            resumed = second.post("/api/runs/resume", json={"runId": parent})
            assert resumed.status_code == 200, resumed.text
            again = second.post("/api/runs/resume", json={"runId": parent})
            assert again.json()["runId"] == resumed.json()["runId"]
            for _ in range(200):
                run = snapshot(second)
                if run["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.01)
            assert run["status"] == "completed", run
            assert effects == [str(tmp_path / "report.txt")]
            assert prompts == ["정산 기준?"] and started == [False, True]
            assert second.post("/api/question", json=response).status_code == 200
            turns = second.post("/api/v1/rpc", json={"method": "sessions.turns", "params": {"sessionId": "resume-session"}}).json()
            assert [t["role"] for t in turns["data"]["turns"]] == ["user", "assistant"]
    finally:
        conversation_wiring._reset_for_tests()
