import asyncio
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from rune.safety.resource_locks import ResourceBusy, path_access


@pytest.fixture(autouse=True)
def private_home(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "rune-home"))


async def test_file_locks_separate_readers_writers_and_unrelated_files(tmp_path):
    async def access(path, write):
        with path_access(str(path), write=write):
            return True

    path = tmp_path / "report.csv"
    with path_access(str(path), write=False):
        assert await asyncio.create_task(access(path, False))
        with pytest.raises(ResourceBusy):
            await asyncio.create_task(access(path, True))
    with path_access(str(path), write=True):
        with pytest.raises(ResourceBusy):
            await asyncio.create_task(access(tmp_path, True))
        assert await asyncio.create_task(access(tmp_path / "other.csv", True))
    assert await access(path, True)


async def test_aliases_and_other_processes_share_workspace_locks(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    script = '''import asyncio,sys
from rune.safety.resource_locks import path_access,ResourceBusy
async def main():
    try:
        with path_access(sys.argv[1],write=True): pass
    except ResourceBusy: return 23
    return 0
sys.exit(asyncio.run(main()))
'''
    with path_access(str(root), write=True):
        result = subprocess.run([sys.executable, "-c", script, str(alias / "data.csv")], timeout=10)
        assert result.returncode == 23
    result = subprocess.run([sys.executable, "-c", script, str(alias / "data.csv")], timeout=10)
    assert result.returncode == 0


async def test_cancelled_owner_releases_its_lock(tmp_path):
    ready = asyncio.Event()

    async def owner():
        with path_access(str(tmp_path), write=True):
            ready.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(owner())
    await ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with path_access(str(tmp_path), write=True):
        pass


async def test_explicit_shell_target_outside_cwd_is_also_locked(tmp_path):
    from rune.safety.resource_locks import command_access

    work = tmp_path / "work"
    work.mkdir()
    target = tmp_path / "shared.txt"

    async def write():
        with command_access(f"printf changed > {target}", str(work)):
            pytest.fail("A conflicting command must not start")

    with path_access(str(target), write=False):
        with pytest.raises(ResourceBusy):
            await asyncio.create_task(write())


async def test_registry_rejects_conflicting_write_before_dispatch(tmp_path, monkeypatch):
    from rune.capabilities.file import FileWriteParams
    from rune.capabilities.registry import CapabilityRegistry
    from rune.capabilities.types import CapabilityDefinition

    registry = CapabilityRegistry()
    write = AsyncMock()
    registry.register(CapabilityDefinition(name="file_write", description="write",
                                           parameters_model=FileWriteParams, execute=write))
    with path_access(str(tmp_path), write=True):
        result = await asyncio.create_task(registry.execute("file_write", {
            "path": str(tmp_path / "data.csv"), "content": "1",
        }))
    assert result.metadata == {"action_status": "not_executed", "resource_busy": True}
    write.assert_not_awaited()


async def test_browser_cannot_be_used_by_another_run():
    from rune.agent.run_control import RunControl, control_scope
    from rune.capabilities.browser.session import BrowserSession, browser_operation, browser_session

    owner, other = RunControl("owner"), RunControl("other")
    session = BrowserSession(bound_control=owner)
    called = []

    @browser_operation
    async def input_action():
        called.append(True)

    async with browser_session(session):
        with control_scope(other), pytest.raises(RuntimeError, match="another execution"):
            await input_action()
        with control_scope(owner):
            await input_action()
    assert called == [True]


async def test_check_drains_large_output_and_never_passes_truncated_output(tmp_path, monkeypatch):
    import shlex

    from rune.safety.verification import run_check

    monkeypatch.setattr("rune.safety.verification.verification_blocker", lambda *_: None)
    command = shlex.join([sys.executable, "-c", "print('x' * 300000)"])
    result = await run_check(command, str(tmp_path), 5)
    assert result.code == 0 and len(result.stdout) == 300001
    result = await run_check(command, str(tmp_path), 5, limit=100)
    assert result.code is None and "capture limit" in result.error


async def test_cancelled_check_kills_process_before_releasing_workspace(tmp_path, monkeypatch):
    import shlex

    from rune.safety.verification import run_check

    monkeypatch.setattr("rune.safety.verification.verification_blocker", lambda *_: None)
    pid_file = tmp_path / "pid"
    script = f"import os,time; from pathlib import Path; Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)"
    task = asyncio.create_task(run_check(shlex.join([sys.executable, "-c", script]), str(tmp_path), 60))
    async with asyncio.timeout(5):
        while not pid_file.exists():
            await asyncio.sleep(0.01)
    pid = int(pid_file.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    with path_access(str(tmp_path), write=True):
        pass


async def test_environment_is_frozen_and_missing_container_never_runs_locally(tmp_path, monkeypatch):
    from rune.config import get_config
    from rune.config.schema import SandboxConfig
    from rune.safety.execution_environment import environment_scope, execution_config
    from rune.safety.verification import run_check

    marker = tmp_path / "unexpected"
    with environment_scope(str(tmp_path), SandboxConfig(backend="container", image="test:fixed")):
        monkeypatch.setattr(get_config().safety.sandbox, "backend", "local")
        monkeypatch.setattr("rune.safety.verification.shutil.which", lambda _: None)
        assert execution_config().backend == "container"
        result = await run_check(f"touch {marker}", str(tmp_path), 1)
    assert result.code is None and not marker.exists()


async def test_check_uses_command_container_mount_and_network_settings(tmp_path, monkeypatch):
    from rune.config.schema import SandboxConfig
    from rune.safety.execution_environment import environment_scope
    from rune.safety.verification import run_check

    cwd = tmp_path / "project"
    cwd.mkdir()
    calls = []
    spawn = asyncio.create_subprocess_exec

    async def fake_docker(*argv, **kwargs):
        calls.append(argv)
        return await spawn("sh", "-c", "printf checked", **kwargs)

    monkeypatch.setattr("rune.safety.verification.verification_blocker", lambda *_: None)
    monkeypatch.setattr("rune.safety.verification.shutil.which", lambda _: "/test/docker")
    monkeypatch.setattr("rune.safety.verification.asyncio.create_subprocess_exec", fake_docker)
    cleanup = AsyncMock(return_value=True)
    monkeypatch.setattr("rune.safety.container_exec.remove_container", cleanup)
    # The private state must live outside the directory mounted into Docker.
    monkeypatch.setenv("RUNE_HOME", str(tmp_path.parent / "private-state"))
    with environment_scope(str(tmp_path), SandboxConfig(backend="container", image="test:fixed")):
        result = await run_check("echo test", str(cwd), 5)
    assert result.code == 0 and result.stdout == b"checked"
    argv = calls[0]
    assert argv[0] == "docker" and "test:fixed" in argv
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--workdir") + 1] == str(cwd)
    assert argv[argv.index("--mount") + 1] == f"type=bind,src={tmp_path},dst={tmp_path}"
    cleanup.assert_awaited_once()


async def test_approval_recovery_requires_new_approval_and_never_repeats_effect(tmp_path, monkeypatch):
    from rune.agent.execution_journal import ExecutionJournal, journal_scope
    from rune.api.approval_recovery import resume_approval
    from rune.api.run_recovery import RunRecovery
    from rune.api.run_snapshot import RunSnapshots
    from rune.api.run_store import RunStore
    from rune.capabilities.file import FileWriteParams
    from rune.capabilities.registry import CapabilityRegistry
    from rune.capabilities.types import CapabilityDefinition
    from rune.safety.approval_request import action_request
    from rune.types import CapabilityResult

    registry = CapabilityRegistry()
    calls = []

    async def write(params):
        calls.append(params.content)
        Path(params.path).write_text(params.content)
        return CapabilityResult(success=True, output="saved")

    registry.register(CapabilityDefinition(name="file_write", description="write",
                                           parameters_model=FileWriteParams, execute=write))
    registry.set_approval_patterns(["file_write"])
    monkeypatch.setattr("rune.capabilities.registry._registry", registry)
    params = FileWriteParams(path=str(tmp_path / "output.txt"), content="saved once").model_dump(mode="json", by_alias=True)
    store = RunStore(tmp_path / "runs.db")
    runs = RunSnapshots(store)
    runs.start("original", "session", "write output")
    runs.record("run_context", {"runId": "original", "workspace": str(tmp_path), "recoveryVersion": 1})
    runs.record("approval_request", {"runId": "original", "id": "old", "action": action_request("file_write", params)})
    runs.close()
    store = RunStore(tmp_path / "runs.db")
    runs = RunSnapshots(store)
    try:
        child, source, records = RunRecovery(runs, store).begin("original")
        assert source["status"] == "interrupted"
        journal = ExecutionJournal(store, child["runId"], str(tmp_path), previous=records)
        entered, proceed = asyncio.Event(), asyncio.Event()

        async def approve(tool, reason):
            assert tool == "file_write" and "saved once" in reason
            entered.set()
            await proceed.wait()
            return True

        async def resume():
            with journal_scope(journal):
                return await resume_approval(source, journal, approve, AsyncMock())

        task = asyncio.create_task(resume())
        await entered.wait()
        assert calls == [] and not (tmp_path / "output.txt").exists()
        proceed.set()
        assert len(await task) == 1
        assert calls == ["saved once"]
        with journal_scope(journal):
            result = await journal.replay_completed("file_write", params)
        assert result.metadata["replayed"] and calls == ["saved once"]
        assert RunRecovery(runs, store).begin("original")[0]["runId"] == child["runId"]
    finally:
        runs.close()


async def test_changed_file_invalidates_approved_action(tmp_path):
    from rune.capabilities.file import FileWriteParams
    from rune.capabilities.registry import CapabilityRegistry
    from rune.capabilities.types import CapabilityDefinition
    from rune.safety.approval_context import approval_granted
    from rune.safety.approval_request import action_request

    target = tmp_path / "report.txt"
    target.write_text("original")
    params = FileWriteParams(path=str(target), content="replace").model_dump(mode="json", by_alias=True)
    action = action_request("file_write", params)
    target.write_text("user change")
    registry = CapabilityRegistry()
    write = AsyncMock()
    registry.register(CapabilityDefinition(name="file_write", description="write",
                                           parameters_model=FileWriteParams, execute=write))
    with approval_granted("file_write", params, revisions=action["revisions"]):
        result = await registry.execute("file_write", params)
    assert not result.success and result.metadata["action_status"] == "not_executed"
    assert target.read_text() == "user change"
    write.assert_not_awaited()


@pytest.mark.parametrize("decision", ["approve_once", "deny"])
def test_api_restart_restores_pending_action_for_review(tmp_path, monkeypatch, decision):
    import time

    from starlette.testclient import TestClient

    from rune.agent.agent_context import AgentContext
    from rune.agent.tool_adapter import ToolAdapterOptions, build_tool_set
    from rune.api import conversation_wiring
    from rune.api.server import create_app
    from rune.capabilities.file import FileWriteParams
    from rune.capabilities.registry import CapabilityRegistry
    from rune.capabilities.types import CapabilityDefinition
    from rune.types import CapabilityResult, CompletionTrace
    from rune.utils.events import EventEmitter

    registry = CapabilityRegistry()
    effects = []
    target = tmp_path / "report.txt"

    async def write(params):
        effects.append(params.content)
        Path(params.path).write_text(params.content)
        return CapabilityResult(success=True, output="saved once")

    registry.register(CapabilityDefinition(name="file_write", description="write",
                                           parameters_model=FileWriteParams, execute=write))
    registry.set_approval_patterns(["file_write"])
    monkeypatch.setattr("rune.capabilities.registry._registry", registry)

    async def prepare(options, **kwargs):
        return AgentContext(goal=options.goal, original_goal=options.goal, channel="web",
                            workspace_root=str(tmp_path), conversation_id=options.conversation_id)

    class Loop(EventEmitter):
        def __init__(self, **kwargs):
            super().__init__()
            self._last_answer_text = ""

        def set_approval_callback(self, callback):
            self.approve = callback

        def set_ask_user_callback(self, callback):
            pass

        async def run(self, goal, **kwargs):
            tools = build_tool_set(ToolAdapterOptions(allowed_tools=["file_write"], approval_callback=self.approve))
            self._last_answer_text = await tools["file_write"].function(path=str(target), content="saved once")
            return CompletionTrace(reason="completed")

    monkeypatch.setattr("rune.agent.agent_context.prepare_agent_context", prepare)
    monkeypatch.setattr("rune.agent.loop.NativeAgentLoop", Loop)
    monkeypatch.setattr("rune.api.run_maintenance.RunMaintenance.enqueue", lambda *args, **kwargs: None)
    monkeypatch.setattr("rune.conversation.store.ConversationStore._embed_new_turns", AsyncMock())

    def wait_run(client, predicate):
        for _ in range(400):
            run = client.get("/api/runs/snapshot", params={"sessionId": "approval-restart"}).json()["run"]
            if run and predicate(run):
                return run
            time.sleep(0.01)
        pytest.fail("The expected execution state was not reached")

    conversation_wiring._reset_for_tests()
    try:
        with TestClient(create_app(), client=("127.0.0.1", 50000)) as client:
            parent = client.post("/api/message", json={"text": "write report", "sessionId": "approval-restart"}).json()["runId"]
            run = wait_run(client, lambda run: run["approval"])
            old = run["approval"]["id"]
            assert run["approval"]["action"]["params"]["content"] == "saved once"
            assert not target.exists()
        conversation_wiring._reset_for_tests()
        with TestClient(create_app(), client=("127.0.0.1", 50000)) as client:
            wait_run(client, lambda run: run["status"] == "interrupted")
            assert client.post("/api/approval", json={"id": old, "decision": "approve_once"}).status_code == 410
            resumed = client.post("/api/runs/resume", json={"runId": parent})
            assert resumed.status_code == 200, resumed.text
            run = wait_run(client, lambda run: run["approval"])
            assert run["approval"]["id"] != old and effects == []
            assert client.post("/api/approval", json={"id": run["approval"]["id"], "decision": decision}).status_code == 200
            run = wait_run(client, lambda run: run["status"] in {"completed", "failed"})
            assert effects == (["saved once"] if decision == "approve_once" else [])
            assert run["status"] == ("completed" if decision == "approve_once" else "failed")
            assert target.exists() == (decision == "approve_once")
    finally:
        conversation_wiring._reset_for_tests()
