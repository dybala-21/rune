"""Exercise the desktop API and agent boundary without sending native input."""

import asyncio
import base64
import time
from io import BytesIO

import httpx
import pytest
from fastapi import FastAPI
from PIL import Image

from rune.agent.run_control import ControlChanged, RunControl, control_scope
from rune.api.computer import Computer, Computers, computer_router
from rune.api.desktop import desktop_router
from rune.computer.protocol import DesktopAction, DesktopError
from rune.computer.session import DesktopManager, DesktopSession, desktop_scope, require_desktop


class Host:
    def __init__(self):
        buffer = BytesIO()
        Image.new("RGB", (20, 10), "white").save(buffer, format="PNG")
        self.image = base64.b64encode(buffer.getvalue()).decode()
        self.calls = []
        self.lock = asyncio.Lock()
        self.number = 0
        self.failure = None
        self.permissions = True
        self.closed = False
        self.started = asyncio.Event()
        self.release = None
        self.epoch = 0
        self.native_decision = None

    def cancel_pending(self):
        self.epoch += 1

    async def request(self, method, params=None, *, guard=None):
        if guard is not None:
            guard()
        self.calls.append((method, params))
        if method == "status":
            return {"accessibility": self.permissions, "screenRecording": self.permissions,
                    "apps": [{"id": "test.editor", "name": "Editor"}]}
        if method in {"grant", "permissions", "acknowledge"}:
            return {}
        if method == "act":
            epoch = self.epoch
            generation = getattr(self, "generation", 0)
            self.native_decision = asyncio.get_running_loop().create_future()
            while not self.native_decision.done():
                if epoch != self.epoch or generation != getattr(self, "generation", 0) or not getattr(self, "connected", True):
                    raise DesktopError("Native review was cancelled before input")
                await asyncio.wait({self.native_decision}, timeout=.01)
            if not self.native_decision.result():
                raise DesktopError("Native input was declined")
            self.started.set()
            if self.release:
                await self.release.wait()
            if epoch != self.epoch:
                raise DesktopError("Native review was cancelled before input")
            if self.failure:
                raise self.failure
        self.number += 1
        return {"app": params.get("app", "test.editor"), "observation": f"view-{self.number}",
                "title": "Test document", "width": 20, "height": 10, "image_base64": self.image,
                "controls": [{"ref": "save", "role": "AXButton", "name": "Save"}]}

    async def close(self):
        self.closed = True


@pytest.fixture
async def desktop():
    host = Host()
    computers = Computers()
    computers.desktop = DesktopManager(host)
    session = await computers.desktop.grant("one", ["test.editor"])
    entry = Computer("one", desktop=session, control=RunControl("run-one"))
    computers.entries["one"] = entry
    app = FastAPI()
    app.include_router(desktop_router(computers, lambda: None))
    app.include_router(computer_router(computers, lambda: None))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://rune.test",
                                headers={"Origin": "http://rune.test"}) as client:
        yield computers, entry, host, client
    await computers.close()


async def run_action(computers, entry):
    async with computers.bind(entry):
        await require_desktop()
        view = await entry.desktop.observe("test.editor")
        return await entry.desktop.act(DesktopAction(observation=view["observation"], action="press", ref="save"))


async def wait_for_review(session):
    async with asyncio.timeout(2):
        while session.pending is None:
            await asyncio.sleep(0.001)
    return session.pending["id"], session.revision


async def allow_native(session):
    async with asyncio.timeout(2):
        while session.host.native_decision is None or session.host.native_decision.done():
            await asyncio.sleep(.001)
    assert session.native_review
    session.host.native_decision.set_result(True)


async def decide(client, action_id, revision, approved=True, session_id="one"):
    return await client.post("/api/desktop/control", json={"sessionId": session_id, "action": "decide",
                             "actionId": action_id, "revision": revision, "approved": approved})


@pytest.mark.parametrize("stop", [False, True])
async def test_native_task_waits_for_access_and_never_enters_execution_early(desktop, monkeypatch, stop):
    from unittest.mock import AsyncMock

    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.loop import NativeAgentLoop
    from rune.computer.session import current_desktop
    from rune.types import CompletionTrace

    computers, entry, host, client = desktop
    await computers.desktop.release(entry.desktop)
    entry.desktop = None
    entered = []
    loop = NativeAgentLoop()
    loop._auto_skill = False
    monkeypatch.setattr(loop, "_build_system_prompt", AsyncMock(return_value="test"))

    async def execute(**kwargs):
        session = current_desktop()
        assert session is entry.desktop
        session.check()
        await session.observe("test.editor")
        entered.append(True)
        return CompletionTrace(reason="completed")

    monkeypatch.setattr(loop, "_execute_loop", execute)

    async def run():
        async with computers.bind(entry):
            return await loop.run("TextEdit에서 문서를 저장해", classification=ClassificationResult(
                goal_type="full", confidence=1, tier=2, intent_categories=frozenset({"desktop", "document"})))

    task = asyncio.create_task(run())
    async with asyncio.timeout(3):
        while not entry.access_requested:
            await asyncio.sleep(0.001)
    assert not entered
    assert (await client.get("/api/desktop/status?sessionId=one")).json()["accessRequested"]
    wrong = await client.post("/api/desktop/control", json={"sessionId": "other", "action": "grant", "apps": ["test.editor"]})
    assert wrong.status_code == 409
    if stop:
        await computers.stop("run-one")
    else:
        response = await client.post("/api/desktop/control", json={"sessionId": "one", "action": "grant", "apps": ["test.editor"]})
        assert response.status_code == 200, response.text
    result = await asyncio.wait_for(task, 3)
    assert result.reason == ("cancelled" if stop else "completed")
    assert entered == ([] if stop else [True])
    assert not entry.access_requested
    assert not any(method == "act" for method, _ in host.calls)
    assert not any(method == "permissions" for method, _ in host.calls)
    if entry.desktop:
        assert entry.desktop.task is None and entry.desktop.control is None


@pytest.mark.parametrize("bad_digest", [False, True])
async def test_published_document_is_downloadable_without_exposing_native_payload(desktop, monkeypatch, tmp_path, bad_digest):
    import hashlib

    from rune.api.files import download_file
    from rune.computer.capabilities import invoke

    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    computers, entry, host, client = desktop
    original_request = host.request
    content = b"{\\rtf1 saved document}"
    path = "/Users/example/Desktop/출시 계획.rtf"

    async def request(method, params=None, **kwargs):
        data = await original_request(method, params, **kwargs)
        if method == "act" and params["action"] == "publish":
            data["artifact"] = {"path": path, "data_base64": base64.b64encode(content).decode(),
                                "sha256": "bad" if bad_digest else hashlib.sha256(content).hexdigest()}
        return data

    host.request = request

    async def publish():
        async with computers.bind(entry):
            await require_desktop()
            view = await entry.desktop.observe("test.editor")
            return await invoke("act", DesktopAction(observation=view["observation"], action="publish"))

    task = asyncio.create_task(publish())
    action_id, revision = await wait_for_review(entry.desktop)
    await allow_native(entry.desktop)
    result = await task
    assert result.success is not bad_digest
    assert not entry.desktop.uncertain
    assert "data_base64" not in result.output
    assert "data_base64" not in str(entry.desktop.view)
    if not bad_digest:
        receipt = result.metadata["receipt"]
        assert receipt["scope"] == "download" and receipt["runId"] == "run-one"
        response = await download_file("one", path)
        assert response.body == content
        assert response.headers["content-disposition"].startswith("attachment;")


async def test_unavailable_routing_cannot_fall_back_to_generic_execution(monkeypatch):
    from unittest.mock import AsyncMock

    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.loop import NativeAgentLoop

    loop = NativeAgentLoop()
    loop._auto_skill = False
    execute = AsyncMock()
    monkeypatch.setattr(loop, "_execute_loop", execute)
    result = await loop.run("TextEdit에서 저장해", classification=ClassificationResult(
        goal_type="full", confidence=0.5, tier=2, available=False,
        reason="Classification unavailable: read_timeout"))
    assert result.reason.startswith("error:") and "read_timeout" in result.reason
    execute.assert_not_awaited()


async def test_expired_native_connection_returns_an_actionable_answer(desktop, monkeypatch):
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.loop import NativeAgentLoop

    computers, entry, host, _ = desktop
    await computers.desktop.release(entry.desktop)
    entry.desktop = None

    async def expired(entry):
        raise DesktopError("Desktop access was not connected in time. No app input was sent.")

    monkeypatch.setattr(computers, "wait_desktop", expired)
    loop = NativeAgentLoop()
    loop._auto_skill = False
    text = []
    loop.on("text_delta", text.append)
    async with computers.bind(entry):
        result = await loop.run("TextEdit에서 저장해", classification=ClassificationResult(
            goal_type="full", confidence=1, tier=2, intent_categories=frozenset({"desktop"})))
    assert result.reason.startswith("error: Desktop access was not connected in time")
    assert "No app input was sent" in "".join(text)
    assert "macOS permissions" in loop._last_answer_text
    assert not any(method == "act" for method, _ in host.calls)


@pytest.mark.parametrize("accessibility,screen", [(False, True), (True, False), (False, False)])
@pytest.mark.parametrize("settings_open", [True, False])
async def test_missing_macos_permissions_open_settings_once_without_execution(desktop, monkeypatch, accessibility, screen, settings_open):
    from unittest.mock import AsyncMock, call

    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.loop import NativeAgentLoop

    computers, entry, host, _ = desktop
    await computers.desktop.release(entry.desktop)
    entry.desktop = None
    request = AsyncMock(side_effect=[
        {"accessibility": accessibility, "screenRecording": screen, "apps": []},
        {"settingsOpened": True} if settings_open else DesktopError("Could not open System Settings."),
    ])
    monkeypatch.setattr(host, "request", request)
    loop = NativeAgentLoop()
    loop._auto_skill = False
    execute = AsyncMock()
    monkeypatch.setattr(loop, "_execute_loop", execute)
    async with asyncio.timeout(2), computers.bind(entry):
        result = await loop.run("Create a document in TextEdit", classification=ClassificationResult(
            goal_type="full", confidence=1, tier=2, intent_categories=frozenset({"desktop"})))
    assert result.reason.startswith("error: Rune Computer needs")
    if settings_open:
        assert "System Settings was opened for you" in loop._last_answer_text
    else:
        assert "Could not open System Settings" in loop._last_answer_text
        assert "System Settings > Privacy & Security" in loop._last_answer_text
        assert "was opened for you" not in loop._last_answer_text
    assert "No app input was sent" in loop._last_answer_text
    assert not entry.access_requested
    assert request.await_args_list == [call("status"), call("permissions", guard=entry.control.check)]
    execute.assert_not_awaited()


async def test_permission_reads_never_open_settings_and_requests_require_same_origin(desktop):
    _, _, host, client = desktop
    host.calls.clear()
    host.permissions = False
    for _ in range(2):
        assert (await client.get("/api/desktop/setup")).status_code == 200
    assert [method for method, _ in host.calls] == ["status", "status"]
    rejected = await client.post("/api/desktop/permissions", json={}, headers={"Origin": "https://other.test"})
    assert rejected.status_code == 403
    assert [method for method, _ in host.calls] == ["status", "status"]
    assert (await client.post("/api/desktop/permissions", json={})).status_code == 200
    assert [method for method, _ in host.calls] == ["status", "status", "permissions"]


async def test_stop_during_permission_check_does_not_open_settings(desktop, monkeypatch):
    computers, entry, host, _ = desktop
    await computers.desktop.release(entry.desktop)
    entry.desktop = None
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def request(method, params=None, **kwargs):
        calls.append(method)
        started.set()
        await release.wait()
        return {"accessibility": False, "screenRecording": True}

    monkeypatch.setattr(host, "request", request)
    async def request_access():
        async with computers.bind(entry):
            await require_desktop()

    task = asyncio.create_task(request_access())
    await asyncio.wait_for(started.wait(), 2)
    await computers.stop("run-one")
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert calls == ["status"]
    assert not entry.access_requested


async def test_pause_after_native_input_does_not_claim_it_was_never_executed(desktop):
    computers, entry, host, client = desktop
    applied, reply = asyncio.Event(), asyncio.Event()
    original_request = host.request

    async def request(method, params=None, **kwargs):
        data = await original_request(method, params, **kwargs)
        if method == "act":
            applied.set()
            await reply.wait()
        return data

    host.request = request
    task = asyncio.create_task(run_action(computers, entry))
    action_id, revision = await wait_for_review(entry.desktop)
    await allow_native(entry.desktop)
    await asyncio.wait_for(applied.wait(), 2)
    entry.control.pause()
    entry.desktop.invalidate()
    reply.set()
    with pytest.raises(DesktopError) as error:
        await task
    assert error.value.outcome == "unknown"
    assert entry.desktop.uncertain


async def test_native_action_requires_one_use_review_and_returns_real_image(desktop):
    computers, entry, host, client = desktop
    task = asyncio.create_task(run_action(computers, entry))
    action_id, revision = await wait_for_review(entry.desktop)
    view = (await client.get("/api/desktop/status?sessionId=one")).json()
    assert view["pending"]["target"]["name"] == "Save"
    assert host.number == 1
    assert (await decide(client, action_id, revision)).status_code == 409
    image = await client.get(f"/api/desktop/frame/{view['observation']}?sessionId=one")
    assert image.headers["cache-control"] == "no-store" and image.content.startswith(b"\x89PNG")
    await allow_native(entry.desktop)
    result = await task
    assert (await decide(client, action_id, revision)).status_code == 409
    assert result["observation"] != view["observation"]
    assert len([method for method, _ in host.calls if method == "act"]) == 1


@pytest.mark.parametrize("change", ["pause", "stop", "revoke", "expiry"])
async def test_control_changes_cancel_native_review_before_input(desktop, change):
    computers, entry, host, client = desktop
    task = asyncio.create_task(run_action(computers, entry))
    action_id, revision = await wait_for_review(entry.desktop)
    if change == "pause":
        response = await client.post("/api/computer/control", json={"sessionId": "one", "lease": entry.lease, "action": "pause"})
        assert response.status_code == 200
    elif change == "stop":
        await computers.stop("run-one")
    elif change == "revoke":
        await computers.desktop.release(entry.desktop)
    else:
        entry.desktop.expires = time.monotonic() - 1
    with pytest.raises((DesktopError, ControlChanged)):
        await task
    assert host.number == 1
    assert (await decide(client, action_id, revision)).status_code in {404, 409}


async def test_other_conversation_origin_and_revision_cannot_approve(desktop):
    computers, entry, host, client = desktop
    task = asyncio.create_task(run_action(computers, entry))
    action_id, revision = await wait_for_review(entry.desktop)
    assert (await decide(client, action_id, revision, session_id="other")).status_code == 404
    assert (await decide(client, action_id, revision + 1)).status_code == 409
    assert (await client.get(f"/api/desktop/frame/{entry.desktop.view['observation']}?sessionId=other")).status_code == 404
    response = await client.post("/api/desktop/control", headers={"Origin": "https://attacker.test"}, json={
        "sessionId": "one", "action": "decide", "actionId": action_id, "revision": revision, "approved": True})
    assert response.status_code == 403
    assert (await decide(client, action_id, revision, False)).status_code == 200
    with pytest.raises(DesktopError, match="cancelled before input"):
        await task
    assert host.number == 1


@pytest.mark.parametrize("outcome", ["not_executed", "unknown"])
async def test_native_errors_preserve_the_difference_between_stale_and_unknown(desktop, outcome):
    computers, entry, host, client = desktop
    host.failure = DesktopError("Window changed", outcome=outcome)
    task = asyncio.create_task(run_action(computers, entry))
    action_id, revision = await wait_for_review(entry.desktop)
    await allow_native(entry.desktop)
    with pytest.raises(DesktopError):
        await task
    assert entry.desktop.uncertain is (outcome == "unknown")
    if outcome == "unknown":
        entry.control.pause()
        assert (await client.post("/api/computer/control", json={"sessionId": "one", "lease": entry.lease, "action": "resume", "acknowledgeUnknown": True})).status_code == 409
        assert (await client.post("/api/desktop/control", json={"sessionId": "one", "revision": entry.desktop.revision,
            "action": "acknowledge", "approved": True})).status_code == 200
        assert not entry.desktop.uncertain


async def test_cancelled_dispatch_is_unknown_and_never_retried(desktop):
    computers, entry, host, client = desktop
    host.release = asyncio.Event()
    task = asyncio.create_task(run_action(computers, entry))
    action_id, revision = await wait_for_review(entry.desktop)
    await allow_native(entry.desktop)
    await host.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert entry.desktop.uncertain
    assert len([method for method, _ in host.calls if method == "act"]) == 1


async def test_pause_cancels_native_dispatch_before_input(desktop):
    computers, entry, host, client = desktop
    host.release = asyncio.Event()
    task = asyncio.create_task(run_action(computers, entry))
    action_id, revision = await wait_for_review(entry.desktop)
    await allow_native(entry.desktop)
    await host.started.wait()
    observed = host.number
    response = await client.post("/api/computer/control", json={
        "sessionId": "one", "lease": entry.lease, "action": "pause"})
    assert response.status_code == 200
    host.release.set()
    with pytest.raises(DesktopError, match="cancelled before input"):
        await task
    assert host.number == observed
    assert not entry.desktop.uncertain


async def test_desktop_grant_excludes_other_app_tasks_but_allows_ordinary_work(desktop):
    computers, entry, host, _ = desktop
    other = await computers.claim("other", "run-other")
    assert other.control.run_id == "run-other"
    before = list(host.calls)
    async with computers.bind(other):
        with pytest.raises(DesktopError, match="Another conversation has desktop access"):
            await require_desktop()
    assert entry.desktop.enabled and host.calls == before


async def test_child_task_cannot_activate_an_inherited_connection(desktop):
    computers, entry, host, _ = desktop
    before = list(host.calls)
    async with computers.bind(entry):
        with pytest.raises(DesktopError, match="active conversation"):
            await asyncio.create_task(require_desktop())
    assert host.calls == before and entry.desktop.task is None


@pytest.mark.parametrize("end", ["disconnect", "expiry"])
async def test_ending_unused_app_access_does_not_stop_ordinary_work(desktop, end):
    computers, entry, _, client = desktop
    async with computers.bind(entry):
        if end == "disconnect":
            response = await client.post("/api/desktop/control", json={
                "sessionId": "one", "action": "disconnect", "revision": entry.desktop.revision})
            assert response.status_code == 200
        else:
            entry.desktop.expires = 0
            await computers.reap()
        assert entry.control.state == "running"
        assert not entry.desktop.enabled


def test_native_control_pipe_rejects_stale_review_and_controller_disconnect(tmp_path):
    import os
    import select
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    if sys.platform != "darwin" or not shutil.which("swiftc"):
        pytest.skip("The native cancellation protocol requires the macOS Swift toolchain")
    harness = tmp_path / "Main.swift"
    harness.write_text('''
import Foundation
struct HostError: Error { let message: String }
@main struct Check {
    static func main() throws {
        let control = try HostControl(fd: Int32(CommandLine.arguments[1])!)
        if CommandLine.arguments.contains("watch") {
            control.watchDisconnect()
            FileHandle.standardOutput.write(Data("watching\\n".utf8))
            Thread.sleep(forTimeInterval: 30)
            return
        }
        while let line = readLine(), let epoch = Int(line) {
            var result = "allowed\\n"
            do { try control.check(epoch) } catch { result = "cancelled\\n" }
            FileHandle.standardOutput.write(Data(result.utf8))
        }
    }
}
''')
    source = Path(__file__).parents[2] / "rune/computer/native/Control.swift"
    executable = tmp_path / "control-test"
    subprocess.run(["swiftc", "-parse-as-library", "-module-cache-path", str(tmp_path / "cache"),
                    str(source), str(harness), "-o", str(executable)], check=True, capture_output=True, timeout=60)
    read_fd, write_fd = os.pipe()
    process = subprocess.Popen([str(executable), str(read_fd)], pass_fds=(read_fd,),
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    os.close(read_fd)

    def check(epoch):
        process.stdin.write(f"{epoch}\n")
        process.stdin.flush()
        assert select.select([process.stdout], [], [], 3)[0], "Native control did not respond"
        return process.stdout.readline().strip()

    try:
        assert check(0) == "allowed"
        os.write(write_fd, b"\x01")
        assert check(0) == "cancelled"
        assert check(1) == "allowed"
        os.close(write_fd)
        write_fd = None
        assert check(1) == "cancelled"
    finally:
        if write_fd is not None:
            os.close(write_fd)
        process.stdin.close()
        process.wait(timeout=3)
        process.stdout.close()

    for named in (False, True):
        if named:
            fifo = tmp_path / "control"
            os.mkfifo(fifo, 0o600)
            write_fd = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
            read_fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        else:
            read_fd, write_fd = os.pipe()
        watcher = subprocess.Popen([str(executable), str(read_fd), "watch"], pass_fds=(read_fd,), stdout=subprocess.PIPE)
        os.close(read_fd)
        try:
            assert select.select([watcher.stdout], [], [], 3)[0], "Disconnect watcher did not start"
            assert watcher.stdout.readline() == b"watching\n"
            os.close(write_fd)
            write_fd = None
            assert watcher.wait(timeout=3) == 0
        finally:
            if write_fd is not None:
                os.close(write_fd)
            if watcher.poll() is None:
                watcher.kill()
                watcher.wait(timeout=3)
            watcher.stdout.close()


async def test_grants_require_permissions_installed_apps_and_an_idle_host():
    host = Host()
    manager = DesktopManager(host)
    host.permissions = False
    with pytest.raises(DesktopError, match="Accessibility"):
        await manager.grant("one", ["test.editor"])
    host.permissions = True
    with pytest.raises(DesktopError, match="installed apps"):
        await manager.grant("one", ["unknown.app"])
    session = await manager.grant("one", ["test.editor"])
    with pytest.raises(DesktopError, match="Disconnect"):
        await manager.grant("two", ["test.editor"])
    await manager.release(session)
    assert not session.enabled and host.closed
    assert (await manager.grant("two", ["test.editor"])).session_id == "two"
    await manager.close()


async def test_run_start_waits_for_an_app_grant_in_progress(desktop):
    computers, entry, _, client = desktop
    computers.finish(entry, "run-one")
    await computers.desktop.release(entry.desktop)
    async with computers.access_lock:
        task = asyncio.create_task(computers.claim("two", "run-two"))
        await asyncio.sleep(0)
        assert not task.done()
    await task
    response = await client.post("/api/desktop/control", json={"sessionId": "one", "action": "grant", "apps": ["test.editor"]})
    assert response.status_code == 409


async def test_desktop_mode_blocks_alternate_transports_even_after_revocation(desktop):
    from rune.agent.tool_adapter import ToolAdapterOptions, build_tool_set
    from rune.capabilities.registry import CapabilityRegistry
    from rune.capabilities.types import CapabilityDefinition
    from rune.computer.capabilities import register_desktop_capabilities
    from rune.computer.session import TOOLS
    from rune.types import CapabilityResult

    computers, entry, _, _ = desktop
    registry = CapabilityRegistry()
    called = []

    async def shell(_):
        called.append(True)
        return CapabilityResult(success=True)

    registry.register(CapabilityDefinition(name="bash_execute", description="Test shell", execute=shell))
    register_desktop_capabilities(registry)
    async with computers.bind(entry):
        await require_desktop()
        tools = build_tool_set(ToolAdapterOptions(allowed_tools=sorted(TOOLS)), registry=registry)
        assert "desktop_observe" in tools and "bash_execute" not in tools
        result = await registry.execute("bash_execute", {})
        assert not result.success
        entry.desktop.revoke()
        assert not (await registry.execute("bash_execute", {})).success
        assert not called
        with pytest.raises(DesktopError):
            await entry.desktop.observe("test.editor")


async def test_child_task_cannot_inherit_a_desktop_grant(desktop):
    computers, entry, _, _ = desktop
    async with computers.bind(entry):
        await require_desktop()
        with pytest.raises(DesktopError, match="active conversation"):
            await asyncio.create_task(entry.desktop.observe("test.editor"))


async def test_unapproved_app_and_old_observation_are_rejected(desktop):
    computers, entry, host, _ = desktop
    async with computers.bind(entry):
        await require_desktop()
        with pytest.raises(DesktopError, match="not allowed"):
            await entry.desktop.observe("unknown.app")
        first = await entry.desktop.observe("test.editor")
        await entry.desktop.observe("test.editor")
        with pytest.raises(DesktopError, match="desktop_observe"):
            await entry.desktop.act(DesktopAction(observation=first["observation"], action="click", x=1, y=1))
    assert not any(method == "act" for method, _ in host.calls)


@pytest.mark.parametrize("tool,params", [
    ("desktop_observe", {"app": "test.editor"}),
    ("desktop_wait", {"app": "test.editor", "condition": {"kind": "title", "text": "Test document"}, "timeout_ms": 0}),
])
async def test_desktop_screenshots_reach_model_but_not_durable_journal(tmp_path, tool, params):
    from rune.agent.execution_journal import ExecutionJournal, journal_scope
    from rune.agent.tool_output import ToolOutput, output_for_model
    from rune.capabilities.registry import CapabilityRegistry
    from rune.computer.capabilities import register_desktop_capabilities

    class Store:
        records = []
        def save_attempt(self, record):
            self.records.append(record)

    host = Host()
    session = DesktopSession("one", host, {"test.editor": "Editor"}, time.monotonic() + 100)
    control = RunControl("run-one")
    store = Store()
    registry = CapabilityRegistry()
    register_desktop_capabilities(registry)
    with control_scope(control), desktop_scope(session, control), journal_scope(ExecutionJournal(store, "run-one", str(tmp_path))):
        result = await registry.execute(tool, params)
    assert result.success
    output = output_for_model(result.output, tool, result)
    assert isinstance(output, ToolOutput) and len(output.images) == 1
    assert all("image_base64" not in (record.get("result") or {}).get("metadata", {}) for record in store.records)


@pytest.mark.parametrize("model", ["gpt-6-astra", "anthropic/claude-opus-5"])
@pytest.mark.parametrize("tool,params", [
    ("desktop_observe", {"app": "test.editor"}),
    ("desktop_wait", {"app": "test.editor", "condition": {"kind": "title", "text": "Test document"}, "timeout_ms": 0}),
])
async def test_native_observations_survive_model_tool_dispatch_and_do_not_cache(desktop, model, tool, params):
    from rune.agent.litellm_adapter import StreamResult
    from rune.agent.tool_adapter import ToolAdapterOptions, build_tool_set
    from rune.agent.tool_output import ToolOutput
    from rune.capabilities.registry import CapabilityRegistry
    from rune.computer.capabilities import register_desktop_capabilities

    computers, entry, host, _ = desktop
    registry = CapabilityRegistry()
    register_desktop_capabilities(registry)
    async with computers.bind(entry):
        await require_desktop()
        tools = build_tool_set(ToolAdapterOptions(enable_guardian=False), registry=registry)
        stream = StreamResult(model=model, messages=[], tool_schemas=[],
                              tool_lookup={name: tool.function for name, tool in tools.items()},
                              max_tokens=512, temperature=0, request_tokens_limit=10000, response_tokens_limit=512)
        for _ in range(2):
            result = await stream._execute_tool(tool, params)
            assert isinstance(result, ToolOutput) and result.images[0].data == host.image
        assert len([method for method, _ in host.calls if method == "observe"]) == 2
        host.failure = DesktopError("Not dispatched")
        # A failed read must remain retryable after the user repairs permissions.
        entry.desktop.apps.clear()
        assert "not allowed" in str(await stream._execute_tool("desktop_observe", {"app": "test.editor"}))
        entry.desktop.apps["test.editor"] = "Editor"
        assert isinstance(await stream._execute_tool("desktop_observe", {"app": "test.editor"}), ToolOutput)


@pytest.mark.parametrize("params", [
    {"action": "click", "x": float("nan"), "y": 1},
    {"action": "click", "x": 1},
    {"action": "press", "ref": "x", "text": "hidden change"},
    {"action": "key", "key": "return", "modifiers": ["command", "command"]},
    {"action": "type", "text": "ok", "script": "run something"},
    {"action": "scroll", "deltaY": 999999},
    {"action": "drag", "x": 1, "y": 1, "endX": 2},
])
def test_native_input_schema_rejects_ambiguous_or_unbounded_commands(params):
    with pytest.raises(ValueError):
        DesktopAction(observation="observed", **params)


async def test_native_pipe_rechecks_control_after_waiting_for_another_request():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from rune.computer.macos import MacHost

    host = MacHost()
    pipe = Mock()
    host.process = SimpleNamespace(returncode=None, stdin=pipe, stdout=Mock())
    allowed = True

    def guard():
        if not allowed:
            raise DesktopError("Access was revoked while waiting")

    async with host.lock:
        task = asyncio.create_task(host.request("act", {"action": "press"}, guard=guard))
        await asyncio.sleep(0)
        allowed = False
    with pytest.raises(DesktopError, match="revoked"):
        await task
    pipe.write.assert_not_called()


@pytest.mark.parametrize("route", ["restart", "reveal"])
async def test_permission_repair_requires_same_origin_and_no_active_work(desktop, route):
    computers, entry, host, client = desktop
    response = await client.post(f"/api/desktop/{route}", headers={"Origin": "https://outside.test"})
    assert response.status_code == 403
    response = await client.post(f"/api/desktop/{route}")
    assert response.status_code == 409
    entry.control = None
    response = await client.post(f"/api/desktop/{route}")
    assert response.status_code == 409  # A connected app still owns this host.
    assert not host.closed


async def test_permission_restart_reports_fresh_status_without_granting_access(desktop, tmp_path):
    computers, entry, host, client = desktop
    await computers.desktop.release(entry.desktop)
    entry.control = None
    host.closed = False
    host.permissions = False
    host.executable = tmp_path / "Rune Computer.app/Contents/MacOS/RuneComputer"
    host.calls.clear()
    response = await client.post("/api/desktop/restart")
    assert response.status_code == 200, response.text
    assert host.closed
    assert host.calls == [("status", None)]
    assert response.json()["accessibility"] is False
    assert response.json()["screenRecording"] is False
    assert response.json()["appPath"] == str(host.executable.parents[2])


@pytest.mark.parametrize("exit_code", [0, 1])
async def test_permission_repair_reveals_only_the_installed_app(desktop, tmp_path, monkeypatch, exit_code):
    from unittest.mock import AsyncMock

    computers, entry, host, client = desktop
    await computers.desktop.release(entry.desktop)
    entry.control = None
    host.closed = False
    host.executable = tmp_path / "Rune Computer.app/Contents/MacOS/RuneComputer"
    host.executable.parent.mkdir(parents=True)
    host.executable.write_bytes(b"test app")
    process = AsyncMock()
    process.wait.return_value = exit_code
    launch = AsyncMock(return_value=process)
    monkeypatch.setattr("rune.api.desktop.asyncio.create_subprocess_exec", launch)
    response = await client.post("/api/desktop/reveal")
    assert response.status_code == (200 if exit_code == 0 else 409)
    assert launch.call_args.args == ("/usr/bin/open", "-R", str(host.executable.parents[2]))
    assert not host.closed


async def test_permission_restart_waits_for_an_inflight_status_request(desktop, tmp_path):
    computers, entry, host, client = desktop
    await computers.desktop.release(entry.desktop)
    entry.control = None
    host.closed = False
    host.executable = tmp_path / "Rune Computer.app/Contents/MacOS/RuneComputer"
    async with host.lock:
        restart = asyncio.create_task(client.post("/api/desktop/restart"))
        await asyncio.sleep(0)
        assert not restart.done() and not host.closed
    response = await asyncio.wait_for(restart, 2)
    assert response.status_code == 200 and host.closed
