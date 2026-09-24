"""Desktop failures must survive transport restarts and final-answer generation."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.loop import NativeAgentLoop
from rune.api.trust import build_trust_payload
from rune.computer.capabilities import EmptyParams, invoke
from rune.computer.protocol import DesktopAction, DesktopError, DesktopTarget
from rune.computer.session import require_desktop
from rune.types import CompletionTrace
from tests.integration.test_desktop_control import allow_native, decide, run_action, wait_for_review
from tests.integration.test_desktop_control import desktop as desktop


async def test_preparation_phase_keeps_app_input_and_workspace_access_separate(desktop, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from rune.agent.tool_adapter import ToolAdapterOptions, build_tool_set
    from rune.capabilities.document_bundle import BundleInspectParams, document_bundle_inspect
    from rune.capabilities.file import register_file_capabilities
    from rune.capabilities.registry import CapabilityRegistry
    from rune.computer.capabilities import PhaseParams, register_desktop_capabilities
    from rune.computer.protocol import DesktopTarget

    computers, entry, host, _ = desktop
    registry = CapabilityRegistry()
    register_file_capabilities(registry)
    register_desktop_capabilities(registry)
    (tmp_path / "notes.txt").write_text("prepared content")
    async with computers.bind(entry):
        await require_desktop()
        session = entry.desktop
        await session.observe("test.editor")
        tools = build_tool_set(ToolAdapterOptions(workspace_root=str(tmp_path), enable_guardian=False), registry)
        assert "unavailable" in await tools["file_read"].function(path="notes.txt")
        assert (await invoke("phase", PhaseParams(phase="prepare"))).success
        assert "prepared content" in await tools["file_read"].function(path="notes.txt")
        assert "BLOCKED" in await tools["file_read"].function(path="../outside.txt")
        monkeypatch.setattr("rune.capabilities.document_bundle.read_snapshot", lambda *args, **kwargs:
                            SimpleNamespace(manifest={"source": {"path": str(tmp_path.parent / "outside.csv")}}))
        bundle = await document_bundle_inspect(BundleInspectParams(directory=str(tmp_path / "bundle")))
        assert not bundle.success and "inside the selected workspace" in bundle.error
        blocked = await invoke("observe", DesktopTarget(app="test.editor"))
        assert not blocked.success
        assert "Return to the app phase" in session.completion_blocker()
        assert (await invoke("phase", PhaseParams(phase="app"))).success
        assert not session.view
        assert (await invoke("observe", DesktopTarget(app="test.editor"))).success
        session.uncertain = True
        assert not (await invoke("phase", PhaseParams(phase="prepare"))).success
        assert all(method != "act" for method, _ in host.calls)


async def test_model_sees_only_the_tools_for_each_desktop_phase(desktop, tmp_path, monkeypatch):
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.tool_adapter import ToolAdapterOptions, build_tool_set
    from rune.capabilities.file import register_file_capabilities
    from rune.capabilities.registry import CapabilityRegistry
    from rune.computer.capabilities import register_desktop_capabilities
    from tests.unit.test_live_streaming import _chunk, _tc

    computers, entry, _, _ = desktop
    registry = CapabilityRegistry()
    register_file_capabilities(registry)
    register_desktop_capabilities(registry)
    (tmp_path / "notes.txt").write_text("prepared content")
    seen = []
    calls = [("desktop_phase", '{"phase":"prepare"}'), ("file_read", '{"path":"notes.txt"}'),
             ("desktop_phase", '{"phase":"app"}'), ("desktop_observe", '{"app":"test.editor"}')]

    async def complete(**params):
        seen.append({tool["function"]["name"] for tool in params["tools"]})
        async def chunks():
            if len(seen) <= len(calls):
                name, args = calls[len(seen) - 1]
                yield _chunk(tool_calls=_tc(name, args), finish="tool_calls")
            else:
                yield _chunk(content="Inspected.", finish="stop")
        return chunks()

    monkeypatch.setattr("rune.agent.litellm_adapter.litellm.acompletion", complete)
    async with computers.bind(entry):
        await require_desktop()
        tools = build_tool_set(ToolAdapterOptions(workspace_root=str(tmp_path), enable_guardian=False), registry)
        agent = LiteLLMAgent("openai/gpt-5.4", tools=list(tools.values()))
        async with agent.run_stream("Prepare and inspect.") as stream:
            assert "".join([part async for part in stream.stream_text()]) == "Inspected."
        assert entry.desktop.completion_blocker() is None
    assert "file_read" not in seen[0] and "desktop_act" in seen[0]
    assert "file_read" in seen[1] and "desktop_act" not in seen[1]
    assert seen[1] == seen[2] and seen[0] == seen[3] == seen[4]


@pytest.mark.parametrize("failure", ["restart", "exit"])
async def test_lost_host_revokes_pending_input_and_preview(desktop, failure):
    computers, entry, host, client = desktop
    task = asyncio.create_task(run_action(computers, entry))
    action_id, revision = await wait_for_review(entry.desktop)
    observation = entry.desktop.view["observation"]
    if failure == "restart":
        host.generation = 1
    else:
        host.connected = False
    status = (await client.get("/api/desktop/status?sessionId=one")).json()
    assert not status["enabled"] and status["expiresIn"] == 0
    assert "Reconnect" in status["connectionError"]
    assert status["pending"] is None and "observation" not in status
    assert (await client.get(f"/api/desktop/frame/{observation}?sessionId=one")).status_code == 404
    assert (await decide(client, action_id, revision)).status_code == 409
    with pytest.raises(DesktopError):
        await asyncio.wait_for(task, 2)
    assert host.number == 1


async def test_reconnect_requires_new_grant_and_stale_session_cannot_cancel_it(desktop):
    computers, entry, host, client = desktop
    old = entry.desktop
    host.generation = 1
    entry.control = None
    response = await client.post("/api/desktop/control", json={
        "sessionId": "one", "action": "grant", "apps": ["test.editor"]})
    assert response.status_code == 200, response.text
    assert response.json()["enabled"] and not response.json()["connectionError"]
    assert entry.desktop is not old and entry.desktop.host_generation == 1
    assert len([method for method, _ in host.calls if method == "grant"]) == 2
    epoch = host.epoch
    old.invalidate()
    await computers.desktop.release(old)
    assert host.epoch == epoch and not host.closed
    assert computers.desktop.owner is entry.desktop


async def test_dead_owner_does_not_block_another_conversation(desktop):
    computers, entry, host, client = desktop
    host.connected = False
    entry.control = None
    other = await computers.claim("two", "run-two")
    assert not entry.desktop.enabled and other.desktop is None
    assert len([method for method, _ in host.calls if method == "grant"]) == 1


@pytest.mark.parametrize("native", [False, True])
async def test_connected_apps_only_activate_for_a_desktop_task(desktop, monkeypatch, native):
    from rune.computer.session import current_desktop

    computers, entry, host, _ = desktop
    session = entry.desktop
    session.last_error = "Previous task failed"
    session.observations = 9
    before = list(host.calls)
    loop = NativeAgentLoop()
    loop._auto_skill = False
    monkeypatch.setattr(loop, "_build_system_prompt", AsyncMock(return_value="test"))

    async def execute(**kwargs):
        assert current_desktop() is (session if native else None)
        if native:
            assert not session.last_error and session.observations == 0
            await session.observe("test.editor")
        return CompletionTrace(reason="completed")

    monkeypatch.setattr(loop, "_execute_loop", execute)
    async with computers.bind(entry):
        assert current_desktop() is None
        trace = await loop.run("test", classification=ClassificationResult(
            goal_type="full" if native else "chat", confidence=1, tier=2,
            intent_categories=frozenset({"desktop"}) if native else frozenset()))
    assert trace.reason == "completed"
    assert session.enabled and session.task is None
    if not native:
        assert host.calls == before
        assert session.last_error == "Previous task failed" and session.observations == 9


async def test_lost_desktop_stops_before_another_model_request(desktop, monkeypatch):
    from rune.agent.litellm_adapter import StreamResult

    computers, entry, host, client = desktop
    completion = AsyncMock(side_effect=AssertionError("Do not ask the model to retry a lost connection"))
    monkeypatch.setattr("litellm.acompletion", completion)
    stream = StreamResult(model="gpt-6-astra", messages=[{"role": "user", "content": "Use the app"}],
        tool_schemas=[], tool_lookup={}, max_tokens=200, temperature=0,
        request_tokens_limit=10000, response_tokens_limit=200)
    monkeypatch.setattr(stream, "_start_artifact_role_classification", lambda: None)
    async with computers.bind(entry):
        await require_desktop()
        host.generation = 1
        with pytest.raises(DesktopError, match="Reconnect"):
            async for _ in stream.stream_text():
                pytest.fail("A lost connection must not produce another model response")
    completion.assert_not_awaited()


@pytest.mark.parametrize("outcome", ["not_executed", "unknown"])
async def test_observing_does_not_erase_failed_input(desktop, outcome):
    computers, entry, host, client = desktop
    host.failure = DesktopError("Input did not confirm", outcome=outcome)

    async def run():
        async with computers.bind(entry):
            await require_desktop()
            view = await entry.desktop.observe("test.editor")
            with pytest.raises(DesktopError):
                await entry.desktop.act(DesktopAction(observation=view["observation"], action="press", ref="save"))
            view = await entry.desktop.observe("test.editor")
            assert entry.desktop.completion_blocker()
            action = DesktopAction(observation=view["observation"], action="key", key="escape")
            if outcome == "unknown":
                assert entry.desktop.uncertain
                with pytest.raises(DesktopError, match="unknown outcome"):
                    await entry.desktop.act(action)
            else:
                host.failure = None
                await entry.desktop.act(action)
                assert entry.desktop.completion_blocker(requires_input=True) is None

    task = asyncio.create_task(run())
    action_id, revision = await wait_for_review(entry.desktop)
    await allow_native(entry.desktop)
    if outcome == "not_executed":
        async with asyncio.timeout(2):
            while entry.desktop.pending is None or entry.desktop.pending["id"] == action_id:
                await asyncio.sleep(0.001)
        action_id, revision = await wait_for_review(entry.desktop)
        await allow_native(entry.desktop)
    await asyncio.wait_for(task, 2)
    assert len([method for method, _ in host.calls if method == "act"]) == (1 if outcome == "unknown" else 2)


@pytest.mark.parametrize("scenario", ["apps_only", "read_failed", "host_lost", "input_missing", "bad_app", "read_ok", "chat"])
async def test_final_answer_cannot_override_desktop_evidence(desktop, monkeypatch, scenario):
    computers, entry, host, client = desktop
    loop = NativeAgentLoop()
    loop._auto_skill = False
    monkeypatch.setattr(loop, "_build_system_prompt", AsyncMock(return_value="test"))

    async def execute(**kwargs):
        if scenario == "chat":
            return CompletionTrace(reason="completed")
        await invoke("apps", EmptyParams())
        if scenario != "apps_only":
            assert (await invoke("observe", DesktopTarget(app="test.editor"))).success
        if scenario == "host_lost":
            host.generation = 1
        elif scenario == "read_failed":
            host.request = AsyncMock(side_effect=DesktopError("Window unavailable"))
            assert not (await invoke("observe", DesktopTarget(app="test.editor"))).success
        elif scenario == "bad_app":
            assert not (await invoke("observe", DesktopTarget(app="other.app"))).success
        return CompletionTrace(reason="completed")

    monkeypatch.setattr(loop, "_execute_loop", execute)
    async with computers.bind(entry):
        trace = await loop.run("test", classification=ClassificationResult(
            goal_type="chat" if scenario == "chat" else "full", confidence=1, tier=2,
            requires_desktop_input=scenario == "input_missing",
            intent_categories=frozenset() if scenario == "chat" else frozenset({"desktop"})))
    trust = build_trust_payload(trace)
    if scenario in {"read_ok", "chat"}:
        assert trust["completionStatus"] == "completed"
    else:
        assert trust["completionStatus"] == "incomplete"
        assert trust["reason"] == "desktop_blocked"
        assert trust["completionCheck"]["name"] == "Desktop outcome"
        assert trust["completionCheck"]["detail"]
        assert not trust["verified"]
        assert not trust["canEscalate"] and not trust["escalationHint"]
        assert "tests" not in trust["honestNote"]


async def test_successful_retry_clears_read_failure_in_same_task(desktop):
    computers, entry, host, client = desktop
    original = host.request
    async with computers.bind(entry):
        await require_desktop()
        host.request = AsyncMock(side_effect=DesktopError("Window unavailable"))
        assert not (await invoke("observe", DesktopTarget(app="test.editor"))).success
        assert entry.desktop.completion_blocker() == "Window unavailable"
        host.request = original
        assert (await invoke("observe", DesktopTarget(app="test.editor"))).success
        assert entry.desktop.completion_blocker() is None


@pytest.mark.parametrize("native", [True, False])
async def test_table_checks_follow_execution_mode_even_if_intent_is_misclassified(desktop, monkeypatch, tmp_path, native):
    from rune.agent.table_acceptance import TableAcceptance
    from rune.computer.session import PREPARATION_TOOLS, TOOLS, desktop_scope

    _, entry, _, _ = desktop
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("RUNE_IN_BEST_OF", "1")
    monkeypatch.setattr("rune.agent.model_traits.supports_vision", lambda model: True)
    options = []
    prompts = []
    monkeypatch.setattr("rune.agent.loop.build_tool_set", lambda opts: options.append(opts) or {})
    monkeypatch.setattr("rune.agent.loop.LiteLLMAgent", lambda **kwargs: prompts.append(kwargs["system_prompt"]))
    loop = NativeAgentLoop()
    # A reused loop must not carry a file check into an app-only task.
    loop._table_acceptance = TableAcceptance("Previous CSV task", required=True)
    with desktop_scope(entry.desktop if native else None, entry.control):
        await loop._execute_loop(
            "Show the app screens in a table", "", [], 0,
            ClassificationResult(goal_type="full", confidence=1, tier=2,
                                 intent_categories=frozenset({"desktop", "table"})),
            context={"workspace_root": str(tmp_path)},
        )
    if native:
        assert options[0].allowed_tools == sorted(TOOLS | PREPARATION_TOOLS)
        assert options[0].table_acceptance is loop._table_acceptance
        assert not loop._table_acceptance.required
        assert await loop._table_acceptance.blocker() is None
    else:
        assert options[0].table_acceptance is loop._table_acceptance
        assert await loop._table_acceptance.blocker()
        assert "table_requirements" in prompts[0] and "table_verify" in prompts[0]
