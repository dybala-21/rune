"""Delayed screens and ineffective inputs must not turn into blind retries."""

import asyncio
import copy
import json
import time
import unicodedata
from io import BytesIO
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from PIL import Image
from pydantic import ValidationError

from rune.agent.litellm_adapter import StreamResult
from rune.computer.capabilities import invoke
from rune.computer.observation import ObservationProgress, match_condition, present, resolve_action
from rune.computer.protocol import (
    DesktopAction,
    DesktopCondition,
    DesktopError,
    DesktopTarget,
    DesktopWait,
)
from rune.computer.session import require_desktop
from tests.integration.test_desktop_control import allow_native
from tests.integration.test_desktop_control import desktop as desktop


def title(text="Ready"):
    return DesktopCondition(kind="title", text=text)


async def approve_next(session, client):
    await allow_native(session)


async def press_save(session, client):
    approval = asyncio.create_task(approve_next(session, client))
    try:
        return await session.act(DesktopAction(observation=session.view["observation"], action="press", ref="save"))
    finally:
        if not approval.done():
            approval.cancel()
        await asyncio.gather(approval, return_exceptions=True)


async def test_unchanged_input_stops_before_third_approval_and_dispatch(desktop):
    computers, entry, host, client = desktop
    async with computers.bind(entry):
        await require_desktop()
        first = await entry.desktop.observe("test.editor")
        for attempt in (1, 2):
            result = await press_save(entry.desktop, client)
            assert result["inputObservation"] == {"change": "no_visible_change", "unchangedAttempts": attempt}
            await entry.desktop.observe("test.editor")
        assert first["observation"] != entry.desktop.view["observation"]
        with pytest.raises(DesktopError, match="same input produced no visible change"):
            await entry.desktop.act(DesktopAction(observation=entry.desktop.view["observation"], action="press", ref="save"))
        assert entry.desktop.pending is None
        assert entry.desktop.completion_blocker(requires_input=True)
        assert not entry.desktop.uncertain
    assert len([method for method, _ in host.calls if method == "act"]) == 2


async def test_inputs_that_change_state_are_not_throttled(desktop):
    computers, entry, host, client = desktop
    original = host.request
    count = 0

    async def request(method, params=None, **kwargs):
        nonlocal count
        data = await original(method, params, **kwargs)
        if method == "act":
            count += 1
        data["title"] = f"Page {count}"
        return data

    host.request = request
    async with computers.bind(entry):
        await require_desktop()
        await entry.desktop.observe("test.editor")
        for _ in range(5):
            result = await press_save(entry.desktop, client)
            assert result["inputObservation"]["change"] == "changed"
        assert entry.desktop.completion_blocker(requires_input=True) is None
    assert count == 5


async def test_delayed_result_resolves_repeat_block_without_another_input(desktop):
    computers, entry, host, client = desktop
    async with computers.bind(entry):
        await require_desktop()
        await entry.desktop.observe("test.editor")
        for _ in range(2):
            await press_save(entry.desktop, client)
        with pytest.raises(DesktopError):
            await entry.desktop.act(DesktopAction(observation=entry.desktop.view["observation"], action="press", ref="save"))
        original = host.request

        async def request(*args, **kwargs):
            data = await original(*args, **kwargs)
            data["title"] = "Ready"
            return data

        host.request = request
        await entry.desktop.wait(DesktopWait(app="test.editor", condition=title(), timeout_ms=0))
        assert entry.desktop.completion_blocker(requires_input=True) is None
        assert entry.desktop.inputs == 2 and entry.desktop.repeat_block is None
    assert len([method for method, _ in host.calls if method == "act"]) == 2


async def test_matching_condition_cannot_clear_unknown_input_outcome(desktop):
    computers, entry, _, _ = desktop
    async with computers.bind(entry):
        await require_desktop()
        entry.desktop.uncertain = True
        entry.desktop.action_failed = True
        await entry.desktop.wait(DesktopWait(app="test.editor", condition=title("Test document"), timeout_ms=0))
        assert entry.desktop.uncertain and entry.desktop.action_failed
        assert entry.desktop.completion_blocker()


async def test_wait_returns_fresh_matching_evidence_without_repeating_input(desktop):
    computers, entry, host, _ = desktop
    original = host.request
    polls = 0

    async def request(method, params=None, **kwargs):
        nonlocal polls
        data = await original(method, params, **kwargs)
        polls += 1
        data["title"] = "Ready" if polls >= 3 else "Loading"
        return data

    host.request = request
    async with computers.bind(entry):
        await require_desktop()
        result = await invoke("wait", DesktopWait(app="test.editor", condition=title(), timeout_ms=1500))
        assert result.success, result.error
        data = json.loads(result.output.split("\n", 1)[1])
        assert data["conditionCheck"]["evidence"] == {"title": "Ready"}
        assert data["conditionCheck"]["polls"] == 3
        assert result.metadata["image_base64"]
        assert data["controls"][0]["ref"] == "e1"
        assert data["observation"] == entry.desktop.view["observation"]
    assert polls == 3 and not any(method == "act" for method, _ in host.calls)


@pytest.mark.parametrize("factory", ["asyncio", "uvloop"])
@pytest.mark.parametrize("matches", [True, False])
def test_wait_deadline_uses_the_running_event_loop(factory, matches):
    from rune.agent.run_control import RunControl, control_scope
    from rune.computer.session import DesktopSession, desktop_scope
    from tests.integration.test_desktop_control import Host

    async def run():
        host = Host()
        original = host.request

        async def request(*args, **kwargs):
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                await host.close()
                raise
            result = await original(*args, **kwargs)
            result["title"] = "Ready" if matches and host.number >= 2 else "Loading"
            return result

        host.request = request
        session = DesktopSession("clock-test", host, {"test.editor": "Editor"}, time.monotonic() + 30)
        control = RunControl("clock-test")
        with control_scope(control), desktop_scope(session, control):
            started = time.monotonic()
            condition = DesktopWait(app="test.editor", condition=title(), timeout_ms=1000 if matches else 350)
            if matches:
                result = await session.wait(condition)
                assert result["conditionCheck"]["polls"] == 2
                assert result["conditionCheck"]["elapsedMs"] >= 250
            else:
                with pytest.raises(DesktopError, match="wait limit"):
                    await session.wait(condition)
                assert time.monotonic() - started >= 0.3
            assert time.monotonic() - started < 1.5
            assert not session.waiting and not session.lock.locked()
            assert not host.closed
            assert all(method == "observe" for method, _ in host.calls)

    loop_factory = asyncio.new_event_loop if factory == "asyncio" else pytest.importorskip("uvloop").new_event_loop
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(run())


@pytest.mark.parametrize("stop", ["timeout", "pause", "revoke", "cancel", "disconnect"])
async def test_wait_is_bounded_and_preserves_control(desktop, stop):
    computers, entry, host, _ = desktop
    started = asyncio.Event()
    original = host.request

    async def request(*args, **kwargs):
        result = await original(*args, **kwargs)
        started.set()
        return result

    host.request = request

    async def run():
        async with computers.bind(entry):
            await require_desktop()
            return await entry.desktop.wait(DesktopWait(app="test.editor", condition=title(), timeout_ms=500 if stop == "timeout" else 10000))

    task = asyncio.create_task(run())
    await started.wait()
    since = time.monotonic()
    if stop == "pause":
        entry.control.pause()
    elif stop == "revoke":
        entry.desktop.revoke()
    elif stop == "cancel":
        task.cancel()
    elif stop == "disconnect":
        host.connected = False
    with pytest.raises((DesktopError, asyncio.CancelledError, RuntimeError)):
        await asyncio.wait_for(task, 1.5)
    assert time.monotonic() - since < 1.5
    assert not entry.desktop.lock.locked() and entry.desktop.pending is None
    assert not entry.desktop.waiting
    assert not any(method == "act" for method, _ in host.calls)


async def test_wait_rejects_unapproved_or_wrong_app_and_does_not_hide_host_error(desktop):
    computers, entry, host, _ = desktop
    async with computers.bind(entry):
        await require_desktop()
        before = len(host.calls)
        result = await invoke("wait", DesktopWait(app="other.app", condition=title()))
        assert not result.success and len(host.calls) == before
        host.request = AsyncMock(side_effect=DesktopError("Permission revoked"))
        result = await invoke("wait", DesktopWait(app="test.editor", condition=title()))
        assert not result.success and result.error == "Permission revoked"
        assert host.request.await_count == 1
        entry.desktop.apps["other.app"] = "Other"
        host.request = AsyncMock(return_value={"app": "other.app", "title": "Ready"})
        result = await invoke("wait", DesktopWait(app="test.editor", condition=title()))
        assert not result.success and "different app" in result.error


async def test_short_refs_resolve_only_against_the_current_observation(desktop):
    computers, entry, host, client = desktop
    async with computers.bind(entry):
        await require_desktop()
        result = await invoke("observe", DesktopTarget(app="test.editor"))
        visible = json.loads(result.output.split("\n", 1)[1])
        action = DesktopAction(observation=visible["observation"], action="press", ref="e1")
        assert resolve_action(action, entry.desktop.view).ref == "save"
        assert entry.desktop.view["controls"][0]["ref"] == "save"
        await entry.desktop.observe("test.editor")
        assert not (await invoke("act", action)).success
        action = action.model_copy(update={"observation": entry.desktop.view["observation"]})
        assert not (await invoke("act", action.model_copy(update={"ref": "e999"}))).success
        assert not any(method == "act" for method, _ in host.calls)
        approval = asyncio.create_task(approve_next(entry.desktop, client))
        result = await invoke("act", action)
        await approval
        assert result.success
        assert [params for method, params in host.calls if method == "act"][0]["ref"] == "save"


def screen(color="white", pixel=None):
    image = Image.new("RGB", (384, 384), color)
    if pixel is not None:
        image.putpixel((0, 0), pixel)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_state_identity_ignores_refs_but_keeps_values_and_pixels():
    progress = ObservationProgress()
    view = {"app": "test.editor", "title": "문서", "observation": "old", "capturedAt": 1,
            "controls": [{"ref": "old", "role": "AXTextArea", "value": "내용"}]}
    assert progress.observe(view, screen())["change"] == "initial"
    view.update(observation="new", capturedAt=2)
    view["controls"][0]["ref"] = "new"
    view["title"] = unicodedata.normalize("NFD", view["title"])
    assert progress.observe(view, screen())["change"] == "unchanged"
    view["controls"][0]["value"] = "다른 내용"
    assert progress.observe(view, screen())["change"] == "changed"
    assert progress.observe(view, screen("black"))["change"] == "changed"


async def test_pixel_noise_cannot_reopen_a_repeated_input(desktop):
    import base64

    computers, entry, host, client = desktop
    original = host.request
    sequence = 0

    async def request(*args, **kwargs):
        nonlocal sequence
        data = await original(*args, **kwargs)
        sequence += 1
        data["image_base64"] = base64.b64encode(screen(pixel=(sequence % 255, 0, 0))).decode()
        return data

    host.request = request
    async with computers.bind(entry):
        await require_desktop()
        await entry.desktop.observe("test.editor")
        for _ in range(2):
            await press_save(entry.desktop, client)
            await entry.desktop.observe("test.editor")
        with pytest.raises(DesktopError, match="same input"):
            await entry.desktop.act(DesktopAction(observation=entry.desktop.view["observation"], action="press", ref="save"))
    assert len([method for method, _ in host.calls if method == "act"]) == 2


def test_canvas_progress_and_small_accessible_values_are_preserved():
    progress = ObservationProgress()
    view = {"app": "canvas", "controls": []}
    progress.observe(view, screen())
    action = DesktopAction(observation="view", action="scroll", deltaY=200)
    for color in ("black", "red", "blue", "white"):
        key = progress.action_key(action, view)
        assert progress.blocker(key) is None
        progress.observe(view, screen(color))
        assert progress.acted(key)["change"] == "changed"
    view["controls"] = [{"role": "AXTextField", "value": "1"}]
    progress.observe(view, screen())
    view["controls"][0]["value"] = "2"
    assert progress.observe(view, screen())["change"] == "changed"


def test_wait_conditions_require_one_matching_control_and_never_treat_missing_as_empty():
    view = {"title": "Loading", "controls": [
        {"role": "AXTextField", "name": "Other", "value": "Saved"},
        {"role": "AXTextArea", "name": "Document", "value": "Draft"}]}
    condition = DesktopCondition(kind="control", role="AXTextArea", name="Document", text="Saved")
    assert match_condition(condition, view) is None
    view["controls"][1]["value"] = "Saved"
    assert match_condition(condition, view)["ref"] == "e2"
    view["controls"][1]["value"] = unicodedata.normalize("NFD", "한글 본문")
    condition = condition.model_copy(update={"text": "한글", "match": "contains"})
    assert match_condition(condition, view)
    condition = condition.model_copy(update={"text": "", "match": "equals"})
    assert match_condition(condition, {"controls": [{"role": "AXTextArea", "name": "Document"}]}) is None


@pytest.mark.parametrize("args", [
    {"kind": "control", "text": "Ready"},
    {"kind": "title", "text": ""},
    {"kind": "title", "text": "Ready", "role": "AXButton"},
    {"kind": "control", "role": "AXTextArea", "text": "", "match": "contains"},
])
def test_ambiguous_wait_conditions_are_rejected(args):
    with pytest.raises(ValidationError):
        DesktopCondition(**args)


@pytest.mark.parametrize("failure", ["tool", "policy", "control"])
async def test_failed_desktop_call_halts_batch_but_answers_every_call_id(desktop, monkeypatch, failure):
    computers, entry, _, _ = desktop
    first = AsyncMock(return_value="ERROR: Control changed" if failure == "control" else "[ERROR] The screen changed")
    next_input = AsyncMock(return_value="must not execute")
    stream = StreamResult(model="gpt-6-astra", messages=[], tool_schemas=[],
        tool_lookup={"desktop_observe": first, "desktop_act": next_input},
        max_tokens=200, temperature=0, request_tokens_limit=10000, response_tokens_limit=200)
    monkeypatch.setattr(stream, "_classify_artifact_roles", AsyncMock())
    if failure == "policy":
        for _ in range(7):
            stream._policy.record_tool_call("desktop_observe")
    async with computers.bind(entry):
        await require_desktop()
        await stream._execute_tool_batch([
            {"id": f"call-{i}", "function": {"name": name, "arguments": "{}"}}
            for i, name in enumerate(["desktop_observe", "desktop_act", "desktop_act"])
        ])
    assert first.await_count == (0 if failure == "policy" else 1)
    next_input.assert_not_awaited()
    messages = [m for m in stream.all_messages() if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in messages] == ["call-0", "call-1", "call-2"]
    assert all("Skipped" in m["content"] for m in messages[1:])


@pytest.mark.parametrize("failure", ["timeout", "read"])
async def test_failed_condition_survives_unrelated_observations_until_its_app_matches(desktop, failure):
    computers, entry, host, _ = desktop
    entry.desktop.apps["other.app"] = "Other"
    original = host.request
    async with computers.bind(entry):
        await require_desktop()
        if failure == "read":
            host.request = AsyncMock(side_effect=DesktopError("Window unavailable"))
        with pytest.raises(DesktopError):
            await entry.desktop.wait(DesktopWait(app="test.editor", condition=title("Saved"), timeout_ms=0))
        host.request = original
        await entry.desktop.observe("test.editor")
        assert entry.desktop.completion_blocker()
        await entry.desktop.wait(DesktopWait(app="test.editor", condition=title("Test document"), timeout_ms=0))
        assert entry.desktop.completion_blocker()

        async def request(*args, **kwargs):
            data = await original(*args, **kwargs)
            data["title"] = "Saved"
            return data

        host.request = request
        await entry.desktop.observe("other.app")
        assert entry.desktop.completion_blocker()
        await entry.desktop.observe("test.editor")
        assert entry.desktop.completion_blocker() is None


async def test_wait_deadline_does_not_cancel_an_inflight_native_reply(desktop):
    computers, entry, host, _ = desktop
    original = host.request

    async def request(*args, **kwargs):
        try:
            await asyncio.sleep(0.08)
        except asyncio.CancelledError:
            await host.close()
            raise
        return await original(*args, **kwargs)

    host.request = request
    async with computers.bind(entry):
        await require_desktop()
        started = time.monotonic()
        with pytest.raises(DesktopError, match="wait limit"):
            await entry.desktop.wait(DesktopWait(app="test.editor", condition=title(), timeout_ms=50))
        assert 0.075 <= time.monotonic() - started < 0.5
        assert not host.closed
        await entry.desktop.observe("test.editor")


def test_truncated_control_equality_checks_full_normalized_value():
    import hashlib

    text = "한" * 500 + "문서의 끝"
    control = {"role": "AXTextArea", "name": "Document", "value": text[:500], "valueTruncated": True,
               "valueSHA256": hashlib.sha256(text.encode()).hexdigest()}
    view = {"controls": [control]}
    condition = DesktopCondition(kind="control", role="AXTextArea", text=text[:500])
    assert match_condition(condition, view) is None
    condition = condition.model_copy(update={"text": unicodedata.normalize("NFD", text)})
    assert match_condition(condition, view)
    control.pop("valueSHA256")
    assert match_condition(condition, view) is None
    assert match_condition(condition.model_copy(update={"text": "한", "match": "contains"}), view)


def test_policy_allows_a_different_tool_after_blocking_a_loop():
    from rune.agent.tool_call_policy import ToolCallPolicy

    policy = ToolCallPolicy()
    for _ in range(7):
        policy.record_tool_call("desktop_observe")
    assert policy.should_block_tool("desktop_observe")
    assert not policy.should_block_tool("desktop_wait")


def test_compact_refs_preserve_control_data_and_reduce_payload():
    view = {"app": "test.editor", "controls": [
        {"ref": str(uuid4()), "role": "AXButton", "name": f"Button {i}", "value": ""}
        for i in range(100)]}
    original = copy.deepcopy(view)
    compact = present(view)
    assert view == original
    assert len(json.dumps(compact)) < len(json.dumps(view)) * 0.8
    for before, after in zip(view["controls"], compact["controls"], strict=True):
        assert {k: v for k, v in before.items() if k != "ref"} == {k: v for k, v in after.items() if k != "ref"}
