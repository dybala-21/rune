"""Verify-on-stop: one bounded reminder when the model finishes with a prose
answer after editing code without running any test since the last edit.

Measured motivation (SWE-bench trace autopsy 2026-07-26): runs were truncated
or self-declared done before any test executed; hermes's equivalent gate was
one of the three scaffold traits behind its higher per-sample fix rate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import rune.agent.litellm_adapter as la
from rune.agent.litellm_adapter import StreamResult


def _delta_chunk(*, content=None, tool_calls=None, finish_reason=None):
    tc_objs = None
    if tool_calls:
        tc_objs = [
            SimpleNamespace(
                index=tc["index"],
                id=tc.get("id", f"tc{tc['index']}"),
                function=SimpleNamespace(
                    name=tc.get("name"), arguments=tc.get("arguments")
                ),
            )
            for tc in tool_calls
        ]
    choice = SimpleNamespace(
        delta=SimpleNamespace(content=content, tool_calls=tc_objs),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice], usage=None)


async def _astream(chunks):
    for c in chunks:
        yield c


def _edit_turn():
    return [
        _delta_chunk(tool_calls=[{
            "index": 0, "name": "file_edit",
            "arguments": '{"path": "src/mod.py", "search": "a", "replace": "b"}',
        }]),
        _delta_chunk(finish_reason="tool_calls"),
    ]


def _bash_turn(command: str):
    import json as _json
    return [
        _delta_chunk(tool_calls=[{
            "index": 0, "name": "bash_execute",
            "arguments": _json.dumps({"command": command}),
        }]),
        _delta_chunk(finish_reason="tool_calls"),
    ]


def _final_turn(text="done, the fix is applied"):
    return [
        _delta_chunk(content=text),
        _delta_chunk(finish_reason="stop"),
    ]


def _make_result():
    async def edit_tool(**_):
        return "edited"

    async def bash_tool(**_):
        return "2 passed in 0.1s"

    return StreamResult(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "fix the bug"}],
        tool_schemas=[
            {"function": {"name": "file_edit"}},
            {"function": {"name": "bash_execute"}},
        ],
        tool_lookup={"file_edit": edit_tool, "bash_execute": bash_tool},
        max_tokens=8192,
        temperature=0.0,
        request_tokens_limit=200000,
        response_tokens_limit=8192,
        max_tool_rounds=20,
    )


def _fake_acompletion(streams):
    it = iter(streams)

    async def fake(**kwargs):
        return _astream(next(it))

    return fake


def _vos_messages(result):
    return [
        m for m in result.all_messages()
        if m.get("role") == "user"
        and m.get("content") == la._VERIFY_ON_STOP_MSG
    ]


@pytest.mark.asyncio
async def test_nudges_once_when_finishing_untested_edit(monkeypatch):
    monkeypatch.delenv("RUNE_VERIFY_ON_STOP", raising=False)
    result = _make_result()
    # edit → tries to finish → nudged → runs tests → finishes.
    streams = [
        _edit_turn(),
        _final_turn(),
        _bash_turn("python -m pytest tests/test_mod.py -q"),
        _final_turn("all tests passed"),
    ]
    monkeypatch.setattr(la.litellm, "acompletion", _fake_acompletion(streams))

    async for _ in result.stream_text():
        pass

    assert len(_vos_messages(result)) == 1
    # final answer survived after the verification round
    assert "passed" in result._collected_text


@pytest.mark.asyncio
async def test_no_nudge_when_tests_ran_after_edit(monkeypatch):
    monkeypatch.delenv("RUNE_VERIFY_ON_STOP", raising=False)
    result = _make_result()
    streams = [
        _edit_turn(),
        _bash_turn("pytest -q"),
        _final_turn(),
    ]
    monkeypatch.setattr(la.litellm, "acompletion", _fake_acompletion(streams))

    async for _ in result.stream_text():
        pass

    assert _vos_messages(result) == []


@pytest.mark.asyncio
async def test_nudge_is_bounded_to_one(monkeypatch):
    monkeypatch.delenv("RUNE_VERIFY_ON_STOP", raising=False)
    result = _make_result()
    # Model ignores the nudge, edits again, and finishes untested — the
    # second finish must NOT be blocked (bounded at one reminder).
    streams = [
        _edit_turn(),
        _final_turn(),
        _edit_turn(),
        _final_turn("shipping anyway"),
    ]
    monkeypatch.setattr(la.litellm, "acompletion", _fake_acompletion(streams))

    async for _ in result.stream_text():
        pass

    assert len(_vos_messages(result)) == 1


@pytest.mark.asyncio
async def test_env_opt_out(monkeypatch):
    monkeypatch.setenv("RUNE_VERIFY_ON_STOP", "0")
    result = _make_result()
    streams = [_edit_turn(), _final_turn()]
    monkeypatch.setattr(la.litellm, "acompletion", _fake_acompletion(streams))

    async for _ in result.stream_text():
        pass

    assert _vos_messages(result) == []


def test_is_test_command_matches_runners():
    assert la._is_test_command("python -m pytest tests/ -q")
    assert la._is_test_command("cd repo && npm test")
    assert la._is_test_command("cargo test --lib")
    assert la._is_test_command("python tests/runtests.py --parallel 1 x")
    assert not la._is_test_command("grep -r pytest docs/")
    assert not la._is_test_command("pip install pytest")
