"""Exploration-round budget in StreamResult.stream_text.

The observed SWE-bench failure mode: a weak model spends every tool round on
read-only exploration (grep/find/read), hits the round cap, and the run ends
with no edit — an empty patch. The budget counts consecutive no-edit tool
rounds; at the budget a steering nudge is injected, and after a grace window a
single file_edit call is forced via tool_choice (narrow edit path).
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


def _read_turn(idx: int):
    """One assistant turn that only calls the read-only file_read tool."""
    return [
        _delta_chunk(tool_calls=[{
            "index": 0, "name": "file_read",
            "arguments": f'{{"path": "src/mod{idx}.py"}}',
        }]),
        _delta_chunk(finish_reason="tool_calls"),
    ]


def _edit_turn():
    return [
        _delta_chunk(tool_calls=[{
            "index": 0, "name": "file_edit",
            "arguments": '{"path": "src/mod.py", "search": "a", "replace": "b"}',
        }]),
        _delta_chunk(finish_reason="tool_calls"),
    ]


_FINAL_TURN = [
    _delta_chunk(content="done"),
    _delta_chunk(finish_reason="stop"),
]


def _make_result(
    explore_budget: int,
    captured: list | None = None,
    max_tool_rounds: int = 20,
):
    async def read_tool(**_):
        return "file contents"

    async def edit_tool(**_):
        return "edited"

    return StreamResult(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "fix the bug"}],
        tool_schemas=[
            {"function": {"name": "file_read"}},
            {"function": {"name": "file_edit"}},
        ],
        tool_lookup={"file_read": read_tool, "file_edit": edit_tool},
        max_tokens=8192,
        temperature=0.0,
        request_tokens_limit=200000,
        response_tokens_limit=8192,
        max_tool_rounds=max_tool_rounds,
        explore_budget=explore_budget,
    )


def _fake_acompletion(streams, captured: list | None = None):
    it = iter(streams)

    async def fake(**kwargs):
        if captured is not None:
            captured.append(kwargs)
        return _astream(next(it))

    return fake


def _nudge_messages(result):
    return [
        m for m in result.all_messages()
        if m.get("role") == "user" and m.get("content") == la._EXPLORE_NUDGE
    ]


@pytest.mark.asyncio
async def test_nudge_after_budget_no_edit_rounds(monkeypatch):
    monkeypatch.delenv("RUNE_EXPLORE_BUDGET", raising=False)
    result = _make_result(explore_budget=2)
    streams = [_read_turn(1), _read_turn(2), _FINAL_TURN]
    monkeypatch.setattr(la.litellm, "acompletion", _fake_acompletion(streams))

    async for _ in result.stream_text():
        pass

    assert len(_nudge_messages(result)) == 1


@pytest.mark.asyncio
async def test_edit_round_resets_budget(monkeypatch):
    monkeypatch.delenv("RUNE_EXPLORE_BUDGET", raising=False)
    result = _make_result(explore_budget=2)
    # read, edit (reset), read — never 2 consecutive no-edit rounds
    streams = [_read_turn(1), _edit_turn(), _read_turn(2), _FINAL_TURN]
    monkeypatch.setattr(la.litellm, "acompletion", _fake_acompletion(streams))

    async for _ in result.stream_text():
        pass

    assert _nudge_messages(result) == []


@pytest.mark.asyncio
async def test_budget_zero_never_nudges(monkeypatch):
    monkeypatch.delenv("RUNE_EXPLORE_BUDGET", raising=False)
    result = _make_result(explore_budget=0)
    streams = [_read_turn(i) for i in range(6)] + [_FINAL_TURN]
    monkeypatch.setattr(la.litellm, "acompletion", _fake_acompletion(streams))

    async for _ in result.stream_text():
        pass

    assert _nudge_messages(result) == []


@pytest.mark.asyncio
async def test_forced_edit_tool_choice_near_cap(monkeypatch):
    monkeypatch.delenv("RUNE_EXPLORE_BUDGET", raising=False)
    monkeypatch.delenv("RUNE_EXPLORE_FORCE_EDIT", raising=False)
    captured: list = []
    # cap=8: nudge at min(1, max(2, 8-6))=1; force at max(1+2, 8-3)=5 —
    # i.e. only near cap exhaustion, never mid-diagnosis.
    result = _make_result(explore_budget=1, max_tool_rounds=8)
    streams = [_read_turn(i) for i in range(5)] + [_FINAL_TURN]
    monkeypatch.setattr(
        la.litellm, "acompletion", _fake_acompletion(streams, captured)
    )

    async for _ in result.stream_text():
        pass

    # The call AFTER the 5th no-edit round must constrain tool_choice to
    # file_edit (one-shot).
    forced = [
        kw for kw in captured
        if kw.get("tool_choice") == {
            "type": "function", "function": {"name": "file_edit"}
        }
    ]
    assert len(forced) == 1
    assert forced[0] is captured[5]
    # And the force message was injected.
    force_msgs = [
        m for m in result.all_messages()
        if m.get("role") == "user" and m.get("content") == la._EXPLORE_FORCE_MSG
    ]
    assert len(force_msgs) == 1


@pytest.mark.asyncio
async def test_no_force_when_cap_headroom_remains(monkeypatch):
    # cap=20 → force_at = 17. Five no-edit rounds must NOT trigger the forced
    # edit (the old fixed nudge+4 schedule forced edits mid-diagnosis).
    monkeypatch.delenv("RUNE_EXPLORE_BUDGET", raising=False)
    monkeypatch.delenv("RUNE_EXPLORE_FORCE_EDIT", raising=False)
    captured: list = []
    result = _make_result(explore_budget=1, max_tool_rounds=20)
    streams = [_read_turn(i) for i in range(5)] + [_FINAL_TURN]
    monkeypatch.setattr(
        la.litellm, "acompletion", _fake_acompletion(streams, captured)
    )

    async for _ in result.stream_text():
        pass

    assert len(_nudge_messages(result)) == 1  # nudge fired
    assert all(
        not isinstance(kw.get("tool_choice"), dict) for kw in captured
    )  # but no forced edit — plenty of cap left


@pytest.mark.asyncio
async def test_nudge_clamped_to_cap_minus_six(monkeypatch):
    # budget=8 but cap=6 → nudge_at = max(2, 0) → min(8, 2) = 2.
    monkeypatch.delenv("RUNE_EXPLORE_BUDGET", raising=False)
    result = _make_result(explore_budget=8, max_tool_rounds=6)
    streams = [_read_turn(i) for i in range(3)] + [_FINAL_TURN]
    monkeypatch.setattr(la.litellm, "acompletion", _fake_acompletion(streams))

    async for _ in result.stream_text():
        pass

    assert len(_nudge_messages(result)) == 1


@pytest.mark.asyncio
async def test_force_edit_env_opt_out(monkeypatch):
    monkeypatch.delenv("RUNE_EXPLORE_BUDGET", raising=False)
    monkeypatch.setenv("RUNE_EXPLORE_FORCE_EDIT", "0")
    captured: list = []
    result = _make_result(explore_budget=1, max_tool_rounds=8)
    streams = [_read_turn(i) for i in range(6)] + [_FINAL_TURN]
    monkeypatch.setattr(
        la.litellm, "acompletion", _fake_acompletion(streams, captured)
    )

    async for _ in result.stream_text():
        pass

    assert len(_nudge_messages(result)) == 1  # nudge still fires
    assert all(
        not isinstance(kw.get("tool_choice"), dict) for kw in captured
    )  # but no forced file_edit call


@pytest.mark.asyncio
async def test_env_zero_disables_even_when_wired_on(monkeypatch):
    monkeypatch.setenv("RUNE_EXPLORE_BUDGET", "0")
    result = _make_result(explore_budget=2)
    streams = [_read_turn(i) for i in range(4)] + [_FINAL_TURN]
    monkeypatch.setattr(la.litellm, "acompletion", _fake_acompletion(streams))

    async for _ in result.stream_text():
        pass

    assert _nudge_messages(result) == []
