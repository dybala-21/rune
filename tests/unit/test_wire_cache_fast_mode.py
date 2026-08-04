"""Wire-level checks for the moving message cache breakpoint and fast mode.

The kwargs actually sent to litellm.acompletion must carry (a) a
cache_control mark on the final message so the transcript prefix is served
from cache on later rounds, and (b) speed="fast" only when RUNE_FAST_MODE is
set for an Anthropic model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import rune.agent.litellm_adapter as la
from rune.agent.litellm_adapter import StreamResult


def _delta_chunk(*, content=None, finish_reason=None):
    choice = SimpleNamespace(
        delta=SimpleNamespace(content=content, tool_calls=None),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice], usage=None)


async def _astream(chunks):
    for c in chunks:
        yield c


_FINAL_TURN = [
    _delta_chunk(content="done"),
    _delta_chunk(finish_reason="stop"),
]


def _make_result(model: str = "anthropic/claude-haiku-4-5"):
    return StreamResult(
        model=model,
        messages=[{"role": "user", "content": "fix the bug"}],
        tool_schemas=[],
        tool_lookup={},
        max_tokens=8192,
        temperature=0.0,
        request_tokens_limit=200000,
        response_tokens_limit=8192,
        max_tool_rounds=5,
    )


def _capture_kwargs(monkeypatch):
    captured: list[dict] = []

    async def fake(**kwargs):
        captured.append(kwargs)
        return _astream(list(_FINAL_TURN))

    monkeypatch.setattr(la.litellm, "acompletion", fake)
    return captured


async def _drain(result):
    async for _ in result.stream_text():
        pass


@pytest.mark.asyncio
async def test_last_wire_message_carries_cache_control(monkeypatch):
    monkeypatch.delenv("RUNE_MSG_CACHE", raising=False)
    captured = _capture_kwargs(monkeypatch)
    await _drain(_make_result())

    last = captured[0]["messages"][-1]
    if last.get("role") == "tool":
        assert last["cache_control"] == {"type": "ephemeral"}
    else:
        assert last["content"][-1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_msg_cache_env_off_leaves_wire_clean(monkeypatch):
    monkeypatch.setenv("RUNE_MSG_CACHE", "0")
    captured = _capture_kwargs(monkeypatch)
    await _drain(_make_result())

    last = captured[0]["messages"][-1]
    assert "cache_control" not in last
    assert isinstance(last["content"], str)


@pytest.mark.asyncio
async def test_fast_mode_reaches_a_model_that_supports_it(monkeypatch):
    monkeypatch.setenv("RUNE_FAST_MODE", "1")
    captured = _capture_kwargs(monkeypatch)
    await _drain(_make_result(model="anthropic/claude-opus-5"))
    assert captured[0]["speed"] == "fast"


@pytest.mark.asyncio
async def test_fast_mode_is_withheld_from_a_model_that_does_not(monkeypatch):
    # Measured: haiku answers `speed` with a 400 and the run makes no
    # progress at all, while finishing fast enough to look like a win.
    monkeypatch.setenv("RUNE_FAST_MODE", "1")
    captured = _capture_kwargs(monkeypatch)
    await _drain(_make_result(model="anthropic/claude-haiku-4-5"))
    assert "speed" not in captured[0]


@pytest.mark.asyncio
async def test_fast_mode_default_off(monkeypatch):
    monkeypatch.delenv("RUNE_FAST_MODE", raising=False)
    captured = _capture_kwargs(monkeypatch)
    await _drain(_make_result())
    assert "speed" not in captured[0]


@pytest.mark.asyncio
async def test_fast_mode_ignored_for_non_anthropic(monkeypatch):
    monkeypatch.setenv("RUNE_FAST_MODE", "1")
    captured = _capture_kwargs(monkeypatch)
    await _drain(_make_result(model="openai/gpt-5.4"))
    assert "speed" not in captured[0]
