"""Tool-call truncation recovery in StreamResult.stream_text.

When a turn hits the output limit mid tool-call arguments (finish_reason
"length" with unparseable arguments), the adapter must NOT execute the broken
call; it must raise both output caps and re-prompt to retry. Verified by driving
the streaming loop with a faked litellm.acompletion.
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
                id=tc.get("id", "tc1"),
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


@pytest.mark.asyncio
async def test_truncated_tool_call_recovers_instead_of_executing(monkeypatch):
    executed: list[str] = []

    async def never_run(**_):
        executed.append("file_write")
        return "wrote"

    # Turn 1: a file_write whose arguments are cut off (invalid JSON) with
    # finish_reason "length". Turn 2: a clean final text answer.
    turn1 = [
        _delta_chunk(tool_calls=[{
            "index": 0, "name": "file_write",
            "arguments": '{"path": "/tmp/a.md"',  # truncated, unparseable
        }]),
        _delta_chunk(finish_reason="length"),
    ]
    turn2 = [_delta_chunk(content="done"), _delta_chunk(finish_reason="stop")]
    streams = iter([turn1, turn2])

    async def fake_acompletion(**_):
        return _astream(next(streams))

    monkeypatch.setattr(la.litellm, "acompletion", fake_acompletion)

    result = StreamResult(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "write a long file"}],
        tool_schemas=[{"function": {"name": "file_write"}}],
        tool_lookup={"file_write": never_run},
        max_tokens=8192,
        temperature=0.0,
        request_tokens_limit=200000,
        response_tokens_limit=8192,
    )

    out = "".join([t async for t in result.stream_text()])

    # The broken file_write was never executed, and the caps were raised.
    assert executed == []
    assert result._response_tokens_limit > 8192
    assert result._max_tokens > 8192
    assert "done" in out
    # A retry nudge was injected.
    assert any(
        m.get("role") == "user" and "cut off by the output limit" in m.get("content", "")
        for m in result.all_messages()
    )
