"""Refusing is a way to finish, and it finishes as a failure.

Two measured shapes motivate this: a test that contradicts the spec it
guards (blocking the test edit only pushed the agent into bending the
source instead), and a request whose premise is wrong (obeying it broke a
documented contract). In both, the correct outcome was to stop and say
what conflicts — and there was no way to express that, so the run always
ended shaped like success.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import rune.agent.litellm_adapter as la
from rune.agent.litellm_adapter import StreamResult
from rune.capabilities.blocked import (
    BLOCKED_MARKER,
    TaskBlockedParams,
    consume_block,
    task_blocked,
)


class TestCapability:
    @pytest.mark.asyncio
    async def test_reports_the_conflict_and_records_it(self):
        consume_block()  # clear anything left by another test
        res = await task_blocked(TaskBlockedParams(
            reason="test_fees expects 8000 for 3 weeks; docs/fees.md says 6000",
            evidence="tests/test_fees.py:18",
        ))
        assert res.success
        assert BLOCKED_MARKER in res.output
        assert "docs/fees.md" in res.output
        assert "tests/test_fees.py:18" in res.output
        assert consume_block().startswith("test_fees expects")

    @pytest.mark.asyncio
    async def test_a_reason_is_required(self):
        consume_block()
        res = await task_blocked(TaskBlockedParams(reason="   "))
        assert not res.success
        assert consume_block() == ""

    @pytest.mark.asyncio
    async def test_the_signal_is_consumed_once(self):
        await task_blocked(TaskBlockedParams(reason="conflict"))
        assert consume_block() == "conflict"
        assert consume_block() == ""

    def test_it_is_registered_as_a_tool(self):
        from rune.capabilities.registry import get_capability_registry
        assert get_capability_registry().get("task_blocked") is not None

    def test_the_outcome_has_an_honest_note(self):
        from rune.agent.escalation import honest_failure_note
        note = honest_failure_note("task_blocked")
        assert note and "cannot be completed" in note


def _delta(*, content=None, tool_calls=None, finish_reason=None):
    tcs = None
    if tool_calls:
        tcs = [SimpleNamespace(
            index=t["index"], id=t.get("id", "tc0"),
            function=SimpleNamespace(name=t.get("name"),
                                     arguments=t.get("arguments")))
            for t in tool_calls]
    return SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=content, tool_calls=tcs),
            finish_reason=finish_reason)],
        usage=None)


async def _astream(chunks):
    for c in chunks:
        yield c


def _fake(streams):
    it = iter(streams)

    async def fake(**kwargs):
        return _astream(next(it))
    return fake


@pytest.mark.asyncio
async def test_abstaining_ends_the_run_immediately(monkeypatch, tmp_path):
    """Nothing follows a refusal — no further tool rounds, and the refusal
    itself is what the run returns."""
    monkeypatch.chdir(tmp_path)
    consume_block()
    later = []

    async def edit(**kw):
        later.append(kw)
        return "edited"

    async def blocked_tool(**kw):
        # the registry wraps capabilities so the model's flat arguments
        # become the params model; mirror that here
        return await task_blocked(TaskBlockedParams(**kw))

    res = StreamResult(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "테스트 통과하게 만들어줘"}],
        tool_schemas=[{"function": {"name": "task_blocked"}},
                      {"function": {"name": "file_edit"}}],
        tool_lookup={"task_blocked": blocked_tool, "file_edit": edit},
        max_tokens=4096, temperature=0.0,
        request_tokens_limit=200000, response_tokens_limit=4096,
        max_tool_rounds=8,
    )
    block_turn = [
        _delta(tool_calls=[{"index": 0, "name": "task_blocked",
                            "arguments": '{"reason": "the test contradicts '
                                         'docs/fees.md"}'}]),
        _delta(finish_reason="tool_calls"),
    ]
    edit_turn = [
        _delta(tool_calls=[{"index": 0, "name": "file_edit",
                            "arguments": '{"path": "a.py", "search": "x", '
                                         '"replace": "y"}'}]),
        _delta(finish_reason="tool_calls"),
    ]
    monkeypatch.setattr(la.litellm, "acompletion",
                        _fake([block_turn, edit_turn, edit_turn]))

    async for _ in res.stream_text():
        pass

    assert later == [], "the run kept editing after refusing"
    assert BLOCKED_MARKER in await res.get_output()
    assert consume_block()
