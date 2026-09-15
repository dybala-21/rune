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


@pytest.mark.asyncio
async def test_pending_verification_requires_tools_then_releases_final_answer(monkeypatch):
    from rune.agent.verification_state import VerificationState

    state = VerificationState()
    state.changed()
    requests = []
    command = "python3 -m unittest -v"

    async def verify(**_):
        output = "test_ok (example.Cases) ... ok\nRan 1 test in 0.001s\nOK"
        state.observe_command(command, True, output, "/fixture")
        return output

    streams = iter([
        [_delta_chunk(tool_calls=[{"index": 0, "name": "bash_execute", "arguments": "{}"}]),
         _delta_chunk(finish_reason="tool_calls")],
        [_delta_chunk(content="The check passed."), _delta_chunk(finish_reason="stop")],
    ])

    async def fake_completion(**kwargs):
        requests.append(kwargs)
        return _astream(next(streams))

    monkeypatch.setattr(la.litellm, "acompletion", fake_completion)
    result = StreamResult(
        model="claude-sonnet-4-5", messages=[{"role": "user", "content": "Verify the change"}],
        tool_schemas=[{"function": {"name": "bash_execute"}}], tool_lookup={"bash_execute": verify},
        max_tokens=1024, temperature=0.0, request_tokens_limit=200000, response_tokens_limit=8192,
        verification_state=lambda: state,
    )
    assert "The check passed." in "".join([part async for part in result.stream_text()])
    assert len(requests) == 2
    assert requests[0]["tool_choice"] == "required"
    assert requests[1].get("tool_choice") != "required"
    ledger = [m for m in result.all_messages() if m.get("content", "").startswith("[Recorded verification evidence]")]
    assert len(ledger) == 2 and '"pending": false' in ledger[-1]["content"]
    assert all(message["role"] == "user" for message in ledger)
    assert requests[1]["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}

    from copy import deepcopy

    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    # Actual provider conversion must leave changing evidence out of the system prefix.
    converter = LiteLLMResponsesTransformationHandler()
    instructions = [converter.convert_chat_completion_messages_to_responses_api(deepcopy(r["messages"]))[1] for r in requests]
    assert instructions[0] == instructions[1]
    assert AnthropicConfig().translate_system_message(deepcopy(requests[1]["messages"])) == []
