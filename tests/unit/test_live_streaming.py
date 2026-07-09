"""Tests for live prose streaming in StreamResult.stream_text.

Native-tool-call cloud models stream content deltas live (answer types out);
local/ollama and guided models stay buffered (their content is tool-call JSON).
When a tool call appears after some prose was streamed, the streamed text is
rolled back so get_output() / collected_text stay correct.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from rune.agent.litellm_adapter import StreamResult

_TOOLS = [{"type": "function",
           "function": {"name": "noop", "parameters": {"type": "object"}}}]


def _chunk(content=None, tool_calls=None, finish=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=content, tool_calls=tool_calls),
            finish_reason=finish,
        )],
        usage=None,
    )


def _usage():
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _tc(name="noop", args="{}"):
    return [SimpleNamespace(index=0, id="call_1",
                            function=SimpleNamespace(name=name, arguments=args))]


def _fake_acompletion(scripts):
    """Return an acompletion that yields scripts[i] on the i-th call."""
    state = {"i": 0}

    async def acompletion(**_):
        chunks = scripts[min(state["i"], len(scripts) - 1)]
        state["i"] += 1

        async def gen():
            for c in chunks:
                yield c
        return gen()

    return acompletion


def _make(model, *, tools=_TOOLS, provider_extra=None):
    return StreamResult(
        model=model,
        messages=[{"role": "system", "content": "s"},
                  {"role": "user", "content": "hi"}],
        tool_schemas=tools,
        tool_lookup={},
        max_tokens=200,
        temperature=0.0,
        request_tokens_limit=100_000,
        response_tokens_limit=200,
        provider_extra=provider_extra,
    )


async def _drive(sr) -> list[str]:
    out = []
    with patch.object(StreamResult, "_execute_tool_batch", new=AsyncMock()):
        async for d in sr.stream_text(delta=True):
            out.append(d)
    return out


class TestLiveStreaming:
    async def test_cloud_model_streams_prose_live(self):
        sr = _make("claude-sonnet-4-5-20250929")
        scripts = [[_chunk("Hello "), _chunk("world"), _chunk("!", finish="stop"),
                    _usage()]]
        with patch("litellm.acompletion", new=_fake_acompletion(scripts)):
            deltas = await _drive(sr)
        # Each content delta arrives as its own yield -> live typing.
        assert deltas == ["Hello ", "world", "!"]
        assert await sr.get_output() == "Hello world!"

    async def test_ollama_model_buffers(self):
        sr = _make("openai/qwen2.5-coder", tools=[],
                   provider_extra={"api_base": "http://localhost:11434/v1"})
        scripts = [[_chunk("Hello "), _chunk("world!", finish="stop"), _usage()]]
        with patch("litellm.acompletion", new=_fake_acompletion(scripts)):
            deltas = await _drive(sr)
        # Buffered: the whole answer is emitted once at end of stream.
        assert deltas == ["Hello world!"]
        assert await sr.get_output() == "Hello world!"

    async def test_prose_before_tool_call_is_rolled_back(self):
        sr = _make("claude-sonnet-4-5-20250929")
        scripts = [
            # Turn 1: prose, then a tool call -> prose is speculative.
            [_chunk("let me check "), _chunk(None, tool_calls=_tc()),
             _chunk(None, finish="tool_calls"), _usage()],
            # Turn 2: the real answer.
            [_chunk("Done.", finish="stop"), _usage()],
        ]
        with patch("litellm.acompletion", new=_fake_acompletion(scripts)):
            deltas = await _drive(sr)
        # The speculative prose was streamed live (terminal can't un-draw)...
        assert "let me check " in deltas
        # ...but it is rolled back, so the committed output is only the answer.
        assert await sr.get_output() == "Done."
