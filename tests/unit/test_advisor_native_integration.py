"""Phase A integration tests — verify LiteLLMAgent + native advisor wiring.

These tests mock ``litellm.acompletion`` and inspect the kwargs to
confirm:
- ``extra_headers`` (anthropic-beta) flows through when configured
- ``advisor_20260301`` schema reaches the API call as a passthrough
- Default path (no extra_headers, no advisor tool) is unchanged
- Synthetic events are extracted from usage and exposed via
  ``LiteLLMAgent.native_advisor_events()``
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rune.agent.advisor.native_tool import build_native_tool_wrapper
from rune.agent.litellm_adapter import LiteLLMAgent, tools_to_openai_schema


def _build_advisor_wrapper():
    from rune.agent.advisor.native_tool import NativeAdvisorConfig
    cfg = NativeAdvisorConfig(
        enabled=True, advisor_model="claude-opus-4-6", max_uses=3,
    )
    return build_native_tool_wrapper(cfg)


class TestSchemaPassthroughAtAdapterLevel:
    """The advisor_20260301 tool must reach litellm in raw form, not
    wrapped in OpenAI's function envelope."""

    def test_advisor_tool_passes_through_unchanged(self):
        wrapper = _build_advisor_wrapper()
        schemas = tools_to_openai_schema([wrapper])
        # Native tool should be the only entry, unwrapped
        assert len(schemas) == 1
        assert schemas[0]["type"] == "advisor_20260301"
        assert schemas[0]["name"] == "advisor"
        assert schemas[0]["model"] == "claude-opus-4-6"
        assert schemas[0]["max_uses"] == 3
        # NOT wrapped in {"type": "function", ...}
        assert "function" not in schemas[0]

    def test_normal_and_advisor_tools_coexist(self):
        wrapper = _build_advisor_wrapper()

        class _NormalTool:
            name = "file_read"
            description = "Read a file"
            json_schema = {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            }

        schemas = tools_to_openai_schema([_NormalTool(), wrapper])
        assert len(schemas) == 2
        # Normal tool is wrapped
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "file_read"
        # Native tool is passthrough
        assert schemas[1]["type"] == "advisor_20260301"


class TestExtraHeadersFlow:
    """LiteLLMAgent.extra_headers must propagate to litellm.acompletion."""

    @pytest.mark.asyncio
    async def test_extra_headers_passed_to_acompletion(self):
        wrapper = _build_advisor_wrapper()
        agent = LiteLLMAgent(
            model="anthropic:claude-haiku-4-5-20251001",
            system_prompt="test",
            tools=[wrapper],
            extra_headers={"anthropic-beta": "advisor-tool-2026-03-01"},
        )

        captured: dict = {}

        async def _fake_completion(**kwargs):
            captured.update(kwargs)
            # Yield one minimal chunk and a final usage chunk
            chunks = [
                SimpleNamespace(
                    choices=[SimpleNamespace(
                        delta=SimpleNamespace(
                            content="ok", tool_calls=None,
                        ),
                        finish_reason="stop",
                    )],
                    usage=None,
                ),
                SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(
                        prompt_tokens=10, completion_tokens=2,
                        iterations=[],
                    ),
                ),
            ]
            for c in chunks:
                yield c

        with patch("litellm.acompletion", side_effect=_fake_completion):
            async with agent.run_stream("hello") as stream:
                async for _ in stream.stream_text():
                    pass

        # extra_headers must have made it to acompletion verbatim
        assert "extra_headers" in captured
        assert captured["extra_headers"] == {
            "anthropic-beta": "advisor-tool-2026-03-01",
        }
        # Tool list should contain the passthrough advisor entry
        tools = captured.get("tools") or []
        native = [t for t in tools if t.get("type") == "advisor_20260301"]
        assert len(native) == 1

    @pytest.mark.asyncio
    async def test_default_agent_has_no_extra_headers(self):
        agent = LiteLLMAgent(
            model="openai:gpt-4o-mini",
            system_prompt="test",
            tools=[],
        )

        captured: dict = {}

        async def _fake_completion(**kwargs):
            captured.update(kwargs)
            chunks = [
                SimpleNamespace(
                    choices=[SimpleNamespace(
                        delta=SimpleNamespace(
                            content="ok", tool_calls=None,
                        ),
                        finish_reason="stop",
                    )],
                    usage=None,
                ),
                SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(
                        prompt_tokens=10, completion_tokens=2,
                    ),
                ),
            ]
            for c in chunks:
                yield c

        with patch("litellm.acompletion", side_effect=_fake_completion):
            async with agent.run_stream("hello") as stream:
                async for _ in stream.stream_text():
                    pass

        # Default path: no extra_headers key in the call
        assert "extra_headers" not in captured

        # No advisor_20260301 in the tool list
        tools = captured.get("tools") or []
        native = [t for t in (tools or []) if t.get("type") == "advisor_20260301"]
        assert len(native) == 0


class TestNativeAdvisorEventExtraction:
    """LiteLLMAgent.native_advisor_events() must reflect synthetic events
    parsed from usage.iterations[]."""

    @pytest.mark.asyncio
    async def test_events_extracted_when_beta_active(self):
        wrapper = _build_advisor_wrapper()
        agent = LiteLLMAgent(
            model="anthropic:claude-haiku-4-5-20251001",
            system_prompt="test",
            tools=[wrapper],
            extra_headers={"anthropic-beta": "advisor-tool-2026-03-01"},
        )

        async def _fake_completion(**kwargs):
            yield SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=2,
                    iterations=[
                        {"type": "message", "output_tokens": 100},
                        {
                            "type": "advisor_message",
                            "model": "claude-opus-4-6",
                            "output_tokens": 280,
                        },
                    ],
                ),
            )

        with patch("litellm.acompletion", side_effect=_fake_completion):
            async with agent.run_stream("hello") as stream:
                async for _ in stream.stream_text():
                    pass

        events = agent.native_advisor_events()
        assert len(events) == 1
        assert events[0]["trigger"] == "native"
        assert events[0]["model"] == "claude-opus-4-6"
        assert events[0]["output_tokens"] == 280

    @pytest.mark.asyncio
    async def test_no_events_when_beta_inactive(self):
        agent = LiteLLMAgent(
            model="openai:gpt-4o-mini",
            system_prompt="test",
            tools=[],
        )

        async def _fake_completion(**kwargs):
            yield SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=2,
                    iterations=[
                        {"type": "advisor_message", "output_tokens": 200},
                    ],
                ),
            )

        with patch("litellm.acompletion", side_effect=_fake_completion):
            async with agent.run_stream("hello") as stream:
                async for _ in stream.stream_text():
                    pass

        # Without the beta header active, the agent does NOT parse
        # iterations for synthetic events (avoids spurious data on
        # non-Anthropic providers)
        assert agent.native_advisor_events() == []
