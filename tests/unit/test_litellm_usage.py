from __future__ import annotations

from types import SimpleNamespace

from rune.agent.litellm_adapter import StreamResult


def _stream_result() -> StreamResult:
    return StreamResult(
        model="gpt-5.4",
        messages=[],
        tool_schemas=[],
        tool_lookup={},
        max_tokens=1024,
        temperature=0.0,
        request_tokens_limit=1000,
        response_tokens_limit=1000,
    )


def test_stream_usage_extracts_openai_breakdown():
    result = _stream_result()

    result._update_usage(
        SimpleNamespace(
            prompt_tokens=1200,
            completion_tokens=300,
            prompt_tokens_details=SimpleNamespace(cached_tokens=200),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=75),
        )
    )

    usage = result.usage()

    assert usage.input_tokens == 1200
    assert usage.output_tokens == 300
    assert usage.cached_input_tokens == 200
    assert usage.reasoning_tokens == 75


def test_stream_usage_extracts_anthropic_cache_write_tokens():
    result = _stream_result()

    result._update_usage(
        {
            "input_tokens": 800,
            "output_tokens": 120,
            "cache_creation_input_tokens": 50,
            "cache_read_input_tokens": 300,
        }
    )

    usage = result.usage()

    assert usage.input_tokens == 800
    assert usage.output_tokens == 120
    assert usage.cached_input_tokens == 300
    assert usage.cache_write_tokens == 50
