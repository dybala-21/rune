from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

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


def _tool_call(call_id: str, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": call_id,
        "function": {
            "name": name,
            "arguments": json.dumps(args or {}),
        },
    }


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

    assert usage.input_tokens == 1150
    assert usage.output_tokens == 120
    assert usage.cached_input_tokens == 300
    assert usage.cache_write_tokens == 50


@pytest.mark.parametrize("written", [0, 300])
@pytest.mark.parametrize("as_object", [False, True])
def test_responses_cache_writes_survive_litellm_conversion(written, as_object):
    from litellm.responses.utils import ResponseAPILoggingUtils
    from openai.types.responses import ResponseUsage

    from rune.llm.usage import preserve_responses_cache_usage, token_counts

    native = {"input_tokens": 1000, "output_tokens": 20, "total_tokens": 1020,
              "input_tokens_details": {"cached_tokens": 600, "cache_write_tokens": written},
              "output_tokens_details": {"reasoning_tokens": 10}}
    preserve_responses_cache_usage()
    original = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage
    preserve_responses_cache_usage()
    assert original is ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage
    normalized = original(ResponseUsage(**native) if as_object else native)
    counts = token_counts(normalized)
    assert counts["cache_write_tokens"] == written and counts["cache_write_reported"]
    assert counts["total_tokens"] == 1020 and counts["reasoning_tokens"] == 10
    result = _stream_result()
    result._update_usage(normalized)
    assert result.usage().cache_write_tokens == written


def test_usage_distinguishes_native_anthropic_input_and_missing_cache_fields():
    from rune.llm.usage import token_counts

    counts = token_counts({"input_tokens": 100, "output_tokens": 20,
                           "cache_read_input_tokens": 600, "cache_creation_input_tokens": 300})
    assert counts["input_tokens"] == 1000 and counts["total_tokens"] == 1020
    counts = token_counts({"prompt_tokens": 100, "completion_tokens": 20})
    assert not counts["cache_write_reported"]


def test_stream_usage_deduplicates_snapshots_but_counts_separate_rounds():
    result = _stream_result()
    result._update_usage({"prompt_tokens": 1000, "completion_tokens": 0})
    result._update_usage({"prompt_tokens": 0, "completion_tokens": 50})
    result._update_usage({"prompt_tokens": 1000, "completion_tokens": 50})
    assert result.usage().input_tokens == 1000 and result.usage().output_tokens == 50
    result._round_usage = None
    result._update_usage({"prompt_tokens": 1000, "completion_tokens": 50})
    assert result.usage().input_tokens == 2000 and result.usage().output_tokens == 100


@pytest.mark.asyncio
async def test_batch_failure_skips_remaining_write_tools(monkeypatch):
    monkeypatch.delenv("RUNE_BENCH_STOP_BATCH_ON_TOOL_FAILURE", raising=False)
    calls: list[str] = []

    async def fail_bash(**_: object) -> str:
        calls.append("bash_execute")
        return "[cmd: false] [exit: 1]\n[ERROR] failed smoke"

    async def write_file(**_: object) -> str:
        calls.append("file_write")
        return "wrote file"

    result = StreamResult(
        model="gpt-5.4",
        messages=[],
        tool_schemas=[],
        tool_lookup={"bash_execute": fail_bash, "file_write": write_file},
        max_tokens=1024,
        temperature=0.0,
        request_tokens_limit=1000,
        response_tokens_limit=1000,
    )

    await result._execute_tool_batch(
        [
            _tool_call("call_1", "bash_execute"),
            _tool_call("call_2", "file_write"),
            _tool_call("call_3", "bash_execute"),
        ]
    )

    tool_messages = [m for m in result.all_messages() if m.get("role") == "tool"]
    assert calls == ["bash_execute"]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_1", "call_2", "call_3"]
    assert "failed smoke" in tool_messages[0]["content"]
    assert tool_messages[1]["content"].startswith("[BLOCKED] Skipped this tool")
    assert tool_messages[2]["content"].startswith("[BLOCKED] Skipped this tool")


@pytest.mark.asyncio
async def test_read_failure_blocks_later_actions_until_the_model_can_review_it(monkeypatch):
    monkeypatch.delenv("RUNE_BENCH_STOP_BATCH_ON_TOOL_FAILURE", raising=False)
    calls: list[str] = []

    async def fail_read(**_: object) -> str:
        calls.append("file_read")
        return "[ERROR] File too large"

    async def run_bash(**_: object) -> str:
        calls.append("bash_execute")
        return "fallback ok"

    result = StreamResult(
        model="gpt-5.4",
        messages=[],
        tool_schemas=[],
        tool_lookup={"file_read": fail_read, "bash_execute": run_bash},
        max_tokens=1024,
        temperature=0.0,
        request_tokens_limit=1000,
        response_tokens_limit=1000,
    )

    await result._execute_tool_batch(
        [
            _tool_call("call_1", "file_read"),
            _tool_call("call_2", "bash_execute"),
        ]
    )

    assert calls == ["file_read"]
    assert result.all_messages()[-1]["content"].startswith("[BLOCKED] Skipped this tool")


@pytest.mark.asyncio
async def test_benchmark_write_exec_cap_skips_after_limit(monkeypatch):
    # Cap = 2 successful write/execute calls per turn; the 3rd is skipped even
    # though nothing failed (the v6/v7 explosion guard).
    monkeypatch.setenv("RUNE_BENCH_MAX_WRITE_EXEC_PER_TURN", "2")
    calls: list[str] = []

    async def write_file(**_: object) -> str:
        calls.append("file_write")
        return "wrote file"

    async def run_bash(**_: object) -> str:
        calls.append("bash_execute")
        return "[cmd: echo ok] [exit: 0]\nok"

    result = StreamResult(
        model="gpt-5.4",
        messages=[],
        tool_schemas=[],
        tool_lookup={"file_write": write_file, "bash_execute": run_bash},
        max_tokens=1024,
        temperature=0.0,
        request_tokens_limit=1000,
        response_tokens_limit=1000,
    )

    await result._execute_tool_batch(
        [
            _tool_call("call_1", "file_write", {"path": "a"}),
            _tool_call("call_2", "bash_execute", {"command": "b"}),
            _tool_call("call_3", "file_write", {"path": "c"}),
            _tool_call("call_4", "bash_execute", {"command": "d"}),
        ]
    )

    tool_messages = [m for m in result.all_messages() if m.get("role") == "tool"]
    # Only the first two write/execute calls actually run.
    assert calls == ["file_write", "bash_execute"]
    # Every tool_call_id is still answered (OpenAI protocol).
    assert [m["tool_call_id"] for m in tool_messages] == [
        "call_1",
        "call_2",
        "call_3",
        "call_4",
    ]
    assert tool_messages[2]["content"].startswith("[BLOCKED] Skipped this tool")
    assert "per-turn write/execute limit (2)" in tool_messages[2]["content"]
    assert tool_messages[3]["content"].startswith("[BLOCKED] Skipped this tool")


@pytest.mark.asyncio
async def test_benchmark_write_exec_cap_off_by_default(monkeypatch):
    # Unset → no cap; all write/execute calls run.
    monkeypatch.delenv("RUNE_BENCH_MAX_WRITE_EXEC_PER_TURN", raising=False)
    monkeypatch.delenv("RUNE_BENCH_STOP_BATCH_ON_TOOL_FAILURE", raising=False)
    calls: list[str] = []

    async def write_file(**_: object) -> str:
        calls.append("file_write")
        return "wrote file"

    result = StreamResult(
        model="gpt-5.4",
        messages=[],
        tool_schemas=[],
        tool_lookup={"file_write": write_file},
        max_tokens=1024,
        temperature=0.0,
        request_tokens_limit=1000,
        response_tokens_limit=1000,
    )

    await result._execute_tool_batch(
        [
            _tool_call("call_1", "file_write", {"path": "a"}),
            _tool_call("call_2", "file_write", {"path": "b"}),
            _tool_call("call_3", "file_write", {"path": "c"}),
        ]
    )

    assert calls == ["file_write", "file_write", "file_write"]
    assert len([m for m in result.all_messages() if m.get("role") == "tool"]) == 3


@pytest.mark.parametrize("method", ["POST", "post", "DELETE"])
async def test_network_writes_wait_for_prior_read_and_failure_stops_next_action(method):
    import asyncio

    calls = []
    async def fetch(**params):
        if params.get("method", "GET") == "GET":
            calls.append("read_started")
            await asyncio.sleep(.01)
            calls.append("read_finished")
            return "source"
        assert calls[-1] == "read_finished"
        calls.append("write_failed")
        return "[ERROR] rejected"

    async def write(**params):
        calls.append("unexpected_write")
        return "saved"

    result = _stream_result()
    result._tool_lookup = {"web_fetch": fetch, "file_write": write}
    await result._execute_tool_batch([
        _tool_call("read", "web_fetch", {"url": "https://example.test/source"}),
        _tool_call("submit", "web_fetch", {"url": "https://example.test/submit", "method": method}),
        _tool_call("later", "file_write", {"path": "receipt.txt", "content": "sent"}),
    ])
    assert calls == ["read_started", "read_finished", "write_failed"]
    assert [m["tool_call_id"] for m in result.all_messages() if m["role"] == "tool"] == ["read", "submit", "later"]
    assert result.all_messages()[-1]["content"].startswith("[BLOCKED] Skipped this tool")


@pytest.mark.parametrize("failed", [True, False])
async def test_parallel_read_failure_blocks_later_write_but_raw_error_text_does_not(failed):
    from rune.agent.tool_output import ReadToolResult

    calls = []
    async def read(**params):
        return "[ERROR] missing" if failed else ReadToolResult("[ERROR] example from source", evidence_key="source")
    async def listing(**params):
        return "source.txt"
    async def write(**params):
        calls.append("write")
        return "saved"
    result = _stream_result()
    result._tool_lookup = {"file_read": read, "file_list": listing, "file_write": write}
    await result._execute_tool_batch([
        _tool_call("read", "file_read"), _tool_call("list", "file_list"), _tool_call("write", "file_write"),
    ])
    assert calls == ([] if failed else ["write"])
    assert len([m for m in result.all_messages() if m["role"] == "tool"]) == 3


async def test_read_metadata_stays_internal_when_provider_copies_messages():
    from copy import deepcopy

    from rune.agent.tool_output import ReadToolResult

    async def read(**params):
        return ReadToolResult('source text', evidence_key='version-1', raw=True)
    result = _stream_result()
    result._tool_lookup = {'file_read': read}
    await result._execute_tool_batch([_tool_call('read', 'file_read')])
    messages = deepcopy(result.all_messages())
    assert type(messages[0]['content']) is str
    assert messages[0]['content'] == 'source text'
    assert result._cached_read_counts == {'version-1': 1}
