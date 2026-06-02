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

    assert usage.input_tokens == 800
    assert usage.output_tokens == 120
    assert usage.cached_input_tokens == 300
    assert usage.cache_write_tokens == 50


@pytest.mark.asyncio
async def test_benchmark_batch_stop_skips_remaining_write_tools(monkeypatch):
    monkeypatch.setenv("RUNE_BENCH_STOP_BATCH_ON_TOOL_FAILURE", "1")
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
async def test_benchmark_batch_stop_does_not_stop_after_read_only_failure(monkeypatch):
    monkeypatch.setenv("RUNE_BENCH_STOP_BATCH_ON_TOOL_FAILURE", "1")
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

    assert calls == ["file_read", "bash_execute"]
    assert result.all_messages()[-1]["content"] == "fallback ok"


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
