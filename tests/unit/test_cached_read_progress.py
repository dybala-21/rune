"""Unchanged cached reads must not consume the entire execution budget."""

import json
from types import SimpleNamespace

import pytest

from rune.agent.litellm_adapter import StreamResult
from rune.agent.tool_output import CachedToolResult


async def run_reads(monkeypatch, actions, results):
    import rune.agent.litellm_adapter as adapter

    monkeypatch.setenv("RUNE_ARTIFACT_PROVENANCE", "0")
    requests = []
    turns = iter(actions)
    reads = iter(results)

    async def read(**params):
        return next(reads)

    async def fresh(**params):
        return "A new observation"

    async def completion(**kwargs):
        requests.append(kwargs)
        name = next(turns)
        calls = [SimpleNamespace(index=0, id=f"call_{len(requests)}",
                 function=SimpleNamespace(name=name, arguments=json.dumps({})))] if name else None

        async def chunks():
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None if name else "Finished", tool_calls=calls),
                finish_reason="tool_calls" if name else "stop")], usage=None)
        return chunks()

    monkeypatch.setattr(adapter.litellm, "acompletion", completion)
    stream = StreamResult(model="gemini/gemini-2.5-flash", messages=[{"role": "user", "content": "Inspect the data"}],
                          tool_schemas=[{"type": "function", "function": {"name": name, "parameters": {
                              "type": "object", "properties": {}}}} for name in ("file_read", "file_list")],
                          tool_lookup={"file_read": read, "file_list": fresh}, max_tokens=4096,
                          temperature=0, request_tokens_limit=200000, response_tokens_limit=4096,
                          max_tool_rounds=10, require_verification=False)
    async for _ in stream.stream_text():
        pass
    return stream, [{t["function"]["name"] for t in r.get("tools") or []} for r in requests]


async def test_repeated_cached_read_is_hidden_then_stopped_if_model_ignores_tools(monkeypatch):
    stream, tools = await run_reads(monkeypatch, ["file_read"] * 6,
                                   [CachedToolResult("same data", "generation:1")] * 6)
    assert "file_read" in tools[0] and "file_read" not in tools[2]
    assert len(tools) == 4
    assert "unchanged cached results" in stream._task_blocked


async def test_fresh_observation_restores_read_tools(monkeypatch):
    stream, tools = await run_reads(monkeypatch, ["file_read", "file_read", "file_list", "file_read", None],
                                   [CachedToolResult("same data", "generation:1")] * 3)
    assert "file_read" not in tools[2] and "file_read" in tools[3]
    assert not stream._task_blocked
    assert await stream.get_output() == "Finished"


@pytest.mark.parametrize("keys", [["file:a", "file:b", "file:c"], ["gen:1", "gen:2", "gen:3"]])
async def test_distinct_cached_reads_remain_available(monkeypatch, keys):
    stream, tools = await run_reads(monkeypatch, ["file_read"] * 3 + [None],
                                   [CachedToolResult("data", key) for key in keys])
    assert all("file_read" in names for names in tools)
    assert not stream._task_blocked


async def test_failed_required_extraction_stops_instead_of_looping_on_blocked_tool(monkeypatch):
    monkeypatch.setenv("RUNE_ARTIFACT_PROVENANCE", "0")
    calls = []

    async def requirements(**params):
        calls.append(params)
        return "[ERROR] Requirement extraction failed"

    stream = StreamResult(model="gemini/gemini-2.5-flash", messages=[], tool_schemas=[],
                          tool_lookup={"table_requirements": requirements}, max_tokens=1000,
                          temperature=0, request_tokens_limit=20000, response_tokens_limit=1000,
                          tool_recovery=lambda: {"table_requirements", "ask_user"})
    for _ in range(4):
        await stream._execute_tool("table_requirements", {"source_path": "source.csv"})
    assert len(calls) == 3
    assert "could not be verified" in stream._task_blocked
