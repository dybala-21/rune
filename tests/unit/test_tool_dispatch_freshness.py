"""Repeated calls must observe edits and reach the tool's execution guards."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rune.agent.cognitive_cache import SessionToolCache
from rune.agent.litellm_adapter import StreamResult
from rune.agent.tool_adapter import ToolAdapterOptions, build_tool_set
from rune.capabilities.bash import register_bash_capabilities
from rune.capabilities.file import get_guardian, register_file_capabilities
from rune.capabilities.registry import CapabilityRegistry


@pytest.fixture(params=["gpt-6-astra", "anthropic/claude-opus-5"])
def stream(request, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / ".rune"))
    monkeypatch.setattr(get_guardian(), "validate_file_path", lambda path: SimpleNamespace(
            allowed=Path(path).resolve().is_relative_to(tmp_path), reason="Outside test workspace",
            requires_approval=False,
    ))
    registry = CapabilityRegistry()
    register_file_capabilities(registry)
    register_bash_capabilities(registry)
    tools = build_tool_set(
        ToolAdapterOptions(enable_guardian=False, cognitive_cache=SessionToolCache()),
        registry=registry,
    )
    return StreamResult(
        model=request.param, messages=[], tool_schemas=[],
        tool_lookup={name: tool.function for name, tool in tools.items()},
        max_tokens=512, temperature=0, request_tokens_limit=10000, response_tokens_limit=512,
    )


async def test_same_test_command_runs_after_edit(stream, tmp_path):
    source = tmp_path / "check.py"
    source.write_text("assert False, 'original failure'\n")
    command = f'"{sys.executable}" -B check.py'
    before = await stream._execute_tool("bash_execute", {"command": command})
    assert "original failure" in before
    written = await stream._execute_tool("file_write", {"path": str(source), "content": "assert True\nprint('verified new code')\n"})
    assert "Written" in written, written
    after = await stream._execute_tool("bash_execute", {"command": command})
    assert "verified new code" in after and "original failure" not in after


async def test_file_read_observes_write_with_session_cache(stream, tmp_path):
    source = tmp_path / "note.txt"
    source.write_text("before edit")
    assert "before edit" in await stream._execute_tool("file_read", {"path": str(source)})
    for _ in range(2):
        assert "CACHE HIT" in await stream._execute_tool("file_read", {"path": str(source)})
    assert stream._stalled_read_tools == {"file_read"}
    written = await stream._execute_tool("file_write", {"path": str(source), "content": "after edit"})
    assert "Written" in written, written
    assert not stream._stalled_read_tools
    assert "after edit" in await stream._execute_tool("file_read", {"path": str(source)})


async def test_distinct_identical_command_calls_both_execute(stream, tmp_path):
    command = "printf 'record\\n' >> receipts.txt"
    for _ in range(2):
        await stream._execute_tool("bash_execute", {"command": command})
    assert (tmp_path / "receipts.txt").read_text().splitlines() == ["record", "record"]


async def test_cached_reads_close_ui_events_without_new_execution(tmp_path):
    from rune.agent.tool_adapter import StallState, _build_typed_tool
    from rune.agent.tool_output import CachedToolResult
    from rune.types import CapabilityResult

    path = str(tmp_path / "cached.txt")
    cache = SessionToolCache()
    params = {"path": path}
    cache.set(cache.generate_key("file_read", params), "file_read", params,
              CapabilityResult(success=True, output="cached content"), 1)
    events = []

    async def start(name, params):
        events.append(("start", name))

    async def end(name, result):
        events.append(("end", result))

    tool = _build_typed_tool(
        cap_def=SimpleNamespace(name="file_read", description="read", parameters_model=None, raw_json_schema=None),
        opts=ToolAdapterOptions(on_tool_start=start, on_tool_end=end),
        reg=CapabilityRegistry(), cache=cache, stall=StallState(),
    )
    for _ in range(2):
        result = await tool.function(**params)
        assert isinstance(result, CachedToolResult) and "CACHE HIT" in result
        assert result.cache_key == cache.generate_key("file_read", params)
    assert [event[0] for event in events] == ["start", "end", "start", "end"]
    assert all(result.metadata == {"cached": True} and result.success for kind, result in events if kind == "end")
