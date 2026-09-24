import json
from unittest.mock import AsyncMock

from rune.agent.tool_catalog import ToolCatalog


async def test_discovery_keeps_core_stable_and_loads_exact_tools_once():
    schemas = [{"type": "function", "function": {"name": name, "description": name + " " + "x" * 1500,
                "parameters": {"type": "object", "properties": {}}}}
               for name in ["file_read", "bash_execute", "table_verify", *[f"service_{n}" for n in range(30)]]]
    assert ToolCatalog.needed(schemas)
    catalog = ToolCatalog(schemas)
    original = json.dumps(catalog.schemas)
    prefix = list(catalog.schemas)
    assert "file_read" in catalog.loaded and "service_12" not in catalog.loaded
    assert len(original) < len(json.dumps(schemas)) / 4
    found = json.loads(await catalog.search("service_12"))
    assert [tool["name"] for tool in found["tools"]] == ["service_12"]
    assert catalog.schemas[:len(prefix)] == prefix
    loaded = json.dumps(catalog.schemas)
    await catalog.search("service_12")
    assert json.dumps(catalog.schemas) == loaded
    assert not ToolCatalog.needed(schemas[:3])


async def test_stream_discovers_a_tool_before_dispatch(monkeypatch):
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.tool_adapter import ToolWrapper
    from tests.unit.test_live_streaming import _chunk, _tc

    export = AsyncMock(return_value="exported")
    tools = [ToolWrapper(name=f"office_export_{n}", description="Export document " + "x" * 1600,
                         function=export) for n in range(30)]
    seen = []

    async def complete(**params):
        seen.append([tool["function"]["name"] for tool in params["tools"]])
        async def chunks():
            if len(seen) == 1:
                yield _chunk(tool_calls=_tc("tool_search", '{"query":"office_export_7"}'), finish="tool_calls")
            elif len(seen) == 2:
                yield _chunk(tool_calls=_tc("office_export_7"), finish="tool_calls")
            else:
                yield _chunk(content="Export completed.", finish="stop")
        return chunks()

    monkeypatch.setattr("rune.agent.litellm_adapter.litellm.acompletion", complete)
    agent = LiteLLMAgent("openai/gpt-5.4", tools=tools)
    async with agent.run_stream("Export a document") as stream:
        assert "not loaded" in await stream._execute_tool("office_export_7", {})
        export.assert_not_awaited()
        assert "".join([part async for part in stream.stream_text()]) == "Export completed."
    assert "office_export_7" not in seen[0] and "office_export_7" in seen[1]
    assert seen[1] == seen[2]
    export.assert_awaited_once()
    agent.update_tools([tools[7]])
    assert agent._tool_catalog is None
    assert [tool["function"]["name"] for tool in agent._tool_schemas] == ["office_export_7"]
    assert set(agent._tool_lookup) == {"office_export_7"}
