import json
from unittest.mock import AsyncMock

import pytest

from rune.agent.tool_catalog import ToolCatalog


@pytest.mark.parametrize("goal_type,intents,immediate", [
    ("full", {"document"}, {"file_read"}),
    ("code_modify", {"coding"}, {"file_read", "code_analyze", "project_map"}),
    ("artifact", {"document", "table"}, {"file_read", "table_requirements", "table_verify"}),
    ("full", set(), {"file_read"}),
])
async def test_task_guidance_defers_unrelated_schemas_without_removing_tools(goal_type, intents, immediate):
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.tool_catalog import deferred_for_task

    names = ["file_read", "code_analyze", "project_map", "table_requirements", "table_verify"]
    schemas = [{"type": "function", "function": {"name": name, "parameters": {"type": "object"}}} for name in names]
    original = json.dumps(schemas)
    classification = ClassificationResult(goal_type=goal_type, confidence=.9, tier=2, intent_categories=frozenset(intents))
    catalog = ToolCatalog(schemas, deferred_tools=deferred_for_task(names, classification))
    assert catalog.loaded == immediate | {"tool_search"}
    for name in set(names) - immediate:
        found = json.loads(await catalog.search(name))
        assert found["tools"][0]["name"] == name and name in catalog.loaded
    assert json.dumps(schemas) == original


def test_agent_keeps_schema_deferral_after_a_tool_update():
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.tool_adapter import ToolWrapper

    tools = [ToolWrapper(name=name, description="x" * 1600, function=AsyncMock())
             for name in ["file_read", "code_analyze", *[f"service_{n}" for n in range(30)]]]
    agent = LiteLLMAgent("openai/gpt-5.4", tools=tools, deferred_tools={"code_analyze"})
    agent.update_tools(tools[:-1])
    assert "file_read" in agent._tool_catalog.loaded
    assert "code_analyze" not in agent._tool_catalog.loaded
    assert "code_analyze" in agent._tool_catalog.all


async def test_small_lookup_catalog_keeps_rendering_tools_available_on_demand():
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.tool_adapter import ToolWrapper
    from rune.agent.tool_catalog import deferred_for_task

    names = ["web_search", "web_fetch", "task_blocked", "file_read", "browser_observe", "browser_navigate"]
    tools = [ToolWrapper(name=name, description="x" * 1000, function=AsyncMock()) for name in names]
    classification = ClassificationResult(goal_type="web", confidence=.99, tier=2, needs_web_evidence=True, lookup_only=True)
    deferred = deferred_for_task(names, classification)
    full = LiteLLMAgent("openai/gpt-5.4", tools=tools, deferred_tools=deferred)
    agent = LiteLLMAgent("openai/gpt-5.4", tools=tools, deferred_tools=deferred, compact_tools=True)
    assert agent._tool_catalog.loaded == {"web_search", "web_fetch", "task_blocked", "tool_search"}
    assert len(json.dumps(agent._tool_schemas)) < len(json.dumps(full._tool_schemas))
    await agent._tool_catalog.search("browser_observe")
    assert "browser_observe" in agent._tool_catalog.loaded
    assert set(agent._tool_lookup) == set(names) | {"tool_search"}


@pytest.mark.parametrize("compact,count", [(False, 1), (True, 1), (False, 30)])
async def test_native_tools_pass_through_with_or_without_discovery(compact, count):
    from rune.agent.advisor.native_tool import NativeAdvisorConfig, build_native_tool_wrapper
    from rune.agent.litellm_adapter import LiteLLMAgent, tools_to_openai_schema
    from rune.agent.tool_adapter import ToolWrapper

    advisor = build_native_tool_wrapper(NativeAdvisorConfig(enabled=True, advisor_model="claude-opus-4-6", max_uses=3))
    native_schema = tools_to_openai_schema([advisor])[0]
    tools = [ToolWrapper(name=name, description="x" * 1600, function=AsyncMock())
             for name in ["web_fetch", *[f"service_{n}" for n in range(count)]]]
    agent = LiteLLMAgent("anthropic/claude-sonnet-4-6", tools=[*tools, advisor],
                        deferred_tools={"service_0", "advisor"}, compact_tools=compact)
    assert [tool for tool in agent._tool_schemas if tool.get("type") != "function"] == [native_schema]
    if compact or count > 24:
        assert "advisor" not in agent._tool_catalog.all
        assert "service_0" not in agent._tool_catalog.loaded
        await agent._tool_catalog.search("service_0")
        assert "service_0" in agent._tool_catalog.loaded
        assert agent._tool_schemas.count(native_schema) == 1
    else:
        assert agent._tool_catalog is None


async def test_browser_guidance_loads_with_browser_tools_and_returns_when_discovery_is_disabled():
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.prompts import PROMPT_BROWSER
    from rune.agent.tool_adapter import ToolWrapper

    tools = [ToolWrapper(name=name, description="x" * 1600, function=AsyncMock())
             for name in ["file_read", "browser_act", "browser_observe", *[f"service_{n}" for n in range(30)]]]
    prompt = "Core instructions\n" + PROMPT_BROWSER + "\nCurrent task"
    agent = LiteLLMAgent("openai/gpt-5.4", system_prompt=prompt, tools=tools)
    assert PROMPT_BROWSER not in agent._system_prompt
    assert "Core instructions" in agent._system_prompt and "Current task" in agent._system_prompt
    assert "guidance" not in json.loads(await agent._tool_catalog.search("service_1"))
    messages = [{"role": "system", "content": agent._system_prompt}, {"role": "user", "content": "Check the page"}]
    assert agent._tool_catalog.with_guidance(messages) is messages
    await agent._tool_catalog.search("browser_act")
    guided = agent._tool_catalog.with_guidance(messages)
    assert PROMPT_BROWSER in guided[0]["content"] and guided[1] == messages[1]
    assert PROMPT_BROWSER not in messages[0]["content"]
    assert agent._tool_catalog.with_guidance(guided) is guided
    assert "browser_act" in agent._tool_catalog.loaded
    assert "guidance" not in json.loads(await agent._tool_catalog.search("browser_act"))
    assert "guidance" not in json.loads(await agent._tool_catalog.search("browser_observe"))
    agent.update_tools(tools[:2])
    assert agent._tool_catalog is None and agent._system_prompt == prompt


async def test_stream_restores_browser_guidance_before_requesting_recovery_tools(monkeypatch):
    from rune.agent.prompts import PROMPT_BROWSER
    from tests.unit.test_live_streaming import _chunk, _make, _usage

    requests = []
    async def complete(**params):
        requests.append(params)
        async def chunks():
            yield _chunk(content="Ready.", finish="stop")
            yield _usage()
        return chunks()
    monkeypatch.setattr("litellm.acompletion", complete)
    catalog = ToolCatalog([{"type": "function", "function": {"name": "browser_observe",
                           "parameters": {"type": "object", "properties": {}}}}])
    stream = _make("openai/gpt-5.4")
    stream._tool_catalog = catalog
    stream._tool_schemas = catalog.schemas
    stream._tool_recovery = lambda: {"browser_observe"}
    stream._require_verification = False
    async for _ in stream.stream_text():
        pass
    assert requests
    assert all(any(m.get("role") == "system" and PROMPT_BROWSER in m.get("content", "")
                   for m in request["messages"]) for request in requests)


async def test_browser_family_includes_input_and_keeps_step_schema_on_vertex():
    import pytest
    from jsonschema import ValidationError, validate
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

    from rune.capabilities.browser.capabilities import register_browser_capabilities
    from rune.capabilities.browser.extended import BrowserBatchParams
    from rune.capabilities.registry import CapabilityRegistry

    registry = CapabilityRegistry()
    register_browser_capabilities(registry)
    schemas = [{"type": "function", "function": {"name": cap.name, "description": cap.description,
                "parameters": cap.parameters_model.model_json_schema()}} for cap in registry.list_all()]
    catalog = ToolCatalog(schemas)
    found = json.loads(await catalog.search("browser"))
    assert {"browser_act", "browser_observe", "browser_batch"} <= {item["name"] for item in found["tools"]}
    original = BrowserBatchParams.model_json_schema()
    converted = VertexGeminiConfig()._map_function([{"type": "function", "function": {
        "name": "browser_batch", "parameters": original,
    }}], {})[0]["function_declarations"][0]["parameters"]
    for schema in (original, converted):
        validate({"actions": [{"type": "act", "params": {"action": "click", "selector": "#calculate"}}]}, schema)
        for invalid in [{"type": "click", "ref": "#calculate"}, {"type": "act", "name": "click", "ref": "#calculate"},
                        {"type": "act", "params": {"selector": "#calculate"}}]:
            with pytest.raises(ValidationError):
                validate({"actions": [invalid]}, schema)
    rejected = await registry.execute("browser_batch", {"actions": [
        {"action": "click", "selector": "#calculate"},
    ]})
    assert not rejected.success and rejected.metadata["action_status"] == "not_executed"
    assert len(rejected.error) < 1200 and "type" in rejected.error


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
        from rune.capabilities.output_prefixes import looks_like_failure_output
        rejected = await stream._execute_tool("office_export_7", {})
        assert "not loaded" in rejected and looks_like_failure_output(rejected)
        export.assert_not_awaited()
        assert "".join([part async for part in stream.stream_text()]) == "Export completed."
    assert "office_export_7" not in seen[0] and "office_export_7" in seen[1]
    assert seen[1] == seen[2]
    export.assert_awaited_once()
    agent.update_tools([tools[7]])
    assert agent._tool_catalog is None
    assert [tool["function"]["name"] for tool in agent._tool_schemas] == ["office_export_7"]
    assert set(agent._tool_lookup) == {"office_export_7"}


async def test_required_reads_are_loaded_and_completion_restores_task_tools(monkeypatch):
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.tool_adapter import ToolWrapper
    from tests.unit.test_live_streaming import _chunk, _tc

    recovery = {"document_read", "table_requirements"}
    reads, writes = AsyncMock(return_value="Policy content"), AsyncMock(return_value="Saved")

    async def requirements(**kwargs):
        nonlocal recovery
        recovery = None
        return "Contract fixed"

    tools = [ToolWrapper(name=name, description=name + " " + "x" * 1500, function=fn) for name, fn in [
        ("document_read", reads), ("table_requirements", requirements), ("file_write", writes),
        *[(f"service_{i}", AsyncMock()) for i in range(30)],
    ]]
    seen = []

    async def complete(**params):
        seen.append({tool["function"]["name"] for tool in params["tools"]})

        async def chunks():
            if len(seen) <= 3:
                yield _chunk(tool_calls=_tc(["document_read", "table_requirements", "file_write"][len(seen) - 1]),
                             finish="tool_calls")
            else:
                yield _chunk(content="Done.", finish="stop")
        return chunks()

    monkeypatch.setattr("rune.agent.litellm_adapter.litellm.acompletion", complete)
    agent = LiteLLMAgent("openai/gpt-5.4", tools=tools)
    async with agent.run_stream("Create a document", tool_recovery=lambda: recovery) as stream:
        assert "".join([part async for part in stream.stream_text()]) == "Done."
    assert seen[:2] == [{"document_read", "table_requirements"}] * 2
    assert {"file_write", "tool_search"} <= seen[2]
    reads.assert_awaited_once()
    writes.assert_awaited_once()


@pytest.mark.parametrize('goal_type,intents,execute', [
    ('artifact', {'document'}, False), ('full', set(), False),
    ('code_modify', {'coding'}, False), ('execution', set(), False),
    ('artifact', {'document'}, True), ('research', {'calculation'}, False),
])
async def test_shell_is_loaded_on_demand_unless_the_task_needs_execution(goal_type, intents, execute):
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.tool_catalog import deferred_for_task

    names = ['file_read', 'file_write', 'file_delete', 'think', 'bash_execute']
    schemas = [{'type': 'function', 'function': {'name': name, 'parameters': {'type': 'object'}}} for name in names]
    classification = ClassificationResult(goal_type=goal_type, confidence=.9, tier=2,
                                         intent_categories=frozenset(intents), requires_execution=execute)
    catalog = ToolCatalog(schemas, deferred_tools=deferred_for_task(names, classification))
    needs_shell = execute or goal_type in {'code_modify', 'execution'} or 'calculation' in intents
    assert ('bash_execute' in catalog.loaded) is needs_shell
    assert {'file_read', 'file_write'} <= catalog.loaded
    for name in names:
        await catalog.search(name)
        assert name in catalog.loaded


async def test_table_guidance_follows_discovery_and_is_not_duplicated():
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.prompts import PROMPT_TABLE
    from rune.agent.tool_adapter import ToolWrapper

    tools = [ToolWrapper(name=name, description='x' * 1600, function=AsyncMock())
             for name in ['file_read', 'table_requirements', 'table_verify', *[f'service_{n}' for n in range(30)]]]
    agent = LiteLLMAgent('openai/gpt-5.4', system_prompt='Core\n' + PROMPT_TABLE,
                        tools=tools, deferred_tools={'table_requirements', 'table_verify'})
    assert PROMPT_TABLE not in agent._system_prompt
    messages = [{'role': 'system', 'content': agent._system_prompt}]
    assert agent._tool_catalog.with_guidance(messages) is messages
    await agent._tool_catalog.search('table_requirements')
    messages = agent._tool_catalog.with_guidance(messages)
    assert PROMPT_TABLE in messages[0]['content']
    await agent._tool_catalog.search('table_verify')
    assert agent._tool_catalog.with_guidance(messages) is messages
    agent.update_tools(tools[:3])
    assert PROMPT_TABLE in agent._system_prompt
