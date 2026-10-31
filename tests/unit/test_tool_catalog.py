import json
from unittest.mock import AsyncMock

import pytest

from rune.agent.tool_catalog import ToolCatalog


@pytest.mark.parametrize("kind", ["web", "research"])
async def test_source_research_defers_other_tools_but_keeps_them_discoverable(kind):
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.tool_adapter import ToolWrapper
    from rune.agent.tool_catalog import deferred_for_task
    from rune.config.defaults import TOOLS_WEB

    classification = ClassificationResult(kind, .9, 2, needs_web_evidence=True, browser_effect="view")
    names = [*TOOLS_WEB, "task_blocked"]
    wrappers = [ToolWrapper(name=name, description=name, function=AsyncMock()) for name in names]
    agent = LiteLLMAgent("openai/gpt-6-astra", tools=wrappers, compact_tools=True,
                        deferred_tools=deferred_for_task(names, classification))
    assert agent._tool_catalog.loaded == {"web_search", "web_fetch", "task_blocked", "tool_search"}
    for name in ("browser_extract", "file_read", "table_verify"):
        await agent._tool_catalog.search(name)
        assert name in agent._tool_catalog.loaded
        assert agent._tool_lookup[name] is next(w.function for w in wrappers if w.name == name)


@pytest.mark.parametrize("overrides", [{"intent_categories": frozenset({"document"})},
                                      {"requires_execution": True}, {"browser_effect": "change"}])
def test_mixed_research_keeps_its_workflow_tools(overrides):
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.tool_catalog import deferred_for_task

    classification = ClassificationResult("web", .9, 2, **{
        "needs_web_evidence": True, "browser_effect": "view", **overrides,
    })
    assert "file_read" not in deferred_for_task(["web_fetch", "file_read"], classification)


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
    classification = ClassificationResult(goal_type="web", confidence=.99, tier=2, needs_web_evidence=True, lookup_only=True, complexity="simple")
    deferred = deferred_for_task(names, classification)
    full = LiteLLMAgent("openai/gpt-5.4", tools=tools)
    agent = LiteLLMAgent("openai/gpt-5.4", tools=tools, deferred_tools=deferred, compact_tools=True)
    assert agent._tool_catalog.loaded == {"web_search", "web_fetch", "task_blocked", "tool_search"}
    assert len(json.dumps(agent._tool_schemas)) < len(json.dumps(full._tool_schemas))
    await agent._tool_catalog.search("browser_observe")
    assert "browser_observe" in agent._tool_catalog.loaded
    assert set(agent._tool_lookup) == set(names) | {"tool_search"}


@pytest.mark.parametrize("needs_sources", [False, True])
async def test_browser_catalog_starts_with_working_tools_and_preserves_discovery(needs_sources):
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.tool_adapter import ToolWrapper
    from rune.agent.tool_catalog import deferred_for_task
    from rune.capabilities.browser.capabilities import register_browser_capabilities
    from rune.capabilities.registry import CapabilityRegistry

    reg = CapabilityRegistry()
    register_browser_capabilities(reg)
    wrappers = [ToolWrapper(name=cap.name, description=cap.description,
                            json_schema=cap.parameters_model.model_json_schema(), function=AsyncMock()) for cap in reg.list_all()]
    wrappers += [ToolWrapper(name=name, description=name, function=AsyncMock())
                 for name in ("ask_user", "task_blocked", "web_search", "web_fetch", "think")]
    names = {t.name for t in wrappers}
    classification = ClassificationResult("browser", .99, 2, needs_web_evidence=needs_sources)
    deferred = deferred_for_task(list(names), classification)
    agent = LiteLLMAgent("openai/gpt-5.4", tools=wrappers, deferred_tools=deferred, compact_tools=True)
    full = LiteLLMAgent("openai/gpt-5.4", tools=wrappers)
    expected = {"browser_navigate", "browser_observe", "browser_act", "browser_extract", "browser_batch",
                "ask_user", "task_blocked", "tool_search"}
    if needs_sources:
        expected.update(("web_search", "web_fetch"))
    assert agent._tool_catalog.loaded == expected
    assert len(json.dumps(agent._tool_schemas)) < len(json.dumps(full._tool_schemas))
    assert agent._tool_lookup["browser_act"] is full._tool_lookup["browser_act"]
    for name in deferred:
        await agent._tool_catalog.search(name)
        assert name in agent._tool_catalog.loaded
        assert agent._tool_lookup[name] is full._tool_lookup[name]
    agent.update_tools([t for t in wrappers if t.name != "think"])
    assert agent._tool_catalog.loaded == expected


async def test_deferred_tool_alias_does_not_load_its_schema_early():
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.tool_adapter import ToolWrapper

    tools = [ToolWrapper(name=name, description=name, function=AsyncMock()) for name in ("web_fetch", "mail.send")]
    agent = LiteLLMAgent("openai/gpt-5.4", tools=tools, deferred_tools={"mail.send"}, compact_tools=True)
    assert agent._tool_catalog.loaded == {"web_fetch", "tool_search"}
    assert "mail__send" in agent._tool_catalog.all
    await agent._tool_catalog.search("mail__send")
    assert "mail__send" in agent._tool_catalog.loaded
    assert agent._tool_lookup["mail__send"] is agent._tool_lookup["mail.send"]


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
    assert "advisor" not in agent._tool_catalog.all
    assert "service_0" not in agent._tool_catalog.loaded
    await agent._tool_catalog.search("service_0")
    assert "service_0" in agent._tool_catalog.loaded
    assert agent._tool_schemas.count(native_schema) == 1


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
    assert guided[:-1] == messages and PROMPT_BROWSER in guided[-1]["content"]
    assert PROMPT_BROWSER not in messages[0]["content"]
    assert agent._tool_catalog.with_guidance(guided) is guided
    assert "browser_act" in agent._tool_catalog.loaded
    assert "guidance" not in json.loads(await agent._tool_catalog.search("browser_act"))
    assert "guidance" not in json.loads(await agent._tool_catalog.search("browser_observe"))
    agent.update_tools(tools[:2])
    assert agent._tool_catalog is None and agent._system_prompt == prompt


@pytest.mark.parametrize("model", ["openai/gpt-6-astra", "azure/responses/gpt-6-astra",
                                   "anthropic/claude-opus-5", "vertex_ai/gemini-3.1-pro-preview", "xai/grok-4.7"])
async def test_stream_restores_browser_guidance_before_requesting_recovery_tools(monkeypatch, model):
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
    stream = _make(model)
    stream._tool_catalog = catalog
    stream._tool_schemas = catalog.schemas
    stream._tool_recovery = lambda: {"browser_observe"}
    stream._require_verification = False
    async for _ in stream.stream_text():
        pass
    assert requests
    def content(message):
        value = message.get("content", "")
        return value if isinstance(value, str) else "".join(part.get("text", "") for part in value)

    assert all(any(m.get("role") == "user" and PROMPT_BROWSER in content(m)
                   for m in request["messages"]) for request in requests)
    assert all(PROMPT_BROWSER not in str(request["messages"][0]) for request in requests)


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
        {"type": "act", "selector": "#calculate"},
    ]})
    assert not rejected.success and rejected.metadata["action_status"] == "not_executed"
    assert len(rejected.error) < 1200 and "params" in rejected.error


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


async def test_unknown_tool_lists_callable_names_without_dispatch():
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.tool_adapter import ToolWrapper

    action = AsyncMock()
    agent = LiteLLMAgent("openai/gpt-5.4", tools=[
        ToolWrapper(name=name, description=name, function=action)
        for name in ("browser_act", "browser_observe", "browser_navigate", "file_read")
    ], deferred_tools={"browser_navigate"}, compact_tools=True)
    async with agent.run_stream("Update the form") as stream:
        result = await stream._execute_tool("BrowserActChecks", {"expected": "3"})
    assert result.code == "unknown_tool"
    assert "browser_act" in result and "browser_observe" in result
    assert "with its exact name" not in result
    action.assert_not_awaited()


@pytest.mark.parametrize("model", [
    "openai/gpt-5.4-mini", "vertex_ai/gemini-2.5-flash", "anthropic/claude-haiku-4-5", "xai/grok-4",
])
async def test_unknown_tool_requires_lookup_before_input_and_restores_tools(monkeypatch, model):
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.tool_adapter import ToolWrapper
    from tests.unit.test_live_streaming import _chunk, _tc, _usage

    write, seen = AsyncMock(return_value="Saved"), []
    calls = [("FileWriteParams", "{}"), ("file_write", "{}"),
             ("tool_search", '{"query":"file_write"}'), ("file_write", "{}")]

    async def complete(**params):
        seen.append(params)
        async def chunks():
            if len(seen) <= len(calls):
                yield _chunk(tool_calls=_tc(*calls[len(seen) - 1]), finish="tool_calls")
            else:
                yield _chunk(content="Saved", finish="stop")
            yield _usage()
        return chunks()

    monkeypatch.setattr("litellm.acompletion", complete)
    agent = LiteLLMAgent(model, compact_tools=True, deferred_tools={"file_read"}, tools=[
        ToolWrapper(name=name, description=name, function=write)
        for name in ("file_read", "file_write", "ask_user", "task_blocked")
    ])
    async with agent.run_stream("Save the file") as stream:
        assert "".join([part async for part in stream.stream_text()]) == "Saved"
    assert len(seen) == 5
    for request in seen[1:3]:
        assert request["tool_choice"] == "required"
        assert {tool["function"]["name"] for tool in request["tools"]} == {
            "tool_search", "ask_user", "task_blocked",
        }
    assert seen[3]["tool_choice"] == "required"
    assert {tool["function"]["name"] for tool in seen[3]["tools"]} == {
        "file_write", "ask_user", "task_blocked",
    }
    assert seen[4].get("tool_choice") != "required"
    assert any(tool["function"]["name"] == "tool_search" for tool in seen[4]["tools"])
    assert not agent._tool_catalog.lookup_required
    assert agent._tool_catalog.recovery_tools() is None
    write.assert_awaited_once()


@pytest.mark.parametrize("result", ["Saved", "[ERROR] Save failed"])
async def test_lookup_selection_ends_at_dispatch_even_when_arguments_are_rejected(result):
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.tool_adapter import ToolWrapper
    from rune.agent.tool_output import RejectedToolResult

    write = AsyncMock(side_effect=[RejectedToolResult("Missing path", "invalid_arguments"), result])
    other = AsyncMock()
    agent = LiteLLMAgent("vertex_ai/gemini-2.5-flash", compact_tools=True, deferred_tools={"file_read"}, tools=[
        ToolWrapper(name="file_write", description="Write a file", function=write),
        ToolWrapper(name="file_read", description="Read a file", function=other),
    ])
    async with agent.run_stream("Save a file") as stream:
        await stream._execute_tool("FileWriteParams", {})
        await stream._execute_tool("tool_search", {"query": "file_write"})
        assert agent._tool_catalog.recovery_tools() == {"file_write"}
        rejected = await stream._execute_tool("OtherWriteParams", {})
        assert rejected.code == "unknown_tool" and "file_write" in rejected
        assert not agent._tool_catalog.lookup_required
        assert agent._tool_catalog.recovery_tools() == {"file_write"}
        blocked = await stream._execute_tool("tool_search", {"query": "file_read"})
        assert "Required recovery is pending" in blocked
        other.assert_not_awaited()
        assert (await stream._execute_tool("file_write", {})).code == "invalid_arguments"
        assert agent._tool_catalog.recovery_tools() is None
        await stream._execute_tool("tool_search", {"query": "file_read"})
        assert "file_read" in agent._tool_catalog.loaded
        assert await stream._execute_tool("file_write", {"path": "result.txt"}) == result
        assert agent._tool_catalog.recovery_tools() is None
    assert write.await_count == 2


async def test_lookup_selection_does_not_carry_into_a_new_turn():
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.tool_adapter import ToolWrapper

    write = AsyncMock()
    agent = LiteLLMAgent("vertex_ai/gemini-2.5-flash", compact_tools=True, deferred_tools={"file_read"}, tools=[
        ToolWrapper(name="file_write", description="Write a file", function=write),
        ToolWrapper(name="file_read", description="Read a file", function=write),
    ])
    async with agent.run_stream("Save a file") as stream:
        await stream._execute_tool("FileWriteParams", {})
        await stream._execute_tool("tool_search", {"query": "file_write"})
        assert agent._tool_catalog.recovery_tools() == {"file_write"}
    async with agent.run_stream("Just explain"):
        assert agent._tool_catalog.recovery_tools() is None
    write.assert_not_awaited()


async def test_repeated_unknown_calls_cannot_restart_successful_lookup(monkeypatch):
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.tool_adapter import ToolWrapper
    from tests.unit.test_live_streaming import _chunk, _tc, _usage

    write, seen = AsyncMock(), []

    async def complete(**params):
        seen.append(params)
        async def chunks():
            name, arguments = ("tool_search", '{"query":"file_write"}') if len(seen) == 2 else ("FileWriteParams", "{}")
            yield _chunk(tool_calls=_tc(name, arguments), finish="tool_calls")
            yield _usage()
        return chunks()

    monkeypatch.setattr("litellm.acompletion", complete)
    agent = LiteLLMAgent("vertex_ai/gemini-2.5-flash", compact_tools=True, deferred_tools={"file_read"}, tools=[
        ToolWrapper(name=name, description=name, function=write) for name in ("file_read", "file_write")
    ])
    async with agent.run_stream("Save a file") as stream:
        async for _ in stream.stream_text():
            pass
        assert "rejected tool calls" in await stream.get_output()
    assert len(seen) == 5
    for request in seen[2:]:
        assert request["tool_choice"] == "required"
        assert [tool["function"]["name"] for tool in request["tools"]] == ["file_write"]
    write.assert_not_awaited()


async def test_unresolved_tool_lookup_stops_without_increasing_retry_limit(monkeypatch):
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.tool_adapter import ToolWrapper
    from tests.unit.test_live_streaming import _chunk, _tc, _usage

    write, seen = AsyncMock(), []
    async def complete(**params):
        seen.append(params)
        async def chunks():
            yield _chunk(tool_calls=_tc(
                "FileWriteParams" if len(seen) == 1 else "tool_search",
                "{}" if len(seen) == 1 else '{"query":"nonexistent"}',
            ), finish="tool_calls")
            yield _usage()
        return chunks()

    monkeypatch.setattr("litellm.acompletion", complete)
    agent = LiteLLMAgent("vertex_ai/gemini-2.5-flash", compact_tools=True, deferred_tools={"file_read"}, tools=[
        ToolWrapper(name=name, description=name, function=write) for name in ("file_read", "file_write")
    ])
    async with agent.run_stream("Save") as stream:
        async for _ in stream.stream_text():
            pass
        text = await stream.get_output()
    assert len(seen) == 3 and "rejected tool calls" in text
    assert agent._tool_catalog.lookup_required
    write.assert_not_awaited()

    async def explain(**params):
        assert params.get("tool_choice") != "required"
        async def chunks():
            yield _chunk(content="An explanation needs no tools.", finish="stop")
            yield _usage()
        return chunks()
    monkeypatch.setattr("litellm.acompletion", explain)
    async with agent.run_stream("Just explain") as stream:
        assert "".join([part async for part in stream.stream_text()]) == "An explanation needs no tools."
    assert not agent._tool_catalog.lookup_required


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
    ('artifact', {'document'}, False), ('full', set(), False), ('full', {'document'}, False),
    ('code_modify', {'coding'}, False), ('execution', set(), False),
    ('artifact', {'document'}, True), ('research', {'calculation'}, False),
])
async def test_shell_deferral_preserves_execution_and_unknown_workflows(goal_type, intents, execute):
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.tool_catalog import deferred_for_task

    names = ['file_read', 'file_write', 'file_delete', 'think', 'bash_execute']
    schemas = [{'type': 'function', 'function': {'name': name, 'parameters': {'type': 'object'}}} for name in names]
    classification = ClassificationResult(goal_type=goal_type, confidence=.9, tier=2,
                                         intent_categories=frozenset(intents), requires_execution=execute)
    catalog = ToolCatalog(schemas, deferred_tools=deferred_for_task(names, classification))
    needs_shell = execute or goal_type in {'code_modify', 'execution'} or 'calculation' in intents or (goal_type == 'full' and not intents)
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
    assert messages[0]['content'] == agent._system_prompt
    assert PROMPT_TABLE in messages[-1]['content']
    await agent._tool_catalog.search('table_verify')
    assert agent._tool_catalog.with_guidance(messages) is messages
    agent.update_tools(tools[:3])
    assert PROMPT_TABLE not in agent._system_prompt
    assert agent._tool_catalog.loaded == {"file_read", "tool_search"}
    await agent._tool_catalog.search("table_verify")
    assert PROMPT_TABLE in agent._tool_catalog.with_guidance([])[-1]["content"]


@pytest.mark.parametrize("goal_type,intents", [("artifact", {"document"}), ("code_modify", {"coding"}),
                                             ("research", set()), ("execution", {"coding"})])
async def test_local_work_defers_web_and_desktop_without_losing_discovery(goal_type, intents):
    from dataclasses import replace

    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.tool_catalog import deferred_for_task

    names = ["file_read", "web_search", "web_fetch", "desktop_open", "desktop_act"]
    c = ClassificationResult(goal_type, .99, 2, intent_categories=frozenset(intents))
    schemas = [{"type": "function", "function": {"name": name}} for name in names]
    catalog = ToolCatalog(schemas, deferred_tools=deferred_for_task(names, c))
    assert catalog.loaded == {"file_read", "tool_search"}
    for name in names[1:]:
        await catalog.search(name)
        assert name in catalog.loaded
    assert not deferred_for_task(names, replace(c, needs_web_evidence=True, requires_desktop_input=True))



async def test_small_local_catalog_honors_deferral_and_keeps_specialized_tools_discoverable():
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.prompts import PROMPT_TABLE
    from rune.agent.tool_adapter import ToolWrapper
    from rune.agent.tool_catalog import deferred_for_task

    names = ["file_read", "code_analyze", "project_map", "web_fetch", "document_read", "table_verify", "memory_search"]
    wrappers = [ToolWrapper(name=name, description=name * 30, function=AsyncMock()) for name in names]
    task = ClassificationResult("research", .95, 2, intent_categories=frozenset({"coding"}))
    agent = LiteLLMAgent("openai/gpt-6-astra", tools=wrappers, system_prompt="Core" + PROMPT_TABLE,
                        deferred_tools=deferred_for_task(names, task))
    assert agent._tool_catalog.loaded == {"file_read", "code_analyze", "project_map", "memory_search", "tool_search"}
    assert PROMPT_TABLE not in agent._system_prompt
    original = json.dumps(agent._tool_schemas)
    agent.update_tools(wrappers)
    assert json.dumps(agent._tool_schemas) == original
    for name in ("web_fetch", "document_read", "table_verify"):
        await agent._tool_catalog.search(name)
        assert agent._tool_lookup[name] is next(w.function for w in wrappers if w.name == name)
    assert PROMPT_TABLE in agent._tool_catalog.with_guidance([])[-1]["content"]



def test_discovery_does_not_add_overhead_to_a_minimal_toolset():
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.tool_adapter import ToolWrapper

    tools = [ToolWrapper(name=name, description=name, function=AsyncMock()) for name in ("file_write", "bash_execute")]
    agent = LiteLLMAgent("openai/gpt-6-astra", tools=tools, deferred_tools={"bash_execute"})
    assert agent._tool_catalog is None
    assert {tool["function"]["name"] for tool in agent._tool_schemas} == {"file_write", "bash_execute"}



def test_mixed_file_work_keeps_its_execution_fallback():
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.tool_adapter import ToolWrapper
    from rune.agent.tool_catalog import deferred_for_task
    from rune.capabilities.bash import BashParams

    task = ClassificationResult("full", .99, 2, output_expectation="file")
    tools = [ToolWrapper(name="file_write", description="Write a file", function=AsyncMock()),
             ToolWrapper(name="bash_execute", description="Run a command", function=AsyncMock(),
                         json_schema=BashParams.model_json_schema())]
    agent = LiteLLMAgent("openai/gpt-6-astra", tools=tools,
                        deferred_tools=deferred_for_task([tool.name for tool in tools], task))
    assert agent._tool_catalog is None
    assert {tool["function"]["name"] for tool in agent._tool_schemas} == {"file_write", "bash_execute"}
