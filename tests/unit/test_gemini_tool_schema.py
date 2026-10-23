from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from rune.agent.litellm_adapter import tools_to_openai_schema
from rune.agent.tool_adapter import ToolAdapterOptions, build_tool_set
from rune.capabilities.registry import CapabilityRegistry
from rune.llm.request_params import compatible_completion
from rune.llm.tool_schema import gemini_tools


def recovery_tools(name="browser_observe"):
    from rune.agent.browser_plan import executor_schema
    from rune.capabilities.ask_user import register_ask_user_capability
    from rune.capabilities.blocked import register_blocked_capability
    from rune.capabilities.browser import register_browser_capabilities

    registry = CapabilityRegistry()
    for register in (register_browser_capabilities, register_ask_user_capability, register_blocked_capability):
        register(registry)
    tools = build_tool_set(ToolAdapterOptions(enable_guardian=False), registry=registry)
    tools[name].json_schema["properties"]["taskRequirements"] = executor_schema()
    return tools_to_openai_schema([tools[key] for key in (name, "ask_user", "task_blocked")])


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("name", ["browser_observe", "browser_batch"])
async def test_recovery_schema_reaches_gemini_without_array_expansion(monkeypatch, streaming, name):
    import json

    import httpx
    import litellm

    calls = []

    async def send(self, request, **kwargs):
        calls.append(json.loads(request.content))
        return httpx.Response(400, request=request, json={"error": {"message": "test boundary"}})

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    tools = recovery_tools(name)
    original = deepcopy(tools)
    params = {"model": "gemini/gemini-2.5-flash", "api_key": "test-key",
              "messages": [{"role": "user", "content": "Correct the plan."}],
              "tools": tools, "tool_choice": "required", "max_tokens": 128,
              "reasoning_effort": "none", "stream": streaming, "num_retries": 0, "max_retries": 0}
    with pytest.raises(litellm.BadRequestError, match="test boundary"):
        response = await compatible_completion(litellm.acompletion, litellm.BadRequestError, params)
        if streaming:
            async for _ in response:
                pass
    assert len(calls) == 1
    assert tools == original
    body = calls[0]
    assert body["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    functions = {item["name"]: item["parameters"] for item in body["tools"][0]["function_declarations"]}
    assert set(functions) == {name, "ask_user", "task_blocked"}
    properties = functions[name]["properties"]
    if name == "browser_observe":
        assert "maxItems" not in properties["checks"]
        assert properties["checks"]["description"].endswith("At most 32 items.")
        assert properties["checks"]["items"]["required"] == ["property", "expected"]
    else:
        assert "maxItems" not in properties["actions"]
        assert properties["actions"]["description"].endswith("At most 20 items.")
        assert properties["actions"]["items"]["anyOf"]
    plan = properties["taskRequirements"]["properties"]["items"]
    assert "maxItems" not in plan and plan["description"] == "At most 16 items."
    assert plan["items"]["required"] == ["requirement", "kind", "expected"]
    assert plan["items"]["properties"]["expected"]["anyOf"][0]["nullable"] is True
    options = functions["ask_user"]["properties"]["options"]["anyOf"][0]
    assert "maxItems" not in options and options["minItems"] == 2
    assert options["description"].endswith("At most 4 items.")


@pytest.mark.parametrize("model,choice", [
    ("gemini/gemini-2.5-flash", None), ("vertex_ai/gemini-2.5-flash", "auto"),
    ("gemini/gemini-2.5-flash", "none"), ("vertex_ai/other-model", "required"),
    ("openai/gpt-5.4-mini", "required"), ("anthropic/claude-haiku-4-5", "required"),
    ("xai/grok-4.7", "required"), ("ollama/local-model", "required"),
])
def test_other_modes_and_providers_keep_their_schemas(model, choice):
    tools = recovery_tools()
    original = deepcopy(tools)
    assert gemini_tools(model, tools, choice) is tools
    assert tools == original


@pytest.mark.parametrize("model", ["gemini/gemini-2.5-flash", "vertex_ai/gemini-2.5-flash",
                                   "vertex_ai_beta/gemini-2.5-flash"])
def test_only_schema_nodes_change_and_named_calls_keep_the_same_contract(model):
    payload = {"maxItems": 7, "items": {"maxItems": 8}}
    schema = {"type": "object", "properties": {
        "maxItems": {"type": "integer", "maximum": 8},
        "items": {"type": "array", "maxItems": 2, "minItems": 1, "items": {"$ref": "#/$defs/Entry"}},
        "metadata": {"type": "object", "default": payload, "examples": [payload]}},
        "$defs": {"Entry": {"anyOf": [{"type": "array", "items": {"type": "string"}, "maxItems": 3},
                                     {"type": "null"}]}},
        "required": ["items"]}
    tools = [{"type": "function", "function": {"name": "inspect", "parameters": schema}}, {"type": "google_search"}]
    original = deepcopy(tools)
    choice = {"type": "function", "function": {"name": "inspect"}}
    wire = gemini_tools(model, tools, choice)
    params = wire[0]["function"]["parameters"]
    assert tools == original and wire[1] == tools[1]
    assert params["properties"]["metadata"] == schema["properties"]["metadata"]
    assert params["properties"]["maxItems"] == schema["properties"]["maxItems"]
    assert params["properties"]["items"]["minItems"] == 1
    assert params["properties"]["items"]["description"] == "At most 2 items."
    assert params["$defs"]["Entry"]["anyOf"][0]["description"] == "At most 3 items."
    assert params["required"] == ["items"]
    assert gemini_tools(model, wire, choice) == wire


@pytest.mark.parametrize("case", ["checks", "requirements", "options"])
async def test_local_limits_reject_oversized_arguments_before_execution(case):
    from rune.agent.browser_plan import configure_tools
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.task_evidence import TaskEvidence
    from rune.capabilities.ask_user import AskUserParams
    from rune.capabilities.browser.capabilities import BrowserObserveParams
    from rune.capabilities.types import CapabilityDefinition

    evidence = TaskEvidence.for_request("Set count to 3 and save", ClassificationResult("browser", .99, 2))
    await evidence.prepare_requirements()
    execute = AsyncMock(side_effect=AssertionError("Invalid arguments must not execute"))
    name = "ask_user" if case == "options" else "browser_observe"
    registry = CapabilityRegistry()
    registry.register(CapabilityDefinition(name=name, description="test", execute=execute,
        parameters_model=AskUserParams if case == "options" else BrowserObserveParams))
    tools = build_tool_set(ToolAdapterOptions(enable_guardian=False, prepare_tool=evidence.prepare_tool), registry)
    configure_tools(evidence, tools)
    schemas = tools_to_openai_schema(list(tools.values()))
    gemini_tools("vertex_ai/gemini-2.5-flash", schemas, "required")
    params = {
        "checks": {"checks": [{"selector": "#count", "property": "value", "expected": "3"}] * 33},
        "requirements": {"taskRequirements": {"items": [
            {"requirement": "count to 3", "kind": "value", "expected": "3"}] * 17}},
        "options": {"question": "Which?", "reason": "Missing input", "options": [{"label": "One"}] * 5},
    }[case]
    result = await tools[name].function(**params)
    assert result.code in {"invalid_arguments", "invalid_requirements"}
    execute.assert_not_awaited()
    assert evidence.browser_actions == 0 and evidence.requirements is None
