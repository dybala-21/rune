"""Azure settings, deployment routing, and SDK requests."""

import json

import httpx
import pytest
from pydantic import ValidationError

from rune.agent.model_traits import reasoning_efforts, supports_vision, traits
from rune.config.schema import LLMConfig
from rune.llm.context import model_scope
from rune.llm.target import ModelCompatibilityError, resolve_target


@pytest.fixture
def azure(monkeypatch):
    from rune.config import get_config

    settings = LLMConfig(activeProvider="azure", activeModel="daily", azureDeployments={
        "daily": "gpt-6-sol", "deep": "gpt-6-astra",
    })
    monkeypatch.setattr(get_config(), "llm", settings)
    for key in ("AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fixture.openai.azure.com/openai/v1/")
    return settings


def test_connection_and_capabilities_are_scoped_to_the_run(azure, monkeypatch):
    with model_scope():
        azure.azure_deployments = {"daily": "gpt-6-astra"}
        monkeypatch.setenv("AZURE_API_BASE", "https://replacement.openai.azure.com")
        monkeypatch.setenv("AZURE_API_KEY", "replacement-key")
        target = resolve_target("azure/daily")
        assert target.key == "azure/daily"
        assert target.kwargs() == {"model": "azure/daily", "api_base": "https://fixture.openai.azure.com",
                                   "api_key": "azure-test-key", "api_version": "v1"}
        assert "azure-test-key" not in repr(target)
        assert "none" in reasoning_efforts("azure/daily")
        assert supports_vision("azure/daily")
        assert traits("azure/responses/daily").responses_api
    assert "none" not in reasoning_efforts("azure/daily")
    assert resolve_target("azure/daily").kwargs()["api_key"] == "replacement-key"


@pytest.mark.parametrize("base", ["", "http://fixture.openai.azure.com", "https://user:secret@fixture.test",
                                  "https://fixture.test/?key=secret", "https://fixture.test/other"])
def test_invalid_endpoints_fail_before_any_request(azure, monkeypatch, base):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", base)
    with pytest.raises(ModelCompatibilityError, match="HTTPS Azure endpoint"):
        resolve_target("azure/daily")


def test_picker_and_auxiliary_calls_use_registered_deployments(azure):
    from rune.agent.failover import build_profiles_from_config
    from rune.llm.client import LLMClient
    from rune.llm.models import known_models
    from rune.types import ModelTier, Provider

    assert [(p, m) for p, m in known_models() if p == "azure"] == [("azure", "daily"), ("azure", "deep")]
    client = LLMClient()
    assert client.resolve_model(ModelTier.BEST, Provider.AZURE) == "deep"
    assert client.resolve_model(ModelTier.FAST, Provider.AZURE) == "daily"
    assert client.resolve_model(ModelTier.CODING, Provider.AZURE) == "daily"
    assert [(p.provider, p.model) for p in build_profiles_from_config()] == [("azure", "daily")]
    azure.azure_deployments = {}
    azure.active_provider = "openai"
    assert not any(p == "azure" for p, _ in known_models())


@pytest.mark.parametrize("deployments", [{"bad/name": "gpt-6-sol"}, {"daily": ""}, {"daily": "openai/gpt-6-sol"}])
def test_deployment_mapping_rejects_paths_and_empty_names(deployments):
    with pytest.raises(ValidationError):
        LLMConfig(azureDeployments=deployments)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("deployment,effort", [("daily", "none"), ("daily", "max"), ("deep", "max")])
async def test_real_sdk_preserves_azure_tool_history_and_reasoning(azure, monkeypatch, deployment, effort, streaming):
    from rune.agent.litellm_adapter import _litellm
    from rune.llm.request_params import compatible_completion

    calls = []

    async def send(self, request, **kwargs):
        calls.append(request)
        return httpx.Response(400, request=request, json={
            "error": {"message": "test boundary", "type": "invalid_request_error"},
        })

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    llm = _litellm()
    params = {**resolve_target(f"azure/{deployment}").kwargs(), "num_retries": 0, "max_retries": 0,
              "max_tokens": 64, "reasoning_effort": effort, "temperature": 0, "stream": streaming,
              "messages": [
                  {"role": "user", "content": "Read the file."},
                  {"role": "assistant", "content": None, "tool_calls": [
                      {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                  ]},
                  {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
              ],
              "tools": [{"type": "function", "function": {"name": "read_file",
                        "parameters": {"type": "object", "properties": {}}}}]}
    with pytest.raises(llm.BadRequestError, match="test boundary"):
        response = await compatible_completion(llm.acompletion, llm.BadRequestError, params)
        if streaming:
            async for _ in response:
                pass
    assert len(calls) == 1
    request = calls[0]
    assert request.url.host == "fixture.openai.azure.com"
    assert request.url.path == "/openai/v1/responses"
    assert request.headers["api-key"] == "azure-test-key"
    body = json.loads(request.content)
    assert body["model"] == deployment
    assert bool(body.get("stream")) == streaming
    assert body["reasoning"]["effort"] == effort
    assert body["max_output_tokens"] == 64 and body["store"] is False
    assert "temperature" not in body
    assert body["tools"][0]["name"] == "read_file"
    assert any(item.get("type") == "function_call_output" and item["call_id"] == "call_1"
               and item["output"] in ("file contents", [{"type": "input_text", "text": "file contents"}])
               for item in body["input"])


@pytest.mark.parametrize("streaming", [False, True])
async def test_azure_response_text_and_usage_survive_sdk_conversion(azure, monkeypatch, streaming):
    from rune.agent.litellm_adapter import _litellm
    from rune.llm.pricing import estimate_request_cost
    from rune.llm.request_params import compatible_completion
    from rune.llm.usage import token_counts

    result = {"id": "resp_fixture", "object": "response", "created_at": 1, "status": "completed",
              "model": "gpt-6-sol", "output": [{"id": "msg_fixture", "type": "message", "status": "completed",
                  "role": "assistant", "content": [{"type": "output_text", "text": "OK", "annotations": []}]}],
              "usage": {"input_tokens": 7, "output_tokens": 1, "total_tokens": 8,
                        "input_tokens_details": {"cached_tokens": 3}, "output_tokens_details": {"reasoning_tokens": 0}}}

    async def send(self, request, **kwargs):
        if not streaming:
            return httpx.Response(200, request=request, json=result)
        events = [
            {"type": "response.created", "response": {**result, "status": "in_progress", "output": []}},
            {"type": "response.output_item.added", "output_index": 0, "item": result["output"][0]},
            {"type": "response.output_text.delta", "item_id": "msg_fixture", "output_index": 0, "content_index": 0, "delta": "OK"},
            {"type": "response.completed", "response": result},
        ]
        return httpx.Response(200, request=request, headers={"content-type": "text/event-stream"},
                              content="".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events))

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    llm = _litellm()
    params = {**resolve_target("azure/daily").kwargs(), "stream": streaming,
              "stream_options": {"include_usage": True},
              "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 32, "num_retries": 0}
    response = await compatible_completion(llm.acompletion, llm.BadRequestError, params)
    if streaming:
        chunks = [chunk async for chunk in response]
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "OK"
        usage = next(chunk.usage for chunk in reversed(chunks) if getattr(chunk, "usage", None))
    else:
        assert response.choices[0].message.content == "OK"
        usage = response.usage
    counts = token_counts(usage)
    assert counts["input_tokens"] == 7 and counts["cached_input_tokens"] == 3 and counts["output_tokens"] == 1
    assert estimate_request_cost("azure/daily", counts, params) is None


async def test_deployment_settings_persist_without_touching_credentials(azure, monkeypatch, tmp_path):
    from rune.api.handlers.config import ConfigPatchRequest, get_config_endpoint, patch_config
    from rune.config import loader
    from rune.config.writer import save_config_values

    path = tmp_path / "config.yaml"
    path.write_text("openai_api_key: ${OPENAI_API_KEY}\nllm:\n  azure_deployments:\n    old: gpt-4o\n")
    monkeypatch.setattr("rune.config.writer.config_file_path", lambda: path)
    assert (await patch_config(ConfigPatchRequest(azureDeployments={"daily": "gpt-6-sol"}))).updated
    assert "${OPENAI_API_KEY}" in path.read_text()
    assert "azure-test-key" not in path.read_text()
    assert "azure_deployments:" not in path.read_text()
    monkeypatch.setattr(loader, "_find_config_file", lambda: path)
    assert loader.load_config(force=True).llm.azure_deployments == {"daily": "gpt-6-sol"}
    state = await get_config_endpoint()
    assert state.azure["hasKey"]
    assert "azure-test-key" not in state.model_dump_json()
    monkeypatch.setattr("rune.config.save_config_values", lambda _: None)
    from fastapi import HTTPException
    with pytest.raises(HTTPException, match="Could not save"):
        await patch_config(ConfigPatchRequest(azureDeployments={"changed": "gpt-6-astra"}))
    assert loader.get_config().llm.azure_deployments == {"daily": "gpt-6-sol"}
    assert save_config_values({"llm.azureDeployments": {}})


def test_azure_keys_are_visible_but_masked_in_environment_settings():
    from rune.api.handlers.env import _categorize_key, _is_rune_key, _mask_value

    for key in ("AZURE_API_KEY", "AZURE_OPENAI_API_KEY"):
        assert _is_rune_key(key) and _categorize_key(key) == "llm"
        assert _mask_value(key, "azure-test-key") != "azure-test-key"
