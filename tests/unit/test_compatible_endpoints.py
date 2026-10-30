"""Ollama on another host and OpenAI-compatible servers (vLLM, LM Studio, llama.cpp)."""

import json

import httpx
import pytest
from pydantic import ValidationError

from rune.config.schema import LLMConfig
from rune.llm import endpoints
from rune.llm.target import ModelCompatibilityError, resolve_target


@pytest.fixture
def servers(monkeypatch):
    from rune.config import get_config

    settings = LLMConfig(activeProvider="openai_compatible", activeModel="gpu/Qwen/Qwen3-Coder-30B", endpoints={
        "gpu": {"baseUrl": "http://192.168.0.20:8000"},
        "router": {"baseUrl": "https://openrouter.example/api/v1", "models": ["qwen/qwen3-coder"], "toolMode": "guided"},
    })
    monkeypatch.setattr(get_config(), "llm", settings)
    for key in ("OLLAMA_API_BASE", "OLLAMA_HOST", "RUNE_ENDPOINT_GPU_API_KEY", "RUNE_ENDPOINT_ROUTER_API_KEY",
                "RUNE_CLOUD_WORKER"):
        monkeypatch.delenv(key, raising=False)
    endpoints.forget_discovered()
    yield settings
    endpoints.forget_discovered()


@pytest.mark.parametrize("variable,value,root", [
    ("OLLAMA_API_BASE", "http://gpu-box:11434/v1/", "http://gpu-box:11434"),
    ("OLLAMA_HOST", "0.0.0.0", "http://localhost:11434"),
    ("OLLAMA_HOST", "10.0.0.7:9999", "http://10.0.0.7:9999"),
    ("OLLAMA_HOST", "https://ollama.example", "https://ollama.example:11434"),
])
def test_ollama_can_run_on_another_host(servers, monkeypatch, variable, value, root):
    monkeypatch.setenv(variable, value)
    assert endpoints.ollama_root() == root
    assert resolve_target("ollama/qwen3:30b").kwargs()["api_base"] == f"{root}/v1"


def test_ollama_defaults_to_this_machine(servers):
    assert resolve_target("ollama/qwen3:30b").kwargs() == {
        "model": "openai/qwen3:30b", "api_base": "http://localhost:11434/v1", "api_key": "ollama"}


def test_endpoint_models_route_to_their_server(servers, monkeypatch):
    target = resolve_target("gpu/Qwen/Qwen3-Coder-30B", "openai_compatible")
    assert target.key == "openai_compatible/gpu/Qwen/Qwen3-Coder-30B"
    # The OpenAI client needs some key even when the server has none.
    assert target.kwargs() == {"model": "openai/Qwen/Qwen3-Coder-30B",
                               "api_base": "http://192.168.0.20:8000/v1", "api_key": "none"}
    monkeypatch.setenv("RUNE_ENDPOINT_ROUTER_API_KEY", "router-secret")
    target = resolve_target("openai_compatible/router/qwen/qwen3-coder")
    assert target.kwargs()["api_key"] == "router-secret"
    assert "router-secret" not in repr(target)


@pytest.mark.parametrize("model", ["missing/qwen", "gpu", "gpu/"])
def test_unknown_endpoint_or_model_fails_before_any_request(servers, model):
    with pytest.raises(ModelCompatibilityError):
        resolve_target(model, "openai_compatible")


@pytest.mark.parametrize("entry", [
    {"baseUrl": "ftp://server/v1"}, {"baseUrl": "http://user:pw@server/v1"},
    {"baseUrl": "http://server/v1?key=1"}, {"baseUrl": "server:8000"},
])
def test_invalid_base_urls_are_rejected(entry):
    with pytest.raises(ValidationError):
        LLMConfig(endpoints={"gpu": entry})


def test_endpoint_names_cannot_shadow_providers():
    with pytest.raises(ValidationError, match="provider names"):
        LLMConfig(endpoints={"ollama": {"baseUrl": "http://localhost:8000"}})
    with pytest.raises(ValidationError):
        LLMConfig(endpoints={"Has Space": {"baseUrl": "http://localhost:8000"}})


@pytest.mark.parametrize("url", ["http://localhost:8000", "http://169.254.169.254/v1", "http://10.0.0.5/v1",
                                 "https://10.0.0.5/v1", "https://metadata.google.internal/v1"])
def test_hosted_workers_only_reach_public_https(monkeypatch, url):
    monkeypatch.setenv("RUNE_CLOUD_WORKER", "1")
    with pytest.raises(ValidationError, match="public HTTPS"):
        LLMConfig(endpoints={"gpu": {"baseUrl": url}})
    assert LLMConfig(endpoints={"gpu": {"baseUrl": "https://api.example.com/v1"}}).endpoints


def test_self_hosted_servers_cost_nothing_and_hosted_ones_are_unpriced(servers):
    from rune.llm.pricing import estimate_request_cost

    usage = {"input_tokens": 1000, "output_tokens": 1000, "cache_write_reported": True}
    local = {**resolve_target("gpu/m", "openai_compatible").kwargs(), "_rune_source_model": "openai_compatible/gpu/m"}
    hosted = {**resolve_target("router/m", "openai_compatible").kwargs(),
              "_rune_source_model": "openai_compatible/router/m"}
    assert estimate_request_cost(local["model"], usage, local) == 0.0
    assert estimate_request_cost(hosted["model"], usage, hosted) is None
    servers.endpoints["router"].free = True
    assert estimate_request_cost(hosted["model"], usage, hosted) == 0.0
    assert endpoints.self_hosted("openai_compatible", "gpu/m")
    assert endpoints.self_hosted("ollama", "qwen3")
    assert not endpoints.self_hosted("openai", "gpt-6-astra")


def test_guided_decoding_follows_the_endpoint_tool_mode(servers, monkeypatch):
    from rune.agent.litellm_adapter import _guided_for_model

    monkeypatch.delenv("RUNE_GUIDED_TOOLS", raising=False)

    def guided(model, provider=None, has_tools=True):
        target = resolve_target(model, provider)
        return _guided_for_model(target.wire_model, has_tools,
                                 {**dict(target.options), "_rune_source_model": target.key})

    assert not guided("gpu/m", "openai_compatible")
    assert guided("router/m", "openai_compatible")
    assert not guided("router/m", "openai_compatible", has_tools=False)
    assert guided("ollama/qwen3:30b")
    assert not guided("ollama/qwen3-coder:480b-cloud")
    # Ollama on another host is still Ollama.
    monkeypatch.setenv("OLLAMA_API_BASE", "http://gpu-box:11434")
    assert guided("ollama/qwen3:30b")
    monkeypatch.setenv("RUNE_GUIDED_TOOLS", "0")
    assert not guided("ollama/qwen3:30b")


def _models_response(request: httpx.Request) -> httpx.Response | None:
    if request.url.path.endswith("/models"):
        return httpx.Response(200, request=request, json={"data": [{"id": "Qwen/Qwen3-Coder-30B"}, {"id": "gpt-oss-20b"}]})
    return None


async def test_discovered_models_reach_the_picker_and_tiers(servers, monkeypatch):
    from rune.llm.client import LLMClient
    from rune.llm.models import known_models, selectable_models
    from rune.types import ModelTier, Provider

    seen = []

    async def send(self, request, **kwargs):
        seen.append(request)
        return _models_response(request) or httpx.Response(404, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setenv("RUNE_ENDPOINT_GPU_API_KEY", "gpu-secret")
    pairs = await selectable_models()
    assert ("openai_compatible", "gpu/Qwen/Qwen3-Coder-30B") in pairs
    assert ("openai_compatible", "gpu/gpt-oss-20b") in pairs
    # Pinned models are used as given; the server is not asked.
    assert ("openai_compatible", "router/qwen/qwen3-coder") in known_models()
    assert [str(r.url) for r in seen] == ["http://192.168.0.20:8000/v1/models"]
    assert seen[0].headers["authorization"] == "Bearer gpu-secret"
    client = LLMClient()
    assert client.resolve_model(ModelTier.FAST, Provider.OPENAI_COMPATIBLE) == "gpu/Qwen/Qwen3-Coder-30B"


async def test_unreachable_server_contributes_no_models(servers, monkeypatch):
    from rune.llm.models import selectable_models

    async def send(self, request, **kwargs):
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    servers.active_provider = servers.active_model = None
    pairs = [pair for pair in await selectable_models() if pair[0] == "openai_compatible"]
    assert pairs == [("openai_compatible", "router/qwen/qwen3-coder")]


@pytest.mark.parametrize("status,body,tools", [
    (200, {"choices": [{"message": {"tool_calls": [{"id": "1", "type": "function",
                                                    "function": {"name": "ping", "arguments": "{}"}}]}}]}, "native"),
    (200, {"choices": [{"message": {"content": "pong"}}]}, "text"),
    (400, {"error": {"message": '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser'}},
     "unsupported"),
    (500, {"error": {"message": "out of memory"}}, "unknown"),
])
async def test_connection_test_reports_tool_support(servers, monkeypatch, status, body, tools):
    sent = []

    async def send(self, request, **kwargs):
        if response := _models_response(request):
            return response
        sent.append(json.loads(request.content))
        return httpx.Response(status, request=request, json=body)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    result = await endpoints.diagnose("gpu")
    assert result["reachable"] and result["tools"] == tools
    assert result["models"] == ["Qwen/Qwen3-Coder-30B", "gpt-oss-20b"]
    assert sent[0]["model"] == "Qwen/Qwen3-Coder-30B" and sent[0]["tool_choice"] == "auto"
    if tools == "unsupported":
        assert "--enable-auto-tool-choice" in result["detail"]


@pytest.mark.parametrize("failure,detail", [
    (httpx.Response(401), "API key was rejected"),
    (httpx.ConnectError("refused"), "ends in /v1"),
])
async def test_connection_test_explains_unreachable_servers(servers, monkeypatch, failure, detail):
    async def send(self, request, **kwargs):
        if isinstance(failure, Exception):
            raise failure
        return httpx.Response(failure.status_code, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    result = await endpoints.diagnose("gpu")
    assert not result["reachable"] and detail in result["detail"]


async def test_real_sdk_sends_tool_requests_to_the_configured_server(servers, monkeypatch):
    from rune.agent.litellm_adapter import _litellm
    from rune.llm.request_params import compatible_completion

    calls = []

    async def send(self, request, **kwargs):
        calls.append(request)
        return httpx.Response(400, request=request, json={"error": {"message": "test boundary"}})

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    params = {**resolve_target("gpu/Qwen/Qwen3-Coder-30B", "openai_compatible").kwargs(),
              "num_retries": 0, "max_retries": 0, "max_tokens": 64,
              "messages": [{"role": "user", "content": "Read the file."}],
              "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]}
    llm = _litellm()
    with pytest.raises(llm.BadRequestError, match="test boundary"):
        await compatible_completion(llm.acompletion, llm.BadRequestError, params)
    assert str(calls[-1].url) == "http://192.168.0.20:8000/v1/chat/completions"
    body = json.loads(calls[-1].content)
    assert body["model"] == "Qwen/Qwen3-Coder-30B" and body["tools"][0]["function"]["name"] == "read_file"


async def test_endpoint_settings_persist_without_storing_keys(servers, monkeypatch, tmp_path):
    from rune.api.handlers.config import ConfigPatchRequest, get_config_endpoint, patch_config
    from rune.api.handlers.env import _categorize_key, _mask_value
    from rune.config import loader

    path = tmp_path / "config.yaml"
    path.write_text("llm: {}\n")
    monkeypatch.setattr("rune.config.writer.config_file_path", lambda: path)
    monkeypatch.setenv("RUNE_ENDPOINT_LAB_API_KEY", "lab-secret")
    request = ConfigPatchRequest(endpoints={"lab": {"baseUrl": "http://lab.local:1234", "toolMode": "guided"}})
    assert (await patch_config(request)).updated
    assert "lab-secret" not in path.read_text()
    monkeypatch.setattr(loader, "_find_config_file", lambda: path)
    saved = loader.load_config(force=True).llm.endpoints["lab"]
    assert saved.base_url == "http://lab.local:1234/v1" and saved.tool_mode == "guided"
    state = (await get_config_endpoint()).local_models
    assert state["endpoints"]["lab"]["hasKey"] and state["endpoints"]["lab"]["selfHosted"]
    assert state["endpoints"]["lab"]["keyName"] == "RUNE_ENDPOINT_LAB_API_KEY"
    assert state["ollama"]["baseUrl"] == "http://localhost:11434"
    assert _categorize_key("RUNE_ENDPOINT_LAB_API_KEY") == "llm"
    assert _mask_value("RUNE_ENDPOINT_LAB_API_KEY", "lab-secret") != "lab-secret"
    with pytest.raises(ValidationError):
        ConfigPatchRequest(endpoints={"openai": {"baseUrl": "http://localhost:1234"}})
