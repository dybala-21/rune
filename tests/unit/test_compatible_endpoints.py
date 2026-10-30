"""Ollama on another host and OpenAI-compatible servers (vLLM, LM Studio, llama.cpp)."""

import json

import httpx
import pytest
from pydantic import ValidationError

from rune.config.schema import CompatibleEndpoints, LLMConfig
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
    endpoints._resolved.clear()
    yield settings
    endpoints.forget_discovered()
    endpoints._resolved.clear()


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
        _strict({"gpu": entry})
    # A config file keeps its other settings and skips only the bad entry.
    assert LLMConfig(endpoints={"gpu": entry}).endpoints == {}


def _strict(value):
    from pydantic import TypeAdapter

    return TypeAdapter(CompatibleEndpoints).validate_python(value)


def test_endpoint_names_cannot_shadow_providers():
    with pytest.raises(ValidationError, match="provider names"):
        _strict({"ollama": {"baseUrl": "http://localhost:8000"}})
    with pytest.raises(ValidationError):
        _strict({"Has Space": {"baseUrl": "http://localhost:8000"}})
    assert LLMConfig(endpoints={"ollama": {"baseUrl": "http://localhost:8000"}}).endpoints == {}


@pytest.mark.parametrize("url", ["http://localhost:8000", "http://169.254.169.254/v1", "http://10.0.0.5/v1",
                                 "https://10.0.0.5/v1", "https://metadata.google.internal/v1"])
def test_hosted_workers_only_reach_public_https(monkeypatch, url):
    monkeypatch.setenv("RUNE_CLOUD_WORKER", "1")
    with pytest.raises(ValidationError, match="public HTTPS"):
        _strict({"gpu": {"baseUrl": url}})
    assert LLMConfig(endpoints={"gpu": {"baseUrl": url}}).endpoints == {}
    assert _strict({"gpu": {"baseUrl": "https://api.example.com/v1"}})


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
    # vLLM 0.32 without --enable-auto-tool-choice.
    (200, {"choices": [{"finish_reason": "stop", "message": {
        "content": '<tool_call>\n{"name": "ping", "arguments": {}}\n</tool_call>'}}]}, "unparsed"),
    (400, {"error": {"message": '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser'}},
     "unsupported"),
    (400, {"error": {"message": "registry.ollama.ai/library/olmo-3:7b-think does not support tools"}}, "unsupported"),
    # llama.cpp b11960 with --no-jinja.
    (500, {"error": {"code": 500, "message": "tools param requires --jinja flag", "type": "server_error"}},
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
        # The model itself may lack tools, so the server's reason is shown.
        assert body["error"]["message"] in result["detail"]
        assert "--enable-auto-tool-choice" in result["detail"] and "Guided" in result["detail"]


@pytest.mark.parametrize("failure,detail", [
    (httpx.Response(401), "API key was rejected"),
    (httpx.ConnectError("refused"), "ending in /v1"),
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


def test_one_bad_endpoint_does_not_discard_the_llm_section(monkeypatch, tmp_path):
    from rune.config import loader

    path = tmp_path / "config.yaml"
    path.write_text("llm:\n  defaultProvider: anthropic\n  costBudgetUsd: 5\n  endpoints:\n"
                    "    Lab:\n      baseUrl: http://localhost:8000\n"
                    "    gpu:\n      baseUrl: http://10.0.0.9:8000\n")
    monkeypatch.setattr(loader, "_find_config_file", lambda: path)
    settings = loader.load_config(force=True).llm
    assert settings.default_provider == "anthropic" and settings.cost_budget_usd == 5
    assert list(settings.endpoints) == ["gpu"]


@pytest.mark.parametrize("settings,expected", [
    ({}, ""),
    ({"models": ["Qwen/Qwen3-32B"]}, "gpu/Qwen/Qwen3-32B"),
    ({"models": ["m1"], "default": "gpu/m2"}, "gpu/m2"),
    ({"models": ["m1"], "default": "gone/m2"}, "gpu/m1"),
])
def test_tiers_never_borrow_another_providers_model(monkeypatch, settings, expected):
    from rune.agent.failover import build_profiles_from_config
    from rune.config import get_config
    from rune.llm.client import LLMClient
    from rune.llm.model_selection import get_effective_model_selection
    from rune.types import ModelTier

    endpoints.forget_discovered()
    entry = {"baseUrl": "http://10.0.0.9:8000", "models": settings.get("models", [])}
    llm = LLMConfig(defaultProvider="openai_compatible", endpoints={"gpu": entry},
                    **({"defaultModel": settings["default"]} if "default" in settings else {}))
    monkeypatch.setattr(get_config(), "llm", llm)
    assert LLMClient().resolve_model(ModelTier.FAST) == expected
    assert get_effective_model_selection().model == expected
    profiles = build_profiles_from_config()
    # Never the OpenAI fallback chain, even with nothing to select.
    assert [(p.provider, p.model) for p in profiles] == [("openai_compatible", expected)]
    if not expected:
        with pytest.raises(ModelCompatibilityError, match="No model is selected"):
            resolve_target(expected, "openai_compatible")


def test_capability_lookups_stay_quiet_and_use_declared_vision(servers, capsys):
    from rune.agent.model_traits import supports_vision
    from rune.llm.reasoning import reasoning_control

    servers.endpoints["gpu"].vision_models = ["Qwen/Qwen2.5-VL-7B-Instruct"]
    assert supports_vision("openai_compatible/gpu/Qwen/Qwen2.5-VL-7B-Instruct")
    assert not supports_vision("openai_compatible/gpu/Qwen/Qwen3-Coder-30B")
    assert reasoning_control("openai_compatible/gpu/Qwen/Qwen3-Coder-30B").efforts == ()
    assert endpoints.capability_model("openai_compatible/gpu/Qwen/Qwen3-Coder-30B") == "openai/Qwen/Qwen3-Coder-30B"
    assert "Provider List" not in capsys.readouterr().out


@pytest.mark.parametrize("host,private", [
    ("localhost", True), ("gpu-box", True), ("lab.local", True), ("nas.home.arpa", True),
    ("10.0.0.5", True), ("100.101.1.1", True), ("169.254.169.254", True), ("::1", True), ("fd00::1", True),
    ("::ffff:169.254.169.254", True), ("0x7f.1", True), ("2130706433", True), ("127.1", True),
    ("8.8.8.8", False), ("2606:4700::1111", False), ("::ffff:8.8.8.8", False), ("api.example.com", False),
])
def test_private_hosts_follow_the_special_purpose_registry(host, private):
    assert endpoints.private_host(host) is private


@pytest.mark.parametrize("answer,allowed", [("93.184.216.34", True), ("10.0.0.5", False), ("169.254.169.254", False)])
def test_hosted_workers_resolve_names_before_connecting(servers, monkeypatch, answer, allowed):
    import socket

    servers.endpoints["router"].base_url = "https://models.example.com/v1"
    monkeypatch.setenv("RUNE_CLOUD_WORKER", "1")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (answer, 443))])
    if allowed:
        assert resolve_target("router/m", "openai_compatible").kwargs()["api_base"] == "https://models.example.com/v1"
    else:
        with pytest.raises(ModelCompatibilityError, match="public HTTPS"):
            resolve_target("router/m", "openai_compatible")


async def test_reasoning_models_get_room_and_a_cut_off_test_is_inconclusive(servers, monkeypatch):
    sent = []

    async def send(self, request, **kwargs):
        if response := _models_response(request):
            return response
        sent.append(json.loads(request.content))
        return httpx.Response(200, request=request, json={"choices": [{
            "finish_reason": "length", "message": {"content": None, "reasoning": "Let me think..."}}]})

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    result = await endpoints.diagnose("gpu")
    assert result["tools"] == "unknown" and "budget" in result["detail"]
    assert sent[0]["max_tokens"] >= 1024


async def test_pinned_models_are_tested_when_the_server_has_no_model_list(servers, monkeypatch):
    async def send(self, request, **kwargs):
        if request.url.path.endswith("/models"):
            return httpx.Response(404, request=request)
        return httpx.Response(200, request=request, json={"choices": [{"finish_reason": "tool_calls", "message": {
            "tool_calls": [{"id": "1", "type": "function", "function": {"name": "ping", "arguments": "{}"}}]}}]})

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    result = await endpoints.diagnose("router")
    assert result["tools"] == "native" and result["model"] == "qwen/qwen3-coder"
    assert "GET /models returned 404" in result["detail"]


async def test_a_down_server_is_retried_sooner_than_a_healthy_list_refreshes(servers, monkeypatch):
    calls = []
    clock = [1000.0]
    monkeypatch.setattr(endpoints.time, "monotonic", lambda: clock[0])

    async def send(self, request, **kwargs):
        calls.append(request)
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    await endpoints.refresh_discovered()
    clock[0] += 10
    await endpoints.refresh_discovered()
    assert len(calls) == 1
    clock[0] += 25
    await endpoints.refresh_discovered()
    assert len(calls) == 2


def test_an_endpoint_at_ollamas_address_keeps_its_tool_mode(servers, monkeypatch):
    from rune.agent.litellm_adapter import _guided_for_model

    monkeypatch.delenv("RUNE_GUIDED_TOOLS", raising=False)
    servers.endpoints["gpu"].base_url = "http://localhost:11434/v1"
    target = resolve_target("gpu/qwen3-coder:30b", "openai_compatible")
    extra = {**dict(target.options), "_rune_source_model": target.key}
    assert not endpoints.is_ollama_request(target.wire_model, extra)
    assert not _guided_for_model(target.wire_model, True, extra)
    servers.endpoints["gpu"].tool_mode = "guided"
    assert _guided_for_model(target.wire_model, True, extra)
    # The Ollama provider itself is unchanged.
    target = resolve_target("ollama/qwen3-coder:30b")
    assert endpoints.is_ollama_request(target.wire_model, {**dict(target.options), "_rune_source_model": target.key})


@pytest.mark.parametrize("model,anthropic", [
    ("anthropic/claude-opus-5", True), ("claude-sonnet-5", True), ("anthropic/claude-haiku-4-5", True),
    ("openai/Qwen3-14B-Claude-4.5-Opus-Distill", False), ("ollama/claude-distill:8b", False),
    ("openai_compatible/gpu/Qwen3-Claude-Opus-Distill", False), ("openai/anthropic/claude-sonnet-4.5", False),
])
def test_only_anthropic_routes_get_anthropic_request_rewrites(servers, model, anthropic):
    from rune.agent.litellm_adapter import _apply_anthropic_tool_cache
    from rune.agent.model_traits import traits

    assert traits(model).anthropic_wire is anthropic
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    assert ("cache_control" in _apply_anthropic_tool_cache(model, tools)[-1]) is anthropic


async def test_hosted_lookups_are_cached_and_async_callers_leave_the_event_loop(servers, monkeypatch):
    import socket
    import threading

    lookups = []

    def getaddrinfo(*args, **kwargs):
        lookups.append(threading.current_thread() is threading.main_thread())
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    servers.endpoints["router"].base_url = "https://models.example.com/v1"
    monkeypatch.setenv("RUNE_CLOUD_WORKER", "1")
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    await endpoints.check_destination_async("https://models.example.com/v1")
    resolve_target("router/m", "openai_compatible")
    resolve_target("router/m", "openai_compatible")
    assert lookups == [False]
