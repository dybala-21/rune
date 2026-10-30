"""Decision backend isolation, authentication, and saved settings."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from rune.agent import decision_router
from rune.agent.goal_classifier import _TIER2_SYSTEM_PROMPT, classify_goal
from rune.agent.timing import capture_timing, timing_snapshot
from rune.config.schema import MicrosoftDecisionConfig, RuneConfig
from rune.llm import system_one
from rune.llm.decision_backends import MICROSOFT, decision_target
from tests.unit.test_decision_routing import payload


@pytest.fixture
def routing(monkeypatch):
    cfg = RuneConfig()
    cfg.llm.active_provider, cfg.llm.active_model = "xai", "grok-4.6"
    cfg.llm.decision_routing.microsoft = MicrosoftDecisionConfig(
        endpoint="https://test-resource.services.ai.azure.com", deployment="Decision-1")
    monkeypatch.setattr("rune.config.get_config", lambda: cfg)
    monkeypatch.setattr("rune.llm.model_selection.get_config", lambda: cfg)
    monkeypatch.setenv("TYPESAFE_API_KEY", "jev-test-key")
    monkeypatch.setenv("AZURE_DECISION_API_KEY", "decision-test-key")
    monkeypatch.setenv("AZURE_API_KEY", "unrelated-chat-key")
    from tests.unit.test_goal_classifier import response, verdict
    client = SimpleNamespace(completion=AsyncMock(return_value=response(verdict())))
    monkeypatch.setattr("rune.llm.client.get_llm_client", lambda: client)
    return cfg, client


def answer(params):
    return {**payload(params["questions"]), "model": params["target"].model}


@pytest.mark.parametrize("backend", ["connected", "jev", "microsoft"])
async def test_selection_uses_only_one_backend_and_accounts_for_usage(routing, monkeypatch, backend):
    cfg, client = routing
    cfg.llm.decision_routing.backend = backend
    request = AsyncMock(side_effect=lambda **kw: answer(kw))
    monkeypatch.setattr(system_one, "_request", request)
    with capture_timing() as run:
        result = await classify_goal("Explain TCP sockets")
    assert result.available and result.decision_backend == backend
    assert request.await_count == int(backend != "connected")
    assert client.completion.await_count == int(backend == "connected")
    if backend != "connected":
        target = decision_target(cfg.llm.decision_routing)
        assert result.decision_model == target.model
        assert request.call_args.kwargs["api_key"] == ("jev-test-key" if backend == "jev" else "decision-test-key")
        usage = timing_snapshot(run)["usage"]
        assert usage["calls"] == 1 and usage["cost_usd"] == pytest.approx(.000084)
        assert target.billing_model in usage["by_model"]
        assert "test-key" not in str(timing_snapshot(run))


async def test_azure_uses_deployment_name_and_api_key(routing, monkeypatch):
    cfg, _ = routing
    target = decision_target(cfg.llm.decision_routing, "microsoft")
    original = httpx.AsyncClient

    def handle(request):
        assert str(request.url) == target.endpoint
        assert request.headers["api-key"] == "decision-test-key"
        assert "authorization" not in request.headers
        body = json.loads(request.content)
        assert body["model"] == "Decision-1"
        return httpx.Response(200, json={**payload(body["questions"]), "model": MICROSOFT.model})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handle), **kw))
    result = await system_one.classify(_TIER2_SYSTEM_PROMPT, json.dumps({"request_to_classify": "Explain TCP sockets"}),
                                       target=target, timeout=1)
    assert result.values["goal_type"] == "chat"


async def test_azure_auth_failure_does_not_disable_jev_or_another_deployment(routing, monkeypatch):
    cfg, client = routing
    cfg.llm.decision_routing.backend = "microsoft"

    async def request(**kw):
        if kw["target"].backend == "microsoft":
            raise system_one.DecisionUnavailable("http_401")
        return answer(kw)

    external = AsyncMock(side_effect=request)
    monkeypatch.setattr(system_one, "_request", external)
    assert (await classify_goal("Explain TCP sockets")).decision_backend == "connected"
    assert decision_router.accelerator_status() == "auth_error"
    cfg.llm.decision_routing.backend = "jev"
    assert (await classify_goal("Explain TCP sockets")).decision_backend == "jev"
    cfg.llm.decision_routing.backend = "microsoft"
    assert (await classify_goal("Explain TCP sockets")).decision_backend == "connected"
    assert external.await_count == 2 and client.completion.await_count == 2
    cfg.llm.decision_routing.microsoft.deployment = "replacement"
    assert decision_router.accelerator_status() == "unverified"
    monkeypatch.delenv("AZURE_DECISION_API_KEY")
    assert decision_router.accelerator_status() == "missing_key"


@pytest.mark.parametrize("cause", ["uncertain", "http_429", "invalid_model"])
async def test_azure_falls_back_once_without_trying_jev(routing, monkeypatch, cause):
    cfg, client = routing
    cfg.llm.decision_routing.backend = "microsoft"

    async def request(**kw):
        assert kw["target"].backend == "microsoft"
        if cause == "uncertain":
            return {**payload(kw["questions"], goal_type="unknown"), "model": MICROSOFT.model}
        raise system_one.DecisionUnavailable(cause)

    external = AsyncMock(side_effect=request)
    monkeypatch.setattr(system_one, "_request", external)
    result = await classify_goal("Explain TCP sockets")
    assert result.available and result.decision_backend == "connected" and result.fallback_reason == cause
    assert client.completion.await_count == external.await_count == 1


@pytest.mark.parametrize("mode", ["local", "automatic_model"])
async def test_azure_respects_local_and_automatic_routing(routing, monkeypatch, mode):
    cfg, client = routing
    cfg.llm.decision_routing.backend = "microsoft"
    if mode == "local":
        cfg.llm.active_provider = "ollama"
    else:
        cfg.llm.active_model = None
    external = AsyncMock()
    monkeypatch.setattr(system_one, "classify", external)
    assert decision_router.accelerator_status() == mode
    assert (await classify_goal("Explain TCP sockets")).decision_backend == "connected"
    external.assert_not_called()
    assert client.completion.await_count == 1


@pytest.mark.parametrize("endpoint", ["http://test.services.ai.azure.com", "https://test.services.ai.azure.com.evil.test",
    "https://user:pass@test.services.ai.azure.com", "https://test.services.ai.azure.com?key=value",
    "https://test.services.ai.azure.com:8443", "https://test.services.ai.azure.com/openai/v1", "https://127.0.0.1"])
def test_azure_endpoint_rejects_untrusted_destinations(endpoint):
    with pytest.raises(ValidationError):
        MicrosoftDecisionConfig(endpoint=endpoint)


async def test_switching_and_partial_updates_preserve_connections(routing, monkeypatch, tmp_path):
    from ruamel.yaml import YAML

    from rune.api.handlers.config import ConfigPatchRequest, patch_config

    cfg, _ = routing
    path = tmp_path / "config.yaml"
    monkeypatch.setattr("rune.config.writer.config_file_path", lambda: path)
    await patch_config(ConfigPatchRequest(decisionRouting={"backend": "microsoft", "microsoft": {
        "endpoint": "https://test-resource.services.ai.azure.com/providers/microsoft/v1/systemone", "deployment": "Decision-1"}}))
    for backend in ("jev", "connected", "microsoft"):
        await patch_config(ConfigPatchRequest(decisionRouting={"backend": backend}))
    await patch_config(ConfigPatchRequest(decisionRouting={"microsoft": {"deployment": "new-deployment"}}))
    persisted = RuneConfig.model_validate(YAML().load(path)).llm.decision_routing
    assert persisted == cfg.llm.decision_routing
    assert persisted.microsoft.endpoint == "https://test-resource.services.ai.azure.com"
    assert persisted.microsoft.deployment == "new-deployment"
    assert "test-key" not in path.read_text()


async def test_missing_azure_setup_cannot_be_enabled(routing, monkeypatch):
    from rune.api.handlers.config import ConfigPatchRequest, patch_config

    cfg, _ = routing
    def save(updates):
        pytest.fail("Incomplete settings must not be saved")

    monkeypatch.setattr("rune.config.save_config_values", save)
    cfg.llm.decision_routing.microsoft.deployment = ""
    with pytest.raises(HTTPException, match="deployment"):
        await patch_config(ConfigPatchRequest(decisionRouting={"backend": "microsoft"}))
    assert cfg.llm.decision_routing.backend == "connected"
    cfg.llm.decision_routing.microsoft.deployment = "Decision-1"
    monkeypatch.delenv("AZURE_DECISION_API_KEY")
    with pytest.raises(HTTPException, match="API key"):
        await patch_config(ConfigPatchRequest(decisionRouting={"backend": "microsoft"}))


async def test_config_reports_each_connection_without_exposing_keys(routing, monkeypatch):
    from rune.api.handlers.config import get_config_endpoint

    monkeypatch.setattr("rune.llm.reasoning.reasoning_control", lambda _: SimpleNamespace(efforts=(), budgets={}))
    cfg, _ = routing
    cfg.llm.decision_routing.backend = "microsoft"
    snapshot = (await get_config_endpoint()).decision_routing
    assert snapshot["effectiveBackend"] == "microsoft" and snapshot["status"] == "unverified"
    assert all(value["hasKey"] for value in snapshot["providers"].values())
    assert "test-key" not in json.dumps(snapshot)


async def test_azure_timeout_cancels_before_falling_back(routing, monkeypatch):
    cfg, client = routing
    cfg.llm.decision_routing.backend = "microsoft"
    cfg.llm.decision_routing.timeout_ms = 250
    cancelled = asyncio.Event()

    async def slow(**kwargs):
        try:
            await asyncio.sleep(30)
        finally:
            cancelled.set()

    monkeypatch.setattr(system_one, "_request", slow)
    result = await classify_goal("Explain TCP sockets")
    assert cancelled.is_set() and result.fallback_reason == "timeout"
    assert client.completion.await_count == 1
    assert decision_router.accelerator_status() == "cooldown"


async def test_azure_respects_cost_limit_before_sending(routing, monkeypatch):
    from rune.agent.request_budget import BudgetExceeded

    cfg, _ = routing
    request = AsyncMock()
    monkeypatch.setattr(system_one, "_request", request)
    owner = SimpleNamespace(_config=SimpleNamespace(cost_budget_usd=0))
    with capture_timing(owner), pytest.raises(BudgetExceeded):
        await system_one.classify(_TIER2_SYSTEM_PROMPT, json.dumps({"request_to_classify": "Explain TCP sockets"}),
                                  target=decision_target(cfg.llm.decision_routing, "microsoft"), timeout=1)
    request.assert_not_called()


async def test_running_request_keeps_its_connection_when_settings_change(routing, monkeypatch):
    from rune.llm.context import model_scope

    cfg, _ = routing
    cfg.llm.decision_routing.backend = "microsoft"
    request = AsyncMock(side_effect=lambda **kw: answer(kw))
    monkeypatch.setattr(system_one, "_request", request)
    with model_scope():
        cfg.llm.decision_routing.backend = "jev"
        cfg.llm.decision_routing.microsoft.deployment = "replacement"
        monkeypatch.setenv("AZURE_DECISION_API_KEY", "new-key")
        assert (await classify_goal("Explain TCP sockets")).decision_backend == "microsoft"
    params = request.call_args.kwargs
    assert params["target"].deployment == "Decision-1" and params["api_key"] == "decision-test-key"
    cfg.llm.decision_routing.backend = "microsoft"
    assert decision_router.accelerator_status() == "unverified"
