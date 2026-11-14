"""Azure settings checks must not confuse catalog access with deployed inference."""

import json

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from starlette.testclient import TestClient

from rune.api.auth import TokenAuthDependency
from rune.api.azure import AzureConnection, AzureTest, _request, azure_router
from rune.config.schema import LLMConfig

BASE = "https://fixture.services.ai.azure.com"
SECRET = "fixture-secret-never-returned"


@pytest.fixture
def connection(monkeypatch, tmp_path):
    from rune.config import get_config

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("RUNE_CLOUD_WORKER", raising=False)
    monkeypatch.delenv("RUNE_REQUIRE_TOKEN", raising=False)
    monkeypatch.setenv("AZURE_API_BASE", BASE)
    monkeypatch.setenv("AZURE_API_KEY", SECRET)
    settings = LLMConfig(azureDeployments={"daily": "Kimi-K2.7-Code", "reasoner": "gpt-6-sol"})
    monkeypatch.setattr(get_config(), "llm", settings)
    return settings


def mock_response(monkeypatch, status=200, data=None):
    calls = []

    async def send(self, request, **kwargs):
        calls.append(request)
        return httpx.Response(status, request=request, json=data)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    return calls


async def test_catalog_is_read_only_and_does_not_register_models(connection, monkeypatch):
    calls = mock_response(monkeypatch, data={"data": [{"id": "z-model"}, {"id": "a-model"}, {"id": "z-model"}]})
    result = await _request(AzureConnection(endpoint=BASE + "/openai/v1/"))
    assert result["ok"] and result["models"] == ["a-model", "z-model"]
    assert "not a list of your deployments" in result["detail"]
    assert len(calls) == 1 and calls[0].method == "GET"
    assert str(calls[0].url) == BASE + "/openai/v1/models"
    assert calls[0].headers["api-key"] == SECRET
    assert not calls[0].content
    assert connection.azure_deployments == {"daily": "Kimi-K2.7-Code", "reasoner": "gpt-6-sol"}
    assert SECRET not in json.dumps(result)


async def test_draft_key_is_used_without_saving(connection, monkeypatch):
    from rune.llm.azure import endpoint
    from rune.llm.context import model_environment

    calls = mock_response(monkeypatch, data={"data": []})
    req = AzureConnection(endpoint="https://other.openai.azure.com", apiKey="new-secret")
    result = await _request(req)
    assert result["ok"] and result["models"] == []
    assert calls[0].headers["api-key"] == "new-secret"
    assert endpoint() == BASE and model_environment("AZURE_API_KEY") == SECRET
    assert "new-secret" not in repr(req) + req.model_dump_json() + json.dumps(result)


async def test_unsupported_catalog_ids_are_reported_without_becoming_deployments(connection, monkeypatch):
    mock_response(monkeypatch, data={"data": [{"id": "Kimi-K2.7-Code"}, {"id": "legacy:model"}]})
    result = await _request(AzureConnection(endpoint=BASE))
    assert result["models"] == ["Kimi-K2.7-Code"] and "1 model IDs" in result["detail"]


@pytest.mark.parametrize("endpoint", ["http://fixture.services.ai.azure.com", "https://evil.test",
    "https://fixture.services.ai.azure.com.evil.test", "https://127.0.0.1", BASE + ":8443",
    BASE + "/other", BASE + "?key=secret", "https://user:secret@fixture.services.ai.azure.com", BASE + "#fragment"])
async def test_invalid_destinations_never_receive_a_key(connection, monkeypatch, endpoint):
    calls = mock_response(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        await _request(AzureConnection(endpoint=endpoint, apiKey=SECRET))
    assert exc.value.status_code == 400 and not calls
    assert SECRET not in str(exc.value)


async def test_saved_key_cannot_be_forwarded_to_another_resource(connection, monkeypatch):
    calls = mock_response(monkeypatch)
    with pytest.raises(HTTPException, match="different resource"):
        await _request(AzureConnection(endpoint="https://other.services.ai.azure.com"))
    assert not calls


@pytest.mark.parametrize("status", [301, 400, 401, 403, 404, 429, 500])
async def test_provider_errors_are_actionable_and_do_not_expose_raw_bodies(connection, monkeypatch, status):
    calls = mock_response(monkeypatch, status, {"error": {"message": SECRET}, "location": "https://evil.test"})
    result = await _request(AzureConnection(endpoint=BASE))
    assert not result["ok"] and result["statusCode"] == status
    assert SECRET not in json.dumps(result) and len(calls) == 1


@pytest.mark.parametrize("failure", [httpx.ReadTimeout(SECRET), httpx.ConnectError(SECRET)])
async def test_network_errors_do_not_expose_request_secrets(connection, monkeypatch, failure):
    async def send(self, request, **kwargs):
        raise failure

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    result = await _request(AzureConnection(endpoint=BASE))
    assert not result["ok"] and SECRET not in json.dumps(result)


@pytest.mark.parametrize("data", [None, {}, {"data": {}}, {"data": [None]}, {"data": [{"id": 1}]}])
async def test_malformed_catalog_is_not_success(connection, monkeypatch, data):
    mock_response(monkeypatch, data=data)
    result = await _request(AzureConnection(endpoint=BASE))
    assert not result["ok"] and "unexpected" in result["detail"]


@pytest.mark.parametrize("model", ["Kimi-K2.7-Code", "DeepSeek-V4-Pro", "gpt-6-sol"])
async def test_deployment_probe_is_one_bounded_request_to_the_exact_alias(connection, monkeypatch, model):
    responses = model == "gpt-6-sol"
    data = ({"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}]}
            if responses else {"choices": [{"finish_reason": "stop", "message": {"content": "OK"}}]})
    calls = mock_response(monkeypatch, data=data)
    result = await _request(AzureTest(endpoint=BASE), deployment="draft-alias", model=model)
    assert result["ok"] and "not tested" in result["detail"]
    assert len(calls) == 1
    request = calls[0]
    assert request.url.path == ("/openai/v1/responses" if responses else "/openai/v1/chat/completions")
    body = json.loads(request.content)
    assert body["model"] == "draft-alias"
    assert body["max_output_tokens" if responses else "max_tokens"] == 256
    assert "temperature" not in body and "tools" not in body
    assert "draft-alias" not in connection.azure_deployments


@pytest.mark.parametrize("data", [{}, {"choices": []}, {"choices": [{"finish_reason": "length", "message": {"content": ""}}]},
    {"choices": [{"finish_reason": "content_filter", "message": {"content": ""}}]}])
async def test_http_200_is_not_sufficient_for_a_successful_deployment_test(connection, monkeypatch, data):
    mock_response(monkeypatch, data=data)
    result = await _request(AzureTest(endpoint=BASE), deployment="daily")
    assert not result["ok"]


@pytest.mark.parametrize("limits", [{"model_request_limit": 0}, {"cost_budget_usd": 0}, {"cost_budget_usd": 1}])
async def test_inference_limits_block_generation_but_allow_credential_checks(connection, monkeypatch, limits):
    for name, value in limits.items():
        setattr(connection, name, value)
    calls = mock_response(monkeypatch, data={"data": []})
    result = await _request(AzureTest(endpoint=BASE), deployment="daily")
    assert not result["ok"] and not calls
    assert (await _request(AzureConnection(endpoint=BASE)))["ok"]
    assert len(calls) == 1 and calls[0].method == "GET"


def test_routes_authenticate_and_validate_before_sending(connection, monkeypatch):
    calls = mock_response(monkeypatch, data={"data": []})
    app = FastAPI()
    app.include_router(azure_router(TokenAuthDependency()))
    with TestClient(app, client=("203.0.113.10", 1234)) as client:
        assert client.post("/api/v1/azure/models", json={"endpoint": BASE}).status_code == 401
    with TestClient(app, client=("127.0.0.1", 1234)) as client:
        assert client.post("/api/v1/azure/models", json={"endpoint": BASE}, headers={"origin": "https://evil.test"}).status_code == 401
        assert client.post("/api/v1/azure/test", json={"endpoint": BASE, "deployment": "bad/path"}).status_code == 422
        assert not calls
        result = client.post("/api/v1/azure/test", json={"endpoint": BASE}).json()
        assert result["ok"] and "models" not in result
        monkeypatch.setenv("RUNE_CLOUD_WORKER", "1")
        monkeypatch.setattr("rune.api.auth.verify_token", lambda _: True)
        response = client.post("/api/v1/azure/models", json={"endpoint": BASE}, headers={"authorization": "Bearer test"})
        assert response.status_code == 403 and len(calls) == 1


def test_running_app_registers_both_azure_routes(connection):
    from rune.api.server import create_app

    paths = {route.path for route in create_app().routes}
    assert {"/api/v1/azure/models", "/api/v1/azure/test", "/api/v1/azure/settings"} <= paths


async def test_client_disconnect_cancels_the_outbound_check(connection, monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from rune.api import azure as handler

    cancelled = asyncio.Event()

    async def delayed(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def disconnected():
        return True

    monkeypatch.setattr(handler, "_request", delayed)
    with pytest.raises(HTTPException) as error:
        await handler._checked(AzureConnection(endpoint=BASE), SimpleNamespace(is_disconnected=disconnected))
    assert error.value.status_code == 499 and cancelled.is_set()


@pytest.mark.parametrize("hosted", [False, True])
async def test_azure_availability_matches_the_route_guard(connection, monkeypatch, hosted):
    from rune.api.handlers.config import get_config_endpoint

    if hosted:
        monkeypatch.setenv("RUNE_CLOUD_WORKER", "1")
    calls = mock_response(monkeypatch, data={"data": []})
    reason = (await get_config_endpoint()).azure["unavailableReason"]
    if hosted:
        assert reason
        with pytest.raises(HTTPException) as error:
            await _request(AzureConnection(endpoint=BASE))
        assert error.value.status_code == 403 and error.value.detail == reason
        assert not calls
    else:
        assert reason is None
        assert (await _request(AzureConnection(endpoint=BASE)))["ok"]
        assert len(calls) == 1
