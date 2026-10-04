"""Credentials must stay behind the broker's approval and destination boundaries."""

import asyncio
import json
import socket
import time
from dataclasses import asdict
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError

from rune.capabilities.connector import connector_request, register_connector_capabilities
from rune.capabilities.registry import CapabilityRegistry
from rune.connectors import transport
from rune.connectors.broker import create_broker
from rune.connectors.models import ConnectorPolicy, ConnectorRequest
from rune.connectors.store import ConnectorStore, authority_key, signature
from rune.safety.approval_context import approval_granted
from rune.types import CapabilityResult


@pytest.fixture
def configured(tmp_path):
    store = ConnectorStore(tmp_path / "broker")
    policy = ConnectorPolicy(name="mail", origin="https://api.example.com", path_prefix="/v1/messages", methods=["GET", "POST"])
    store.put(policy, "test-secret-never-return")
    key = authority_key(tmp_path / "broker", create=True)
    return store, policy, key


def action(store, **kwargs):
    return ConnectorRequest(connector="mail", origin="https://api.example.com", revision=store.list()[0]["revision"], path="/v1/messages", **kwargs)


def signed(key, operation, body, nonce="a" * 64, expires=None):
    expires = expires or int(time.time()) + 30
    return {"x-rune-nonce": nonce, "x-rune-expires": str(expires),
            "x-rune-signature": signature(key, operation, body, nonce, expires)}


@pytest.mark.parametrize("origin", ["http://api.example.com", "https://user:pass@example.com", "https://example.com/path", "https://example.com:444", "https://example.com#x", "https://example.com./"])
def test_connector_origin_is_exact(origin):
    with pytest.raises(ValidationError):
        ConnectorPolicy(name="test", origin=origin, methods=["GET"])


@pytest.mark.parametrize("path", ["//evil.test/x", "/v1/../admin", "/v1/%2e%2e/admin", "/v1/%252fadmin", "/v1\\admin", "/v1?q=x", "https://evil.test", "/v1/./messages"])
def test_ambiguous_paths_are_rejected(path):
    with pytest.raises(ValidationError):
        ConnectorRequest(connector="test", origin="https://api.example.com", revision="a" * 32, path=path)


async def test_signature_replay_and_policy_change(configured, monkeypatch):
    store, policy, key = configured
    send = AsyncMock(return_value=CapabilityResult(success=True, output="sent"))
    monkeypatch.setattr("rune.connectors.broker.send_request", send)
    app = create_broker(store, key)
    body = action(store, method="POST", body='{"reply":"Hello"}').model_dump_json().encode()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://broker") as client:
        bad = await client.post("/request", content=body + b" ", headers=signed(key, "request", body))
        assert bad.status_code == 401 and send.await_count == 0
        first = await client.post("/request", content=body, headers=signed(key, "request", body))
        assert first.json()["success"]
        # Replay protection survives broker process restart.
        restarted = ConnectorStore(store.path.parent)
        assert not restarted.redeem("a" * 64, int(time.time()) + 30)
        second = await client.post("/request", content=body, headers=signed(key, "request", body))
        assert second.status_code == 409 and send.await_count == 1
        store.put(policy, "test-replacement-secret")
        changed = await client.post("/request", content=body, headers=signed(key, "request", body, "b" * 64))
        assert changed.json()["metadata"]["action_status"] == "not_executed"
        assert send.await_count == 1
        listed = await client.post("/list", content=b"{}", headers=signed(key, "list", b"{}", "c" * 64))
        assert "secret" not in listed.text
        assert listed.json()["connectors"][0]["origin"] == policy.origin


async def test_expired_and_cross_operation_signatures(configured):
    store, _, key = configured
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_broker(store, key)), base_url="http://broker") as client:
        assert (await client.post("/list", content=b"{}", headers=signed(key, "request", b"{}"))).status_code == 401
        assert (await client.post("/list", content=b"{}", headers=signed(key, "list", b"{}", expires=int(time.time()) - 1))).status_code == 409


async def test_registry_requires_exact_one_use_approval(configured, monkeypatch):
    store, _, _ = configured
    monkeypatch.setenv("RUNE_APPROVAL_MODE", "standard")
    call = AsyncMock(return_value=asdict(CapabilityResult(success=True, output="ok")))
    monkeypatch.setattr("rune.capabilities.connector.call_broker", call)
    registry = CapabilityRegistry()
    register_connector_capabilities(registry)
    params = action(store, method="POST").model_dump()
    assert not (await registry.execute("connector_request", params)).success
    assert not (await connector_request(ConnectorRequest(**params))).success
    with approval_granted("connector_request", {**params, "body": "different"}):
        assert not (await registry.execute("connector_request", params)).success
    with approval_granted("connector_request", params):
        assert (await registry.execute("connector_request", params)).success
        assert not (await registry.execute("connector_request", params)).success
    assert call.await_count == 1


@pytest.mark.parametrize("addresses", [["127.0.0.1"], ["169.254.169.254"], ["::1"], ["::ffff:127.0.0.1"], ["8.8.8.8", "10.0.0.1"], ["224.0.0.1"]])
async def test_private_and_mixed_dns_are_blocked(addresses, monkeypatch):
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "getaddrinfo", AsyncMock(return_value=[(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 443)) for ip in addresses]))
    with pytest.raises(ValueError):
        await transport.public_address("api.example.com")


async def test_pinned_tls_no_redirects_or_secret_echo(configured, monkeypatch):
    store, policy, _ = configured
    secret = store.get("mail")[1]
    clients = []
    seen = []
    reply = {"status": 200}

    def handle(request):
        seen.append(request)
        return httpx.Response(reply["status"], headers={"Location": "https://attacker.example"}, text=f"result: {secret}")

    original = httpx.AsyncClient
    def factory(**kwargs):
        clients.append(kwargs)
        return original(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(transport, "public_address", AsyncMock(return_value="93.184.215.14"))
    monkeypatch.setattr(transport.httpx, "AsyncClient", factory)
    result = await transport.send_request(policy, secret, action(store))
    assert result.success and secret not in result.output and "[redacted]" in result.output
    assert seen[0].url.host == "93.184.215.14"
    assert seen[0].headers["host"] == "api.example.com"
    assert seen[0].extensions["sni_hostname"] == "api.example.com"
    assert seen[0].headers["authorization"] == f"Bearer {secret}"
    assert clients[0]["trust_env"] is False and clients[0]["follow_redirects"] is False
    reply["status"] = 302
    result = await transport.send_request(policy, secret, action(store))
    assert not result.success and len(seen) == 2 and secret not in json.dumps(asdict(result))
    wrong = action(store).model_copy(update={"path": "/v1/messages-other"})
    assert (await transport.send_request(policy, secret, wrong)).metadata["action_status"] == "not_executed"
    assert len(seen) == 2
    wrong_origin = action(store).model_copy(update={"origin": "https://other.example.com"})
    assert not (await transport.send_request(policy, secret, wrong_origin)).success
    assert len(seen) == 2
    reply["status"] = 202
    pending = await transport.send_request(policy, secret, action(store, method="POST"))
    assert not pending.success and pending.metadata["action_status"] == "unknown"
    for status in (302, 500, 409):
        reply["status"] = status
        count = len(seen)
        failed = await transport.send_request(policy, secret, action(store, method="POST"))
        assert not failed.success and failed.metadata["action_status"] == "unknown"
        assert len(seen) == count + 1


async def test_broker_transport_failure_does_not_retry_or_leak(configured, monkeypatch):
    store, _, _ = configured
    call = AsyncMock(side_effect=httpx.ReadTimeout("test-secret-never-return"))
    monkeypatch.setattr("rune.capabilities.connector.call_broker", call)
    with approval_granted():
        result = await connector_request(action(store, method="POST"))
    assert result.metadata["action_status"] == "unknown" and call.await_count == 1
    assert "test-secret-never-return" not in result.error


def test_private_files_and_no_plaintext_listing(configured):
    store, _, _ = configured
    assert store.path.stat().st_mode & 0o077 == 0
    assert store.path.parent.stat().st_mode & 0o077 == 0
    assert "test-secret-never-return" not in json.dumps(store.list())
    key = store.path.parent / "controller.key"
    key.chmod(0o644)
    with pytest.raises(ValueError):
        authority_key(key.parent)


def test_empty_broker_does_not_add_model_tools(configured, monkeypatch):
    from rune.connectors.client import broker_available
    store, _, _ = configured
    monkeypatch.setenv("RUNE_BROKER_HOME", str(store.path.parent))
    assert not broker_available()
    store.remove("mail")
    assert not (store.path.parent / "configured").exists()
