"""A caller's credentials select exactly one worker; requests are never replayed."""

import json
from urllib.parse import urlencode

import httpx
import pytest

from rune.cloud.gateway import create_gateway
from rune.cloud.store import CloudStore


class Body(httpx.AsyncByteStream):
    def __init__(self, data):
        self.data = data

    async def __aiter__(self):
        yield self.data


@pytest.fixture
def tenants(tmp_path):
    store = CloudStore(tmp_path / "cloud")
    alice = store.register("alice", "http://127.0.0.2:18789", "rune_alice_internal")
    bob = store.register("bob", "http://127.0.0.3:18789", "rune_bob_internal")
    return store, alice, bob


async def test_owner_routing_and_header_isolation(tenants):
    store, alice, bob = tenants
    requests = []
    def upstream(request):
        requests.append(request)
        return httpx.Response(200, stream=Body(json.dumps({"host": request.url.host}).encode()), headers={"content-type": "application/json", "set-cookie": "stolen=token", "X-Owner": "untrusted"})

    app = create_gateway(store, "https://rune.example")
    async with app.router.lifespan_context(app):
        await app.state.upstream.aclose()
        app.state.upstream = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://rune.example") as client:
            assert (await client.post("/api/message", json={})).status_code == 401
            first = await client.post("/api/message?sessionId=same", headers={"Authorization": f"Bearer {alice}", "X-Owner": "bob", "Cookie": "secret=client", "X-Forwarded-Host": "evil"}, json={"owner": "bob"})
            second = await client.post("/api/message?sessionId=same", headers={"Authorization": f"Bearer {bob}"}, json={})
            assert first.json()["host"] == "127.0.0.2" and second.json()["host"] == "127.0.0.3"
            assert requests[0].headers["authorization"] == "Bearer rune_alice_internal"
            assert not {"cookie", "x-owner", "x-forwarded-host"} & requests[0].headers.keys()
            assert "set-cookie" not in first.headers and "x-owner" not in first.headers
            assert first.headers["cache-control"] == "no-store"
            assert (await client.get("/api/message?token=oops", headers={"Authorization": f"Bearer {alice}"})).status_code == 400
        await app.state.upstream.aclose()


async def test_cookie_login_csrf_logout_and_revocation(tenants):
    store, alice, _ = tenants
    app = create_gateway(store, "https://rune.example")
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://rune.example") as client:
            page = await client.get("/")
            assert "Access token" in page.text
            body = urlencode({"token": alice})
            assert (await client.post("/cloud/session", content=body, headers={"origin": "https://evil.example"})).status_code == 403
            login = await client.post("/cloud/session", content=body, headers={"origin": "https://rune.example"})
            assert login.status_code == 303
            cookie = login.headers["set-cookie"]
            assert "Secure" in cookie and "HttpOnly" in cookie and "SameSite=strict" in cookie
            assert (await client.post("/api/v1/auth/bootstrap")).status_code == 401
            assert (await client.post("/api/v1/auth/bootstrap", headers={"origin": "https://evil.example"})).status_code == 401
            assert (await client.post("/api/v1/auth/bootstrap", headers={"origin": "https://rune.example"})).status_code == 200
            assert (await client.post("/cloud/logout", headers={"origin": "https://rune.example"})).status_code == 303
            assert (await client.post("/api/v1/auth/bootstrap", headers={"origin": "https://rune.example"})).status_code == 401
            store.disable("alice")
            assert (await client.post("/cloud/session", headers={"authorization": f"Bearer {alice}"})).status_code == 401


async def test_failed_write_is_not_retried_and_response_is_sanitized(tenants):
    store, alice, _ = tenants
    requests = []
    def upstream(request):
        requests.append(request)
        raise httpx.ReadError("rune_alice_internal")

    app = create_gateway(store, "https://rune.example")
    async with app.router.lifespan_context(app):
        await app.state.upstream.aclose()
        app.state.upstream = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://rune.example") as client:
            response = await client.post("/api/message", headers={"authorization": f"Bearer {alice}"}, json={"goal": "send"})
        assert response.status_code == 502 and len(requests) == 1
        assert "may have started" in response.text and "rune_alice_internal" not in response.text
        await app.state.upstream.aclose()


def test_reassignment_rotates_login_and_sessions(tenants):
    store, alice, _ = tenants
    session = store.session("alice")
    replacement = store.register("alice", "http://127.0.0.4:18789", "rune_new")
    assert store.authenticate(alice) is None and store.authenticate(session, session=True) is None
    assert store.authenticate(replacement)["endpoint"] == "http://127.0.0.4:18789"
    assert alice not in store.path.read_bytes().decode(errors="ignore")


def test_machine_reservations_are_persistent_and_unique(tenants):
    store, _, _ = tenants
    alice, bob = store.reserve("alice"), store.reserve("bob")
    assert alice["slot"] != bob["slot"] and alice["name"] != bob["name"]
    assert CloudStore(store.root).reserve("alice") == alice
    assert json.dumps(alice).find("token") == -1


@pytest.mark.parametrize("origin", ["http://rune.example", "https://rune.example/path", "https://user:pass@rune.example", "https://rune.example?token=x"])
def test_gateway_requires_explicit_https_origin(tenants, origin):
    with pytest.raises(ValueError):
        create_gateway(tenants[0], origin)


def test_worker_websocket_never_follows_redirects():
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    from rune.cloud.gateway import WorkerConnection
    error = InvalidStatus(Response(302, "Found", Headers({"Location": "wss://elsewhere.example/ws"})))
    assert WorkerConnection("ws://127.0.0.1/ws").process_redirect(error) is error


async def test_bootstrap_storage_scope_follows_owner_and_worker_credentials(tenants):
    store, alice, bob = tenants
    app = create_gateway(store, "https://rune.example")
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://rune.example") as client:
            async def bootstrap(token):
                return await client.post("/api/v1/auth/bootstrap", headers={"Authorization": f"Bearer {token}"})

            first = await bootstrap(alice)
            assert first.headers["cache-control"] == "no-store"
            scope = first.json()["storageScope"]
            assert len(scope) == 64 and all(character in "0123456789abcdef" for character in scope)
            assert (await bootstrap(alice)).json()["storageScope"] == scope
            assert (await bootstrap(bob)).json()["storageScope"] != scope
            assert "rune_alice_internal" not in first.text and "alice" not in scope
            assert (await bootstrap("invalid")).status_code == 401
            replacement = store.register("alice", "http://127.0.0.4:18789", "rune_new")
            assert (await bootstrap(replacement)).json()["storageScope"] != scope


async def test_account_switch_cannot_send_old_tabs_actions_to_another_worker(tenants):
    store, alice, bob = tenants
    requests = []

    def upstream(request):
        requests.append(request)
        return httpx.Response(200, stream=Body(b'{}'), headers={"content-type": "application/json"})

    app = create_gateway(store, "https://rune.example")
    async with app.router.lifespan_context(app):
        await app.state.upstream.aclose()
        app.state.upstream = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://rune.example") as client:
            first = await client.post("/api/v1/auth/bootstrap", headers={"Authorization": f"Bearer {alice}"})
            alice_scope = first.json()["storageScope"]
            headers = {"Authorization": f"Bearer {bob}", "X-Rune-Account": alice_scope}
            bootstrap = await client.post("/api/v1/auth/bootstrap", headers=headers)
            assert bootstrap.status_code == 200
            bob_scope = bootstrap.json()["storageScope"]
            assert bob_scope != alice_scope
            assert (await client.post("/api/message", json={"goal": "send"}, headers=headers)).status_code == 401
            for query in [alice_scope, f"{bob_scope}&accountScope={alice_scope}", ""]:
                response = await client.get(f"/api/events?accountScope={query}", headers={"Authorization": f"Bearer {bob}"})
                assert response.status_code == 401
            assert requests == []
            headers["X-Rune-Account"] = bob_scope
            assert (await client.post("/api/message", json={}, headers=headers)).status_code == 200
            assert len(requests) == 1 and "x-rune-account" not in requests[0].headers
        await app.state.upstream.aclose()
