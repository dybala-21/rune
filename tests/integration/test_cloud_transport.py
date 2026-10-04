"""Exercise real local sockets; no model provider or user credentials are needed."""

import asyncio
import contextlib
import json
import socket
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import StreamingResponse
from websockets.asyncio.client import connect

from rune.capabilities.connector import ConnectorListParams, connector_list, connector_request
from rune.cloud.gateway import create_gateway
from rune.cloud.store import CloudStore
from rune.connectors.broker import create_broker
from rune.connectors.models import ConnectorPolicy, ConnectorRequest
from rune.connectors.store import ConnectorStore, authority_key
from rune.safety.approval_context import approval_granted
from rune.types import CapabilityResult


@asynccontextmanager
async def serving(app, *, uds=None):
    listener = None
    config = uvicorn.Config(app, host="127.0.0.1", port=0, uds=str(uds) if uds else None, log_level="error", access_log=False)
    server = uvicorn.Server(config)
    server.capture_signals = contextlib.nullcontext
    if not uds:
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    task = asyncio.create_task(server.serve(sockets=[listener] if listener else None))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("Server exited before startup")
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{port}" if listener else str(uds)
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, 5)
        except TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if listener:
            listener.close()


async def test_tool_to_separate_broker_socket(tmp_path, monkeypatch):
    root = tmp_path / "broker"
    monkeypatch.setenv("RUNE_BROKER_HOME", str(root))
    store = ConnectorStore(root)
    store.put(ConnectorPolicy(name="mail", origin="https://api.example.com", methods=["POST"]), "synthetic-credential-only")
    requests = []
    async def send(policy, secret, request):
        requests.append((policy, secret, request))
        return CapabilityResult(success=True, output='{"id":"draft-1"}')
    monkeypatch.setattr("rune.connectors.broker.send_request", send)
    # Unix socket paths have a much shorter limit than filesystem paths.
    with tempfile.TemporaryDirectory(prefix="rn-sock-", dir="/tmp") as directory:
        socket_path = Path(directory) / "broker.sock"
        monkeypatch.setenv("RUNE_CONNECTOR_SOCKET", str(socket_path))
        async with serving(create_broker(store, authority_key(root, create=True)), uds=socket_path):
            listed = await connector_list(ConnectorListParams())
            assert listed.success and "synthetic-credential-only" not in listed.output
            revision = json.loads(listed.output)["connectors"][0]["revision"]
            request = ConnectorRequest(connector="mail", origin="https://api.example.com", revision=revision, path="/drafts", method="POST", body='{"text":"Hello"}')
            assert not (await connector_request(request)).success
            with approval_granted():
                result = await connector_request(request)
            assert result.success and json.loads(result.output)["id"] == "draft-1"
            assert len(requests) == 1 and requests[0][1] == "synthetic-credential-only"


async def test_real_gateway_streams_without_buffering_and_keeps_owners_separate(tmp_path):
    release = asyncio.Event()
    worker = FastAPI()
    requests = []

    @worker.get("/events")
    async def events(request: Request):
        requests.append(request.headers.get("authorization"))
        async def stream():
            yield "data: first\n\n"
            await release.wait()
            yield "data: last\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream")

    @worker.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        try:
            text = await socket.receive_text()
            await socket.send_json({"token": socket.headers["authorization"].removeprefix("Bearer "), "text": text})
        finally:
            await socket.close()

    store = CloudStore(tmp_path / "cloud")
    async with serving(worker) as first, serving(worker) as second:
        alice = store.register("alice", first, "rune_alice")
        bob = store.register("bob", second, "rune_bob")
        app = create_gateway(store, "https://rune.example")
        async with serving(app) as gateway:
            async with httpx.AsyncClient(timeout=3) as client:
                started = time.monotonic()
                async with client.stream("GET", gateway + "/events", headers={"authorization": f"Bearer {alice}"}) as response:
                    lines = response.aiter_lines()
                    assert await anext(lines) == "data: first"
                    assert time.monotonic() - started < 2
                    release.set()
                    assert "data: last" in [line async for line in lines]
            for token, internal in ((alice, "rune_alice"), (bob, "rune_bob")):
                async with connect(gateway.replace("http:", "ws:") + "/ws", additional_headers={"Authorization": f"Bearer {token}"}, proxy=None) as socket:
                    await socket.send("continue")
                    result = json.loads(await socket.recv())
                    assert result == {"token": internal, "text": "continue"}
            store.disable("alice")
            async with httpx.AsyncClient() as client:
                denied = await client.get(gateway + "/events", headers={"authorization": f"Bearer {alice}"})
                assert denied.status_code == 401
    assert requests == ["Bearer rune_alice"]
