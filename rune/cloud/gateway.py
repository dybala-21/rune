"""Authenticate once, then stream to that owner's controller without replaying writes."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from rune.cloud.store import CloudStore

_COOKIE = "__Host-rune_session"
_HEADERS = {"content-type", "accept", "x-client-id", "last-event-id", "range", "if-none-match"}
_RESPONSE_HEADERS = {"content-type", "content-disposition", "content-range", "content-encoding", "etag", "accept-ranges"}
_LOGIN = """<!doctype html><html lang="en"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rune sign in</title><body><main><h1>Rune</h1><form method="post" action="/cloud/session">
<label>Access token <input type="password" name="token" required autocomplete="current-password"></label>
<button type="submit">Sign in</button></form></main></body></html>"""


class WorkerConnection(connect):
    def process_redirect(self, exc: Exception) -> Exception:
        return exc


def create_gateway(store: CloudStore, public_origin: str) -> FastAPI:
    origin = urlsplit(public_origin)
    if origin.scheme != "https" or not origin.netloc or origin.path or origin.query or origin.fragment or origin.username:
        raise ValueError("Public origin must be an HTTPS origin without a path")

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False,
                                     timeout=httpx.Timeout(15, read=None),
                                     limits=httpx.Limits(max_connections=200, max_keepalive_connections=50)) as client:
            app.state.upstream = client
            yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def identity(request: Request | WebSocket) -> dict | None:
        header = request.headers.get("authorization", "")
        if header:
            scheme, _, token = header.partition(" ")
            return store.authenticate(token) if scheme.lower() == "bearer" else None
        token = request.cookies.get(_COOKIE, "")
        if not token:
            return None
        supplied = request.headers.get("origin")
        unsafe = isinstance(request, WebSocket) or request.method not in {"GET", "HEAD"}
        if (unsafe and supplied != public_origin) or supplied and supplied != public_origin:
            return None
        return store.authenticate(token, session=True)

    @app.post("/cloud/session")
    async def login(request: Request):
        if request.headers.get("origin") not in {None, public_origin}:
            raise HTTPException(403)
        body = await bounded_body(request, 4096)
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not token:
            # Browser form logins must have a same-origin submission.
            if request.headers.get("origin") != public_origin:
                raise HTTPException(403)
            token = parse_qs(body.decode()).get("token", [""])[0]
        worker = store.authenticate(token)
        if worker is None:
            raise HTTPException(401, "Invalid access token")
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(_COOKIE, store.session(worker["owner"]), secure=True, httponly=True,
                            samesite="strict", max_age=8 * 3600, path="/")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/cloud/logout")
    async def logout(request: Request):
        if identity(request) is None:
            raise HTTPException(401)
        store.logout(request.cookies.get(_COOKIE, ""))
        response = RedirectResponse("/", status_code=303)
        response.delete_cookie(_COOKIE, secure=True, httponly=True, samesite="strict")
        return response

    @app.api_route("/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def forward(path: str, request: Request):
        worker = identity(request)
        if worker is None:
            if path == "" and request.method == "GET":
                return HTMLResponse(_LOGIN, headers={"Cache-Control": "no-store", "Content-Security-Policy": "default-src 'none'; form-action 'self'; frame-ancestors 'none'"})
            raise HTTPException(401, "Sign in to Rune")
        if path == "api/v1/auth/bootstrap":
            return {"ok": True}
        if "token" in request.query_params:
            raise HTTPException(400, "Tokens must not appear in URLs")
        body = await bounded_body(request, 32 * 1024 * 1024)
        url = httpx.URL(worker["endpoint"]).copy_with(raw_path=request.scope["raw_path"] +
                                                     (b"?" + request.scope["query_string"] if request.scope["query_string"] else b""))
        headers = {k: v for k, v in request.headers.items() if k in _HEADERS}
        headers.update(authorization=f"Bearer {worker['upstream_token']}", **{"accept-encoding": "identity"})
        try:
            async with asyncio.timeout(30):
                upstream = await app.state.upstream.send(app.state.upstream.build_request(request.method, url, headers=headers, content=body), stream=True)
        except (httpx.HTTPError, TimeoutError) as exc:
            raise HTTPException(502, "Worker connection failed; a submitted action may have started. Check its run before retrying.") from exc

        async def chunks():
            checked_at = time.monotonic()
            try:
                async with asyncio.timeout(max(0, worker.get("expires", time.time() + 8 * 3600) - time.time())):
                    async for chunk in upstream.aiter_raw():
                        if time.monotonic() - checked_at >= 1:
                            if identity(request) is None:
                                return
                            checked_at = time.monotonic()
                        yield chunk
            except TimeoutError:
                return
            finally:
                await upstream.aclose()

        response_headers = {k: v for k, v in upstream.headers.items() if k in _RESPONSE_HEADERS}
        response_headers.update({"Cache-Control": "no-store", "X-Accel-Buffering": "no", "X-Content-Type-Options": "nosniff"})
        if upstream.is_redirect:
            await upstream.aclose()
            return Response("Worker redirect refused", status_code=502)
        return StreamingResponse(chunks(), status_code=upstream.status_code, headers=response_headers)

    @app.websocket("/ws")
    async def websocket(socket: WebSocket):
        worker = identity(socket)
        if worker is None:
            await socket.close(code=4401)
            return
        base = urlsplit(worker["endpoint"])
        url = f"{'wss' if base.scheme == 'https' else 'ws'}://{base.netloc}/ws"
        tasks = []
        try:
            async with WorkerConnection(url, proxy=None, open_timeout=15, max_size=2 * 1024 * 1024,
                                        additional_headers={"Authorization": f"Bearer {worker['upstream_token']}"}) as upstream:
                await socket.accept()

                async def to_worker():
                    while True:
                        message = await socket.receive()
                        if message["type"] == "websocket.disconnect" or identity(socket) is None:
                            return
                        await upstream.send(message.get("text") if message.get("text") is not None else message["bytes"])

                async def to_client():
                    async for message in upstream:
                        if identity(socket) is None:
                            return
                        await (socket.send_text(message) if isinstance(message, str) else socket.send_bytes(message))

                tasks = [asyncio.create_task(to_worker()), asyncio.create_task(to_client())]
                await asyncio.wait(tasks, timeout=max(0, worker.get("expires", time.time() + 8 * 3600) - time.time()), return_when=asyncio.FIRST_COMPLETED)
        except (OSError, TimeoutError, WebSocketException):
            if socket.application_state.name != "DISCONNECTED":
                await socket.close(code=1013, reason="Worker connection unavailable")
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if socket.application_state.name == "CONNECTED":
                await socket.close()

    return app


async def bounded_body(request: Request, limit: int) -> bytes:
    body = bytearray()
    try:
        async with asyncio.timeout(60):
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > limit:
                    raise HTTPException(413)
    except TimeoutError as exc:
        raise HTTPException(408, "Request body timed out before forwarding") from exc
    return bytes(body)
