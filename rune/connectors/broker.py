"""Authenticated, single-use requests over a private Unix socket."""

from __future__ import annotations

import hmac
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from rune.connectors.models import ConnectorRequest
from rune.connectors.store import ConnectorStore, signature
from rune.connectors.transport import send_request


def create_broker(store: ConnectorStore, key: bytes) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.post("/{operation}")
    async def dispatch(operation: str, request: Request):
        if operation not in {"list", "request"}:
            raise HTTPException(404)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 2 * 1024 * 1024:
                raise HTTPException(413)
        nonce = request.headers.get("x-rune-nonce", "")
        try:
            expires = int(request.headers.get("x-rune-expires", "0"))
        except ValueError as exc:
            raise HTTPException(401) from exc
        expected = signature(key, operation, bytes(body), nonce, expires)
        if not hmac.compare_digest(expected, request.headers.get("x-rune-signature", "")):
            raise HTTPException(401)
        if not store.redeem(nonce, expires):
            raise HTTPException(409, "Expired or already used request")
        if operation == "list":
            return {"connectors": store.list()}
        try:
            action = ConnectorRequest.model_validate_json(body)
            policy, secret, revision = store.get(action.connector)
        except (ValidationError, ValueError) as exc:
            raise HTTPException(400, "Invalid connector request") from exc
        if action.revision != revision:
            return {"success": False, "error": "Connector changed; list it again and obtain fresh approval",
                    "metadata": {"action_status": "not_executed"}}
        return asdict(await send_request(policy, secret, action))

    return app
