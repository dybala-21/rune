"""Controller-side broker client. Never retry a request after uncertain delivery."""

from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path

import httpx

from rune.connectors.store import authority_key, broker_home, signature


class BrokerUnavailable(RuntimeError):
    pass


def broker_available() -> bool:
    root = broker_home()
    return (root / "configured").is_file() and Path(os.environ.get("RUNE_CONNECTOR_SOCKET", str(root / "broker.sock"))).is_socket()


async def call_broker(operation: str, params: dict) -> dict:
    root = broker_home()
    socket = Path(os.environ.get("RUNE_CONNECTOR_SOCKET", str(root / "broker.sock")))
    if not socket.is_socket():
        raise BrokerUnavailable("Connector broker is not running; use rune connector serve")
    try:
        key = authority_key(root)
    except (OSError, ValueError) as exc:
        raise BrokerUnavailable("Connector authority is unavailable") from exc
    body = json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
    nonce, expires = secrets.token_hex(32), int(time.time()) + 45
    headers = {"x-rune-nonce": nonce, "x-rune-expires": str(expires),
               "x-rune-signature": signature(key, operation, body, nonce, expires),
               "content-type": "application/json"}
    transport = httpx.AsyncHTTPTransport(uds=str(socket), retries=0)
    async with httpx.AsyncClient(transport=transport, trust_env=False, timeout=40) as client:
        response = await client.post(f"http://broker/{operation}", content=body, headers=headers)
        response.raise_for_status()
        return response.json()
