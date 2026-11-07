"""Controller-side broker client. Never retry a request after uncertain delivery.

The controller only holds a signing key, never a connector credential. The broker is
reached over a local Unix socket by default. When ``RUNE_CONNECTOR_URL`` is set, the
broker runs outside this machine (for example on a hosted VM's host), and the socket
is not consulted at all, so a stale local broker can never shadow the remote one.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

import httpx

from rune.connectors.store import authority_key, broker_home, signature
from rune.utils.logger import get_logger

log = get_logger(__name__)


class BrokerUnavailable(RuntimeError):
    pass


def remote_broker() -> str | None:
    """Return the configured remote broker origin, or None for the local socket broker."""
    value = os.environ.get("RUNE_CONNECTOR_URL", "").strip()
    if not value:
        return None
    from rune.utils.service_url import service_url

    try:
        return service_url(value, allow_path=False)
    except ValueError as exc:
        raise BrokerUnavailable(f"RUNE_CONNECTOR_URL is not usable: {exc}") from exc


def hosted_approval() -> bool:
    """Whether a remote broker decides connector approvals, outside this machine.

    A local install approves each call here. A hosted worker can't be trusted to: the broker
    keeps the request until the owner approves it on a page this worker can't reach.
    """
    try:
        return remote_broker() is not None
    except BrokerUnavailable:
        return False


def _socket(root: Path) -> Path:
    return Path(os.environ.get("RUNE_CONNECTOR_SOCKET", str(root / "broker.sock")))


def broker_available() -> bool:
    root = broker_home()
    try:
        remote = remote_broker()
    except BrokerUnavailable as exc:
        log.warning("connector_broker_misconfigured", error=str(exc))
        return False
    if remote:
        # Listing is answered by the remote broker; locally we can only check the signing key.
        return (root / "controller.key").is_file()
    return (root / "configured").is_file() and _socket(root).is_socket()


async def call_broker(operation: str, params: dict[str, Any]) -> dict[str, Any]:
    root = broker_home()
    remote = remote_broker()
    if remote:
        transport = httpx.AsyncHTTPTransport(retries=0)
        base = remote
    else:
        socket = _socket(root)
        if not socket.is_socket():
            raise BrokerUnavailable("Connector broker is not running; use rune connector serve")
        transport = httpx.AsyncHTTPTransport(uds=str(socket), retries=0)
        base = "http://broker"
    try:
        key = authority_key(root)
    except (OSError, ValueError) as exc:
        raise BrokerUnavailable("Connector authority is unavailable") from exc
    body = json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
    nonce, expires = secrets.token_hex(32), int(time.time()) + 45
    headers = {"x-rune-nonce": nonce, "x-rune-expires": str(expires),
               "x-rune-signature": signature(key, operation, body, nonce, expires),
               "content-type": "application/json"}
    if key_id := os.environ.get("RUNE_CONNECTOR_KEY_ID", "").strip():
        # Lets a multi-tenant broker pick the verification key. Not secret: a wrong id only fails the HMAC.
        headers["x-rune-key-id"] = key_id
    # trust_env=False keeps the browsing-lane proxy (HTTP(S)_PROXY) out of the credentialed path.
    async with httpx.AsyncClient(transport=transport, trust_env=False, timeout=40, follow_redirects=False) as client:
        response = await client.post(f"{base}/{operation}", content=body, headers=headers)
        # Any non-2xx, including a redirect, is uncertain delivery for writes; callers report it as such.
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result
