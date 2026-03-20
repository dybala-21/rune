"""Webhook server for WhatsApp interactive messages (Cloud API).

Ported from src/channels/whatsapp-interaction-server.ts - handles
GET verification challenges and POST inbound message events,
with optional HMAC signature verification.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from aiohttp import web

from rune.utils.fast_serde import json_decode
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Event type

@dataclass(slots=True)
class WhatsAppMessageEvent:
    event_id: str
    from_number: str
    text: str
    timestamp: int = 0
    profile_name: str = ""
    phone_number_id: str = ""
    message_type: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


# Handler protocol

class WhatsAppInteractionHandler(Protocol):
    async def on_message(self, event: WhatsAppMessageEvent) -> None: ...


# Configuration

@dataclass(slots=True)
class WhatsAppInteractionServerConfig:
    host: str = "0.0.0.0"
    port: int = 8081
    path: str = "/whatsapp"
    verify_token: str = ""
    verify_signature: bool = False
    app_secret: str = ""
    max_body_bytes: int = 1_000_000
    dedupe_window_sec: float = 300.0


# Server

class WhatsAppInteractionServer:
    """HTTP server that receives WhatsApp Cloud API webhooks."""

    def __init__(
        self,
        config: WhatsAppInteractionServerConfig,
        handler: WhatsAppInteractionHandler,
    ) -> None:
        self._config = config
        self._handler = handler
        self._runner: web.AppRunner | None = None
        self._seen_events: dict[str, float] = {}

    async def start(self) -> None:
        if self._runner is not None:
            return
        if not self._config.verify_token.strip():
            raise RuntimeError("WHATSAPP_VERIFY_TOKEN is required")
        if self._config.verify_signature and not self._config.app_secret.strip():
            raise RuntimeError(
                "WHATSAPP_APP_SECRET is required when verify_signature=True"
            )

        app = web.Application(client_max_size=self._config.max_body_bytes)
        app.router.add_get(self._config.path, self._handle_verify)
        app.router.add_post(self._config.path, self._handle_post)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._config.host, self._config.port)
        await site.start()
        self._runner = runner
        log.info(
            "WhatsApp interaction server started",
            host=self._config.host,
            port=self._config.port,
        )

    async def stop(self) -> None:
        if self._runner is None:
            return
        await self._runner.cleanup()
        self._runner = None
        self._seen_events.clear()
        log.info("WhatsApp interaction server stopped")

    @property
    def is_running(self) -> bool:
        return self._runner is not None

    def get_address(self) -> dict[str, Any]:
        return {
            "host": self._config.host,
            "port": self._config.port,
            "path": self._config.path,
        }

    # -- GET verification ---------------------------------------------------

    async def _handle_verify(self, request: web.Request) -> web.Response:
        mode = request.query.get("hub.mode", "")
        token = request.query.get("hub.verify_token", "")
        challenge = request.query.get("hub.challenge", "")

        if mode == "subscribe" and token == self._config.verify_token:
            return web.Response(text=challenge, content_type="text/plain")
        return web.Response(status=403, text="Forbidden")

    # -- POST inbound events ------------------------------------------------

    async def _handle_post(self, request: web.Request) -> web.Response:
        raw_body = await request.read()

        if self._config.verify_signature:
            sig_header = request.headers.get("X-Hub-Signature-256", "")
            if not self._verify_hmac(raw_body, sig_header):
                return web.json_response({"error": "Invalid signature"}, status=403)

        try:
            body: dict[str, Any] = json_decode(raw_body)
        except (json.JSONDecodeError, ValueError):
            return web.json_response({"error": "Invalid JSON"}, status=400)

        events = _extract_message_events(body)
        for evt in events:
            if not self._is_duplicate(evt.event_id):
                asyncio.create_task(self._handler.on_message(evt))

        return web.Response(text="EVENT_RECEIVED", content_type="text/plain")

    # -- HMAC verification --------------------------------------------------

    def _verify_hmac(self, body: bytes, sig_header: str) -> bool:
        if not sig_header.startswith("sha256="):
            return False
        expected = sig_header[7:]
        computed = hmac.new(
            self._config.app_secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(computed, expected)

    # -- deduplication ------------------------------------------------------

    def _is_duplicate(self, event_id: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._config.dedupe_window_sec
        self._seen_events = {k: v for k, v in self._seen_events.items() if v > cutoff}
        if event_id in self._seen_events:
            return True
        self._seen_events[event_id] = now
        return False


# Payload parsing

def _extract_message_events(body: dict[str, Any]) -> list[WhatsAppMessageEvent]:
    """Extract message events from a WhatsApp Cloud API webhook payload."""
    results: list[WhatsAppMessageEvent] = []
    entries = body.get("entry") or []
    for entry in entries:
        changes = entry.get("changes") or []
        for change in changes:
            value = change.get("value") or {}
            metadata = value.get("metadata") or {}
            phone_number_id = metadata.get("phone_number_id") or ""
            contacts = value.get("contacts") or []
            contact_map: dict[str, str] = {}
            for c in contacts:
                wa_id = c.get("wa_id") or ""
                name = (c.get("profile") or {}).get("name") or ""
                if wa_id:
                    contact_map[wa_id] = name

            messages = value.get("messages") or []
            for msg in messages:
                msg_id = msg.get("id") or ""
                from_number = msg.get("from") or ""
                msg_type = msg.get("type") or "text"
                ts_str = msg.get("timestamp") or "0"
                try:
                    ts = int(ts_str)
                except ValueError:
                    ts = 0

                text = ""
                if msg_type == "text":
                    text = (msg.get("text") or {}).get("body") or ""
                elif msg_type == "button":
                    text = (msg.get("button") or {}).get("text") or ""
                elif msg_type == "interactive":
                    interactive = msg.get("interactive") or {}
                    itype = interactive.get("type") or ""
                    if itype == "button_reply":
                        text = (interactive.get("button_reply") or {}).get("title") or ""
                    elif itype == "list_reply":
                        text = (interactive.get("list_reply") or {}).get("title") or ""

                if not text.strip():
                    continue

                results.append(
                    WhatsAppMessageEvent(
                        event_id=msg_id,
                        from_number=from_number,
                        text=text.strip(),
                        timestamp=ts,
                        profile_name=contact_map.get(from_number, ""),
                        phone_number_id=phone_number_id,
                        message_type=msg_type,
                        raw=msg,
                    )
                )
    return results
