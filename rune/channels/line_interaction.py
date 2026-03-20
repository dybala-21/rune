"""Webhook server for LINE postback events (Messaging API).

Ported from src/channels/line-interaction-server.ts - receives
LINE webhook POST requests, verifies the ``X-Line-Signature``
HMAC header, and dispatches message/postback events.
"""

from __future__ import annotations

import asyncio
import base64
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
class LineMessageEvent:
    event_id: str
    sender_id: str
    sender_name: str = ""
    user_id: str = ""
    text: str = ""
    timestamp: int = 0
    reply_token: str = ""
    source_type: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


# Handler protocol

class LineInteractionHandler(Protocol):
    async def on_message(self, event: LineMessageEvent) -> None: ...


# Configuration

@dataclass(slots=True)
class LineInteractionServerConfig:
    host: str = "0.0.0.0"
    port: int = 8082
    path: str = "/line"
    verify_signature: bool = True
    channel_secret: str = ""
    max_body_bytes: int = 1_000_000
    dedupe_window_sec: float = 300.0


# Server

class LineInteractionServer:
    """HTTP server that receives LINE Messaging API webhooks."""

    def __init__(
        self,
        config: LineInteractionServerConfig,
        handler: LineInteractionHandler,
    ) -> None:
        self._config = config
        self._handler = handler
        self._runner: web.AppRunner | None = None
        self._seen_events: dict[str, float] = {}

    async def start(self) -> None:
        if self._runner is not None:
            return
        if self._config.verify_signature and not self._config.channel_secret.strip():
            raise RuntimeError(
                "LINE_CHANNEL_SECRET is required when verify_signature=True"
            )

        app = web.Application(client_max_size=self._config.max_body_bytes)
        app.router.add_post(self._config.path, self._handle_request)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._config.host, self._config.port)
        await site.start()
        self._runner = runner
        log.info(
            "LINE interaction server started",
            host=self._config.host,
            port=self._config.port,
        )

    async def stop(self) -> None:
        if self._runner is None:
            return
        await self._runner.cleanup()
        self._runner = None
        self._seen_events.clear()
        log.info("LINE interaction server stopped")

    @property
    def is_running(self) -> bool:
        return self._runner is not None

    def get_address(self) -> dict[str, Any]:
        return {
            "host": self._config.host,
            "port": self._config.port,
            "path": self._config.path,
        }

    # -- request handling ---------------------------------------------------

    async def _handle_request(self, request: web.Request) -> web.Response:
        raw_body = await request.read()

        # Signature verification
        if self._config.verify_signature:
            sig_header = request.headers.get("X-Line-Signature", "")
            if not self._verify_signature(raw_body, sig_header):
                return web.json_response({"error": "Invalid signature"}, status=403)

        try:
            body: dict[str, Any] = json_decode(raw_body)
        except (json.JSONDecodeError, ValueError):
            return web.json_response({"error": "Invalid JSON"}, status=400)

        events = body.get("events") or []
        for raw_evt in events:
            parsed = _parse_event(raw_evt)
            if parsed and not self._is_duplicate(parsed.event_id):
                asyncio.create_task(self._handler.on_message(parsed))

        return web.json_response({}, status=200)

    # -- signature verification ---------------------------------------------

    def _verify_signature(self, body: bytes, sig_header: str) -> bool:
        """Verify the X-Line-Signature HMAC-SHA256 header."""
        if not sig_header:
            return False
        computed = base64.b64encode(
            hmac.new(
                self._config.channel_secret.encode("utf-8"),
                body,
                hashlib.sha256,
            ).digest()
        ).decode("ascii")
        return hmac.compare_digest(computed, sig_header)

    # -- deduplication ------------------------------------------------------

    def _is_duplicate(self, event_id: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._config.dedupe_window_sec
        self._seen_events = {k: v for k, v in self._seen_events.items() if v > cutoff}
        if event_id in self._seen_events:
            return True
        self._seen_events[event_id] = now
        return False


# Event parsing

def _parse_event(raw: dict[str, Any]) -> LineMessageEvent | None:
    """Parse a LINE webhook event into a :class:`LineMessageEvent`."""
    evt_type = raw.get("type") or ""
    source = raw.get("source") or {}
    user_id = source.get("userId") or ""
    sender_id = source.get("groupId") or source.get("roomId") or user_id
    source_type = source.get("type") or ""

    text = ""
    if evt_type == "message":
        message = raw.get("message") or {}
        if message.get("type") == "text":
            text = message.get("text") or ""
    elif evt_type == "postback":
        postback = raw.get("postback") or {}
        text = postback.get("data") or ""
    else:
        return None

    if not text.strip():
        return None

    event_id = raw.get("webhookEventId") or raw.get("message", {}).get("id") or ""

    return LineMessageEvent(
        event_id=event_id,
        sender_id=sender_id,
        user_id=user_id,
        text=text.strip(),
        timestamp=raw.get("timestamp") or 0,
        reply_token=raw.get("replyToken") or "",
        source_type=source_type,
        raw=raw,
    )
