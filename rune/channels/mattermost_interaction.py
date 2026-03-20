"""Webhook server for Mattermost interactive messages.

Ported from src/channels/mattermost-interaction-server.ts - receives
slash command / outgoing webhook payloads from Mattermost, with
optional token verification and event deduplication.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import parse_qs

from aiohttp import web

from rune.utils.fast_serde import json_decode
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Event type

@dataclass(slots=True)
class MattermostMessageEvent:
    event_id: str
    user_id: str
    user_name: str
    text: str
    timestamp: int = 0
    channel_id: str = ""
    channel_name: str = ""
    team_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


# Handler protocol

class MattermostInteractionHandler(Protocol):
    async def on_message(self, event: MattermostMessageEvent) -> None: ...


# Configuration

@dataclass(slots=True)
class MattermostInteractionServerConfig:
    host: str = "0.0.0.0"
    port: int = 8083
    path: str = "/mattermost"
    verify_token: str = ""
    max_body_bytes: int = 1_000_000
    dedupe_window_sec: float = 300.0
    ack_text: str = "Received. Working on it now."


# Server

class MattermostInteractionServer:
    """HTTP server that receives Mattermost webhook/slash-command payloads."""

    def __init__(
        self,
        config: MattermostInteractionServerConfig,
        handler: MattermostInteractionHandler,
    ) -> None:
        self._config = config
        self._handler = handler
        self._runner: web.AppRunner | None = None
        self._seen_events: dict[str, float] = {}

    async def start(self) -> None:
        if self._runner is not None:
            return

        app = web.Application(client_max_size=self._config.max_body_bytes)
        app.router.add_post(self._config.path, self._handle_request)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._config.host, self._config.port)
        await site.start()
        self._runner = runner
        log.info(
            "Mattermost interaction server started",
            host=self._config.host,
            port=self._config.port,
        )

    async def stop(self) -> None:
        if self._runner is None:
            return
        await self._runner.cleanup()
        self._runner = None
        self._seen_events.clear()
        log.info("Mattermost interaction server stopped")

    @property
    def is_running(self) -> bool:
        return self._runner is not None

    def get_address(self) -> dict[str, Any]:
        return {
            "host": self._config.host,
            "port": self._config.port,
            "path": self._config.path,
        }

    def get_runtime_info(self) -> dict[str, Any]:
        return {"verify_token": bool(self._config.verify_token)}

    # -- request handling ---------------------------------------------------

    async def _handle_request(self, request: web.Request) -> web.Response:
        raw_body = await request.read()
        content_type = (request.content_type or "").lower()

        payload = _parse_mattermost_payload(raw_body, content_type)
        if payload is None:
            return web.json_response({"error": "Invalid payload"}, status=400)

        # Token verification
        if self._config.verify_token and payload.get("token") != self._config.verify_token:
            return web.json_response({"error": "Unauthorized"}, status=401)

        event = _parse_message_event(payload)
        if event is None:
            return web.json_response(
                {"text": self._config.ack_text}, status=200
            )

        if not self._is_duplicate(event.event_id):
            asyncio.create_task(self._handler.on_message(event))

        return web.json_response({"text": self._config.ack_text}, status=200)

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

def _parse_mattermost_payload(
    raw_body: bytes,
    content_type: str,
) -> dict[str, Any] | None:
    """Parse a Mattermost payload from either JSON or form-urlencoded body."""
    if "application/json" in content_type:
        try:
            return json_decode(raw_body)
        except (json.JSONDecodeError, ValueError):
            return None

    if "application/x-www-form-urlencoded" in content_type:
        try:
            parsed = parse_qs(raw_body.decode("utf-8"), keep_blank_values=True)
            return {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}
        except (UnicodeDecodeError, ValueError):
            return None

    # Try JSON as fallback
    try:
        return json_decode(raw_body)
    except (json.JSONDecodeError, ValueError):
        return None


def _parse_message_event(
    payload: dict[str, Any],
) -> MattermostMessageEvent | None:
    """Extract a message event from a Mattermost payload."""
    text = str(payload.get("text") or "").strip()
    if not text:
        return None

    user_id = str(payload.get("user_id") or "")
    user_name = str(payload.get("user_name") or "")
    trigger_word = str(payload.get("trigger_word") or "")

    # For outgoing webhooks, strip the trigger word from the text
    if trigger_word and text.startswith(trigger_word):
        text = text[len(trigger_word):].strip()

    if not text:
        return None

    # Build a stable event id from available fields
    event_id = str(
        payload.get("post_id")
        or payload.get("trigger_id")
        or payload.get("timestamp")
        or ""
    )

    ts = 0
    raw_ts = payload.get("timestamp")
    if raw_ts is not None:
        with contextlib.suppress(ValueError, TypeError):
            ts = int(raw_ts)

    return MattermostMessageEvent(
        event_id=event_id,
        user_id=user_id,
        user_name=user_name,
        text=text,
        timestamp=ts,
        channel_id=str(payload.get("channel_id") or ""),
        channel_name=str(payload.get("channel_name") or ""),
        team_id=str(payload.get("team_id") or ""),
        raw=payload,
    )
