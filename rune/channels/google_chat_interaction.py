"""Webhook server for Google Chat interactive cards.

Ported from src/channels/google-chat-interaction-server.ts - receives
interaction events from the Google Chat App HTTP endpoint, returns a
synchronous ACK, and forwards MESSAGE/ACTION events for async handling.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from aiohttp import web

from rune.channels.google_chat_security import verify_google_chat_bearer_token
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Event types

@dataclass(slots=True)
class GoogleChatMessageEvent:
    event_id: str
    space_name: str
    thread_name: str
    user_id: str
    user_name: str
    text: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GoogleChatActionEvent:
    event_id: str
    event_type: str
    space_name: str
    thread_name: str
    user_id: str
    user_name: str
    action_method_name: str
    parameters: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


# Handler protocol

class GoogleChatInteractionHandler(Protocol):
    async def on_message(self, event: GoogleChatMessageEvent) -> None: ...
    async def on_action(self, event: GoogleChatActionEvent) -> None: ...


# Configuration

@dataclass(slots=True)
class GoogleChatInteractionServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    path: str = "/google-chat"
    verify_requests: bool = True
    audience: str = ""
    max_body_bytes: int = 1_000_000
    dedupe_window_sec: float = 300.0
    message_ack_text: str = ""
    added_to_space_text: str = "RUNE is connected. Mention me or send a DM to start."


# Server

class GoogleChatInteractionServer:
    """HTTP server that receives Google Chat interaction webhooks."""

    def __init__(
        self,
        config: GoogleChatInteractionServerConfig,
        handler: GoogleChatInteractionHandler,
    ) -> None:
        self._config = config
        self._handler = handler
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._seen_events: dict[str, float] = {}
        self._background_tasks: set[asyncio.Task[None]] = set()

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        if self._runner is not None:
            return

        if self._config.verify_requests and not self._config.audience:
            raise RuntimeError(
                "GOOGLE_CHAT_AUTH_AUDIENCE is required when verify_requests is enabled"
            )

        app = web.Application(client_max_size=self._config.max_body_bytes)
        app.router.add_post(self._config.path, self._handle_request)
        self._app = app

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._config.host, self._config.port)
        await site.start()
        self._runner = runner

        log.info(
            "Google Chat interaction server started",
            host=self._config.host,
            port=self._config.port,
            path=self._config.path,
        )

    async def stop(self) -> None:
        if self._runner is None:
            return
        await self._runner.cleanup()
        self._runner = None
        self._app = None
        self._seen_events.clear()
        log.info("Google Chat interaction server stopped")

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
        # Verify bearer token
        if self._config.verify_requests:
            auth_header = request.headers.get("Authorization", "")
            token = _extract_bearer(auth_header)
            if not token:
                return _json_response(401, {"error": "Missing bearer token"})

            result = await verify_google_chat_bearer_token(
                token, audience=self._config.audience
            )
            if not result.ok:
                return _json_response(403, {"error": result.reason or "Forbidden"})

        try:
            body: dict[str, Any] = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _json_response(400, {"error": "Invalid JSON"})

        event_type = body.get("type") or body.get("eventType") or ""

        # ADDED_TO_SPACE
        if event_type == "ADDED_TO_SPACE":
            return _json_response(200, {"text": self._config.added_to_space_text})

        # CARD_CLICKED / ACTION
        if event_type in ("CARD_CLICKED",) and body.get("action"):
            action_event = _parse_action_event(body)
            if action_event and not self._is_duplicate(action_event.event_id):
                task = asyncio.create_task(self._handler.on_action(action_event))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            return _json_response(200, {"text": self._config.message_ack_text})

        # MESSAGE
        if event_type == "MESSAGE":
            msg_event = _parse_message_event(body)
            if msg_event and not self._is_duplicate(msg_event.event_id):
                task = asyncio.create_task(self._handler.on_message(msg_event))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            return _json_response(200, {"text": self._config.message_ack_text})

        return _json_response(200, {"text": ""})

    # -- deduplication ------------------------------------------------------

    def _is_duplicate(self, event_id: str) -> bool:
        now = time.monotonic()
        # Purge old entries
        cutoff = now - self._config.dedupe_window_sec
        self._seen_events = {
            k: v for k, v in self._seen_events.items() if v > cutoff
        }
        if event_id in self._seen_events:
            return True
        self._seen_events[event_id] = now
        return False


# Helpers

def _extract_bearer(header: str) -> str:
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return ""


def _json_response(status: int, data: dict[str, Any]) -> web.Response:
    return web.json_response(data, status=status)


def _parse_message_event(body: dict[str, Any]) -> GoogleChatMessageEvent | None:
    msg = body.get("message") or {}
    text = msg.get("argumentText") or msg.get("text") or ""
    if not text.strip():
        return None
    sender = msg.get("sender") or body.get("user") or {}
    space = body.get("space") or {}
    thread = msg.get("thread") or {}
    return GoogleChatMessageEvent(
        event_id=msg.get("name") or "",
        space_name=space.get("name") or "",
        thread_name=thread.get("name") or "",
        user_id=sender.get("name") or "",
        user_name=sender.get("displayName") or "",
        text=text.strip(),
        raw=body,
    )


def _parse_action_event(body: dict[str, Any]) -> GoogleChatActionEvent | None:
    action = body.get("action") or {}
    params_list = action.get("parameters") or []
    params: dict[str, str] = {}
    for p in params_list:
        key = p.get("key")
        val = p.get("value")
        if key:
            params[key] = str(val or "")
    user = body.get("user") or {}
    space = body.get("space") or {}
    return GoogleChatActionEvent(
        event_id=body.get("eventTime") or "",
        event_type=body.get("type") or "",
        space_name=space.get("name") or "",
        thread_name="",
        user_id=user.get("name") or "",
        user_name=user.get("displayName") or "",
        action_method_name=action.get("actionMethodName") or "",
        parameters=params,
        raw=body,
    )
