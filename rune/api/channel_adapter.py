"""Bridge API clients to Gateway's ChannelAdapter interface.

Ported from src/api/api-channel-adapter.ts - translates between
REST/SSE API clients and the internal channel adapter protocol.
Manages pending approvals, questions, and SSE event streaming.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)


DEFAULT_APPROVAL_TIMEOUT = 60.0  # seconds
DEFAULT_QUESTION_TIMEOUT = 120.0  # seconds


@dataclass
class PendingApproval:
    """A pending approval request waiting for client response."""
    future: asyncio.Future[dict[str, Any]]
    timer: asyncio.TimerHandle | None = None


@dataclass
class PendingQuestion:
    """A pending question waiting for client response."""
    future: asyncio.Future[dict[str, Any]]
    timer: asyncio.TimerHandle | None = None


@dataclass
class SseClient:
    """Tracks a connected SSE client."""
    client_id: str
    queue: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)
    connected: bool = True


class ApiChannelAdapter:
    """Bridges API clients to the internal channel adapter interface.

    Responsibilities:
    - Manage connected SSE clients and their event queues.
    - Stream events to clients via SSE.
    - Handle approval and question request/response flows.
    - Forward incoming messages to the gateway.
    """

    def __init__(self) -> None:
        self.name = "api"
        self._status = "disconnected"
        self._clients: dict[str, SseClient] = {}
        self._pending_approvals: dict[str, PendingApproval] = {}
        self._pending_questions: dict[str, PendingQuestion] = {}
        self._message_handler: Callable[..., Any] | None = None

        # Callbacks (set by the server or gateway)
        self.on_client_connect: Callable[[str], None] | None = None
        self.on_client_disconnect: Callable[[str], None] | None = None

    @property
    def status(self) -> str:
        return self._status

    async def connect(self) -> None:
        """Mark the adapter as connected."""
        self._status = "connected"
        log.info("api_channel_connected")

    async def disconnect(self) -> None:
        """Disconnect and clean up all pending requests."""
        # Resolve pending approvals with denial
        for _approval_id, pending in self._pending_approvals.items():
            if pending.timer:
                pending.timer.cancel()
            if not pending.future.done():
                pending.future.set_result(
                    {"decision": "deny", "approved": False, "timedOut": True}
                )
        self._pending_approvals.clear()

        # Resolve pending questions with empty answer
        for _question_id, pending in self._pending_questions.items():
            if pending.timer:
                pending.timer.cancel()
            if not pending.future.done():
                pending.future.set_result({"answer": ""})
        self._pending_questions.clear()

        # Disconnect all clients
        for client in self._clients.values():
            client.connected = False
        self._clients.clear()

        self._status = "disconnected"
        log.info("api_channel_disconnected")

    def on_message(self, handler: Callable[..., Any]) -> None:
        """Register a handler for incoming client messages."""
        self._message_handler = handler

    # Client management

    def add_client(self, client_id: str) -> SseClient:
        """Register a new SSE client."""
        client = SseClient(client_id=client_id)
        self._clients[client_id] = client
        log.info("api_client_connected", client_id=client_id)
        if self.on_client_connect:
            self.on_client_connect(client_id)
        return client

    def remove_client(self, client_id: str) -> None:
        """Remove a disconnected SSE client."""
        client = self._clients.pop(client_id, None)
        if client:
            client.connected = False
            log.info("api_client_disconnected", client_id=client_id)
            if self.on_client_disconnect:
                self.on_client_disconnect(client_id)

    def get_client(self, client_id: str) -> SseClient | None:
        return self._clients.get(client_id)

    def get_session_count(self) -> int:
        return len(self._clients)

    # Event streaming

    def stream_event(self, client_id: str, event: dict[str, Any]) -> None:
        """Push an SSE event to a specific client's queue."""
        client = self._clients.get(client_id)
        if client and client.connected:
            try:
                client.queue.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("sse_queue_full", client_id=client_id)

    def broadcast_event(self, event: dict[str, Any]) -> None:
        """Push an SSE event to all connected clients."""
        for client in self._clients.values():
            if client.connected:
                with contextlib.suppress(asyncio.QueueFull):
                    client.queue.put_nowait(event)

    async def event_generator(self, client_id: str) -> AsyncGenerator[dict[str, Any]]:
        """Async generator yielding SSE events for a client.

        Used by the SSE endpoint to stream events.
        """
        client = self._clients.get(client_id)
        if not client:
            return

        try:
            while client.connected:
                try:
                    event = await asyncio.wait_for(client.queue.get(), timeout=30.0)
                    yield event
                except TimeoutError:
                    # Send heartbeat
                    yield {"event": "heartbeat", "data": {}}
        finally:
            self.remove_client(client_id)

    # Message handling

    async def send_message(self, recipient_id: str, content: str) -> None:
        """Send a completion message to a client."""
        self.stream_event(recipient_id, {
            "event": "agent_complete",
            "data": {"success": True, "answer": content, "durationMs": 0},
        })

    async def send_text_delta(self, recipient_id: str, text: str) -> str:
        """Send a streaming text delta to a client. Returns a message ID."""
        message_id = uuid.uuid4().hex
        self.stream_event(recipient_id, {
            "event": "text_delta",
            "data": {"text": text, "messageId": message_id},
        })
        return message_id

    async def send_notification(
        self,
        recipient_id: str,
        title: str,
        body: str,
    ) -> None:
        """Send a notification event to a client."""
        self.stream_event(recipient_id, {
            "event": "notification",
            "data": {"title": title, "body": body},
        })

    def handle_client_message(
        self,
        client_id: str,
        text: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> None:
        """Process an incoming message from an API client."""
        if self._message_handler:
            self._message_handler({
                "sender_id": client_id,
                "text": text,
                "channel": "api",
                "attachments": attachments or [],
            })

    # Approval flow

    async def send_approval(
        self,
        recipient_id: str,
        info: dict[str, Any],
        timeout: float = DEFAULT_APPROVAL_TIMEOUT,
    ) -> dict[str, Any]:
        """Send an approval request and wait for the client's response.

        Args:
            recipient_id: The client to send the approval to.
            info: Approval details (command, riskLevel, reason, etc.).
            timeout: Seconds to wait before auto-denying.

        Returns:
            Approval response dict with ``decision``, ``approved``, etc.
        """
        approval_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()

        def _on_timeout() -> None:
            self._pending_approvals.pop(approval_id, None)
            if not future.done():
                future.set_result({"decision": "deny", "approved": False, "timedOut": True})

        timer = loop.call_later(timeout, _on_timeout)
        self._pending_approvals[approval_id] = PendingApproval(future=future, timer=timer)

        # Send the approval request event to the client
        self.stream_event(recipient_id, {
            "event": "approval_request",
            "data": {"approvalId": approval_id, **info},
        })

        return await future

    def resolve_approval(
        self,
        approval_id: str,
        decision: str,
        user_guidance: str | None = None,
    ) -> bool:
        """Resolve a pending approval request.

        Returns True if the approval was found and resolved.
        """
        pending = self._pending_approvals.pop(approval_id, None)
        if not pending:
            return False
        if pending.timer:
            pending.timer.cancel()
        if not pending.future.done():
            pending.future.set_result({
                "decision": decision,
                "approved": decision != "deny",
                "userGuidance": user_guidance,
            })
        return True

    # Question flow

    async def send_question(
        self,
        recipient_id: str,
        prompt: str,
        options: dict[str, Any] | None = None,
        timeout: float = DEFAULT_QUESTION_TIMEOUT,
    ) -> dict[str, Any]:
        """Send a question to a client and wait for their answer.

        Returns:
            Dict with ``answer`` and optionally ``selectedIndex``.
        """
        question_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()

        def _on_timeout() -> None:
            self._pending_questions.pop(question_id, None)
            if not future.done():
                future.set_result({"answer": ""})

        timer = loop.call_later(timeout, _on_timeout)
        self._pending_questions[question_id] = PendingQuestion(future=future, timer=timer)

        self.stream_event(recipient_id, {
            "event": "question_request",
            "data": {"questionId": question_id, "prompt": prompt, **(options or {})},
        })

        return await future

    def resolve_question(
        self,
        question_id: str,
        answer: str,
        selected_index: int | None = None,
    ) -> bool:
        """Resolve a pending question.

        Returns True if the question was found and resolved.
        """
        pending = self._pending_questions.pop(question_id, None)
        if not pending:
            return False
        if pending.timer:
            pending.timer.cancel()
        if not pending.future.done():
            result: dict[str, Any] = {"answer": answer}
            if selected_index is not None:
                result["selectedIndex"] = selected_index
            pending.future.set_result(result)
        return True
