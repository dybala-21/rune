"""Channel type definitions for RUNE.

Ported from src/channels/types.ts - base types, adapter ABC,
and response formatter for all channel implementations.

Provides common infrastructure that all channel adapters share:
- Authorization (allowed_users filtering)
- Approval flow with ApprovalResponse (approve_once/approve_always/deny)
- Pending approval/question lifecycle management
- Connection status tracking
- Reconnection with exponential backoff
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)


# Priority

class Priority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class ApprovalDecision(StrEnum):
    APPROVE_ONCE = "approve_once"
    APPROVE_ALWAYS = "approve_always"
    DENY = "deny"


@dataclass(slots=True)
class ApprovalResponse:
    """Result of an approval request."""

    decision: ApprovalDecision = ApprovalDecision.DENY
    approved: bool = False
    timed_out: bool = False
    user_guidance: str = ""


# Messages

@dataclass(slots=True)
class IncomingMessage:
    """A message received from a channel."""

    channel_id: str
    sender_id: str
    text: str
    id: str = ""
    sender_name: str = ""
    timestamp: float = 0.0
    attachments: list[dict[str, Any]] = field(default_factory=list)
    reply_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OutgoingMessage:
    """A message to be sent to a channel."""

    text: str
    attachments: list[dict[str, Any]] = field(default_factory=list)
    reply_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Notification:
    """A lightweight notification from a channel."""

    channel_id: str
    sender_id: str
    text: str
    priority: Priority = Priority.NORMAL


# Callback type

type MessageCallback = Callable[[IncomingMessage], Awaitable[None]]


# Reconnection defaults

_DEFAULT_MAX_RECONNECT_ATTEMPTS = 5
_DEFAULT_BASE_RECONNECT_DELAY = 5.0  # seconds


# ChannelAdapter ABC

class ChannelAdapter(ABC):
    """Abstract base for all channel adapters (TUI, Telegram, Discord, etc.).

    Provides common infrastructure:
    - ``allowed_users``: user authorization filtering
    - ``_pending_approvals`` / ``_pending_questions``: Future lifecycle
    - ``_status``: connection status tracking (disconnected/connecting/connected/error)
    - ``_schedule_reconnect()``: exponential backoff reconnection
    - ``_cleanup_pending()``: auto-deny approvals, resolve questions on disconnect
    """

    def __init__(
        self,
        *,
        allowed_users: list[str] | None = None,
        max_reconnect_attempts: int = _DEFAULT_MAX_RECONNECT_ATTEMPTS,
        base_reconnect_delay: float = _DEFAULT_BASE_RECONNECT_DELAY,
    ) -> None:
        self._on_message: MessageCallback | None = None
        self._allowed_users: list[str] | None = allowed_users
        self._pending_approvals: dict[str, asyncio.Future[ApprovalResponse]] = {}
        self._pending_questions: dict[str, asyncio.Future[str]] = {}
        # Connection status
        self._status: str = "disconnected"
        # Reconnection state
        self._max_reconnect_attempts = max_reconnect_attempts
        self._base_reconnect_delay = base_reconnect_delay
        self._reconnect_attempts: int = 0
        self._reconnect_task: asyncio.Task[None] | None = None

    # -- Abstract interface -------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this channel adapter."""
        ...

    @abstractmethod
    async def start(self) -> None:
        """Start the adapter (connect, begin polling/listening)."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the adapter (disconnect, clean up)."""
        ...

    @abstractmethod
    async def send(self, channel_id: str, message: OutgoingMessage) -> str | None:
        """Send a message to a channel.

        Returns the platform message ID if available, or ``None``.
        """
        ...

    @abstractmethod
    async def send_approval(self, channel_id: str, description: str, approval_id: str) -> None:
        """Send an approval request to the channel."""
        ...

    @abstractmethod
    async def ask_question(self, channel_id: str, question: str, options: list[str] | None = None, timeout: float = 300.0) -> str:
        """Ask a question via the channel and return the answer."""
        ...

    # -- Properties ---------------------------------------------------------

    @property
    def status(self) -> str:
        """Connection status: disconnected | connecting | connected | error."""
        return self._status

    @property
    def on_message(self) -> MessageCallback | None:
        return self._on_message

    @on_message.setter
    def on_message(self, callback: MessageCallback | None) -> None:
        self._on_message = callback

    # -- Optional overrides -------------------------------------------------

    async def edit_message(
        self, channel_id: str, message_id: str, new_text: str
    ) -> None:
        raise NotImplementedError(
            f"{self.name} does not support message editing"
        )

    async def delete_message(self, channel_id: str, message_id: str) -> None:
        raise NotImplementedError(
            f"{self.name} does not support message deletion"
        )

    async def send_notification(self, channel_id: str, text: str) -> None:
        await self.send(channel_id, OutgoingMessage(text=text))

    async def send_typing_indicator(self, channel_id: str) -> None:
        pass

    # -- Authorization (common) ---------------------------------------------

    def check_authorization(self, sender_id: str) -> bool:
        """Check if a user is authorized.

        Returns True if no allowed_users configured (open access)
        or if the sender_id is in the allowed list.
        Subclasses can override for platform-specific logic.
        """
        if not self._allowed_users:
            return True
        return sender_id in self._allowed_users

    # -- Approval flow (common) ---------------------------------------------

    async def wait_for_approval(
        self,
        approval_id: str,
        timeout: float = 60.0,
    ) -> ApprovalResponse:
        """Wait for an approval response.

        Creates a Future, stores it in ``_pending_approvals``, and awaits it.
        Auto-denies on timeout. Subclasses resolve the Future when the user
        clicks a button (via callback query, interaction, etc.).
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ApprovalResponse] = loop.create_future()
        self._pending_approvals[approval_id] = future
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            self._pending_approvals.pop(approval_id, None)
            log.warning(
                "approval_timeout",
                adapter=self.name,
                approval_id=approval_id,
            )
            return ApprovalResponse(
                decision=ApprovalDecision.DENY,
                approved=False,
                timed_out=True,
            )

    def _resolve_approval(
        self,
        approval_id: str,
        decision: ApprovalDecision,
    ) -> bool:
        """Resolve a pending approval Future. Returns True if resolved.

        Call this from platform-specific callback handlers (button clicks etc.).
        """
        future = self._pending_approvals.pop(approval_id, None)
        if future is None or future.done():
            return False
        future.set_result(ApprovalResponse(
            decision=decision,
            approved=decision != ApprovalDecision.DENY,
            timed_out=False,
        ))
        return True

    # -- Pending lifecycle (common) -----------------------------------------

    def _cleanup_pending(self) -> None:
        """Auto-deny all pending approvals and resolve pending questions.

        Call this from ``stop()`` to ensure no dangling Futures.
        """
        for future in self._pending_approvals.values():
            if not future.done():
                future.set_result(ApprovalResponse(
                    decision=ApprovalDecision.DENY,
                    approved=False,
                    timed_out=True,
                ))
        self._pending_approvals.clear()

        for future in self._pending_questions.values():
            if not future.done():
                future.set_result("")
        self._pending_questions.clear()

    # -- Reconnection (common) ----------------------------------------------

    def _schedule_reconnect(self) -> None:
        """Schedule a reconnection attempt with exponential backoff.

        Subclasses should call this when a connection error occurs.
        Override ``_do_connect()`` for the actual connection logic.
        """
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            log.error(
                "max_reconnect_reached",
                adapter=self.name,
                attempts=self._reconnect_attempts,
            )
            return
        if self._status == "disconnected":
            return
        if self._reconnect_task is not None:
            return

        self._reconnect_attempts += 1
        delay = self._base_reconnect_delay * (2 ** (self._reconnect_attempts - 1))
        log.info(
            "reconnect_scheduled",
            adapter=self.name,
            attempt=self._reconnect_attempts,
            delay_sec=delay,
        )
        self._reconnect_task = asyncio.create_task(self._do_reconnect(delay))

    async def _do_reconnect(self, delay: float) -> None:
        """Wait, then call ``_do_connect()``. Override ``_do_connect()`` in subclasses."""
        try:
            await asyncio.sleep(delay)
            self._reconnect_task = None
            self._status = "connecting"
            await self._do_connect()
            self._status = "connected"
            self._reconnect_attempts = 0
            log.info("reconnected", adapter=self.name)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("reconnect_failed", adapter=self.name, error=str(exc))
            self._reconnect_task = None
            self._schedule_reconnect()

    async def _do_connect(self) -> None:
        """Establish the actual connection. Override in subclasses that use reconnection."""
        raise NotImplementedError

    def _cancel_reconnect(self) -> None:
        """Cancel any pending reconnect task. Call from ``stop()``."""
        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            self._reconnect_task = None


# ResponseFormatter ABC

class ResponseFormatter(ABC):
    """Formats agent responses for a specific channel."""

    @abstractmethod
    def format_text(self, text: str) -> str: ...

    @abstractmethod
    def format_code(self, code: str, language: str = "") -> str: ...

    @abstractmethod
    def format_error(self, error: str) -> str: ...
