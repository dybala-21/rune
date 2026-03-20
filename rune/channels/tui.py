"""TUI Channel Adapter for RUNE.

Ported from src/channels/tui.ts - bridges the Textual-based TUI application
with the channel system so the agent can send/receive through the terminal UI.
"""

from __future__ import annotations

from rune.channels.types import (
    ApprovalDecision,
    ChannelAdapter,
    IncomingMessage,
    OutgoingMessage,
    ResponseFormatter,
)
from rune.utils.events import EventEmitter
from rune.utils.logger import get_logger

log = get_logger(__name__)

_TUI_CHANNEL_ID = "tui:local"
_TUI_SENDER_ID = "user:local"


class TUIResponseFormatter(ResponseFormatter):
    """Formats responses for Rich/Textual terminal display."""

    def format_text(self, text: str) -> str:
        return text

    def format_code(self, code: str, language: str = "") -> str:
        lang_tag = language or ""
        return f"```{lang_tag}\n{code}\n```"

    def format_error(self, error: str) -> str:
        return f"[bold red]Error:[/bold red] {error}"


class TUIChannelAdapter(ChannelAdapter):
    """Channel adapter that integrates with the Textual TUI application.

    Uses an EventEmitter to bridge messages between the channel system and
    the UI layer without a direct import dependency on the Textual App.
    """

    def __init__(self) -> None:
        super().__init__()
        self._emitter = EventEmitter()
        self._formatter = TUIResponseFormatter()
        self._running = False

    # -- ChannelAdapter interface -------------------------------------------

    @property
    def name(self) -> str:
        return "tui"

    async def start(self) -> None:
        self._running = True
        log.info("tui_channel_started")

    async def stop(self) -> None:
        self._running = False
        self._cleanup_pending()
        log.info("tui_channel_stopped")

    async def send(self, channel_id: str, message: OutgoingMessage) -> str | None:
        """Send an outgoing message by emitting a UI event."""
        if not self._running:
            return None

        formatted = self._formatter.format_text(message.text)
        await self._emitter.emit(
            "ui:message",
            {
                "channel_id": channel_id,
                "text": formatted,
                "attachments": message.attachments,
                "reply_to": message.reply_to,
                "metadata": message.metadata,
            },
        )
        return None

    # -- TUI-specific API ---------------------------------------------------

    @property
    def emitter(self) -> EventEmitter:
        """Expose the emitter so the UI app can subscribe to events."""
        return self._emitter

    @property
    def formatter(self) -> TUIResponseFormatter:
        return self._formatter

    async def handle_user_input(self, text: str) -> None:
        """Called by the TUI app when the user submits text."""
        if not self._running or self._on_message is None:
            return

        msg = IncomingMessage(
            channel_id=_TUI_CHANNEL_ID,
            sender_id=_TUI_SENDER_ID,
            text=text,
        )
        await self._on_message(msg)

    async def send_approval(self, channel_id: str, description: str, approval_id: str) -> None:
        """Send an approval request via the TUI.

        Emits a UI prompt event for the user to approve or deny.
        """
        await self._emitter.emit(
            "ui:approval_request",
            {
                "approval_id": approval_id,
                "description": description,
                "channel_id": channel_id,
            },
        )

    async def ask_question(self, channel_id: str, question: str, options: list[str] | None = None, timeout: float = 300.0) -> str:
        """Ask a question via the TUI. Sends the question and returns empty string."""
        text = question
        if options:
            text += "\n" + "\n".join(f"• {opt}" for opt in options)
        await self._emitter.emit(
            "ui:message",
            {
                "channel_id": channel_id,
                "text": text,
                "attachments": [],
                "reply_to": None,
                "metadata": {"type": "question"},
            },
        )
        return ""

    async def request_approval(
        self,
        description: str,
        *,
        timeout_seconds: float = 300.0,
    ) -> bool:
        """Request user approval via the TUI.

        Emits a UI prompt event and waits for the user's decision.
        Uses base-class wait_for_approval() which returns ApprovalResponse.
        """
        approval_id = f"approval-{id(description)}"

        await self._emitter.emit(
            "ui:approval_request",
            {
                "approval_id": approval_id,
                "description": description,
                "timeout_seconds": timeout_seconds,
            },
        )

        response = await self.wait_for_approval(approval_id, timeout=timeout_seconds)
        return response.approved

    async def resolve_approval(self, approval_id: str, approved: bool) -> None:
        """Called by the UI when the user responds to an approval prompt."""
        decision = ApprovalDecision.APPROVE_ONCE if approved else ApprovalDecision.DENY
        self._resolve_approval(approval_id, decision)

    async def send_status(self, status: str) -> None:
        """Send a status update to the TUI (thinking, acting, etc.)."""
        await self._emitter.emit("ui:status", {"status": status})

    async def send_progress(
        self, current: int, total: int, label: str = ""
    ) -> None:
        """Send a progress update to the TUI."""
        await self._emitter.emit(
            "ui:progress",
            {"current": current, "total": total, "label": label},
        )
