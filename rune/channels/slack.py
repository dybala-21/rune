"""Slack Channel Adapter for RUNE.

Ported from src/channels/slack.ts - uses slack-sdk with Socket Mode
for real-time messaging and Block Kit formatting.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from rune.channels.types import (
    ApprovalDecision,
    ChannelAdapter,
    IncomingMessage,
    OutgoingMessage,
    ResponseFormatter,
)
from rune.utils.logger import get_logger

log = get_logger(__name__)


def _get_slack_bot_token() -> str:
    """Resolve Slack bot token from environment variables.

    Checks TS-compatible name first, then Python-specific fallback.
    """
    return (
        os.environ.get("SLACK_BOT_TOKEN")
        or os.environ.get("RUNE_SLACK_BOT_TOKEN")
        or ""
    )


def _get_slack_app_token() -> str:
    """Resolve Slack app token from environment variables.

    Checks TS-compatible name first, then Python-specific fallback.
    """
    return (
        os.environ.get("SLACK_APP_TOKEN")
        or os.environ.get("RUNE_SLACK_APP_TOKEN")
        or ""
    )


def _get_slack_default_channel() -> str:
    """Resolve Slack default channel from environment variables."""
    return (
        os.environ.get("SLACK_DEFAULT_CHANNEL")
        or os.environ.get("RUNE_SLACK_DEFAULT_CHANNEL")
        or ""
    )


class SlackResponseFormatter(ResponseFormatter):
    """Formats responses using Slack mrkdwn syntax."""

    def format_text(self, text: str) -> str:
        return text

    def format_code(self, code: str, language: str = "") -> str:
        return f"```\n{code}\n```"

    def format_error(self, error: str) -> str:
        return f":x: *Error:* {error}"


class SlackAdapter(ChannelAdapter):
    """Channel adapter for Slack via slack-sdk Socket Mode."""

    def __init__(self, bot_token: str, app_token: str) -> None:
        super().__init__()
        self._bot_token = bot_token
        self._app_token = app_token
        self._web_client: Any = None
        self._socket_handler: Any = None
        self._formatter = SlackResponseFormatter()
        self._running = False
        self._bot_user_id: str | None = None

    @property
    def name(self) -> str:
        return "slack"

    async def start(self) -> None:
        try:
            from slack_sdk.socket_mode.aiohttp import SocketModeClient
            from slack_sdk.web.async_client import AsyncWebClient
        except ImportError as exc:
            raise ImportError(
                "slack-sdk and aiohttp are required for the Slack adapter. "
                "Install with: pip install slack-sdk aiohttp"
            ) from exc

        self._web_client = AsyncWebClient(token=self._bot_token)

        # Resolve bot user ID
        try:
            auth_response = await self._web_client.auth_test()
            self._bot_user_id = auth_response.get("user_id")
            log.info("slack_auth_ok", bot_user_id=self._bot_user_id)
        except Exception as exc:
            log.error("slack_auth_failed", error=str(exc))

        self._socket_handler = SocketModeClient(
            app_token=self._app_token,
            web_client=self._web_client,
        )

        # Register Socket Mode event handler
        self._socket_handler.socket_mode_request_listeners.append(
            self._handle_socket_event
        )

        await self._socket_handler.connect()
        self._running = True
        log.info("slack_adapter_started")

    async def stop(self) -> None:
        self._running = False
        self._cleanup_pending()
        if self._socket_handler is not None:
            await self._socket_handler.disconnect()
            self._socket_handler = None
        if self._web_client is not None:
            session = getattr(self._web_client, "session", None)
            if session is not None:
                await session.close()
            self._web_client = None
        log.info("slack_adapter_stopped")

    async def send(self, channel_id: str, message: OutgoingMessage) -> str | None:
        if self._web_client is None:
            log.warning("slack_send_no_client")
            return None

        text = self._formatter.format_text(message.text)

        # Build Block Kit blocks if metadata requests it
        blocks = message.metadata.get("blocks")
        if blocks is None:
            blocks = self._text_to_blocks(text)

        kwargs: dict[str, Any] = {
            "channel": channel_id,
            "text": text,  # Fallback for notifications
            "blocks": blocks,
        }
        if message.reply_to:
            kwargs["thread_ts"] = message.reply_to

        try:
            await self._web_client.chat_postMessage(**kwargs)
        except Exception as exc:
            log.error("slack_send_failed", error=str(exc))

        # Send attachments as file uploads
        for attachment in message.attachments:
            await self._send_attachment(channel_id, attachment)

        return None

    async def edit_message(
        self, channel_id: str, message_id: str, new_text: str
    ) -> None:
        if self._web_client is None:
            return
        try:
            await self._web_client.chat_update(
                channel=channel_id,
                ts=message_id,
                text=new_text,
                blocks=self._text_to_blocks(new_text),
            )
        except Exception as exc:
            log.error("slack_edit_failed", error=str(exc))

    async def delete_message(self, channel_id: str, message_id: str) -> None:
        if self._web_client is None:
            return
        try:
            await self._web_client.chat_delete(
                channel=channel_id,
                ts=message_id,
            )
        except Exception as exc:
            log.error("slack_delete_failed", error=str(exc))

    # -- Internals ----------------------------------------------------------

    async def _handle_socket_event(self, client: Any, req: Any) -> None:
        """Handle a Socket Mode request (event, interaction, etc.)."""
        from slack_sdk.socket_mode.response import SocketModeResponse

        # Acknowledge the event immediately
        await client.send_socket_mode_response(
            SocketModeResponse(envelope_id=req.envelope_id)
        )

        if req.type == "events_api":
            await self._handle_events_api(req.payload)
        elif req.type == "interactive":
            await self._handle_interactive(req.payload)

    async def _handle_events_api(self, payload: dict[str, Any]) -> None:
        """Handle Slack Events API payloads."""
        event = payload.get("event", {})
        event_type = event.get("type")

        if event_type != "message":
            return
        # Ignore bot messages
        if event.get("bot_id") or event.get("user") == self._bot_user_id:
            return
        # Ignore message subtypes (edits, deletes, etc.)
        if event.get("subtype"):
            return
        if self._on_message is None:
            return

        attachments = self._extract_attachments(event)

        incoming = IncomingMessage(
            channel_id=event.get("channel", ""),
            sender_id=event.get("user", ""),
            text=event.get("text", ""),
            attachments=attachments,
            reply_to=event.get("thread_ts"),
            metadata={
                "channel_name": self.name,
                "ts": event.get("ts", ""),
                "team": payload.get("team_id", ""),
            },
        )

        # Resolve any pending question Future for this channel
        question_future = self._pending_questions.pop(event.get("channel", ""), None)
        if question_future is not None and not question_future.done():
            question_future.set_result(event.get("text", ""))

        await self._on_message(incoming)

    async def _handle_interactive(self, payload: dict[str, Any]) -> None:
        """Handle interactive payloads (button clicks, etc.)."""
        actions = payload.get("actions", [])
        for action in actions:
            action_id = action.get("action_id", "")
            log.info("slack_interaction", action_id=action_id)

            # Handle approval button clicks
            if action_id.startswith("approve:") or action_id.startswith("deny:"):
                approved = action_id.startswith("approve:")
                approval_id = action_id.split(":", 1)[1]
                decision = ApprovalDecision.APPROVE_ONCE if approved else ApprovalDecision.DENY
                self._resolve_approval(approval_id, decision)
                # Update the original message to reflect the decision
                channel_id = payload.get("channel", {}).get("id", "")
                message_ts = payload.get("message", {}).get("ts", "")
                user_name = payload.get("user", {}).get("username", "someone")
                if channel_id and message_ts and self._web_client is not None:
                    status = "Approved" if approved else "Denied"
                    try:
                        await self._web_client.chat_update(
                            channel=channel_id,
                            ts=message_ts,
                            text=f"Approval {approval_id}: *{status}* by {user_name}",
                            blocks=[],
                        )
                    except Exception as exc:
                        log.warning("slack_approval_update_failed", error=str(exc))

    def _extract_attachments(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract file attachments from a Slack event."""
        files = event.get("files", [])
        return [
            {
                "type": f.get("filetype", "file"),
                "url": f.get("url_private", ""),
                "filename": f.get("name", ""),
                "size": f.get("size", 0),
                "mimetype": f.get("mimetype", ""),
            }
            for f in files
        ]

    def _text_to_blocks(self, text: str) -> list[dict[str, Any]]:
        """Convert plain text to Slack Block Kit section blocks."""
        # Split into sections at double newlines for better formatting
        sections = text.split("\n\n") if "\n\n" in text else [text]
        blocks: list[dict[str, Any]] = []
        for section in sections:
            if not section.strip():
                continue
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": section.strip()[:3000],  # Block text limit
                },
            })
        return blocks or [{"type": "section", "text": {"type": "mrkdwn", "text": " "}}]

    async def _send_attachment(
        self, channel_id: str, attachment: dict[str, Any]
    ) -> None:
        """Upload a file attachment to the channel."""
        if self._web_client is None:
            return
        try:
            file_path = attachment.get("path")
            if file_path:
                await self._web_client.files_upload_v2(
                    channel=channel_id,
                    file=file_path,
                    title=attachment.get("title", ""),
                    initial_comment=attachment.get("caption", ""),
                )
        except Exception as exc:
            log.error("slack_attachment_failed", error=str(exc))

    async def ask_question(self, channel_id: str, question: str, options: list[str] | None = None, timeout: float = 300.0) -> str:
        """Ask a question via Slack and wait for a reply.

        Sends the question, then awaits a Future that is resolved when the
        next message arrives in the same channel.  Returns the reply text,
        or ``""`` on timeout.
        """
        if self._web_client is None:
            return ""
        text = question
        if options:
            text += "\n" + "\n".join(f"• {opt}" for opt in options)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_questions[channel_id] = future

        try:
            await self._web_client.chat_postMessage(
                channel=channel_id,
                text=text,
                blocks=self._text_to_blocks(text),
            )
        except Exception as exc:
            log.error("slack_ask_question_failed", error=str(exc))
            self._pending_questions.pop(channel_id, None)
            return ""

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            self._pending_questions.pop(channel_id, None)
            log.warning("slack_ask_question_timeout", channel_id=channel_id)
            return ""

    async def send_approval(
        self,
        channel_id: str,
        description: str,
        approval_id: str,
    ) -> None:
        """Send Block Kit interactive message for an approval request."""
        if self._web_client is None:
            return

        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":lock: *Approval Required*\n\n{description}",
                },
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "style": "primary",
                        "action_id": f"approve:{approval_id}",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Deny"},
                        "style": "danger",
                        "action_id": f"deny:{approval_id}",
                    },
                ],
            },
        ]

        try:
            await self._web_client.chat_postMessage(
                channel=channel_id,
                text=f"Approval Required: {description}",
                blocks=blocks,
            )
        except Exception as exc:
            log.error("slack_approval_blocks_failed", error=str(exc))

