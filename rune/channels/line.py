"""LINE Messaging API Channel Adapter for RUNE.

Uses the LINE Messaging API for webhook signature verification, reply/push
message support, and Flex Messages for approval workflows.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from typing import Any

from rune.channels.types import (
    ApprovalDecision,
    ChannelAdapter,
    IncomingMessage,
    OutgoingMessage,
    ResponseFormatter,
)
from rune.utils.fast_serde import json_decode
from rune.utils.logger import get_logger

log = get_logger(__name__)


def _get_line_channel_access_token() -> str:
    """Resolve LINE channel access token from environment variables.

    Checks TS-compatible name first, then Python-specific fallback.
    """
    return (
        os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
        or os.environ.get("RUNE_LINE_CHANNEL_ACCESS_TOKEN")
        or ""
    )


def _get_line_channel_secret() -> str:
    """Resolve LINE channel secret from environment variables."""
    return (
        os.environ.get("LINE_CHANNEL_SECRET")
        or os.environ.get("RUNE_LINE_CHANNEL_SECRET")
        or ""
    )


def _get_line_allowed_users() -> list[str]:
    """Resolve LINE allowed users from environment variables."""
    raw = (
        os.environ.get("LINE_ALLOWED_USERS")
        or os.environ.get("RUNE_LINE_ALLOWED_USERS")
        or ""
    )
    return [u.strip() for u in raw.split(",") if u.strip()] if raw else []


def _get_line_default_recipient() -> str:
    """Resolve LINE default recipient from environment variables."""
    return (
        os.environ.get("LINE_DEFAULT_RECIPIENT")
        or os.environ.get("RUNE_LINE_DEFAULT_RECIPIENT")
        or ""
    )


def _get_line_webhook_config() -> dict[str, Any]:
    """Resolve LINE webhook configuration from environment variables."""
    return {
        "enabled": (
            os.environ.get("LINE_WEBHOOK_ENABLED")
            or os.environ.get("RUNE_LINE_WEBHOOK_ENABLED")
            or "false"
        ).lower() == "true",
        "host": (
            os.environ.get("LINE_WEBHOOK_HOST")
            or os.environ.get("RUNE_LINE_WEBHOOK_HOST")
            or "127.0.0.1"
        ),
        "port": int(
            os.environ.get("LINE_WEBHOOK_PORT")
            or os.environ.get("RUNE_LINE_WEBHOOK_PORT")
            or "8791"
        ),
        "path": (
            os.environ.get("LINE_WEBHOOK_PATH")
            or os.environ.get("RUNE_LINE_WEBHOOK_PATH")
            or "/line/webhook"
        ),
        "verify_signature": (
            os.environ.get("LINE_VERIFY_SIGNATURE")
            or os.environ.get("RUNE_LINE_VERIFY_SIGNATURE")
            or "true"
        ).lower() == "true",
    }

# LINE message length limit
_MAX_MESSAGE_LENGTH = 5000

# LINE Messaging API base URL
_API_BASE = "https://api.line.me/v2/bot"


class LINEResponseFormatter(ResponseFormatter):
    """Formats responses for LINE (plain text, no native markdown)."""

    def format_text(self, text: str) -> str:
        return text

    def format_code(self, code: str, language: str = "") -> str:
        # LINE does not support markdown; use plain formatting
        return f"[Code]\n{code}"

    def format_error(self, error: str) -> str:
        return f"[Error] {error}"


class LINEAdapter(ChannelAdapter):
    """Channel adapter for LINE via the Messaging API."""

    def __init__(
        self,
        channel_access_token: str,
        channel_secret: str,
        *,
        listen_host: str = "0.0.0.0",
        listen_port: int = 8082,
    ) -> None:
        super().__init__()
        self._channel_access_token = channel_access_token
        self._channel_secret = channel_secret
        self._listen_host = listen_host
        self._listen_port = listen_port
        self._http_session: Any = None
        self._web_app: Any = None
        self._web_runner: Any = None
        self._formatter = LINEResponseFormatter()

    @property
    def name(self) -> str:
        return "line"

    async def start(self) -> None:
        try:
            import aiohttp.web
        except ImportError as exc:
            raise ImportError(
                "aiohttp is required for the LINE adapter webhook server. "
                "Install with: pip install aiohttp"
            ) from exc

        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for the LINE adapter. "
                "Install with: pip install httpx"
            ) from exc

        self._http_session = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self._channel_access_token}",
                "Content-Type": "application/json",
            }
        )

        # Set up webhook server
        self._web_app = aiohttp.web.Application()
        self._web_app.router.add_post("/line/webhook", self._handle_webhook)

        self._web_runner = aiohttp.web.AppRunner(self._web_app)
        await self._web_runner.setup()
        site = aiohttp.web.TCPSite(
            self._web_runner, self._listen_host, self._listen_port
        )
        await site.start()

        log.info(
            "line_adapter_started",
            host=self._listen_host,
            port=self._listen_port,
        )

    async def stop(self) -> None:
        self._cleanup_pending()
        if self._web_runner is not None:
            await self._web_runner.cleanup()
            self._web_runner = None
        self._web_app = None
        if self._http_session is not None:
            await self._http_session.aclose()
            self._http_session = None
        log.info("line_adapter_stopped")

    async def send(self, channel_id: str, message: OutgoingMessage) -> str | None:
        """Send a push message to a user/group/room."""
        if self._http_session is None:
            log.warning("line_send_no_session")
            return None

        text = self._formatter.format_text(message.text)

        # Build messages payload
        messages = self._build_text_messages(text)

        # If metadata contains flex messages, use those instead
        flex = message.metadata.get("flex")
        if flex is not None:
            messages = [{"type": "flex", "altText": text[:400], "contents": flex}]

        await self._push_message(channel_id, messages)

        # Send media attachments
        for attachment in message.attachments:
            media_msg = self._build_media_message(attachment)
            if media_msg:
                await self._push_message(channel_id, [media_msg])

        return None

    async def reply(
        self, reply_token: str, message: OutgoingMessage
    ) -> None:
        """Send a reply message using a reply token (must be used within 1 min)."""
        if self._http_session is None:
            log.warning("line_reply_no_session")
            return

        text = self._formatter.format_text(message.text)
        messages = self._build_text_messages(text)

        flex = message.metadata.get("flex")
        if flex is not None:
            messages = [{"type": "flex", "altText": text[:400], "contents": flex}]

        await self._api_post(
            f"{_API_BASE}/message/reply",
            {"replyToken": reply_token, "messages": messages},
        )

    # -- Internals ----------------------------------------------------------

    async def _api_post(
        self, url: str, body: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Make a POST request to the LINE Messaging API."""
        if self._http_session is None:
            return None
        try:
            resp = await self._http_session.post(url, json=body)
            if resp.status_code == 200:
                return resp.json()
            else:
                log.error(
                    "line_api_error",
                    status=resp.status_code,
                    error=resp.text,
                )
                return None
        except Exception as exc:
            log.error("line_api_request_failed", error=str(exc))
            return None

    async def _push_message(
        self, to: str, messages: list[dict[str, Any]]
    ) -> None:
        """Send a push message to a user/group/room."""
        # LINE allows max 5 messages per request
        for i in range(0, len(messages), 5):
            batch = messages[i : i + 5]
            await self._api_post(
                f"{_API_BASE}/message/push",
                {"to": to, "messages": batch},
            )

    async def _handle_webhook(self, request: Any) -> Any:
        """Handle an incoming LINE webhook event."""
        import aiohttp.web

        body_bytes = await request.read()

        # Verify signature
        signature = request.headers.get("X-Line-Signature", "")
        if not self._verify_signature(body_bytes, signature):
            log.warning("line_webhook_invalid_signature")
            return aiohttp.web.Response(status=401, text="Invalid signature")

        payload = json_decode(body_bytes)

        for event in payload.get("events", []):
            event_type = event.get("type", "")
            if event_type == "message":
                await self._handle_message_event(event)
            elif event_type == "postback":
                await self._handle_postback_event(event)
            elif event_type == "follow":
                log.info(
                    "line_follow_event",
                    user_id=event.get("source", {}).get("userId"),
                )
            elif event_type == "unfollow":
                log.info(
                    "line_unfollow_event",
                    user_id=event.get("source", {}).get("userId"),
                )

        return aiohttp.web.json_response({"status": "ok"})

    def _verify_signature(self, body: bytes, signature: str) -> bool:
        """Verify the X-Line-Signature header using HMAC-SHA256."""
        digest = hmac.new(
            self._channel_secret.encode(),
            body,
            hashlib.sha256,
        ).digest()
        expected = base64.b64encode(digest).decode()
        return hmac.compare_digest(expected, signature)

    async def _handle_message_event(self, event: dict[str, Any]) -> None:
        """Convert a LINE message event to IncomingMessage and dispatch."""
        if self._on_message is None:
            return

        source = event.get("source", {})
        message = event.get("message", {})
        msg_type = message.get("type", "")

        text = ""
        attachments: list[dict[str, Any]] = []

        if msg_type == "text":
            text = message.get("text", "")
        elif msg_type in ("image", "video", "audio", "file"):
            attachments.append({
                "type": msg_type,
                "message_id": message.get("id", ""),
                "filename": message.get("fileName", ""),
                "file_size": message.get("fileSize", 0),
                "content_url": f"{_API_BASE}/message/{message.get('id', '')}/content",
            })
        elif msg_type == "sticker":
            text = f"[Sticker: {message.get('packageId', '')}/{message.get('stickerId', '')}]"
        elif msg_type == "location":
            text = (
                f"[Location: {message.get('title', '')} "
                f"({message.get('latitude', '')}, {message.get('longitude', '')})]"
            )

        # Determine channel_id based on source type
        source_type = source.get("type", "")
        if source_type == "group":
            channel_id = source.get("groupId", "")
        elif source_type == "room":
            channel_id = source.get("roomId", "")
        else:
            channel_id = source.get("userId", "")

        incoming = IncomingMessage(
            channel_id=channel_id,
            sender_id=source.get("userId", ""),
            text=text,
            attachments=attachments,
            reply_to=None,
            metadata={
                "channel_name": self.name,
                "message_id": message.get("id", ""),
                "reply_token": event.get("replyToken", ""),
                "source_type": source_type,
                "timestamp": event.get("timestamp", 0),
            },
        )

        await self._on_message(incoming)

    async def _handle_postback_event(self, event: dict[str, Any]) -> None:
        """Handle a postback event (used for approval button clicks).

        Postback data format: ``action=approve&approval_id=<id>`` or
        ``action=deny&approval_id=<id>``.  Resolves the pending approval
        future so ``wait_for_approval`` can return.
        """
        source = event.get("source", {})
        postback = event.get("postback", {})
        data = postback.get("data", "")

        log.info(
            "line_postback",
            user_id=source.get("userId"),
            data=data,
        )

        # Parse postback data (URL-encoded key=value pairs)
        params: dict[str, str] = {}
        for pair in data.split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                params[k] = v

        action = params.get("action", "")
        approval_id = params.get("approval_id", "")

        if action in ("approve", "deny") and approval_id:
            approved = action == "approve"
            decision = ApprovalDecision.APPROVE_ONCE if approved else ApprovalDecision.DENY
            self._resolve_approval(approval_id, decision)

            # Reply with decision confirmation using the reply token
            reply_token = event.get("replyToken", "")
            if reply_token:
                status = "Approved" if approved else "Denied"
                await self._api_post(
                    f"{_API_BASE}/message/reply",
                    {
                        "replyToken": reply_token,
                        "messages": [
                            {"type": "text", "text": f"Approval {approval_id}: {status}"}
                        ],
                    },
                )

    def _build_text_messages(self, text: str) -> list[dict[str, Any]]:
        """Build LINE text message objects, splitting if necessary."""
        chunks = _split_text(text, _MAX_MESSAGE_LENGTH)
        return [{"type": "text", "text": chunk} for chunk in chunks]

    def _build_media_message(
        self, attachment: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Build a LINE media message from an attachment dict."""
        att_type = attachment.get("type", "")
        url = attachment.get("url", "")
        preview_url = attachment.get("preview_url", url)

        if att_type == "image" and url:
            return {
                "type": "image",
                "originalContentUrl": url,
                "previewImageUrl": preview_url,
            }
        elif att_type == "video" and url:
            return {
                "type": "video",
                "originalContentUrl": url,
                "previewImageUrl": preview_url,
            }
        elif att_type == "audio" and url:
            return {
                "type": "audio",
                "originalContentUrl": url,
                "duration": attachment.get("duration", 60000),
            }

        return None

    async def ask_question(self, channel_id: str, question: str, options: list[str] | None = None, timeout: float = 300.0) -> str:
        """Ask a question via LINE. Sends the question and returns empty string."""
        if self._http_session is None:
            return ""
        text = question
        if options:
            text += "\n" + "\n".join(f"• {opt}" for opt in options)
        messages = self._build_text_messages(text)
        await self._push_message(channel_id, messages)
        return ""

    async def send_approval(
        self,
        channel_id: str,
        description: str,
        approval_id: str,
    ) -> None:
        """Send a Flex Message for approval requests."""
        if self._http_session is None:
            return

        flex_contents: dict[str, Any] = {
            "type": "bubble",
            "header": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": "Approval Required",
                        "weight": "bold",
                        "size": "lg",
                        "color": "#1DB446",
                    }
                ],
            },
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": description,
                        "wrap": True,
                        "size": "md",
                        "color": "#666666",
                    },
                    {
                        "type": "separator",
                        "margin": "lg",
                    },
                    {
                        "type": "text",
                        "text": f"ID: {approval_id}",
                        "size": "xs",
                        "color": "#AAAAAA",
                        "margin": "md",
                    },
                ],
            },
            "footer": {
                "type": "box",
                "layout": "horizontal",
                "spacing": "md",
                "contents": [
                    {
                        "type": "button",
                        "style": "primary",
                        "color": "#1DB446",
                        "action": {
                            "type": "postback",
                            "label": "Approve",
                            "data": f"action=approve&approval_id={approval_id}",
                            "displayText": "Approved",
                        },
                    },
                    {
                        "type": "button",
                        "style": "primary",
                        "color": "#DD4444",
                        "action": {
                            "type": "postback",
                            "label": "Deny",
                            "data": f"action=deny&approval_id={approval_id}",
                            "displayText": "Denied",
                        },
                    },
                ],
            },
        }

        messages = [
            {
                "type": "flex",
                "altText": f"Approval Required: {description}",
                "contents": flex_contents,
            }
        ]

        await self._push_message(channel_id, messages)


def _split_text(text: str, max_length: int) -> list[str]:
    """Split text into chunks that fit within max_length."""
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, max_length)
        if split_at <= 0:
            split_at = max_length
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks
