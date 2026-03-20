"""WhatsApp Business API Channel Adapter for RUNE.

Uses the WhatsApp Business Cloud API for webhook verification, message
sending via REST API, media support, and template messages for approvals.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any

from rune.channels.types import (
    ChannelAdapter,
    IncomingMessage,
    OutgoingMessage,
    ResponseFormatter,
)
from rune.utils.fast_serde import json_decode
from rune.utils.logger import get_logger

log = get_logger(__name__)


def _get_whatsapp_access_token() -> str:
    """Resolve WhatsApp access token from environment variables.

    Checks TS-compatible name first, then Python-specific fallback.
    """
    return (
        os.environ.get("WHATSAPP_ACCESS_TOKEN")
        or os.environ.get("RUNE_WHATSAPP_ACCESS_TOKEN")
        or ""
    )


def _get_whatsapp_phone_number_id() -> str:
    """Resolve WhatsApp phone number ID from environment variables."""
    return (
        os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
        or os.environ.get("RUNE_WHATSAPP_PHONE_NUMBER_ID")
        or ""
    )


def _get_whatsapp_verify_token() -> str:
    """Resolve WhatsApp webhook verify token from environment variables."""
    return (
        os.environ.get("WHATSAPP_VERIFY_TOKEN")
        or os.environ.get("RUNE_WHATSAPP_VERIFY_TOKEN")
        or ""
    )


def _get_whatsapp_app_secret() -> str:
    """Resolve WhatsApp app secret from environment variables."""
    return (
        os.environ.get("WHATSAPP_APP_SECRET")
        or os.environ.get("RUNE_WHATSAPP_APP_SECRET")
        or ""
    )


def _get_whatsapp_allowed_users() -> list[str]:
    """Resolve WhatsApp allowed users from environment variables."""
    raw = (
        os.environ.get("WHATSAPP_ALLOWED_USERS")
        or os.environ.get("RUNE_WHATSAPP_ALLOWED_USERS")
        or ""
    )
    return [u.strip() for u in raw.split(",") if u.strip()] if raw else []


def _get_whatsapp_default_recipient() -> str:
    """Resolve WhatsApp default recipient from environment variables."""
    return (
        os.environ.get("WHATSAPP_DEFAULT_RECIPIENT")
        or os.environ.get("RUNE_WHATSAPP_DEFAULT_RECIPIENT")
        or ""
    )


def _get_whatsapp_webhook_config() -> dict[str, Any]:
    """Resolve WhatsApp webhook configuration from environment variables."""
    return {
        "enabled": (
            os.environ.get("WHATSAPP_WEBHOOK_ENABLED")
            or os.environ.get("RUNE_WHATSAPP_WEBHOOK_ENABLED")
            or "false"
        ).lower() == "true",
        "host": (
            os.environ.get("WHATSAPP_WEBHOOK_HOST")
            or os.environ.get("RUNE_WHATSAPP_WEBHOOK_HOST")
            or "127.0.0.1"
        ),
        "port": int(
            os.environ.get("WHATSAPP_WEBHOOK_PORT")
            or os.environ.get("RUNE_WHATSAPP_WEBHOOK_PORT")
            or "8790"
        ),
        "path": (
            os.environ.get("WHATSAPP_WEBHOOK_PATH")
            or os.environ.get("RUNE_WHATSAPP_WEBHOOK_PATH")
            or "/whatsapp/webhook"
        ),
        "verify_signature": (
            os.environ.get("WHATSAPP_VERIFY_SIGNATURE")
            or os.environ.get("RUNE_WHATSAPP_VERIFY_SIGNATURE")
            or "true"
        ).lower() == "true",
    }

# WhatsApp message length limit
_MAX_MESSAGE_LENGTH = 4096

# WhatsApp Cloud API base URL
_API_BASE = "https://graph.facebook.com/v18.0"


class WhatsAppResponseFormatter(ResponseFormatter):
    """Formats responses for WhatsApp (plain text with limited formatting)."""

    def format_text(self, text: str) -> str:
        return text

    def format_code(self, code: str, language: str = "") -> str:
        return f"```{code}```"

    def format_error(self, error: str) -> str:
        return f"*Error:* {error}"


class WhatsAppAdapter(ChannelAdapter):
    """Channel adapter for WhatsApp via the Business Cloud API."""

    def __init__(
        self,
        access_token: str,
        phone_number_id: str,
        *,
        verify_token: str = "",
        app_secret: str | None = None,
        listen_host: str = "0.0.0.0",
        listen_port: int = 8081,
    ) -> None:
        super().__init__()
        self._access_token = access_token
        self._phone_number_id = phone_number_id
        self._verify_token = verify_token
        self._app_secret = app_secret
        self._listen_host = listen_host
        self._listen_port = listen_port
        self._http_session: Any = None
        self._web_app: Any = None
        self._web_runner: Any = None
        self._formatter = WhatsAppResponseFormatter()

    @property
    def name(self) -> str:
        return "whatsapp"

    async def start(self) -> None:
        try:
            import aiohttp.web
        except ImportError as exc:
            raise ImportError(
                "aiohttp is required for the WhatsApp adapter webhook server. "
                "Install with: pip install aiohttp"
            ) from exc

        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for the WhatsApp adapter. "
                "Install with: pip install httpx"
            ) from exc

        self._http_session = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json",
            }
        )

        # Set up webhook server
        self._web_app = aiohttp.web.Application()
        self._web_app.router.add_get("/whatsapp/webhook", self._handle_verify)
        self._web_app.router.add_post("/whatsapp/webhook", self._handle_webhook)

        self._web_runner = aiohttp.web.AppRunner(self._web_app)
        await self._web_runner.setup()
        site = aiohttp.web.TCPSite(
            self._web_runner, self._listen_host, self._listen_port
        )
        await site.start()

        log.info(
            "whatsapp_adapter_started",
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
        log.info("whatsapp_adapter_stopped")

    async def send(self, channel_id: str, message: OutgoingMessage) -> str | None:
        if self._http_session is None:
            log.warning("whatsapp_send_no_session")
            return None

        text = self._formatter.format_text(message.text)

        # Send text message
        chunks = _split_text(text, _MAX_MESSAGE_LENGTH)
        for chunk in chunks:
            body = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": channel_id,
                "type": "text",
                "text": {"preview_url": False, "body": chunk},
            }

            # Context (reply) support
            if message.reply_to:
                body["context"] = {"message_id": message.reply_to}

            await self._api_post(
                f"{_API_BASE}/{self._phone_number_id}/messages", body
            )

        # Send media attachments
        for attachment in message.attachments:
            await self._send_media(channel_id, attachment)

        return None

    # -- Internals ----------------------------------------------------------

    async def _api_post(self, url: str, body: dict[str, Any]) -> dict[str, Any] | None:
        """Make a POST request to the WhatsApp Cloud API."""
        if self._http_session is None:
            return None
        try:
            resp = await self._http_session.post(url, json=body)
            data = resp.json()
            if resp.status_code >= 400:
                log.error(
                    "whatsapp_api_error",
                    status=resp.status_code,
                    error=data,
                )
                return None
            return data
        except Exception as exc:
            log.error("whatsapp_api_request_failed", error=str(exc))
            return None

    async def _handle_verify(self, request: Any) -> Any:
        """Handle WhatsApp webhook verification (GET request)."""
        import aiohttp.web

        mode = request.query.get("hub.mode", "")
        token = request.query.get("hub.verify_token", "")
        challenge = request.query.get("hub.challenge", "")

        if mode == "subscribe" and token == self._verify_token:
            log.info("whatsapp_webhook_verified")
            return aiohttp.web.Response(text=challenge)

        log.warning("whatsapp_webhook_verify_failed")
        return aiohttp.web.Response(status=403, text="Forbidden")

    async def _handle_webhook(self, request: Any) -> Any:
        """Handle an incoming WhatsApp webhook notification (POST request)."""
        import aiohttp.web

        body_bytes = await request.read()

        # Verify signature if app_secret is configured
        if self._app_secret:
            signature = request.headers.get("X-Hub-Signature-256", "")
            if not self._verify_signature(body_bytes, signature):
                log.warning("whatsapp_webhook_invalid_signature")
                return aiohttp.web.Response(status=401, text="Invalid signature")

        payload = json_decode(body_bytes)

        # Process entries
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                if change.get("field") == "messages":
                    await self._process_messages(value)

        return aiohttp.web.Response(text="OK")

    def _verify_signature(self, body: bytes, signature: str) -> bool:
        """Verify the X-Hub-Signature-256 header."""
        if not self._app_secret:
            return True
        expected = (
            "sha256="
            + hmac.new(
                self._app_secret.encode(),
                body,
                hashlib.sha256,
            ).hexdigest()
        )
        return hmac.compare_digest(expected, signature)

    async def _process_messages(self, value: dict[str, Any]) -> None:
        """Process messages from a webhook notification value."""
        if self._on_message is None:
            return

        contacts = {
            c["wa_id"]: c.get("profile", {}).get("name", "")
            for c in value.get("contacts", [])
        }
        phone_number_id = (
            value.get("metadata", {}).get("phone_number_id", "")
        )

        for msg in value.get("messages", []):
            msg_type = msg.get("type", "")
            text = ""
            attachments: list[dict[str, Any]] = []

            if msg_type == "text":
                text = msg.get("text", {}).get("body", "")
            elif msg_type in ("image", "video", "audio", "document", "sticker"):
                media_obj = msg.get(msg_type, {})
                text = media_obj.get("caption", "")
                attachments.append({
                    "type": msg_type,
                    "media_id": media_obj.get("id", ""),
                    "mime_type": media_obj.get("mime_type", ""),
                    "filename": media_obj.get("filename", ""),
                    "sha256": media_obj.get("sha256", ""),
                })
            elif msg_type == "interactive":
                interactive = msg.get("interactive", {})
                resp_type = interactive.get("type", "")
                if resp_type == "button_reply":
                    text = interactive.get("button_reply", {}).get("id", "")
                elif resp_type == "list_reply":
                    text = interactive.get("list_reply", {}).get("id", "")

            sender = msg.get("from", "")
            context = msg.get("context", {})

            incoming = IncomingMessage(
                channel_id=sender,
                sender_id=sender,
                text=text,
                attachments=attachments,
                reply_to=context.get("id"),
                metadata={
                    "channel_name": self.name,
                    "message_id": msg.get("id", ""),
                    "timestamp": msg.get("timestamp", ""),
                    "type": msg_type,
                    "phone_number_id": phone_number_id,
                    "sender_name": contacts.get(sender, ""),
                },
            )

            # Mark message as read
            await self._mark_read(msg.get("id", ""))

            await self._on_message(incoming)

    async def _mark_read(self, message_id: str) -> None:
        """Mark a message as read."""
        if not message_id:
            return
        await self._api_post(
            f"{_API_BASE}/{self._phone_number_id}/messages",
            {
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message_id,
            },
        )

    async def _send_media(
        self, channel_id: str, attachment: dict[str, Any]
    ) -> None:
        """Send a media attachment to the recipient."""
        media_type = attachment.get("type", "document")
        if media_type not in ("image", "video", "audio", "document", "sticker"):
            media_type = "document"

        body: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": channel_id,
            "type": media_type,
        }

        # Support both media ID and URL
        media_payload: dict[str, Any] = {}
        if attachment.get("media_id"):
            media_payload["id"] = attachment["media_id"]
        elif attachment.get("url"):
            media_payload["link"] = attachment["url"]
        if attachment.get("caption"):
            media_payload["caption"] = attachment["caption"]
        if attachment.get("filename"):
            media_payload["filename"] = attachment["filename"]

        body[media_type] = media_payload

        await self._api_post(
            f"{_API_BASE}/{self._phone_number_id}/messages", body
        )

    async def ask_question(self, channel_id: str, question: str, options: list[str] | None = None, timeout: float = 300.0) -> str:
        """Ask a question via WhatsApp. Sends the question and returns empty string."""
        if self._http_session is None:
            return ""
        text = question
        if options:
            text += "\n" + "\n".join(f"• {opt}" for opt in options)
        body = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": channel_id,
            "type": "text",
            "text": {"preview_url": False, "body": text},
        }
        await self._api_post(f"{_API_BASE}/{self._phone_number_id}/messages", body)
        return ""

    async def send_approval(
        self,
        channel_id: str,
        description: str,
        approval_id: str,
    ) -> None:
        """Send an interactive approval message using template or buttons."""
        if self._http_session is None:
            return

        # Send interactive buttons for approval
        body: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": channel_id,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "header": {
                    "type": "text",
                    "text": "Approval Required",
                },
                "body": {
                    "text": description,
                },
                "footer": {
                    "text": f"Approval ID: {approval_id}",
                },
                "action": {
                    "buttons": [
                        {
                            "type": "reply",
                            "reply": {
                                "id": f"approve:{approval_id}",
                                "title": "Approve",
                            },
                        },
                        {
                            "type": "reply",
                            "reply": {
                                "id": f"deny:{approval_id}",
                                "title": "Deny",
                            },
                        },
                    ]
                },
            },
        }

        result = await self._api_post(
            f"{_API_BASE}/{self._phone_number_id}/messages", body
        )
        if result is None:
            log.error("whatsapp_approval_template_failed")

    async def send_template_message(
        self,
        channel_id: str,
        template_name: str,
        language_code: str = "en",
        components: list[dict[str, Any]] | None = None,
    ) -> None:
        """Send a pre-approved template message."""
        body: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "to": channel_id,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": language_code},
            },
        }
        if components:
            body["template"]["components"] = components

        await self._api_post(
            f"{_API_BASE}/{self._phone_number_id}/messages", body
        )


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
