"""Google Chat Channel Adapter for RUNE.

Uses Google API client for webhook-based message handling, card-based
approval UI, and space/thread routing.
"""

from __future__ import annotations

import asyncio
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


def _get_google_chat_webhook_url() -> str:
    """Resolve Google Chat webhook URL from environment variables.

    Checks TS-compatible name first, then Python-specific fallback.
    """
    return (
        os.environ.get("GOOGLE_CHAT_WEBHOOK_URL")
        or os.environ.get("RUNE_GOOGLE_CHAT_WEBHOOK_URL")
        or ""
    )


def _get_google_chat_service_account_key_file() -> str:
    """Resolve Google Chat service account key file path from environment variables."""
    return (
        os.environ.get("GOOGLE_CHAT_SERVICE_ACCOUNT_KEY_FILE")
        or os.environ.get("RUNE_GOOGLE_CHAT_SERVICE_ACCOUNT_KEY_FILE")
        or ""
    )


def _get_google_chat_service_account_key_json() -> str:
    """Resolve Google Chat service account key JSON from environment variables."""
    return (
        os.environ.get("GOOGLE_CHAT_SERVICE_ACCOUNT_KEY_JSON")
        or os.environ.get("RUNE_GOOGLE_CHAT_SERVICE_ACCOUNT_KEY_JSON")
        or ""
    )


def _get_google_chat_service_account_email() -> str:
    """Resolve Google Chat service account email from environment variables."""
    return (
        os.environ.get("GOOGLE_CHAT_SERVICE_ACCOUNT_EMAIL")
        or os.environ.get("RUNE_GOOGLE_CHAT_SERVICE_ACCOUNT_EMAIL")
        or ""
    )


def _get_google_chat_service_account_private_key() -> str:
    """Resolve Google Chat service account private key from environment variables."""
    return (
        os.environ.get("GOOGLE_CHAT_SERVICE_ACCOUNT_PRIVATE_KEY")
        or os.environ.get("RUNE_GOOGLE_CHAT_SERVICE_ACCOUNT_PRIVATE_KEY")
        or ""
    )


def _get_google_chat_service_account_token_uri() -> str:
    """Resolve Google Chat service account token URI from environment variables."""
    return (
        os.environ.get("GOOGLE_CHAT_SERVICE_ACCOUNT_TOKEN_URI")
        or os.environ.get("RUNE_GOOGLE_CHAT_SERVICE_ACCOUNT_TOKEN_URI")
        or "https://oauth2.googleapis.com/token"
    )


def _get_google_chat_default_space() -> str:
    """Resolve Google Chat default space from environment variables."""
    return (
        os.environ.get("GOOGLE_CHAT_DEFAULT_SPACE")
        or os.environ.get("RUNE_GOOGLE_CHAT_DEFAULT_SPACE")
        or ""
    )


def _get_google_chat_interaction_config() -> dict[str, Any]:
    """Resolve Google Chat interaction (bidirectional) configuration."""
    return {
        "enabled": (
            os.environ.get("GOOGLE_CHAT_INTERACTION_ENABLED")
            or os.environ.get("RUNE_GOOGLE_CHAT_INTERACTION_ENABLED")
            or "false"
        ).lower() == "true",
        "host": (
            os.environ.get("GOOGLE_CHAT_INTERACTION_HOST")
            or os.environ.get("RUNE_GOOGLE_CHAT_INTERACTION_HOST")
            or "127.0.0.1"
        ),
        "port": int(
            os.environ.get("GOOGLE_CHAT_INTERACTION_PORT")
            or os.environ.get("RUNE_GOOGLE_CHAT_INTERACTION_PORT")
            or "8787"
        ),
        "path": (
            os.environ.get("GOOGLE_CHAT_INTERACTION_PATH")
            or os.environ.get("RUNE_GOOGLE_CHAT_INTERACTION_PATH")
            or "/google-chat/interactions"
        ),
        "verify_requests": (
            os.environ.get("GOOGLE_CHAT_VERIFY_REQUESTS")
            or os.environ.get("RUNE_GOOGLE_CHAT_VERIFY_REQUESTS")
            or "true"
        ).lower() == "true",
        "auth_audience": (
            os.environ.get("GOOGLE_CHAT_AUTH_AUDIENCE")
            or os.environ.get("RUNE_GOOGLE_CHAT_AUTH_AUDIENCE")
            or ""
        ),
        "sync_ack_text": (
            os.environ.get("GOOGLE_CHAT_SYNC_ACK_TEXT")
            or os.environ.get("RUNE_GOOGLE_CHAT_SYNC_ACK_TEXT")
            or "Received. Working on it now."
        ),
    }

# Google Chat message length limit
_MAX_MESSAGE_LENGTH = 4096


class GoogleChatResponseFormatter(ResponseFormatter):
    """Formats responses for Google Chat text formatting."""

    def format_text(self, text: str) -> str:
        return text

    def format_code(self, code: str, language: str = "") -> str:
        return f"```\n{code}\n```"

    def format_error(self, error: str) -> str:
        return f"*Error:* {error}"


class GoogleChatAdapter(ChannelAdapter):
    """Channel adapter for Google Chat via Google API client and webhooks."""

    def __init__(
        self,
        service_account_path: str,
        project_id: str,
        *,
        webhook_secret: str | None = None,
        listen_host: str = "0.0.0.0",
        listen_port: int = 8080,
    ) -> None:
        super().__init__()
        self._service_account_path = service_account_path
        self._project_id = project_id
        self._webhook_secret = webhook_secret
        self._listen_host = listen_host
        self._listen_port = listen_port
        self._chat_service: Any = None
        self._web_app: Any = None
        self._web_runner: Any = None
        self._formatter = GoogleChatResponseFormatter()

    @property
    def name(self) -> str:
        return "google_chat"

    async def start(self) -> None:
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise ImportError(
                "google-api-python-client and google-auth are required for the "
                "Google Chat adapter. Install with: "
                "pip install google-api-python-client google-auth"
            ) from exc

        try:
            import aiohttp.web
        except ImportError as exc:
            raise ImportError(
                "aiohttp is required for the Google Chat webhook server. "
                "Install with: pip install aiohttp"
            ) from exc

        # Build the Chat API service
        credentials = service_account.Credentials.from_service_account_file(
            self._service_account_path,
            scopes=["https://www.googleapis.com/auth/chat.bot"],
        )
        self._chat_service = build("chat", "v1", credentials=credentials)

        # Set up webhook server
        self._web_app = aiohttp.web.Application()
        self._web_app.router.add_post("/google-chat/webhook", self._handle_webhook)

        self._web_runner = aiohttp.web.AppRunner(self._web_app)
        await self._web_runner.setup()
        site = aiohttp.web.TCPSite(
            self._web_runner, self._listen_host, self._listen_port
        )
        await site.start()

        log.info(
            "google_chat_adapter_started",
            host=self._listen_host,
            port=self._listen_port,
        )

    async def stop(self) -> None:
        self._cleanup_pending()
        if self._web_runner is not None:
            await self._web_runner.cleanup()
            self._web_runner = None
        self._web_app = None
        self._chat_service = None
        log.info("google_chat_adapter_stopped")

    async def send(self, channel_id: str, message: OutgoingMessage) -> str | None:
        if self._chat_service is None:
            log.warning("google_chat_send_no_service")
            return None

        text = self._formatter.format_text(message.text)

        # Build request body
        body: dict[str, Any] = {"text": text}

        # If there are cards in metadata, use them
        cards = message.metadata.get("cards")
        if cards is not None:
            body["cardsV2"] = cards

        # Thread routing: if reply_to is set, use it as the thread key
        kwargs: dict[str, Any] = {}
        if message.reply_to:
            body["thread"] = {"name": message.reply_to}
            kwargs["messageReplyOption"] = "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"

        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._chat_service.spaces()
                .messages()
                .create(
                    parent=channel_id,
                    body=body,
                    **kwargs,
                )
                .execute(),
            )
        except Exception as exc:
            log.error("google_chat_send_failed", error=str(exc))

        return None

    async def edit_message(
        self, channel_id: str, message_id: str, new_text: str
    ) -> None:
        if self._chat_service is None:
            return
        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._chat_service.spaces()
                .messages()
                .update(
                    name=message_id,
                    updateMask="text",
                    body={"text": new_text},
                )
                .execute(),
            )
        except Exception as exc:
            log.error("google_chat_edit_failed", error=str(exc))

    async def delete_message(self, channel_id: str, message_id: str) -> None:
        if self._chat_service is None:
            return
        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._chat_service.spaces()
                .messages()
                .delete(name=message_id)
                .execute(),
            )
        except Exception as exc:
            log.error("google_chat_delete_failed", error=str(exc))

    # -- Internals ----------------------------------------------------------

    async def _handle_webhook(self, request: Any) -> Any:
        """Handle an incoming Google Chat webhook event."""
        import aiohttp.web

        # Verify webhook signature if secret is configured
        if self._webhook_secret:
            signature = request.headers.get("X-Goog-Signature", "")
            body_bytes = await request.read()
            if not self._verify_signature(body_bytes, signature):
                log.warning("google_chat_webhook_invalid_signature")
                return aiohttp.web.Response(status=401, text="Invalid signature")
            payload = json_decode(body_bytes)
        else:
            payload = await request.json()

        event_type = payload.get("type", "")

        if event_type == "MESSAGE":
            await self._handle_message_event(payload)
        elif event_type == "CARD_CLICKED":
            await self._handle_card_click(payload)
        elif event_type == "ADDED_TO_SPACE":
            log.info(
                "google_chat_added_to_space",
                space=payload.get("space", {}).get("name"),
            )

        return aiohttp.web.json_response({})

    def _verify_signature(self, body: bytes, signature: str) -> bool:
        """Verify the webhook request signature."""
        if not self._webhook_secret:
            return True
        expected = hmac.new(
            self._webhook_secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    async def _handle_message_event(self, payload: dict[str, Any]) -> None:
        """Convert a Google Chat MESSAGE event to IncomingMessage and dispatch."""
        if self._on_message is None:
            return

        message = payload.get("message", {})
        sender = message.get("sender", {})
        space = payload.get("space", {})
        thread = message.get("thread", {})

        text = message.get("argumentText", "") or message.get("text", "")
        attachments = self._extract_attachments(message)

        incoming = IncomingMessage(
            channel_id=space.get("name", ""),
            sender_id=sender.get("name", ""),
            text=text.strip(),
            attachments=attachments,
            reply_to=thread.get("name"),
            metadata={
                "channel_name": self.name,
                "message_name": message.get("name", ""),
                "space_type": space.get("type", ""),
                "space_display_name": space.get("displayName", ""),
                "sender_display_name": sender.get("displayName", ""),
                "sender_email": sender.get("email", ""),
                "thread_name": thread.get("name", ""),
            },
        )

        # Resolve any pending question Future for this space
        space_name = space.get("name", "")
        question_future = self._pending_questions.pop(space_name, None)
        if question_future is not None and not question_future.done():
            question_future.set_result(text.strip())

        await self._on_message(incoming)

    async def _handle_card_click(self, payload: dict[str, Any]) -> None:
        """Handle a card button click (used for approval flows)."""
        action = payload.get("action", {})
        action_method = action.get("actionMethodName", "")
        parameters = {
            p["key"]: p["value"] for p in action.get("parameters", [])
        }

        log.info(
            "google_chat_card_click",
            action_method=action_method,
            parameters=parameters,
        )

        # Resolve pending approval futures
        if action_method in ("approve", "deny"):
            approval_id = parameters.get("approval_id", "")
            if approval_id:
                approved = action_method == "approve"
                decision = ApprovalDecision.APPROVE_ONCE if approved else ApprovalDecision.DENY
                self._resolve_approval(approval_id, decision)

                # Send a follow-up message with the decision
                sender = payload.get("user", {})
                user_name = sender.get("displayName", "someone")
                status = "Approved" if approved else "Denied"
                space = payload.get("space", {}).get("name", "")
                if space and self._chat_service is not None:
                    try:
                        await asyncio.get_running_loop().run_in_executor(
                            None,
                            lambda: self._chat_service.spaces()
                            .messages()
                            .create(
                                parent=space,
                                body={"text": f"Approval {approval_id}: *{status}* by {user_name}"},
                            )
                            .execute(),
                        )
                    except Exception as exc:
                        log.warning("google_chat_approval_update_failed", error=str(exc))

    def _extract_attachments(
        self, message: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract attachments from a Google Chat message."""
        raw_attachments = message.get("attachment", [])
        return [
            {
                "type": att.get("contentType", "file"),
                "name": att.get("contentName", ""),
                "download_uri": att.get("downloadUri", ""),
                "source": att.get("source", ""),
            }
            for att in raw_attachments
        ]

    async def ask_question(self, channel_id: str, question: str, options: list[str] | None = None, timeout: float = 300.0) -> str:
        """Ask a question via Google Chat and wait for a reply.

        Sends the question, then awaits a Future that is resolved when the
        next message arrives in the same space.  Returns the reply text,
        or ``""`` on timeout.
        """
        if self._chat_service is None:
            return ""
        text = question
        if options:
            text += "\n" + "\n".join(f"• {opt}" for opt in options)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_questions[channel_id] = future

        try:
            await loop.run_in_executor(
                None,
                lambda: self._chat_service.spaces()
                .messages()
                .create(parent=channel_id, body={"text": text})
                .execute(),
            )
        except Exception as exc:
            log.error("google_chat_ask_question_failed", error=str(exc))
            self._pending_questions.pop(channel_id, None)
            return ""

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            self._pending_questions.pop(channel_id, None)
            log.warning("google_chat_ask_question_timeout", channel_id=channel_id)
            return ""

    async def send_approval(
        self,
        channel_id: str,
        description: str,
        approval_id: str,
    ) -> None:
        """Send a card-based approval UI to the specified space."""
        if self._chat_service is None:
            return

        card_body: dict[str, Any] = {
            "cardsV2": [
                {
                    "cardId": f"approval-{approval_id}",
                    "card": {
                        "header": {
                            "title": "Approval Required",
                            "subtitle": f"ID: {approval_id}",
                            "imageUrl": "",
                            "imageType": "CIRCLE",
                        },
                        "sections": [
                            {
                                "header": "Details",
                                "widgets": [
                                    {
                                        "textParagraph": {
                                            "text": description,
                                        }
                                    }
                                ],
                            },
                            {
                                "widgets": [
                                    {
                                        "buttonList": {
                                            "buttons": [
                                                {
                                                    "text": "Approve",
                                                    "color": {
                                                        "red": 0.0,
                                                        "green": 0.6,
                                                        "blue": 0.3,
                                                        "alpha": 1.0,
                                                    },
                                                    "onClick": {
                                                        "action": {
                                                            "actionMethodName": "approve",
                                                            "parameters": [
                                                                {
                                                                    "key": "approval_id",
                                                                    "value": approval_id,
                                                                }
                                                            ],
                                                        }
                                                    },
                                                },
                                                {
                                                    "text": "Deny",
                                                    "color": {
                                                        "red": 0.8,
                                                        "green": 0.0,
                                                        "blue": 0.0,
                                                        "alpha": 1.0,
                                                    },
                                                    "onClick": {
                                                        "action": {
                                                            "actionMethodName": "deny",
                                                            "parameters": [
                                                                {
                                                                    "key": "approval_id",
                                                                    "value": approval_id,
                                                                }
                                                            ],
                                                        }
                                                    },
                                                },
                                            ]
                                        }
                                    }
                                ],
                            },
                        ],
                    },
                }
            ],
            "text": f"Approval Required: {description}",
        }

        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._chat_service.spaces()
                .messages()
                .create(parent=channel_id, body=card_body)
                .execute(),
            )
        except Exception as exc:
            log.error("google_chat_approval_card_failed", error=str(exc))


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
