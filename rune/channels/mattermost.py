"""Mattermost Channel Adapter for RUNE.

Uses the Mattermost WebSocket API for real-time message handling, REST API
for sending, slash commands, and interactive message buttons for approvals.
"""

from __future__ import annotations

import asyncio
import contextlib
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


def _get_mattermost_base_url() -> str:
    """Resolve Mattermost base URL from environment variables.

    Checks TS-compatible name first, then Python-specific fallback.
    """
    return (
        os.environ.get("MATTERMOST_BASE_URL")
        or os.environ.get("RUNE_MATTERMOST_BASE_URL")
        or ""
    )


def _get_mattermost_bot_token() -> str:
    """Resolve Mattermost bot token from environment variables."""
    return (
        os.environ.get("MATTERMOST_BOT_TOKEN")
        or os.environ.get("RUNE_MATTERMOST_BOT_TOKEN")
        or ""
    )


def _get_mattermost_team_name() -> str:
    """Resolve Mattermost team name from environment variables."""
    return (
        os.environ.get("MATTERMOST_TEAM_NAME")
        or os.environ.get("RUNE_MATTERMOST_TEAM_NAME")
        or ""
    )


def _get_mattermost_allowed_users() -> list[str]:
    """Resolve Mattermost allowed users from environment variables."""
    raw = (
        os.environ.get("MATTERMOST_ALLOWED_USERS")
        or os.environ.get("RUNE_MATTERMOST_ALLOWED_USERS")
        or ""
    )
    return [u.strip() for u in raw.split(",") if u.strip()] if raw else []


def _get_mattermost_default_recipient() -> str:
    """Resolve Mattermost default recipient from environment variables."""
    return (
        os.environ.get("MATTERMOST_DEFAULT_RECIPIENT")
        or os.environ.get("RUNE_MATTERMOST_DEFAULT_RECIPIENT")
        or ""
    )


def _get_mattermost_webhook_config() -> dict[str, Any]:
    """Resolve Mattermost webhook configuration from environment variables."""
    return {
        "enabled": (
            os.environ.get("MATTERMOST_WEBHOOK_ENABLED")
            or os.environ.get("RUNE_MATTERMOST_WEBHOOK_ENABLED")
            or "false"
        ).lower() == "true",
        "host": (
            os.environ.get("MATTERMOST_WEBHOOK_HOST")
            or os.environ.get("RUNE_MATTERMOST_WEBHOOK_HOST")
            or "127.0.0.1"
        ),
        "port": int(
            os.environ.get("MATTERMOST_WEBHOOK_PORT")
            or os.environ.get("RUNE_MATTERMOST_WEBHOOK_PORT")
            or "8795"
        ),
        "path": (
            os.environ.get("MATTERMOST_WEBHOOK_PATH")
            or os.environ.get("RUNE_MATTERMOST_WEBHOOK_PATH")
            or "/mattermost/webhook"
        ),
        "verify_token": (
            os.environ.get("MATTERMOST_VERIFY_TOKEN")
            or os.environ.get("RUNE_MATTERMOST_VERIFY_TOKEN")
            or ""
        ),
        "ack_text": (
            os.environ.get("MATTERMOST_WEBHOOK_ACK_TEXT")
            or os.environ.get("RUNE_MATTERMOST_WEBHOOK_ACK_TEXT")
            or "Received. Working on it now."
        ),
    }

# Mattermost message length limit
_MAX_MESSAGE_LENGTH = 16383


class MattermostResponseFormatter(ResponseFormatter):
    """Formats responses using Mattermost markdown syntax."""

    def format_text(self, text: str) -> str:
        return text

    def format_code(self, code: str, language: str = "") -> str:
        return f"```{language}\n{code}\n```"

    def format_error(self, error: str) -> str:
        return f"**Error:** {error}"


class MattermostAdapter(ChannelAdapter):
    """Channel adapter for Mattermost via WebSocket and REST API."""

    def __init__(
        self,
        url: str,
        token: str,
        *,
        listen_host: str = "0.0.0.0",
        listen_port: int = 8083,
    ) -> None:
        super().__init__()
        # Normalise URL (strip trailing slash)
        self._url = url.rstrip("/")
        self._token = token
        self._listen_host = listen_host
        self._listen_port = listen_port
        self._http_session: Any = None
        self._ws: Any = None
        self._ws_task: asyncio.Task[None] | None = None
        self._web_app: Any = None
        self._web_runner: Any = None
        self._formatter = MattermostResponseFormatter()
        self._bot_user_id: str | None = None
        self._running = False
        self._seq = 0

    @property
    def name(self) -> str:
        return "mattermost"

    async def start(self) -> None:
        try:
            import aiohttp.web
        except ImportError as exc:
            raise ImportError(
                "aiohttp is required for the Mattermost adapter webhook server and WebSocket. "
                "Install with: pip install aiohttp"
            ) from exc

        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for the Mattermost adapter. "
                "Install with: pip install httpx"
            ) from exc

        self._http_session = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            }
        )

        # Resolve bot user ID
        try:
            me = await self._api_get("/api/v4/users/me")
            if me:
                self._bot_user_id = me.get("id")
                log.info("mattermost_auth_ok", bot_user_id=self._bot_user_id)
        except Exception as exc:
            log.error("mattermost_auth_failed", error=str(exc))

        # Start WebSocket connection
        self._running = True
        self._ws_task = asyncio.create_task(self._ws_loop())

        # Set up interactive message endpoint (for button clicks / slash cmds)
        self._web_app = aiohttp.web.Application()
        self._web_app.router.add_post(
            "/mattermost/actions", self._handle_action
        )
        self._web_app.router.add_post(
            "/mattermost/slash", self._handle_slash_command
        )

        self._web_runner = aiohttp.web.AppRunner(self._web_app)
        await self._web_runner.setup()
        site = aiohttp.web.TCPSite(
            self._web_runner, self._listen_host, self._listen_port
        )
        await site.start()

        log.info(
            "mattermost_adapter_started",
            url=self._url,
            host=self._listen_host,
            port=self._listen_port,
        )

    async def stop(self) -> None:
        self._running = False
        self._cleanup_pending()

        if self._ws is not None:
            await self._ws.close()
            self._ws = None

        if self._ws_task is not None:
            self._ws_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ws_task
            self._ws_task = None

        if self._web_runner is not None:
            await self._web_runner.cleanup()
            self._web_runner = None
        self._web_app = None

        if self._http_session is not None:
            await self._http_session.aclose()
            self._http_session = None

        log.info("mattermost_adapter_stopped")

    async def send(self, channel_id: str, message: OutgoingMessage) -> str | None:
        if self._http_session is None:
            log.warning("mattermost_send_no_session")
            return None

        text = self._formatter.format_text(message.text)

        # Build post body
        body: dict[str, Any] = {
            "channel_id": channel_id,
            "message": text,
        }

        # Thread routing via root_id
        if message.reply_to:
            body["root_id"] = message.reply_to

        # Attach props (interactive attachments, etc.)
        props = message.metadata.get("props")
        if props is not None:
            body["props"] = props

        chunks = _split_text(text, _MAX_MESSAGE_LENGTH)
        for i, chunk in enumerate(chunks):
            post_body = {**body, "message": chunk}
            if i > 0:
                # Only the first chunk keeps the root_id for thread routing
                post_body.pop("props", None)
            await self._api_post("/api/v4/posts", post_body)

        # Send file attachments
        for attachment in message.attachments:
            await self._send_attachment(channel_id, attachment)

        return None

    async def edit_message(
        self, channel_id: str, message_id: str, new_text: str
    ) -> None:
        if self._http_session is None:
            return
        try:
            await self._api_put(
                f"/api/v4/posts/{message_id}",
                {"id": message_id, "message": new_text},
            )
        except Exception as exc:
            log.error("mattermost_edit_failed", error=str(exc))

    async def delete_message(self, channel_id: str, message_id: str) -> None:
        if self._http_session is None:
            return
        try:
            await self._api_delete(f"/api/v4/posts/{message_id}")
        except Exception as exc:
            log.error("mattermost_delete_failed", error=str(exc))

    # -- REST helpers -------------------------------------------------------

    async def _api_get(self, path: str) -> dict[str, Any] | None:
        """Make a GET request to the Mattermost REST API."""
        if self._http_session is None:
            return None
        try:
            resp = await self._http_session.get(f"{self._url}{path}")
            if resp.status_code >= 400:
                log.error(
                    "mattermost_api_error",
                    method="GET",
                    path=path,
                    status=resp.status_code,
                    error=resp.text,
                )
                return None
            return resp.json()
        except Exception as exc:
            log.error("mattermost_api_request_failed", error=str(exc))
            return None

    async def _api_post(
        self, path: str, body: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Make a POST request to the Mattermost REST API."""
        if self._http_session is None:
            return None
        try:
            resp = await self._http_session.post(
                f"{self._url}{path}", json=body
            )
            if resp.status_code >= 400:
                log.error(
                    "mattermost_api_error",
                    method="POST",
                    path=path,
                    status=resp.status_code,
                    error=resp.text,
                )
                return None
            return resp.json()
        except Exception as exc:
            log.error("mattermost_api_request_failed", error=str(exc))
            return None

    async def _api_put(
        self, path: str, body: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Make a PUT request to the Mattermost REST API."""
        if self._http_session is None:
            return None
        try:
            resp = await self._http_session.put(
                f"{self._url}{path}", json=body
            )
            if resp.status_code >= 400:
                log.error(
                    "mattermost_api_error",
                    method="PUT",
                    path=path,
                    status=resp.status_code,
                    error=resp.text,
                )
                return None
            return resp.json()
        except Exception as exc:
            log.error("mattermost_api_request_failed", error=str(exc))
            return None

    async def _api_delete(self, path: str) -> bool:
        """Make a DELETE request to the Mattermost REST API."""
        if self._http_session is None:
            return False
        try:
            resp = await self._http_session.delete(f"{self._url}{path}")
            if resp.status_code >= 400:
                log.error(
                    "mattermost_api_error",
                    method="DELETE",
                    path=path,
                    status=resp.status_code,
                    error=resp.text,
                )
                return False
            return True
        except Exception as exc:
            log.error("mattermost_api_request_failed", error=str(exc))
            return False

    # -- WebSocket ----------------------------------------------------------

    async def _ws_loop(self) -> None:
        """Maintain a persistent WebSocket connection to Mattermost."""
        import aiohttp

        ws_url = self._url.replace("https://", "wss://").replace(
            "http://", "ws://"
        )
        ws_url = f"{ws_url}/api/v4/websocket"

        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(ws_url) as ws:
                        self._ws = ws

                        # Authenticate
                        self._seq += 1
                        await ws.send_json({
                            "seq": self._seq,
                            "action": "authentication_challenge",
                            "data": {"token": self._token},
                        })

                        log.info("mattermost_ws_connected")

                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._handle_ws_message(
                                    json_decode(msg.data)
                                )
                            elif msg.type in (
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.ERROR,
                            ):
                                break

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._running:
                    log.warning(
                        "mattermost_ws_disconnected",
                        error=str(exc),
                    )
                    await asyncio.sleep(5)  # Reconnect delay

        self._ws = None

    async def _handle_ws_message(self, data: dict[str, Any]) -> None:
        """Handle a WebSocket event from Mattermost."""
        event = data.get("event", "")

        if event == "posted":
            await self._handle_posted_event(data)
        elif event == "post_edited":
            log.debug("mattermost_post_edited", data=data)
        elif event == "post_deleted":
            log.debug("mattermost_post_deleted", data=data)

    async def _handle_posted_event(self, data: dict[str, Any]) -> None:
        """Handle a 'posted' WebSocket event."""
        if self._on_message is None:
            return

        raw_post = data.get("data", {}).get("post", "")
        if isinstance(raw_post, str):
            try:
                post = json_decode(raw_post)
            except ValueError:
                return
        else:
            post = raw_post

        # Ignore bot's own messages
        user_id = post.get("user_id", "")
        if user_id == self._bot_user_id:
            return

        # Extract file IDs as attachments
        file_ids = post.get("file_ids") or []
        attachments = [
            {
                "type": "file",
                "file_id": fid,
                "url": f"{self._url}/api/v4/files/{fid}",
            }
            for fid in file_ids
        ]

        channel_id = post.get("channel_id", "")
        sender_name = data.get("data", {}).get("sender_name", "")

        incoming = IncomingMessage(
            channel_id=channel_id,
            sender_id=user_id,
            text=post.get("message", ""),
            attachments=attachments,
            reply_to=post.get("root_id") or None,
            metadata={
                "channel_name": self.name,
                "post_id": post.get("id", ""),
                "team_id": data.get("data", {}).get("team_id", ""),
                "channel_type": data.get("data", {}).get("channel_type", ""),
                "sender_name": sender_name,
            },
        )

        await self._on_message(incoming)

    # -- Interactive messages / slash commands -------------------------------

    async def _handle_action(self, request: Any) -> Any:
        """Handle interactive message button clicks."""
        import aiohttp.web

        try:
            payload = await request.json()
        except Exception:
            return aiohttp.web.Response(status=400, text="Bad request")

        context = payload.get("context", {})
        action_id = context.get("action_id", "")
        user_id = payload.get("user_id", "")

        log.info(
            "mattermost_action",
            action_id=action_id,
            user_id=user_id,
        )

        # Resolve pending approval futures from interactive message buttons.
        # action_id format: "approve:<approval_id>" or "deny:<approval_id>"
        if ":" in action_id:
            decision, approval_id = action_id.split(":", 1)
            if decision in ("approve", "deny"):
                approved = decision == "approve"
                d = ApprovalDecision.APPROVE_ONCE if approved else ApprovalDecision.DENY
                self._resolve_approval(approval_id, d)

                status = "Approved" if approved else "Denied"
                return aiohttp.web.json_response({
                    "update": {"message": f"Approval {approval_id}: **{status}**"}
                })

        # Respond with an update to acknowledge the action
        return aiohttp.web.json_response({"update": {"message": "Action received."}})

    async def _handle_slash_command(self, request: Any) -> Any:
        """Handle incoming slash commands."""
        import aiohttp.web

        try:
            data = await request.post()
        except Exception:
            return aiohttp.web.Response(status=400, text="Bad request")

        command = data.get("command", "")
        text = data.get("text", "")
        user_id = data.get("user_id", "")
        channel_id = data.get("channel_id", "")

        log.info(
            "mattermost_slash_command",
            command=command,
            user_id=user_id,
            channel_id=channel_id,
        )

        if self._on_message is not None:
            incoming = IncomingMessage(
                channel_id=channel_id,
                sender_id=user_id,
                text=f"{command} {text}".strip(),
                metadata={
                    "channel_name": self.name,
                    "type": "slash_command",
                    "command": command,
                    "trigger_id": data.get("trigger_id", ""),
                    "team_id": data.get("team_id", ""),
                },
            )
            await self._on_message(incoming)

        return aiohttp.web.json_response({
            "response_type": "ephemeral",
            "text": "Processing your command...",
        })

    async def _send_attachment(
        self, channel_id: str, attachment: dict[str, Any]
    ) -> None:
        """Upload a file attachment to the channel."""
        if self._http_session is None:
            return
        try:
            file_path = attachment.get("path")
            if not file_path:
                return

            filename = attachment.get("filename", "file")
            with open(file_path, "rb") as f:
                files = {"files": (filename, f)}
                data = {"channel_id": channel_id}

                # Use a separate header set (no Content-Type for multipart)
                resp = await self._http_session.post(
                    f"{self._url}/api/v4/files",
                    data=data,
                    files=files,
                    headers={"Authorization": f"Bearer {self._token}"},
                )
                if resp.status_code < 400:
                    result = resp.json()
                    file_infos = result.get("file_infos", [])
                    if file_infos:
                        file_ids = [fi["id"] for fi in file_infos]
                        await self._api_post(
                            "/api/v4/posts",
                            {
                                "channel_id": channel_id,
                                "message": "",
                                "file_ids": file_ids,
                            },
                        )
                else:
                    log.error(
                        "mattermost_attachment_upload_failed",
                        status=resp.status_code,
                        error=resp.text,
                    )
        except Exception as exc:
            log.error("mattermost_attachment_failed", error=str(exc))

    async def ask_question(self, channel_id: str, question: str, options: list[str] | None = None, timeout: float = 300.0) -> str:
        """Ask a question via Mattermost. Sends the question and returns empty string."""
        if self._http_session is None:
            return ""
        text = question
        if options:
            text += "\n" + "\n".join(f"• {opt}" for opt in options)
        await self._api_post("/api/v4/posts", {"channel_id": channel_id, "message": text})
        return ""

    async def send_approval(
        self,
        channel_id: str,
        description: str,
        approval_id: str,
    ) -> None:
        """Send interactive message buttons for an approval request."""
        if self._http_session is None:
            return

        body: dict[str, Any] = {
            "channel_id": channel_id,
            "message": "",
            "props": {
                "attachments": [
                    {
                        "fallback": f"Approval Required: {description}",
                        "color": "#FFA500",
                        "title": "Approval Required",
                        "text": description,
                        "actions": [
                            {
                                "id": f"approve-{approval_id}",
                                "name": "Approve",
                                "integration": {
                                    "url": (
                                        f"http://{self._listen_host}:"
                                        f"{self._listen_port}"
                                        "/mattermost/actions"
                                    ),
                                    "context": {
                                        "action_id": f"approve:{approval_id}",
                                    },
                                },
                                "style": "good",
                            },
                            {
                                "id": f"deny-{approval_id}",
                                "name": "Deny",
                                "integration": {
                                    "url": (
                                        f"http://{self._listen_host}:"
                                        f"{self._listen_port}"
                                        "/mattermost/actions"
                                    ),
                                    "context": {
                                        "action_id": f"deny:{approval_id}",
                                    },
                                },
                                "style": "danger",
                            },
                        ],
                    }
                ]
            },
        }

        result = await self._api_post("/api/v4/posts", body)
        if result is None:
            log.error("mattermost_approval_buttons_failed")


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
