"""Telegram through rune-relay, for a hosted machine: no bot token here and no open connection.

On a hosted machine the owner's bot is held on the host (rune-bot's rune-relay). Updates arrive
from the host's control loop over the `channel.deliver` RPC, after the relay has kept only those
from the chat the owner linked. Replies go to the relay, signed with the broker key like connector
requests, and the relay sends them only into that chat. Nothing here keeps a connection open, so
the machine can sleep with Telegram connected, and closing its open web (the broker's taint)
doesn't cut it off.

The relay may hand an update over twice (it marks it delivered after this answers); the last few
update ids are remembered so it runs once.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import time
from collections import OrderedDict
from typing import Any

import httpx

from rune.channels.telegram import _MAX_MESSAGE_LENGTH, TelegramResponseFormatter, _split_text
from rune.channels.types import ApprovalDecision, ChannelAdapter, IncomingMessage, OutgoingMessage
from rune.utils.logger import get_logger

log = get_logger(__name__)

RELAY_URL = "RUNE_CHANNEL_RELAY_URL"
_SEEN = 1000
_SIGNED_FOR = 45


class RelayUnavailable(RuntimeError):
    pass


def relay_url() -> str | None:
    value = os.environ.get(RELAY_URL, "").strip()
    return value.rstrip("/") or None


class RelayTelegramAdapter(ChannelAdapter):
    def __init__(self, url: str, *, client: httpx.AsyncClient | None = None) -> None:
        # The relay already keeps everyone but the owner out.
        super().__init__(allowed_users=None)
        self._url = url
        self._client = client
        self._formatter = TelegramResponseFormatter()
        self._seen: OrderedDict[int, None] = OrderedDict()
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def name(self) -> str:
        return "telegram"

    async def start(self) -> None:
        # "relayed", not "connected": the host's control loop doesn't keep the machine awake for it.
        self._status = "relayed"
        log.info("telegram_relay_adapter_started")

    async def stop(self) -> None:
        self._status = "disconnected"
        self._cleanup_pending()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # -- Relay --------------------------------------------------------------

    async def _call(self, method: str, params: dict[str, Any]) -> Any:
        from rune.connectors.store import authority_key, broker_home, signature

        try:
            key = authority_key(broker_home())
        except (OSError, ValueError) as exc:
            raise RelayUnavailable("The relay signing key is unavailable") from exc
        body = json.dumps({"method": method, "params": params}, separators=(",", ":")).encode()
        nonce, expires = secrets.token_hex(32), int(time.time()) + _SIGNED_FOR
        headers = {"x-rune-key-id": os.environ.get("RUNE_CONNECTOR_KEY_ID", "").strip(), "x-rune-nonce": nonce,
                   "x-rune-expires": str(expires), "x-rune-signature": signature(key, "telegram", body, nonce, expires),
                   "content-type": "application/json"}
        try:
            if self._client is not None:
                response = await self._client.post(f"{self._url}/telegram", content=body, headers=headers)
            else:
                # trust_env=False keeps the browsing-lane proxy off the relay's link.
                async with httpx.AsyncClient(trust_env=False, timeout=30, follow_redirects=False) as client:
                    response = await client.post(f"{self._url}/telegram", content=body, headers=headers)
        except httpx.HTTPError as exc:
            raise RelayUnavailable(f"The relay could not be reached ({type(exc).__name__})") from exc
        if response.status_code != 200:
            raise RelayUnavailable(f"The relay refused {method} ({response.status_code})")
        answer = response.json()
        if not answer.get("ok"):
            raise RelayUnavailable(str(answer.get("error") or f"Telegram refused {method}"))
        return answer.get("result")

    async def _try(self, method: str, params: dict[str, Any]) -> Any:
        try:
            return await self._call(method, params)
        except RelayUnavailable as exc:
            log.error("telegram_relay_call_failed", method=method, error=str(exc))
            return None

    # -- Delivery -----------------------------------------------------------

    async def deliver(self, updates: list[dict[str, Any]]) -> int:
        """Take updates the host's control loop hands over; returns how many were new.

        Each is handled in the background: a message can start a run that takes far longer than the
        control loop should wait."""
        fresh = 0
        for update in updates:
            update_id = update.get("update_id") if isinstance(update, dict) else None
            if not isinstance(update_id, int) or update_id in self._seen:
                continue
            self._seen[update_id] = None
            while len(self._seen) > _SEEN:
                self._seen.popitem(last=False)
            fresh += 1
            task = asyncio.create_task(self._handle(update))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        return fresh

    async def _handle(self, update: dict[str, Any]) -> None:
        try:
            if isinstance(update.get("callback_query"), dict):
                await self._handle_callback(update["callback_query"])
            elif isinstance(update.get("message"), dict):
                await self._handle_message(update["message"])
        except Exception as exc:
            log.error("telegram_relay_update_failed", error=str(exc)[:200])

    async def _handle_message(self, message: dict[str, Any]) -> None:
        if self._on_message is None:
            return
        chat_id = str((message.get("chat") or {}).get("id", ""))
        sender = message.get("from") or {}
        text = message.get("text") or message.get("caption") or ""
        reply = message.get("reply_to_message") or {}
        incoming = IncomingMessage(
            channel_id=chat_id,
            sender_id=str(sender.get("id", "unknown")),
            text=text,
            attachments=_attachments(message),
            reply_to=str(reply["message_id"]) if "message_id" in reply else None,
            metadata={"channel_name": self.name, "message_id": str(message.get("message_id", "")),
                      "chat_type": (message.get("chat") or {}).get("type"), "username": sender.get("username"),
                      "relayed": True},
        )
        question = self._pending_questions.pop(chat_id, None)
        if question is not None and not question.done():
            question.set_result(text)
        await self.send_typing_indicator(chat_id)
        await self._on_message(incoming)

    async def _handle_callback(self, query: dict[str, Any]) -> None:
        data = str(query.get("data") or "")
        query_id = str(query.get("id") or "")
        parts = data.split(":", 2)
        if parts[0] != "approval" or len(parts) < 3:
            await self._try("answerCallbackQuery", {"callback_query_id": query_id})
            return
        action, approval_id = parts[1], parts[2]
        decision = {"allow": ApprovalDecision.APPROVE_ONCE, "allow_once": ApprovalDecision.APPROVE_ONCE,
                    "allow_always": ApprovalDecision.APPROVE_ALWAYS}.get(action, ApprovalDecision.DENY)
        self._resolve_approval(approval_id, decision)
        text = "Denied" if decision == ApprovalDecision.DENY else "Approved"
        await self._try("answerCallbackQuery", {"callback_query_id": query_id, "text": f"{text}!"})
        message = query.get("message") or {}
        chat_id = (message.get("chat") or {}).get("id")
        if isinstance(chat_id, int) and "message_id" in message:
            name = (query.get("from") or {}).get("username") or "someone"
            await self._try("editMessageText", {"chat_id": chat_id, "message_id": message["message_id"],
                                                "text": f"{text} by {name}"})

    # -- Sending ------------------------------------------------------------

    async def send(self, channel_id: str, message: OutgoingMessage) -> str | None:
        text = self._formatter.format_text(message.text)
        sent = None
        for chunk in _split_text(text, _MAX_MESSAGE_LENGTH):
            sent = await self._try("sendMessage", {"chat_id": int(channel_id), "text": chunk, "parse_mode": "HTML"})
        if message.attachments:
            # Files don't go through the relay yet.
            await self._try("sendMessage", {"chat_id": int(channel_id),
                                            "text": f"({len(message.attachments)} file(s) are in the Rune app.)"})
        return str(sent["message_id"]) if isinstance(sent, dict) and "message_id" in sent else None

    async def edit_message(self, channel_id: str, message_id: str, new_text: str) -> None:
        await self._try("editMessageText", {"chat_id": int(channel_id), "message_id": int(message_id), "text": new_text})

    async def delete_message(self, channel_id: str, message_id: str) -> None:
        await self._try("deleteMessage", {"chat_id": int(channel_id), "message_id": int(message_id)})

    async def send_typing_indicator(self, channel_id: str) -> None:
        with contextlib.suppress(ValueError):
            await self._try("sendChatAction", {"chat_id": int(channel_id), "action": "typing"})

    async def ask_question(self, channel_id: str, question: str, options: list[str] | None = None,
                           timeout: float = 300.0) -> str:
        text = question + ("\n" + "\n".join(f"• {o}" for o in options) if options else "")
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._pending_questions[channel_id] = future
        if await self._try("sendMessage", {"chat_id": int(channel_id), "text": text}) is None:
            self._pending_questions.pop(channel_id, None)
            return ""
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            self._pending_questions.pop(channel_id, None)
            return ""

    async def send_approval(self, channel_id: str, description: str, approval_id: str) -> None:
        shown = description[:200] + "..." if len(description) > 200 else description
        keyboard = {"inline_keyboard": [[
            {"text": "Allow", "callback_data": f"approval:allow_once:{approval_id}"},
            {"text": "Always allow", "callback_data": f"approval:allow_always:{approval_id}"},
            {"text": "Deny", "callback_data": f"approval:deny:{approval_id}"},
        ]]}
        await self._try("sendMessage", {"chat_id": int(channel_id), "text": f"Approval required\n\n{shown}",
                                        "reply_markup": keyboard})


def _attachments(message: dict[str, Any]) -> list[dict[str, Any]]:
    """What was attached, described; the files themselves stay with Telegram for now."""
    found: list[dict[str, Any]] = []
    if isinstance(message.get("document"), dict):
        d = message["document"]
        found.append({"type": "document", "file_id": d.get("file_id"), "file_name": d.get("file_name"),
                      "mime_type": d.get("mime_type"), "file_size": d.get("file_size")})
    if isinstance(message.get("photo"), list) and message["photo"]:
        p = message["photo"][-1]
        found.append({"type": "photo", "file_id": p.get("file_id"), "width": p.get("width"), "height": p.get("height")})
    for kind in ("audio", "voice"):
        if isinstance(message.get(kind), dict):
            found.append({"type": kind, "file_id": message[kind].get("file_id"),
                          "duration": message[kind].get("duration")})
    return found
