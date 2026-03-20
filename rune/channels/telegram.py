"""Telegram Channel Adapter for RUNE.

Ported from src/channels/telegram.ts - uses aiogram 3.x for bot API,
long polling, file attachments, typing indicators, and inline-keyboard
approval flows. Leverages ChannelAdapter base for authorization,
reconnection, pending lifecycle, and status tracking.
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
from rune.utils.logger import get_logger

log = get_logger(__name__)


def _get_telegram_token() -> str:
    """Resolve Telegram bot token from environment variables."""
    return (
        os.environ.get("TELEGRAM_BOT_TOKEN")
        or os.environ.get("RUNE_TELEGRAM_TOKEN")
        or ""
    )


_MAX_MESSAGE_LENGTH = 4096


class TelegramResponseFormatter(ResponseFormatter):
    """Formats responses for Telegram's HTML parse mode."""

    def format_text(self, text: str) -> str:
        # Escape HTML entities to prevent parse errors
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return text

    def format_code(self, code: str, language: str = "") -> str:
        code = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<pre>{code}</pre>"

    def format_error(self, error: str) -> str:
        error = error.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"❌ <b>Error:</b> {error}"


class TelegramAdapter(ChannelAdapter):
    """Channel adapter for Telegram via aiogram.

    Uses base-class infrastructure for authorization, reconnection,
    pending lifecycle, and connection status.
    """

    def __init__(
        self,
        token: str,
        *,
        allowed_users: list[int] | None = None,
        polling_timeout: int = 30,
    ) -> None:
        # Convert int user IDs to str for base class
        allowed_str = [str(uid) for uid in allowed_users] if allowed_users else None
        super().__init__(allowed_users=allowed_str)
        self._token = token
        self._polling_timeout = polling_timeout
        self._bot: Any = None
        self._dispatcher: Any = None
        self._polling_task: asyncio.Task[None] | None = None
        self._formatter = TelegramResponseFormatter()

    @property
    def name(self) -> str:
        return "telegram"

    # -- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        self._status = "connecting"
        try:
            await self._do_connect()
            self._status = "connected"
            self._reconnect_attempts = 0
            log.info("telegram_adapter_started")
        except Exception as exc:
            self._status = "error"
            log.error("telegram_start_failed", error=str(exc))
            self._schedule_reconnect()
            raise

    async def _do_connect(self) -> None:
        """Establish bot connection and start polling."""
        try:
            from aiogram import Bot, Dispatcher
            from aiogram.client.default import DefaultBotProperties
            from aiogram.enums import ParseMode
            from aiogram.types import CallbackQuery
            from aiogram.types import Message as TGMessage
        except ImportError as exc:
            raise ImportError(
                "aiogram is required for the Telegram adapter. "
                "Install with: pip install aiogram"
            ) from exc

        # Clean up previous bot if reconnecting
        if self._bot is not None:
            with contextlib.suppress(Exception):
                await self._bot.session.close()

        self._bot = Bot(
            token=self._token,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        )
        self._dispatcher = Dispatcher()

        @self._dispatcher.message()
        async def _on_tg_message(message: TGMessage) -> None:
            await self._handle_telegram_message(message)

        @self._dispatcher.callback_query()
        async def _on_callback_query(callback_query: CallbackQuery) -> None:
            await self._handle_callback_query(callback_query)

        self._polling_task = asyncio.create_task(
            self._run_polling_with_reconnect()
        )

    async def _run_polling_with_reconnect(self) -> None:
        """Run polling; on failure, trigger base-class reconnection."""
        while self._status not in ("disconnected",):
            try:
                await self._dispatcher.start_polling(self._bot, handle_signals=False)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._status == "disconnected":
                    return
                self._status = "error"
                log.error("telegram_polling_error", error=str(exc))
                self._schedule_reconnect()
                return

    async def stop(self) -> None:
        self._status = "disconnected"
        self._cancel_reconnect()
        self._cleanup_pending()

        if self._dispatcher is not None:
            with contextlib.suppress(Exception):
                await self._dispatcher.stop_polling()
        if self._polling_task is not None:
            self._polling_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._polling_task
            self._polling_task = None
        if self._bot is not None:
            with contextlib.suppress(Exception):
                await self._bot.session.close()
            self._bot = None
        self._dispatcher = None
        log.info("telegram_adapter_stopped")

    # -- Send / Edit / Delete -----------------------------------------------

    async def send(self, channel_id: str, message: OutgoingMessage) -> str | None:
        if self._bot is None:
            log.warning("telegram_send_no_bot")
            return None

        text = self._formatter.format_text(message.text)
        chunks = _split_text(text, _MAX_MESSAGE_LENGTH)
        for chunk in chunks:
            try:
                await self._bot.send_message(chat_id=int(channel_id), text=chunk)
            except Exception as exc:
                log.error("telegram_send_failed", error=str(exc))

        for attachment in message.attachments:
            await self._send_attachment(channel_id, attachment)
        return None

    async def edit_message(
        self, channel_id: str, message_id: str, new_text: str
    ) -> None:
        if self._bot is None:
            return
        try:
            await self._bot.edit_message_text(
                chat_id=int(channel_id),
                message_id=int(message_id),
                text=new_text,
            )
        except Exception as exc:
            log.error("telegram_edit_failed", error=str(exc))

    async def delete_message(self, channel_id: str, message_id: str) -> None:
        if self._bot is None:
            return
        try:
            await self._bot.delete_message(
                chat_id=int(channel_id),
                message_id=int(message_id),
            )
        except Exception as exc:
            log.error("telegram_delete_failed", error=str(exc))

    # -- Message handling ---------------------------------------------------

    async def _handle_telegram_message(self, message: Any) -> None:
        if self._on_message is None:
            return

        # Base-class authorization check
        sender_id = str(message.from_user.id) if message.from_user else "unknown"
        if not self.check_authorization(sender_id):
            username = getattr(message.from_user, "username", None) or "unknown"
            log.warning(
                "telegram_unauthorized_user",
                user_id=sender_id,
                username=username,
            )
            with contextlib.suppress(Exception):
                await self._bot.send_message(
                    chat_id=message.chat.id,
                    text="⛔ 이 봇은 승인된 사용자만 사용할 수 있습니다.",
                )
            return

        text = message.text or message.caption or ""
        attachments = self._extract_attachments(message)

        incoming = IncomingMessage(
            channel_id=str(message.chat.id),
            sender_id=sender_id,
            text=text,
            attachments=attachments,
            reply_to=str(message.reply_to_message.message_id)
            if message.reply_to_message
            else None,
            metadata={
                "channel_name": self.name,
                "message_id": str(message.message_id),
                "chat_type": message.chat.type,
                "username": getattr(message.from_user, "username", None),
            },
        )

        # Resolve pending question if awaiting reply in this chat
        chat_id_str = str(message.chat.id)
        question_future = self._pending_questions.pop(chat_id_str, None)
        if question_future is not None and not question_future.done():
            question_future.set_result(text)

        await self._send_typing(chat_id_str)
        await self._on_message(incoming)

    def _extract_attachments(self, message: Any) -> list[dict[str, Any]]:
        attachments: list[dict[str, Any]] = []
        if message.document:
            attachments.append({
                "type": "document",
                "file_id": message.document.file_id,
                "file_name": message.document.file_name,
                "mime_type": message.document.mime_type,
                "file_size": message.document.file_size,
            })
        if message.photo:
            largest = message.photo[-1]
            attachments.append({
                "type": "photo",
                "file_id": largest.file_id,
                "width": largest.width,
                "height": largest.height,
            })
        if message.audio:
            attachments.append({
                "type": "audio",
                "file_id": message.audio.file_id,
                "file_name": message.audio.file_name,
                "duration": message.audio.duration,
            })
        if message.voice:
            attachments.append({
                "type": "voice",
                "file_id": message.voice.file_id,
                "duration": message.voice.duration,
            })
        return attachments

    # -- Typing / File download ---------------------------------------------

    async def send_typing_indicator(self, channel_id: str) -> None:
        await self._send_typing(channel_id)

    async def _send_typing(self, chat_id: str) -> None:
        if self._bot is None:
            return
        with contextlib.suppress(Exception):
            await self._bot.send_chat_action(chat_id=int(chat_id), action="typing")

    async def download_file(self, file_id: str) -> bytes:
        if self._bot is None:
            raise RuntimeError("Telegram bot is not connected")
        file = await self._bot.get_file(file_id)
        if not file.file_path:
            raise RuntimeError(f"Telegram returned no file_path for {file_id}")
        url = f"https://api.telegram.org/file/bot{self._token}/{file.file_path}"
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise RuntimeError(f"Telegram file download failed: HTTP {resp.status_code}")
            return resp.content

    async def _send_attachment(self, channel_id: str, attachment: dict[str, Any]) -> None:
        if self._bot is None:
            return
        try:
            from aiogram.types import FSInputFile
            file_path = attachment.get("path")
            if file_path:
                input_file = FSInputFile(file_path)
                await self._bot.send_document(
                    chat_id=int(channel_id),
                    document=input_file,
                    caption=attachment.get("caption", ""),
                )
        except Exception as exc:
            log.error("telegram_attachment_failed", error=str(exc))

    # -- Ask question -------------------------------------------------------

    async def ask_question(
        self,
        channel_id: str,
        question: str,
        options: list[str] | None = None,
        timeout: float = 300.0,
    ) -> str:
        if self._bot is None:
            return ""
        text = question
        if options:
            text += "\n" + "\n".join(f"• {opt}" for opt in options)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_questions[channel_id] = future

        try:
            await self._bot.send_message(chat_id=int(channel_id), text=text)
        except Exception as exc:
            log.error("telegram_ask_question_failed", error=str(exc))
            self._pending_questions.pop(channel_id, None)
            return ""

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            self._pending_questions.pop(channel_id, None)
            log.warning("telegram_ask_question_timeout", channel_id=channel_id)
            return ""

    # -- Approval flow (3-button) -------------------------------------------

    async def send_approval(
        self,
        channel_id: str,
        description: str,
        approval_id: str,
    ) -> None:
        if self._bot is None:
            return
        try:
            from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

            desc_display = description[:200] + "..." if len(description) > 200 else description

            keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="✅ 허용",
                            callback_data=f"approval:allow_once:{approval_id}",
                        ),
                        InlineKeyboardButton(
                            text="🔒 항상 허용",
                            callback_data=f"approval:allow_always:{approval_id}",
                        ),
                        InlineKeyboardButton(
                            text="❌ 거부",
                            callback_data=f"approval:deny:{approval_id}",
                        ),
                    ]
                ]
            )
            await self._bot.send_message(
                chat_id=int(channel_id),
                text=f"🔐 *Approval Required*\n\n{desc_display}",
                reply_markup=keyboard,
            )
        except Exception as exc:
            log.error("telegram_approval_keyboard_failed", error=str(exc))

    async def _handle_callback_query(self, callback_query: Any) -> None:
        data = callback_query.data or ""
        log.info("telegram_callback_query", data=data)

        if data.startswith("approval:"):
            parts = data.split(":", 2)
            if len(parts) < 3:
                with contextlib.suppress(Exception):
                    await callback_query.answer()
                return

            action = parts[1]
            approval_id = parts[2]

            # Map action to decision
            if action in ("allow", "allow_once"):
                decision = ApprovalDecision.APPROVE_ONCE
            elif action == "allow_always":
                decision = ApprovalDecision.APPROVE_ALWAYS
            else:
                decision = ApprovalDecision.DENY

            # Use base-class resolver
            self._resolve_approval(approval_id, decision)

            # Update message with result
            if decision != ApprovalDecision.DENY:
                status_emoji = "✅"
                status_text = "Approved"
            else:
                status_emoji = "❌"
                status_text = "Denied"

            user_name = getattr(callback_query.from_user, "username", None) or "someone"
            try:
                await callback_query.answer(text=f"{status_text}!")
                if callback_query.message and self._bot is not None:
                    await self._bot.edit_message_text(
                        chat_id=callback_query.message.chat.id,
                        message_id=callback_query.message.message_id,
                        text=f"{status_emoji} {status_text} by {user_name}",
                        reply_markup=None,
                    )
            except Exception as exc:
                log.warning("telegram_approval_update_failed", error=str(exc))
        else:
            with contextlib.suppress(Exception):
                await callback_query.answer()


def _split_text(text: str, max_length: int) -> list[str]:
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
