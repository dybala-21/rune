"""Discord Channel Adapter for RUNE.

Ported from src/channels/discord.ts - uses discord.py for bot integration,
embed support, and button-based approval flows.
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


def _get_discord_token() -> str:
    """Resolve Discord bot token from environment variables.

    Checks TS-compatible name first, then Python-specific fallback.
    """
    return (
        os.environ.get("DISCORD_BOT_TOKEN")
        or os.environ.get("RUNE_DISCORD_TOKEN")
        or ""
    )

# Discord message length limit
_MAX_MESSAGE_LENGTH = 2000


class DiscordResponseFormatter(ResponseFormatter):
    """Formats responses for Discord markdown."""

    def format_text(self, text: str) -> str:
        return text

    def format_code(self, code: str, language: str = "") -> str:
        return f"```{language}\n{code}\n```"

    def format_error(self, error: str) -> str:
        return f"**Error:** {error}"


class DiscordAdapter(ChannelAdapter):
    """Channel adapter for Discord via discord.py."""

    def __init__(self, token: str) -> None:
        super().__init__()
        self._token = token
        self._client: Any = None
        self._run_task: asyncio.Task[None] | None = None
        self._formatter = DiscordResponseFormatter()
        self._ready_event = asyncio.Event()

    @property
    def name(self) -> str:
        return "discord"

    async def start(self) -> None:
        try:
            import discord
        except ImportError as exc:
            raise ImportError(
                "discord.py is required for the Discord adapter. "
                "Install with: pip install discord.py"
            ) from exc

        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True

        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready() -> None:
            log.info(
                "discord_ready",
                user=str(self._client.user),
                guilds=len(self._client.guilds),
            )
            self._ready_event.set()

        @self._client.event
        async def on_message(message: discord.Message) -> None:
            await self._handle_discord_message(message)

        self._run_task = asyncio.create_task(self._client.start(self._token))
        # Wait for the client to become ready (with timeout)
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=30.0)
        except TimeoutError:
            log.warning("discord_ready_timeout")

        log.info("discord_adapter_started")

    async def stop(self) -> None:
        self._cleanup_pending()
        if self._client is not None:
            await self._client.close()
        if self._run_task is not None:
            self._run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._run_task
            self._run_task = None
        self._client = None
        self._ready_event.clear()
        log.info("discord_adapter_stopped")

    async def send(self, channel_id: str, message: OutgoingMessage) -> str | None:
        if self._client is None:
            log.warning("discord_send_no_client")
            return None

        channel = self._client.get_channel(int(channel_id))
        if channel is None:
            try:
                channel = await self._client.fetch_channel(int(channel_id))
            except Exception as exc:
                log.error("discord_channel_not_found", channel_id=channel_id, error=str(exc))
                return None

        text = self._formatter.format_text(message.text)

        # Check if we should use an embed
        if message.metadata.get("embed"):
            await self._send_embed(channel, message)
            return None

        # Split long messages - capture the last sent message ID
        last_message_id: str | None = None
        chunks = _split_text(text, _MAX_MESSAGE_LENGTH)
        for chunk in chunks:
            try:
                kwargs: dict[str, Any] = {"content": chunk}
                if message.reply_to:
                    try:
                        ref_msg = await channel.fetch_message(int(message.reply_to))
                        kwargs["reference"] = ref_msg
                    except Exception:
                        pass  # Non-critical
                sent_message = await channel.send(**kwargs)
                last_message_id = str(sent_message.id)
            except Exception as exc:
                log.error("discord_send_failed", error=str(exc))

        # Send attachments as files
        for attachment in message.attachments:
            await self._send_attachment(channel, attachment)

        return last_message_id

    async def edit_message(
        self, channel_id: str, message_id: str, new_text: str
    ) -> None:
        if self._client is None:
            return
        try:
            channel = self._client.get_channel(int(channel_id))
            if channel is None:
                channel = await self._client.fetch_channel(int(channel_id))
            msg = await channel.fetch_message(int(message_id))
            await msg.edit(content=new_text)
        except Exception as exc:
            log.error("discord_edit_failed", error=str(exc))

    async def delete_message(self, channel_id: str, message_id: str) -> None:
        if self._client is None:
            return
        try:
            channel = self._client.get_channel(int(channel_id))
            if channel is None:
                channel = await self._client.fetch_channel(int(channel_id))
            msg = await channel.fetch_message(int(message_id))
            await msg.delete()
        except Exception as exc:
            log.error("discord_delete_failed", error=str(exc))

    async def send_typing_indicator(self, channel_id: str) -> None:
        """Send typing indicator to Discord channel."""
        if self._client is None:
            return
        try:
            channel = self._client.get_channel(int(channel_id))
            if channel and hasattr(channel, 'trigger_typing'):
                await channel.trigger_typing()
        except Exception:
            pass

    # -- Internals ----------------------------------------------------------

    async def _handle_discord_message(self, message: Any) -> None:
        """Convert a Discord message to IncomingMessage and dispatch."""
        # Ignore messages from the bot itself
        if message.author == self._client.user:
            return
        if message.author.bot:
            return
        if self._on_message is None:
            return

        attachments = [
            {
                "type": "file",
                "url": att.url,
                "filename": att.filename,
                "size": att.size,
                "content_type": att.content_type,
            }
            for att in message.attachments
        ]

        incoming = IncomingMessage(
            channel_id=str(message.channel.id),
            sender_id=str(message.author.id),
            text=message.content,
            attachments=attachments,
            reply_to=str(message.reference.message_id)
            if message.reference
            else None,
            metadata={
                "channel_name": self.name,
                "message_id": str(message.id),
                "guild_id": str(message.guild.id) if message.guild else None,
                "username": message.author.name,
                "display_name": message.author.display_name,
            },
        )

        # Resolve any pending question Future for this channel
        channel_id_str = str(message.channel.id)
        question_future = self._pending_questions.pop(channel_id_str, None)
        if question_future is not None and not question_future.done():
            question_future.set_result(message.content)

        # Show typing indicator
        async with message.channel.typing():
            await self._on_message(incoming)

    async def _send_embed(self, channel: Any, message: OutgoingMessage) -> None:
        """Send a message as a Discord embed."""
        try:
            import discord

            embed_data = message.metadata.get("embed", {})
            embed = discord.Embed(
                title=embed_data.get("title", ""),
                description=embed_data.get("description", message.text),
                color=discord.Color(embed_data.get("color", 0x5865F2)),
            )
            for fld in embed_data.get("fields", []):
                embed.add_field(
                    name=fld.get("name", ""),
                    value=fld.get("value", ""),
                    inline=fld.get("inline", False),
                )
            if embed_data.get("footer"):
                embed.set_footer(text=embed_data["footer"])
            await channel.send(embed=embed)
        except Exception as exc:
            log.error("discord_embed_failed", error=str(exc))
            # Fallback to plain text
            await channel.send(content=message.text[:_MAX_MESSAGE_LENGTH])

    async def _send_attachment(
        self, channel: Any, attachment: dict[str, Any]
    ) -> None:
        """Send a file attachment to the channel."""
        try:
            import discord

            file_path = attachment.get("path")
            if file_path:
                file = discord.File(file_path)
                await channel.send(file=file)
        except Exception as exc:
            log.error("discord_attachment_failed", error=str(exc))

    async def ask_question(self, channel_id: str, question: str, options: list[str] | None = None, timeout: float = 300.0) -> str:
        """Ask a question via Discord and wait for a reply.

        Sends the question, then awaits a Future that is resolved when the
        next message arrives in the same channel.  Returns the reply text,
        or ``""`` on timeout.
        """
        if self._client is None:
            return ""
        channel = self._client.get_channel(int(channel_id))
        if channel is None:
            try:
                channel = await self._client.fetch_channel(int(channel_id))
            except Exception:
                return ""
        text = question
        if options:
            text += "\n" + "\n".join(f"• {opt}" for opt in options)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_questions[channel_id] = future

        try:
            await channel.send(content=text[:_MAX_MESSAGE_LENGTH])
        except Exception as exc:
            log.error("discord_ask_question_failed", error=str(exc))
            self._pending_questions.pop(channel_id, None)
            return ""

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            self._pending_questions.pop(channel_id, None)
            log.warning("discord_ask_question_timeout", channel_id=channel_id)
            return ""

    async def send_approval(
        self,
        channel_id: str,
        description: str,
        approval_id: str,
    ) -> None:
        """Send interactive buttons for an approval request."""
        if self._client is None:
            return
        try:
            import discord
            from discord.ui import Button, View

            channel = self._client.get_channel(int(channel_id))
            if channel is None:
                channel = await self._client.fetch_channel(int(channel_id))

            view = View(timeout=300)
            adapter = self

            approve_btn = Button(
                style=discord.ButtonStyle.success,
                label="Approve",
                custom_id=f"approve:{approval_id}",
            )
            deny_btn = Button(
                style=discord.ButtonStyle.danger,
                label="Deny",
                custom_id=f"deny:{approval_id}",
            )

            async def _on_approve(interaction: discord.Interaction) -> None:
                adapter._resolve_approval(approval_id, ApprovalDecision.APPROVE_ONCE)
                user_name = interaction.user.display_name if interaction.user else "someone"
                await interaction.response.edit_message(
                    content=f"Approval {approval_id}: **Approved** by {user_name}",
                    embed=None,
                    view=None,
                )

            async def _on_deny(interaction: discord.Interaction) -> None:
                adapter._resolve_approval(approval_id, ApprovalDecision.DENY)
                user_name = interaction.user.display_name if interaction.user else "someone"
                await interaction.response.edit_message(
                    content=f"Approval {approval_id}: **Denied** by {user_name}",
                    embed=None,
                    view=None,
                )

            approve_btn.callback = _on_approve
            deny_btn.callback = _on_deny

            view.add_item(approve_btn)
            view.add_item(deny_btn)

            embed = discord.Embed(
                title="Approval Required",
                description=description,
                color=discord.Color.orange(),
            )
            await channel.send(embed=embed, view=view)
        except Exception as exc:
            log.error("discord_approval_buttons_failed", error=str(exc))


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
