"""Channel registration and routing for RUNE.

Ported from src/channels/registry.ts - singleton registry, auto-discovery
from environment variables, and bulk start/stop.
"""

from __future__ import annotations

import os

from rune.channels.types import ChannelAdapter
from rune.utils.logger import get_logger

log = get_logger(__name__)


class ChannelRegistry:
    """Central registry for all channel adapters."""

    __slots__ = ("_adapters",)

    def __init__(self) -> None:
        self._adapters: dict[str, ChannelAdapter] = {}

    def register(self, adapter: ChannelAdapter) -> None:
        """Register a channel adapter."""
        name = adapter.name
        if name in self._adapters:
            log.warning("channel_already_registered", name=name)
        self._adapters[name] = adapter
        log.info("channel_registered", name=name)

    def unregister(self, name: str) -> None:
        """Remove a channel adapter by name."""
        removed = self._adapters.pop(name, None)
        if removed:
            log.info("channel_unregistered", name=name)
        else:
            log.warning("channel_not_found", name=name)

    def get(self, name: str) -> ChannelAdapter | None:
        """Get a channel adapter by name."""
        return self._adapters.get(name)

    def list(self) -> list[str]:
        """List all registered channel names."""
        return list(self._adapters.keys())

    async def start_all(self) -> None:
        """Start all registered channel adapters."""
        for name, adapter in self._adapters.items():
            try:
                await adapter.start()
                log.info("channel_started", name=name)
            except Exception as exc:
                log.error("channel_start_failed", name=name, error=str(exc))

    async def stop_all(self) -> None:
        """Stop all registered channel adapters."""
        for name, adapter in self._adapters.items():
            try:
                await adapter.stop()
                log.info("channel_stopped", name=name)
            except Exception as exc:
                log.error("channel_stop_failed", name=name, error=str(exc))


# Singleton

_registry: ChannelRegistry | None = None


def get_channel_registry() -> ChannelRegistry:
    """Return the singleton ChannelRegistry instance."""
    global _registry
    if _registry is None:
        _registry = ChannelRegistry()
    return _registry


# Auto-discovery

def _allowed_from_env(var: str) -> list[str] | None:
    """Parse a comma-separated allowlist. ``None`` means open access.

    These four channels serve webhooks on 0.0.0.0 and had no authorization at
    all: the registry never passed them an allowlist and they never checked one,
    so anyone who could reach the port could drive the agent.
    """
    raw = os.environ.get(var, "")
    users = [u.strip() for u in raw.split(",") if u.strip()]
    if not users:
        # Kept open for compatibility with existing setups, but a webhook on
        # 0.0.0.0 with no allowlist hands full tool access to whoever reaches
        # the port, so say so rather than starting quietly.
        log.warning(
            "channel_open_to_all",
            allowlist_var=var,
            hint=f"set {var} to restrict who can drive the agent",
        )
    return users or None


def auto_discover_channels() -> list[str]:
    """Discover and register channel adapters based on environment variables.

    Checks for:
        - RUNE_TELEGRAM_TOKEN  → TelegramAdapter
        - RUNE_DISCORD_TOKEN   → DiscordAdapter
        - RUNE_SLACK_BOT_TOKEN → SlackAdapter

    Returns the list of discovered channel names.
    """
    registry = get_channel_registry()
    discovered: list[str] = []

    telegram_token = os.environ.get("RUNE_TELEGRAM_TOKEN")
    if telegram_token:
        try:
            from rune.channels.telegram import TelegramAdapter

            # Parse allowed users from comma-separated env var
            allowed_users_raw = os.environ.get("RUNE_TELEGRAM_ALLOWED_USERS", "")
            allowed_users: list[int] | None = None
            if allowed_users_raw.strip():
                allowed_users = [
                    int(uid.strip())
                    for uid in allowed_users_raw.split(",")
                    if uid.strip().lstrip("-").isdigit()
                ]

            adapter = TelegramAdapter(
                token=telegram_token,
                allowed_users=allowed_users,
            )
            registry.register(adapter)
            discovered.append(adapter.name)
        except Exception as exc:
            log.error("telegram_discovery_failed", error=str(exc))

    discord_token = os.environ.get("RUNE_DISCORD_TOKEN")
    if discord_token:
        try:
            from rune.channels.discord import DiscordAdapter

            d_allowed_raw = os.environ.get("RUNE_DISCORD_ALLOWED_USERS", "")
            d_allowed = [u.strip() for u in d_allowed_raw.split(",")
                         if u.strip()] or None
            adapter = DiscordAdapter(token=discord_token,
                                     allowed_users=d_allowed)
            registry.register(adapter)
            discovered.append(adapter.name)
        except Exception as exc:
            log.error("discord_discovery_failed", error=str(exc))

    slack_bot_token = os.environ.get("RUNE_SLACK_BOT_TOKEN")
    slack_app_token = os.environ.get("RUNE_SLACK_APP_TOKEN", "")
    if slack_bot_token:
        try:
            from rune.channels.slack import SlackAdapter

            s_allowed_raw = os.environ.get("RUNE_SLACK_ALLOWED_USERS", "")
            s_allowed = [u.strip() for u in s_allowed_raw.split(",")
                         if u.strip()] or None
            adapter = SlackAdapter(
                bot_token=slack_bot_token, app_token=slack_app_token,
                allowed_users=s_allowed,
            )
            registry.register(adapter)
            discovered.append(adapter.name)
        except Exception as exc:
            log.error("slack_discovery_failed", error=str(exc))

    # WhatsApp
    whatsapp_token = os.environ.get("RUNE_WHATSAPP_TOKEN")
    whatsapp_phone_id = os.environ.get("RUNE_WHATSAPP_PHONE_NUMBER_ID", "")
    if whatsapp_token:
        try:
            from rune.channels.whatsapp import WhatsAppAdapter

            adapter = WhatsAppAdapter(
                access_token=whatsapp_token,
                phone_number_id=whatsapp_phone_id,
                verify_token=os.environ.get("RUNE_WHATSAPP_VERIFY_TOKEN", ""),
                app_secret=os.environ.get("RUNE_WHATSAPP_APP_SECRET"),
                allowed_users=_allowed_from_env("RUNE_WHATSAPP_ALLOWED_USERS"),
            )
            registry.register(adapter)
            discovered.append(adapter.name)
        except Exception as exc:
            log.error("whatsapp_discovery_failed", error=str(exc))

    # Mattermost
    mattermost_url = os.environ.get("RUNE_MATTERMOST_URL")
    mattermost_token = os.environ.get("RUNE_MATTERMOST_TOKEN")
    if mattermost_url and mattermost_token:
        try:
            from rune.channels.mattermost import MattermostAdapter

            adapter = MattermostAdapter(
                url=mattermost_url,
                token=mattermost_token,
                allowed_users=_allowed_from_env("RUNE_MATTERMOST_ALLOWED_USERS"),
            )
            registry.register(adapter)
            discovered.append(adapter.name)
        except Exception as exc:
            log.error("mattermost_discovery_failed", error=str(exc))

    # LINE
    line_access_token = os.environ.get("RUNE_LINE_CHANNEL_ACCESS_TOKEN")
    line_secret = os.environ.get("RUNE_LINE_CHANNEL_SECRET")
    if line_access_token and line_secret:
        try:
            from rune.channels.line import LINEAdapter

            adapter = LINEAdapter(
                channel_access_token=line_access_token,
                channel_secret=line_secret,
                allowed_users=_allowed_from_env("RUNE_LINE_ALLOWED_USERS"),
            )
            registry.register(adapter)
            discovered.append(adapter.name)
        except Exception as exc:
            log.error("line_discovery_failed", error=str(exc))

    # Google Chat
    google_chat_creds = os.environ.get("RUNE_GOOGLE_CHAT_CREDENTIALS")
    google_chat_project = os.environ.get("RUNE_GOOGLE_CHAT_PROJECT_ID", "")
    if google_chat_creds:
        try:
            from rune.channels.google_chat import GoogleChatAdapter

            adapter = GoogleChatAdapter(
                service_account_path=google_chat_creds,
                project_id=google_chat_project,
                allowed_users=_allowed_from_env("RUNE_GOOGLE_CHAT_ALLOWED_USERS"),
            )
            registry.register(adapter)
            discovered.append(adapter.name)
        except Exception as exc:
            log.error("google_chat_discovery_failed", error=str(exc))

    log.info("channels_discovered", channels=discovered)
    return discovered
