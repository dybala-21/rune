"""A message a chat app did not take is never counted as sent.

Every adapter used to log a failed or impossible send and return as if it had gone. The gateway then
counted a scheduled message as delivered: it reached neither the chat nor the app's "From RUNE"
conversation, and the owner was never told.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rune.api import conversation_wiring
from rune.channels.discord import DiscordAdapter
from rune.channels.google_chat import GoogleChatAdapter
from rune.channels.line import LINEAdapter
from rune.channels.mattermost import MattermostAdapter
from rune.channels.registry import ChannelRegistry
from rune.channels.slack import SlackAdapter
from rune.channels.telegram import TelegramAdapter
from rune.channels.telegram_relay import RelayTelegramAdapter, RelayUnavailable
from rune.channels.tui import TUIChannelAdapter
from rune.channels.types import NotSent, OutgoingMessage
from rune.channels.whatsapp import WhatsAppAdapter
from rune.daemon.gateway import ChannelGateway, GatewayNotification


def _refuse(*args, **kwargs):
    raise RuntimeError("Forbidden: bot was blocked by the user")


async def _refuse_async(*args, **kwargs):
    _refuse()


class _Session:
    """An HTTP session the platform answers with an error."""

    async def post(self, url, json=None):
        return SimpleNamespace(status_code=400, text="bad request", json=lambda: {"error": "bad request"})


def _direct_telegram():
    adapter = TelegramAdapter("123:abc")
    adapter._bot = SimpleNamespace(send_message=_refuse_async)
    return adapter


def _relayed_telegram():
    adapter = RelayTelegramAdapter("http://10.0.0.1:4110")

    async def unreachable(method, params):
        raise RelayUnavailable("The relay could not be reached (ConnectError)")

    adapter._call = unreachable
    return adapter


def _with(adapter, **attributes):
    for name, value in attributes.items():
        setattr(adapter, name, value)
    return adapter


NOT_CONNECTED = {
    "telegram": lambda: TelegramAdapter("123:abc"),
    "discord": lambda: DiscordAdapter(token="x"),
    "slack": lambda: SlackAdapter(bot_token="x", app_token="y"),
    "line": lambda: LINEAdapter("token", "secret"),
    "whatsapp": lambda: WhatsAppAdapter("token", "123"),
    "mattermost": lambda: MattermostAdapter("https://chat.example", "token"),
    "google_chat": lambda: GoogleChatAdapter("/nonexistent.json", "project"),
    "tui": TUIChannelAdapter,
}

REFUSED = {
    "telegram": _direct_telegram,
    "telegram-relay": _relayed_telegram,
    "discord": lambda: _with(DiscordAdapter(token="x"),
                             _client=SimpleNamespace(get_channel=lambda _id: SimpleNamespace(send=_refuse_async))),
    "slack": lambda: _with(SlackAdapter(bot_token="x", app_token="y"),
                           _web_client=SimpleNamespace(chat_postMessage=_refuse_async)),
    "line": lambda: _with(LINEAdapter("token", "secret"), _http_session=_Session()),
    "whatsapp": lambda: _with(WhatsAppAdapter("token", "123"), _http_session=_Session()),
    "mattermost": lambda: _with(MattermostAdapter("https://chat.example", "token"), _http_session=_Session()),
    "google_chat": lambda: _with(GoogleChatAdapter("/nonexistent.json", "project"),
                                 _chat_service=SimpleNamespace(spaces=_refuse)),
}


@pytest.mark.parametrize("make", NOT_CONNECTED.values(), ids=NOT_CONNECTED.keys())
async def test_a_chat_app_with_no_connection_says_it_sent_nothing(make):
    with pytest.raises(NotSent):
        await make().send("777", OutgoingMessage(text="Watch\nwent down"))


@pytest.mark.parametrize("make", REFUSED.values(), ids=REFUSED.keys())
async def test_a_message_the_platform_refused_is_not_sent(make):
    with pytest.raises(NotSent):
        await make().send("777", OutgoingMessage(text="Watch\nwent down"))


@pytest.fixture
def conversations(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    monkeypatch.setattr(conversation_wiring, "_conv_manager", None)
    monkeypatch.setattr(conversation_wiring, "_conv_manager_failed", False)
    return lambda: conversation_wiring.get_conv_manager()._store


@pytest.mark.parametrize("make", [_direct_telegram, _relayed_telegram], ids=["direct", "relayed"])
async def test_a_scheduled_message_telegram_did_not_take_is_kept_in_the_app(conversations, make):
    registry = ChannelRegistry()
    registry.register(make())
    gateway = ChannelGateway(registry)
    gateway._default_recipients["telegram"] = "777"
    await gateway.route_notification(GatewayNotification(title="Watch", body="went down", channel="telegram",
                                                         source="cron"))
    assert await conversations().load("777:777") is None, "not kept as a turn the chat showed"
    assert (await conversations().load("rune:inbox")).turns[-1].content == "Watch\nwent down"


async def test_a_reply_telegram_refused_is_not_sent_through_another_chat_app():
    registry = ChannelRegistry()
    registry.register(_direct_telegram())
    slack_posts: list[dict] = []

    async def post(**kwargs):
        slack_posts.append(kwargs)

    registry.register(_with(SlackAdapter(bot_token="x", app_token="y"),
                            _web_client=SimpleNamespace(chat_postMessage=post)))
    await ChannelGateway(registry)._send_response("777", "777", "Done", channel_name="telegram")
    assert slack_posts == [], "a Telegram chat id means nothing on Slack, or someone else"
