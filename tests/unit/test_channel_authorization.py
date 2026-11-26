"""Every inbound channel answers only the senders its owner allowed.

Discord and Slack once accepted an allowed_users list and never consulted it. A channel with no
list was then open to anyone: a Telegram bot token alone let any Telegram user who found the bot
drive the owner's agent, and the first to write became where RUNE's scheduled messages went.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from rune.channels.discord import DiscordAdapter
from rune.channels.registry import ChannelRegistry
from rune.channels.slack import SlackAdapter
from rune.channels.telegram import TelegramAdapter
from rune.daemon.gateway import ChannelGateway


class TestConstructorAcceptsAllowedUsers:
    def test_discord(self):
        a = DiscordAdapter(token="x", allowed_users=["42"])
        assert a.check_authorization("42") is True
        assert a.check_authorization("99") is False

    def test_slack(self):
        a = SlackAdapter(bot_token="x", app_token="y", allowed_users=["U1"])
        assert a.check_authorization("U1") is True
        assert a.check_authorization("U2") is False


class TestNobodyWithoutList:
    def test_discord(self):
        assert DiscordAdapter(token="x").check_authorization("anyone") is False

    def test_slack(self):
        assert SlackAdapter(bot_token="x", app_token="y").check_authorization("anyone") is False

    def test_telegram(self):
        assert TelegramAdapter("123:abc").check_authorization("anyone") is False


def _telegram_message(user_id: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(from_user=SimpleNamespace(id=user_id, username=None),
                           chat=SimpleNamespace(id=user_id, type="private"), text=text, caption=None,
                           reply_to_message=None, message_id=1, document=None, photo=None, audio=None,
                           voice=None)


class _Scheduler:
    def __init__(self) -> None:
        self.goals: list[str] = []

    async def execute(self, goal, sender_id):
        self.goals.append(goal)
        return "ok"


async def _write(allowed_users: list[int] | None, user_id: int, tmp_path):
    sent: list[dict] = []

    async def send_message(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(message_id=2)

    async def send_chat_action(**kwargs): ...

    adapter = TelegramAdapter("123:abc", allowed_users=allowed_users)
    adapter._bot = SimpleNamespace(send_message=send_message, send_chat_action=send_chat_action)
    registry = ChannelRegistry()
    registry.register(adapter)
    scheduler = _Scheduler()
    gateway = ChannelGateway(registry, agent_scheduler=scheduler,
                             recipients_path=tmp_path / "channel-recipients.json")
    await gateway.start()
    try:
        await adapter._handle_telegram_message(_telegram_message(user_id, "run ls ~"))
        await asyncio.sleep(0.05)
    finally:
        await gateway.stop()
    return scheduler.goals, gateway._default_recipients.get("telegram"), sent


async def test_with_a_token_alone_a_stranger_is_refused_told_their_id_and_not_made_the_recipient(tmp_path):
    goals, recipient, sent = await _write(None, 5551234, tmp_path)
    assert goals == [] and recipient is None
    (refusal,) = sent
    assert "5551234" in refusal["text"] and "RUNE_TELEGRAM_ALLOWED_USERS" in refusal["text"]


async def test_the_owner_once_listed_is_answered_and_is_where_messages_go(tmp_path):
    goals, recipient, _sent = await _write([5551234], 5551234, tmp_path)
    assert goals == ["run ls ~"] and recipient == "5551234"


def _passed_on() -> tuple[list, object]:
    received: list = []

    async def on_message(incoming):
        received.append(incoming)

    return received, on_message


async def test_discord_tells_a_refused_sender_their_id_and_the_setting():
    received, on_message = _passed_on()
    replies: list[str] = []

    async def send(text):
        replies.append(text)

    adapter = DiscordAdapter(token="x")
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=1))
    adapter._on_message = on_message
    await adapter._handle_discord_message(SimpleNamespace(
        author=SimpleNamespace(id=4242, bot=False, name="stranger"), channel=SimpleNamespace(send=send)))
    assert received == []
    (reply,) = replies
    assert "4242" in reply and "RUNE_DISCORD_ALLOWED_USERS" in reply


async def test_slack_tells_a_refused_sender_their_id_and_the_setting():
    received, on_message = _passed_on()
    posts: list[dict] = []

    async def post(**kwargs):
        posts.append(kwargs)

    adapter = SlackAdapter(bot_token="x", app_token="y")
    adapter._web_client = SimpleNamespace(chat_postMessage=post)
    adapter._on_message = on_message
    await adapter._handle_events_api({"event": {"type": "message", "user": "U9", "channel": "D1", "text": "hi"}})
    assert received == []
    (reply,) = posts
    assert reply["channel"] == "D1" and "U9" in reply["text"] and "RUNE_SLACK_ALLOWED_USERS" in reply["text"]
