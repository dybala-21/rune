"""What RUNE sends first stays in the owner's conversation, and the app lists that conversation."""

from __future__ import annotations

import pytest

from rune.api import conversation_wiring
from rune.channels.registry import ChannelRegistry
from rune.channels.types import ChannelAdapter, OutgoingMessage
from rune.daemon import gateway as gateway_module
from rune.daemon import owner_thread
from rune.daemon.gateway import ChannelGateway, GatewayNotification


class Chat(ChannelAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.sent: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "telegram"

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def send(self, channel_id: str, message: OutgoingMessage) -> str | None:
        self.sent.append((channel_id, message.text))
        return None

    async def send_approval(self, channel_id, description, approval_id) -> None: ...

    async def ask_question(self, channel_id, question, options=None, timeout=300.0) -> str:
        return ""


@pytest.fixture
def owner(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    monkeypatch.setattr(conversation_wiring, "_conv_manager", None)
    monkeypatch.setattr(conversation_wiring, "_conv_manager_failed", False)
    chat = Chat()
    registry = ChannelRegistry()
    registry.register(chat)
    gateway = ChannelGateway(registry)
    gateway._default_recipients["telegram"] = "777"
    monkeypatch.setattr(gateway_module, "_gateway", gateway, raising=False)
    monkeypatch.setattr(gateway_module, "get_gateway", lambda: gateway)
    return gateway, chat


async def test_a_notification_is_kept_as_an_assistant_turn_of_the_owner_s_chat(owner):
    gateway, chat = owner
    await gateway.route_notification(GatewayNotification(title="Morning briefing", body="2 events today", channel="telegram"))
    assert chat.sent == [("777", "Morning briefing\n2 events today")]
    conv = await conversation_wiring.get_conv_manager()._store.load("777:777")
    assert [(t.role, t.content) for t in conv.turns] == [("assistant", "Morning briefing\n2 events today")]


async def test_the_gateway_and_the_app_share_one_copy_of_the_conversation(owner):
    gateway, _ = owner
    assert await gateway._get_conv_manager() is conversation_wiring.get_conv_manager()


async def test_the_app_lists_the_owner_s_chat_beside_its_own(owner):
    gateway, _ = owner
    await gateway.route_notification(GatewayNotification(title="Briefing", body="today", channel="telegram"))
    manager = conversation_wiring.get_conv_manager()
    manager.start_conversation("someone-else")
    convs, total = await conversation_wiring.list_web_conversations_page()
    assert [c.id for c in convs] == ["777:777"] and total == 1
    assert [c.id for c in await conversation_wiring.list_web_conversations()] == ["777:777"]


async def test_an_undelivered_message_is_not_put_in_the_chat_s_conversation(owner):
    gateway, chat = owner
    gateway._default_recipients.clear()
    await gateway.route_notification(GatewayNotification(title="Briefing", body="today", channel="telegram"))
    assert chat.sent == []
    assert await conversation_wiring.get_conv_manager()._store.load("777:777") is None


async def test_a_message_no_chat_app_took_is_kept_in_the_app_s_rune_conversation(owner):
    gateway, chat = owner
    gateway._default_recipients.clear()
    from rune.channels.types import LinkButton

    await gateway.route_notification(GatewayNotification(
        title="Morning briefing", body="Gmail isn't connected.", channel="telegram", source="cron",
        buttons=[LinkButton("Connect Gmail", "https://approve.example/connect/gmail-read")]))
    conv = await conversation_wiring.get_conv_manager()._store.load("rune:inbox")
    assert conv.title == "From RUNE"
    assert conv.turns[-1].content == ("Morning briefing\nGmail isn't connected.\n\n"
                                      "Connect Gmail: https://approve.example/connect/gmail-read")
    assert "rune:inbox" in [c.id for c in (await conversation_wiring.list_web_conversations_page())[0]]


async def test_a_plan_s_progress_is_not_kept(owner):
    gateway, _ = owner
    gateway._default_recipients.clear()
    await gateway.route_notification(GatewayNotification(title="Plan", body="step 2", source="orchestrator"))
    assert await conversation_wiring.get_conv_manager()._store.load("rune:inbox") is None


async def test_a_suggestion_is_not_kept_twice_since_the_app_shows_it_as_a_card(owner):
    gateway, _ = owner
    gateway._default_recipients.clear()
    await gateway.route_notification(GatewayNotification(title="rune", body="Leave at 8?", source="sugg-1"))
    assert await conversation_wiring.get_conv_manager()._store.load("rune:inbox") is None


async def test_the_kept_turn_has_the_links_the_chat_showed_as_buttons(owner):
    from rune.channels.types import LinkButton

    gateway, _ = owner
    await gateway.route_notification(GatewayNotification(
        title="Briefing", body="today", channel="telegram",
        buttons=[LinkButton("Connect Gmail", "https://approve.example/connect/gmail-read")]))
    conv = await conversation_wiring.get_conv_manager()._store.load("777:777")
    assert conv.turns[-1].content == "Briefing\ntoday\n\nConnect Gmail: https://approve.example/connect/gmail-read"


@pytest.mark.parametrize("state", ["reaches", "no-recipient", "not-connected", "failed-to-connect"])
async def test_where_messages_go_is_said_as_the_gateway_then_routes_one(owner, state):
    """Said when a briefing or a watch is set: a chat app is named only if a message would reach it."""
    gateway, chat = owner
    if state == "no-recipient":
        gateway._default_recipients.clear()
    elif state == "not-connected":
        gateway._registry.unregister("telegram")
    elif state == "failed-to-connect":
        # As Telegram's adapter is when its start or its polling failed: a send then does nothing.
        chat._status = "error"
    said = owner_thread.where("telegram")
    await gateway.route_notification(GatewayNotification(title="Watch", body="went down", channel="telegram",
                                                         source="cron"))
    inbox = await conversation_wiring.get_conv_manager()._store.load("rune:inbox")
    assert (said == "Messages go to telegram.") is bool(chat.sent) is (inbox is None), said
    if inbox is not None:
        assert said.startswith('Messages are kept in the app, in the conversation "From RUNE"'), said
        assert inbox.turns[-1].content == "Watch\nwent down"


def test_set_with_no_gateway_running_where_messages_go_is_left_to_when_one_is_sent(monkeypatch):
    monkeypatch.setattr(gateway_module, "get_gateway", lambda: None)
    assert owner_thread.where("telegram") == ("Messages go to telegram if it reaches the owner when one is sent; if "
                                              'not, they are kept in the app, in the conversation "From RUNE".')
