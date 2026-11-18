"""Link buttons under a message: Telegram shows them, other channels get the links written out."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from rune.channels.registry import ChannelRegistry
from rune.channels.telegram import TelegramAdapter
from rune.channels.telegram_relay import RelayTelegramAdapter
from rune.channels.types import ChannelAdapter, LinkButton, OutgoingMessage
from rune.daemon.gateway import ChannelGateway, GatewayNotification

CONNECT = LinkButton("Connect Gmail", "https://approve.example/connect/gmail-read")


class Relay:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append((body["method"], body["params"]))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": len(self.calls)}})

    def sent(self) -> list[dict]:
        return [p for m, p in self.calls if m == "sendMessage"]


@pytest.fixture
def relayed(monkeypatch):
    from rune.connectors.store import authority_key, broker_home

    authority_key(broker_home(), create=True)
    monkeypatch.setenv("RUNE_CONNECTOR_KEY_ID", "tenant-1")
    fake = Relay()
    adapter = RelayTelegramAdapter("http://10.0.0.1:4110",
                                   client=httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)))
    return adapter, fake


async def test_the_relay_puts_link_buttons_under_the_message(relayed):
    adapter, fake = relayed
    await adapter.send("777", OutgoingMessage(text="Morning", buttons=[CONNECT, LinkButton("Open Rune", "https://app.example/")]))
    assert fake.sent()[-1]["reply_markup"] == {"inline_keyboard": [
        [{"text": "Connect Gmail", "url": CONNECT.url}], [{"text": "Open Rune", "url": "https://app.example/"}]]}


async def test_a_long_message_carries_the_buttons_on_its_last_part_only(relayed):
    adapter, fake = relayed
    await adapter.send("777", OutgoingMessage(text="x" * 9000, buttons=[CONNECT]))
    parts = fake.sent()
    assert len(parts) == 3
    assert ["reply_markup" in p for p in parts] == [False, False, True]


async def test_only_https_links_become_buttons(relayed):
    adapter, fake = relayed
    await adapter.send("777", OutgoingMessage(text="hi", buttons=[
        LinkButton("Call", "tel:+8210"), LinkButton("Plain", "http://example.com"), LinkButton("", "https://x.example")]))
    assert "reply_markup" not in fake.sent()[-1]


async def test_a_plain_message_has_no_keyboard(relayed):
    adapter, fake = relayed
    await adapter.send("777", OutgoingMessage(text="hi"))
    assert "reply_markup" not in fake.sent()[-1]


async def test_a_notification_through_the_gateway_keeps_its_buttons(relayed):
    adapter, fake = relayed
    registry = ChannelRegistry()
    registry.register(adapter)
    gateway = ChannelGateway(registry)
    gateway._default_recipients["telegram"] = "777"
    await gateway.route_notification(GatewayNotification(title="Briefing", body="today", channel="telegram",
                                                         buttons=[CONNECT]))
    assert fake.sent()[-1]["reply_markup"] == {"inline_keyboard": [[{"text": "Connect Gmail", "url": CONNECT.url}]]}


class Plain(ChannelAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.sent: list[OutgoingMessage] = []

    @property
    def name(self) -> str:
        return "plain"

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def send(self, channel_id: str, message: OutgoingMessage) -> str | None:
        self.sent.append(message)
        return None

    async def send_approval(self, channel_id: str, description: str, approval_id: str) -> None: ...

    async def ask_question(self, channel_id, question, options=None, timeout=300.0) -> str:
        return ""


async def test_a_channel_without_buttons_gets_the_links_written_out():
    plain = Plain()
    await plain.send_notification("1", "Briefing", [CONNECT, LinkButton("Call", "tel:+8210")])
    assert plain.sent[-1].text == f"Briefing\n\nConnect Gmail: {CONNECT.url}"
    assert plain.sent[-1].buttons == []


async def test_the_direct_telegram_adapter_sends_url_buttons():
    calls: list[dict] = []

    async def send_message(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(message_id=1)

    adapter = TelegramAdapter("123:abc")
    adapter._bot = SimpleNamespace(send_message=send_message)
    await adapter.send("777", OutgoingMessage(text="Morning", buttons=[CONNECT]))
    (row,) = calls[-1]["reply_markup"].inline_keyboard
    assert [(b.text, b.url) for b in row] == [("Connect Gmail", CONNECT.url)]


async def test_a_message_telegram_refuses_over_its_buttons_still_goes_with_the_links_as_text(relayed):
    adapter, fake = relayed
    plain = fake.handler

    def refuse_buttons(request: httpx.Request) -> httpx.Response:
        if "reply_markup" in json.loads(request.content)["params"]:
            fake.calls.append(("refused", {}))
            return httpx.Response(200, json={"ok": False, "error": "Bad Request: BUTTON_URL_INVALID"})
        return plain(request)

    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(refuse_buttons))
    await adapter.send("777", OutgoingMessage(text="Morning", buttons=[CONNECT]))
    (sent,) = fake.sent()
    assert "reply_markup" not in sent and sent["text"] == f"Morning\n\nConnect Gmail: {CONNECT.url}"


async def test_the_direct_adapter_resends_without_buttons_when_telegram_refuses_them():
    calls: list[dict] = []

    async def send_message(**kwargs):
        if "reply_markup" in kwargs:
            raise RuntimeError("Bad Request: BUTTON_URL_INVALID")
        calls.append(kwargs)
        return SimpleNamespace(message_id=1)

    adapter = TelegramAdapter("123:abc")
    adapter._bot = SimpleNamespace(send_message=send_message)
    await adapter.send("777", OutgoingMessage(text="Morning", buttons=[CONNECT]))
    assert calls == [{"chat_id": 777, "text": f"Morning\n\nConnect Gmail: {CONNECT.url}"}]
