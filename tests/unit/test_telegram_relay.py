"""Telegram through rune-relay on a hosted machine: updates handed over, replies signed to the relay.

The machine holds no bot token and keeps no connection open, so it can sleep with Telegram on.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json

import httpx
import pytest

from rune.channels.telegram_relay import RelayTelegramAdapter
from rune.channels.types import ApprovalDecision, IncomingMessage, OutgoingMessage


class Relay:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 99}})

    def calls(self) -> list[tuple[str, dict]]:
        return [(json.loads(r.content)["method"], json.loads(r.content)["params"]) for r in self.requests]


@pytest.fixture
def relay(monkeypatch):
    from rune.connectors.store import authority_key, broker_home

    authority_key(broker_home(), create=True)
    monkeypatch.setenv("RUNE_CONNECTOR_KEY_ID", "tenant-1")
    fake = Relay()
    adapter = RelayTelegramAdapter("http://10.0.0.1:4110",
                                   client=httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)))
    return adapter, fake


def message(update_id: int, text: str = "hello", chat: int = 777) -> dict:
    return {"update_id": update_id, "message": {"message_id": 5, "text": text, "chat": {"id": chat, "type": "private"},
                                                "from": {"id": 777, "username": "alice"}}}


async def drain(adapter: RelayTelegramAdapter) -> None:
    while adapter._tasks:
        await asyncio.gather(*list(adapter._tasks))


async def test_it_reports_relayed_so_the_host_lets_the_machine_sleep(relay):
    adapter, _ = relay
    await adapter.start()
    assert adapter.status == "relayed"


async def test_a_handed_over_message_reaches_the_gateway_once(relay):
    adapter, _ = relay
    seen: list[IncomingMessage] = []

    async def on_message(incoming: IncomingMessage) -> None:
        seen.append(incoming)

    adapter.on_message = on_message
    assert await adapter.deliver([message(1), message(1), {"update_id": "x"}]) == 1
    assert await adapter.deliver([message(1), message(2, "again")]) == 1, "handed over twice, run once"
    await drain(adapter)
    assert [(m.channel_id, m.sender_id, m.text) for m in seen] == [("777", "777", "hello"), ("777", "777", "again")]
    assert seen[0].metadata["relayed"] is True


async def test_replies_are_signed_with_the_broker_key_for_the_relay(relay):
    adapter, fake = relay
    await adapter.send("777", OutgoingMessage(text="a < b"))
    request = fake.requests[-1]
    assert str(request.url) == "http://10.0.0.1:4110/telegram"
    from rune.connectors.store import authority_key, broker_home

    expected = hmac.new(authority_key(broker_home()), b"\n".join((
        b"telegram", request.headers["x-rune-expires"].encode(), request.headers["x-rune-nonce"].encode(),
        request.content)), hashlib.sha256).hexdigest()
    assert request.headers["x-rune-signature"] == expected and request.headers["x-rune-key-id"] == "tenant-1"
    assert fake.calls()[-1] == ("sendMessage", {"chat_id": 777, "text": "a &lt; b", "parse_mode": "HTML"})


async def test_a_long_reply_is_split_like_telegram_needs(relay):
    adapter, fake = relay
    await adapter.send("777", OutgoingMessage(text="x" * 5000))
    assert [len(p["text"]) for m, p in fake.calls() if m == "sendMessage"] == [4096, 904]


async def test_an_approval_button_resolves_the_waiting_approval(relay):
    adapter, fake = relay
    waiting = asyncio.create_task(adapter.wait_for_approval("ap1", timeout=5))
    await asyncio.sleep(0)
    await adapter.deliver([{"update_id": 3, "callback_query": {
        "id": "cb1", "data": "approval:allow_once:ap1", "from": {"id": 777, "username": "alice"},
        "message": {"message_id": 8, "chat": {"id": 777}}}}])
    await drain(adapter)
    assert (await waiting).decision == ApprovalDecision.APPROVE_ONCE
    assert ("answerCallbackQuery", {"callback_query_id": "cb1", "text": "Approved!"}) in fake.calls()


async def test_a_question_is_answered_by_the_next_message(relay):
    adapter, _ = relay
    adapter.on_message = None
    asked = asyncio.create_task(adapter.ask_question("777", "Which one?", timeout=5))
    await asyncio.sleep(0.01)
    adapter.on_message = _ignore
    await adapter.deliver([message(4, "the second")])
    await drain(adapter)
    assert await asked == "the second"


async def _ignore(_incoming: IncomingMessage) -> None:
    return None


def test_with_a_relay_the_hosted_machine_ignores_a_bot_token_of_its_own(monkeypatch):
    from rune.channels import registry as registry_mod

    monkeypatch.setattr(registry_mod, "_registry", None)
    monkeypatch.setenv("RUNE_CHANNEL_RELAY_URL", "http://10.0.0.1:4110/")
    monkeypatch.setenv("RUNE_TELEGRAM_TOKEN", "123:abc")
    assert "telegram" in registry_mod.auto_discover_channels()
    assert isinstance(registry_mod.get_channel_registry().get("telegram"), RelayTelegramAdapter)


async def test_channels_list_says_relayed_not_connected(monkeypatch, relay):
    from rune.api.handlers.channels import list_channels
    from rune.channels import registry as registry_mod

    adapter, _ = relay
    await adapter.start()
    monkeypatch.setattr(registry_mod, "_registry", None)
    registry_mod.get_channel_registry().register(adapter)
    listed = await list_channels()
    assert [(c.name, c.status) for c in listed.channels] == [("api", "connected"), ("telegram", "relayed")]


def test_channel_messages_count_as_active_runs(monkeypatch):
    from rune.daemon import gateway as gateway_mod

    class Gw:
        _working = {"777:777"}
        _sender_queues: dict = {}

    monkeypatch.setattr(gateway_mod, "_gateway_instance", Gw())
    assert gateway_mod.busy_channel_runs() == ["channel:777:777"]
    monkeypatch.setattr(gateway_mod, "_gateway_instance", None)
    assert gateway_mod.busy_channel_runs() == []
