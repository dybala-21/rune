"""Approval buttons in a chat answer the run that is waiting on them, through the gateway.

The gateway shows a request with the channel's buttons and with numbered choices; either answer
settles it, and a button pressed on a request nothing waits for says so instead of "Approved".
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from rune.channels.registry import ChannelRegistry
from rune.channels.telegram import TelegramAdapter
from rune.channels.telegram_relay import RelayTelegramAdapter
from rune.daemon.gateway import ChannelGateway


class Relay:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append((body["method"], body["params"]))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": len(self.calls)}})

    def buttons(self) -> list[dict]:
        return [p["reply_markup"] for m, p in self.calls if m == "sendMessage" and "reply_markup" in p]


@pytest.fixture
def relayed(monkeypatch):
    from rune.connectors.store import authority_key, broker_home

    authority_key(broker_home(), create=True)
    monkeypatch.setenv("RUNE_CONNECTOR_KEY_ID", "tenant-1")
    fake = Relay()
    adapter = RelayTelegramAdapter("http://10.0.0.1:4110",
                                   client=httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)))
    registry = ChannelRegistry()
    registry.register(adapter)
    return ChannelGateway(registry), adapter, fake


async def shown(fake: Relay) -> dict[str, str]:
    for _ in range(100):
        if fake.buttons():
            row = fake.buttons()[-1]["inline_keyboard"][0]
            return {b["text"]: b["callback_data"] for b in row}
        await asyncio.sleep(0.01)
    raise AssertionError("no buttons were sent")


async def press(adapter: RelayTelegramAdapter, data: str, update_id: int = 10) -> None:
    await adapter.deliver([{"update_id": update_id, "callback_query": {
        "id": f"cb{update_id}", "data": data, "from": {"id": 777, "username": "alice"},
        "message": {"message_id": 1, "chat": {"id": 777}}}}])
    while adapter._tasks:
        await asyncio.gather(*list(adapter._tasks))


def ask(gateway: ChannelGateway, adapter, timeout: float = 5) -> asyncio.Task:
    return asyncio.create_task(gateway.request_approval("777:777", adapter.name, "777",
                                                        command="POST https://example.com", timeout=timeout))


@pytest.mark.parametrize(("label", "approved", "decision"), [
    ("Allow", True, "approve_once"), ("Always allow", True, "approve_always"), ("Deny", False, "deny")])
async def test_a_pressed_button_answers_the_waiting_run(relayed, label, approved, decision):
    gateway, adapter, fake = relayed
    waiting = ask(gateway, adapter)
    buttons = await shown(fake)
    assert all(len(data.encode()) <= 64 for data in buttons.values()), "Telegram keeps 64 bytes of callback data"
    await press(adapter, buttons[label])
    answer = await asyncio.wait_for(waiting, 1)
    assert (answer.approved, answer.decision, answer.timed_out) == (approved, decision, False)
    assert ("editMessageText", {"chat_id": 777, "message_id": 1,
                                "text": f"{'Approved' if approved else 'Denied'} by alice"}) in fake.calls
    assert gateway._pending_approvals == {} and adapter._pending_approvals == {}


async def test_a_typed_answer_still_works_and_its_buttons_then_say_so(relayed):
    gateway, adapter, fake = relayed
    waiting = ask(gateway, adapter)
    buttons = await shown(fake)
    assert gateway._resolve_pending_approval("777:777", "1")
    assert (await asyncio.wait_for(waiting, 1)).approved
    assert adapter._pending_approvals == {}
    await press(adapter, buttons["Deny"])
    assert ("answerCallbackQuery", {"callback_query_id": "cb10", "text": "This request is no longer waiting"}) in fake.calls
    assert not [p for m, p in fake.calls if m == "editMessageText"], "never shown as approved or denied"


async def test_a_button_pressed_after_the_request_timed_out_settles_nothing(relayed):
    gateway, adapter, fake = relayed
    waiting = ask(gateway, adapter, timeout=0.05)
    buttons = await shown(fake)
    answer = await waiting
    assert (answer.approved, answer.timed_out) == (False, True)
    await press(adapter, buttons["Allow"])
    assert ("answerCallbackQuery", {"callback_query_id": "cb10", "text": "This request is no longer waiting"}) in fake.calls
    assert ("editMessageReplyMarkup", {"chat_id": 777, "message_id": 1, "reply_markup": {"inline_keyboard": []}}) in fake.calls


async def test_buttons_of_an_earlier_request_do_not_answer_a_newer_one(relayed):
    gateway, adapter, fake = relayed
    first = ask(gateway, adapter)
    old = await shown(fake)
    second = ask(gateway, adapter)
    with pytest.raises(RuntimeError):
        await first
    for _ in range(100):
        if len(fake.buttons()) == 2:
            break
        await asyncio.sleep(0.01)
    await press(adapter, old["Allow"])
    assert not second.done()
    new = await shown(fake)
    await press(adapter, new["Deny"], update_id=11)
    assert (await asyncio.wait_for(second, 1)).approved is False


class Callback(SimpleNamespace):
    async def answer(self, text: str | None = None) -> None:
        self.answered = text


class Bot:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def send_message(self, **kwargs):
        self.calls.append(("send_message", kwargs))
        return SimpleNamespace(message_id=len(self.calls))

    async def edit_message_text(self, **kwargs):
        self.calls.append(("edit_message_text", kwargs))

    async def edit_message_reply_markup(self, **kwargs):
        self.calls.append(("edit_message_reply_markup", kwargs))


async def test_the_direct_telegram_adapter_answers_the_waiting_run_too():
    adapter = TelegramAdapter("123:abc")
    adapter._bot = Bot()
    registry = ChannelRegistry()
    registry.register(adapter)
    gateway = ChannelGateway(registry)
    sent: list[str] = []

    async def send_approval(channel_id: str, description: str, approval_id: str) -> None:
        sent.append(approval_id)

    async def send(channel_id, message):
        return None

    adapter.send_approval = send_approval
    adapter.send = send
    waiting = ask(gateway, adapter)
    for _ in range(100):
        if sent:
            break
        await asyncio.sleep(0.01)
    message = SimpleNamespace(chat=SimpleNamespace(id=777), message_id=3)
    pressed = Callback(data=f"approval:allow_once:{sent[0]}", from_user=SimpleNamespace(username="alice"), message=message)
    await adapter._handle_callback_query(pressed)
    assert (await asyncio.wait_for(waiting, 1)).approved
    assert pressed.answered == "Approved!"
    late = Callback(data=f"approval:deny:{sent[0]}", from_user=SimpleNamespace(username="bob"), message=message)
    await adapter._handle_callback_query(late)
    assert late.answered == "This request is no longer waiting"
    assert adapter._bot.calls[-1][0] == "edit_message_reply_markup"
