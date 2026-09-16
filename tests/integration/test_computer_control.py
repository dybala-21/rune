import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from rune.api.computer import Computers, computer_router
from rune.capabilities.browser.capabilities import (
    BrowserActParams,
    BrowserObserveParams,
    browser_act,
    browser_observe,
)
from rune.capabilities.browser.core import _get_browser
from rune.capabilities.browser.helpers import extract_interactive_elements
from rune.capabilities.browser.session import current_session, with_browser_session


@pytest.fixture
async def computer():
    computers = Computers()
    entry = await computers.claim("expenses", "run1")
    async with computers.bind(entry):
        _, page = await _get_browser("managed")
        await page.set_content('''<title>Expenses</title>
            <label><input id="approved" type="checkbox">Approved</label>
            <button id="save" onclick="window.saves++">Save</button><script>window.saves=0</script>''')
        old_ref = next(el.ref for el in await extract_interactive_elements(page) if el.name == "Save")
    app = FastAPI()
    app.include_router(computer_router(computers, lambda: None))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield computers, entry, page, old_ref, client
    await computers.close()


async def command(client, entry, action, **kwargs):
    return await client.post("/api/computer/control", json={"sessionId": entry.session_id,
        "lease": entry.lease, "action": action, **kwargs})


async def test_handoff_invalidates_old_plans_and_manual_commands_are_consumed_once(computer):
    computers, entry, page, old_ref, client = computer
    assert (await command(client, entry, "takeover")).status_code == 409
    assert (await command(client, entry, "pause")).json()["state"] == "paused"
    assert (await command(client, entry, "takeover")).json()["state"] == "manual"
    view = (await client.get("/api/computer/state", params={"sessionId": "expenses"})).json()
    ref = next(el["ref"] for el in view["controls"] if el["name"] == "Save")
    assert ref != old_ref
    request = {"sessionId": "expenses", "lease": entry.lease, "frameId": view["frameId"], "ref": ref, "action": "click"}
    assert (await client.post("/api/computer/action", json=request)).status_code == 200
    assert (await client.post("/api/computer/action", json=request)).status_code == 409
    assert await page.evaluate("window.saves") == 1
    assert (await command(client, entry, "resume", instruction="Only check approval")).status_code == 200
    async with computers.bind(entry):
        assert not (await browser_act(BrowserActParams(action="click", selector=ref))).success
        assert (await browser_observe(BrowserObserveParams())).success
        assert not (await browser_act(BrowserActParams(action="click", selector=ref))).success
    assert await page.evaluate("window.saves") == 1


async def test_preview_and_control_cannot_cross_conversations_or_leases(computer):
    computers, entry, _, _, client = computer
    view = await computers.snapshot(entry)
    assert (await client.get(f"/api/computer/frame/{view['frameId']}", params={"sessionId": "other"})).status_code == 404
    assert (await client.post("/api/computer/control", json={"sessionId": "expenses", "lease": entry.lease - 1, "action": "pause"})).status_code == 409
    async with computers.bind(entry):
        with pytest.raises(RuntimeError, match="active run"):
            await computers.claim("expenses", "overlap")


async def test_a_preview_in_progress_does_not_reject_the_next_turn(computer):
    computers, entry, _, _, _ = computer
    computers.finish(entry, "run1")
    async with entry.lock:
        task = asyncio.create_task(computers.claim("expenses", "run2"))
        await asyncio.sleep(0)
        assert not task.done()
    assert await task is entry
    assert entry.control.run_id == "run2"


@pytest.mark.parametrize("now", [60.0, 3600.0])
async def test_conversation_survives_runs_but_expiry_releases_resources(computer, monkeypatch, now):
    computers, entry, page, _, _ = computer
    original = entry.browser.browser
    computers.finish(entry, "run1")
    resumed = await computers.claim("expenses", "run2")
    assert resumed is entry

    @with_browser_session
    async def run():
        assert current_session() is entry.browser
        assert current_session().page is page

    async with computers.bind(resumed):
        await run()
    assert original.is_connected()
    computers.finish(entry, "run2")
    monkeypatch.setattr("rune.api.computer.time", SimpleNamespace(monotonic=lambda: now))
    entry.touched = now
    await computers.reap()
    assert original.is_connected() and computers.entries["expenses"] is entry

    entry.touched = now - computers.idle_seconds - 1
    await computers.reap()
    assert not original.is_connected() and not computers.entries


async def test_cancelled_click_keeps_unknown_outcome_until_user_checks_it(computer, monkeypatch):
    from playwright.async_api import ElementHandle

    computers, entry, page, _, client = computer
    click = ElementHandle.click

    async def cancelled_after_click(self, **kwargs):
        await click(self, **kwargs)
        raise asyncio.CancelledError

    monkeypatch.setattr(ElementHandle, "click", cancelled_after_click)
    async with computers.bind(entry):
        with pytest.raises(asyncio.CancelledError):
            await browser_act(BrowserActParams(action="click", selector="#save"))
    assert entry.browser.uncertain_action and await page.evaluate("window.saves") == 1
    assert (await command(client, entry, "pause")).status_code == 200
    assert (await command(client, entry, "resume")).status_code == 409
    assert (await command(client, entry, "resume", acknowledgeUnknown=True)).status_code == 200
    assert not entry.browser.uncertain_action


async def test_manual_control_rejects_a_replaced_target(computer):
    _, entry, page, _, client = computer
    await command(client, entry, "pause")
    await command(client, entry, "takeover")
    view = (await client.get("/api/computer/state", params={"sessionId": "expenses"})).json()
    ref = next(el["ref"] for el in view["controls"] if el["name"] == "Save")
    await page.locator("#save").evaluate("el => el.outerHTML = el.outerHTML")
    result = await client.post("/api/computer/action", json={"sessionId": "expenses", "lease": entry.lease,
        "frameId": view["frameId"], "ref": ref, "action": "click"})
    assert result.status_code == 409 and await page.evaluate("window.saves") == 0
