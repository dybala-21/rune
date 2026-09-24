"""Exercise browser actions against Chromium, including failed and stale targets."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from rune.agent.tool_output import ToolOutput, output_for_model
from rune.capabilities.browser.capabilities import (
    BrowserActParams,
    BrowserObserveParams,
    BrowserScreenshotParams,
    browser_act,
    browser_observe,
    browser_screenshot,
    register_browser_capabilities,
)
from rune.capabilities.browser.core import _get_browser
from rune.capabilities.browser.extended import BrowserBatchParams, browser_batch
from rune.capabilities.browser.helpers import extract_interactive_elements
from rune.capabilities.browser.network import get_network_monitor
from rune.capabilities.browser.session import browser_operation, browser_session, current_session
from rune.capabilities.registry import CapabilityRegistry


@pytest.fixture
async def screen(tmp_path, monkeypatch):
    pytest.importorskip("playwright.async_api")
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    async with browser_session():
        browser, page = await _get_browser("managed")
        await page.set_content('''
            <title>Expense review</title>
            <label><input id="approved" type="checkbox">Approved</label>
            <button id="save" onclick="window.saves++">Save</button>
            <output id="count"></output>
            <script>window.saves = 0;</script>
        ''')
        yield page
    assert not browser.is_connected()


async def reference(page, name):
    elements = await extract_interactive_elements(page)
    return next(el.ref for el in elements if el.name == name)


async def test_same_url_actions_dispatch_once_and_report_control_state(screen):
    result = await browser_act(BrowserActParams(action="click", selector=await reference(screen, "Approved")))
    assert result.success and result.metadata["control_state"]["checked"] is True
    assert await screen.locator("#approved").is_checked()
    saved = await browser_act(BrowserActParams(action="click", selector=await reference(screen, "Save")))
    assert saved.success and saved.metadata["action_status"] == "dispatched"
    assert not saved.metadata["page_changed"]
    assert await screen.evaluate("window.saves") == 1


async def test_resume_observes_the_live_page_without_repeating_a_click(screen, tmp_path):
    from rune.agent.execution_journal import ExecutionJournal, reconcile
    from rune.api.run_snapshot import RunSnapshots
    from rune.api.run_store import RunStore

    store = RunStore(tmp_path / "resume.db")
    runs = RunSnapshots(store)
    runs.start("before", "one", "save")
    runs.start("after", "one", "continue")
    try:
        params = BrowserActParams(action="click", selector=await reference(screen, "Save"))
        first = ExecutionJournal(store, "before", str(tmp_path))
        assert (await first.execute("browser_act", params.model_dump(), lambda: browser_act(params))).success
        assert await screen.evaluate("window.saves") == 1
        records = reconcile(store.attempts("before"), browser=current_session())
        next_run = ExecutionJournal(store, "after", str(tmp_path), previous=records)
        replay = await next_run.execute("browser_act", params.model_dump(), lambda: browser_act(params))
        assert replay.metadata["replayed"]
        assert await screen.evaluate("window.saves") == 1
        assert not (await browser_act(params)).success
        assert (await browser_observe(BrowserObserveParams())).success
        assert not current_session().needs_observation
        runs.start("again", "one", "continue again")
        records = reconcile([*records, *store.attempts("after")], browser=current_session())
        again = ExecutionJournal(store, "again", str(tmp_path), previous=records)
        replay = await again.execute("browser_act", params.model_dump(), lambda: browser_act(params))
        assert replay.metadata == {"replayed": True}
        assert "Observe the current page" in replay.output
        assert await screen.evaluate("window.saves") == 1
    finally:
        runs.close()


@pytest.mark.parametrize("change", ["replacement", "new_document", "duplicate"])
async def test_stale_or_ambiguous_reference_never_selects_a_replacement(screen, change):
    ref = await reference(screen, "Save")
    if change == "replacement":
        await screen.locator("#save").evaluate("el => el.outerHTML = el.outerHTML")
    elif change == "new_document":
        await screen.reload()
        await screen.set_content('<button id="save" onclick="window.saves++">Save</button><script>window.saves=0</script>')
    else:
        await screen.locator("#save").evaluate("el => el.after(el.cloneNode(true))")
    result = await browser_act(BrowserActParams(action="click", selector=ref))
    assert not result.success and result.metadata["action_status"] == "not_executed"
    assert await screen.evaluate("window.saves") == 0
    if change == "duplicate":
        result = await browser_act(BrowserActParams(action="click", selector="button"))
        assert not result.success
        assert await screen.evaluate("window.saves") == 0


async def test_overlay_is_not_dismissed_or_clicked_through(screen):
    await screen.evaluate('''() => {
        window.consent = 0;
        const overlay = document.createElement('div');
        overlay.setAttribute('role', 'dialog');
        overlay.style = 'position:fixed;inset:0;z-index:9999;background:white';
        overlay.innerHTML = '<button onclick="window.consent++;this.parentNode.remove()">Accept all</button>';
        document.body.append(overlay);
    }''')
    assert (await browser_observe(BrowserObserveParams())).success
    result = await browser_act(BrowserActParams(action="click", selector="#save"))
    assert not result.success
    assert await screen.evaluate("window.saves") == 0
    assert await screen.evaluate("window.consent") == 0
    assert await screen.get_by_role("dialog").is_visible()


async def test_lost_acknowledgement_does_not_repeat_a_write(screen, monkeypatch):
    from playwright.async_api import ElementHandle

    click = ElementHandle.click

    async def click_then_disconnect(self, **kwargs):
        await click(self, **kwargs)
        raise TimeoutError("response lost after dispatch")

    monkeypatch.setattr(ElementHandle, "click", click_then_disconnect)
    result = await browser_act(BrowserActParams(action="click", selector="#save"))
    assert not result.success and result.metadata["action_status"] == "unknown"
    assert await screen.evaluate("window.saves") == 1
    assert (await browser_observe(BrowserObserveParams())).success
    retry = await browser_act(BrowserActParams(action="click", selector="#save"))
    assert not retry.success and retry.metadata["action_status"] == "not_executed"
    assert await screen.evaluate("window.saves") == 1


async def test_screenshots_are_unique_and_writes_respect_isolation(screen, tmp_path, monkeypatch):
    first = await browser_screenshot(BrowserScreenshotParams())
    await screen.locator("body").evaluate("el => el.style.background='blue'")
    second = await browser_screenshot(BrowserScreenshotParams())
    assert first.success and second.success and first.metadata["path"] != second.metadata["path"]
    first_image = output_for_model(first.output, "browser_screenshot", first)
    second_image = output_for_model(second.output, "browser_screenshot", second)
    assert isinstance(first_image, ToolOutput) and isinstance(second_image, ToolOutput)
    assert first_image.images[0].data != second_image.images[0].data
    monkeypatch.setenv("RUNE_ISOLATION_ROOT", str(tmp_path / "workspace"))
    denied = await browser_screenshot(BrowserScreenshotParams(path=str(tmp_path / "outside.png")))
    assert not denied.success and not (tmp_path / "outside.png").exists()
    # macOS puts pytest's default temp root under /var, which Guardian protects.
    with tempfile.TemporaryDirectory(dir="/tmp") as workspace:
        monkeypatch.setenv("RUNE_ISOLATION_ROOT", workspace)
        path = Path(workspace) / "screen.png"
        allowed = await browser_screenshot(BrowserScreenshotParams(path=str(path)))
        assert allowed.success and path.exists()


async def test_browser_batch_stops_after_failure_and_preserves_images(screen, monkeypatch):
    reg = CapabilityRegistry()
    register_browser_capabilities(reg)
    monkeypatch.setattr("rune.capabilities.registry.get_capability_registry", lambda: reg)
    result = await browser_batch(BrowserBatchParams(actions=[
        {"type": "act", "params": {"action": "click", "selector": "#missing"}},
        {"type": "act", "params": {"action": "click", "selector": "#save"}},
    ]))
    assert not result.success and "Not executed" in result.output
    assert await screen.evaluate("window.saves") == 0
    captured = await browser_batch(BrowserBatchParams(actions=[{"type": "screenshot"}]))
    assert isinstance(output_for_model(captured.output, "browser_batch", captured), ToolOutput)


async def test_batch_keeps_unchanged_nodes_but_rejects_replacements(screen, monkeypatch):
    reg = CapabilityRegistry()
    register_browser_capabilities(reg)
    monkeypatch.setattr("rune.capabilities.registry.get_capability_registry", lambda: reg)
    elements = {meta.name: meta.ref for meta in await extract_interactive_elements(screen)}
    result = await browser_batch(BrowserBatchParams(actions=[
        {"type": "act", "params": {"action": "check", "selector": elements["Approved"]}},
        {"type": "act", "params": {"action": "click", "selector": elements["Save"]}},
    ]))
    assert result.success and await screen.evaluate("window.saves") == 1
    assert await screen.locator("#approved").is_checked()
    await screen.locator("#save").evaluate("el => el.outerHTML = el.outerHTML")
    await browser_observe(BrowserObserveParams())
    stale = await browser_act(BrowserActParams(action="click", selector=elements["Save"]))
    assert not stale.success and await screen.evaluate("window.saves") == 1


async def test_concurrent_runs_own_pages_references_monitors_and_cleanup(screen):
    original_session = current_session()
    original_ref = await reference(screen, "Save")
    original_monitor = get_network_monitor()

    async def other_run():
        async with browser_session() as session:
            browser, other = await _get_browser("managed")
            await other.set_content('<button onclick="window.saves++">Save</button><script>window.saves=0</script>')
            assert current_session() is session and session is not original_session
            assert get_network_monitor() is not original_monitor
            await reference(other, "Save")
            result = await browser_act(BrowserActParams(action="click", selector=original_ref))
            assert not result.success and await other.evaluate("window.saves") == 0
        assert not browser.is_connected()

    await asyncio.create_task(other_run())
    result = await browser_act(BrowserActParams(action="click", selector=original_ref))
    assert result.success and await screen.evaluate("window.saves") == 1
    assert current_session() is original_session


async def test_explicit_open_replaces_a_closed_page_without_reusing_references(screen):
    ref = await reference(screen, "Save")
    await screen.close()
    observed = await browser_observe(BrowserObserveParams())
    assert not observed.success
    _, reopened = await _get_browser("managed")
    await reopened.set_content("<button>Save</button>")
    assert await reference(reopened, "Save") != ref
    result = await browser_act(BrowserActParams(action="click", selector=ref))
    assert not result.success and result.metadata["action_status"] == "not_executed"


async def test_cancellation_closes_owned_browser_and_operations_serialize(screen):
    order = []
    entered, release = asyncio.Event(), asyncio.Event()

    @browser_operation
    async def operation(first):
        order.append((first, "start"))
        if first:
            entered.set()
            await release.wait()
        order.append((first, "end"))

    first = asyncio.create_task(operation(True))
    await entered.wait()
    second = asyncio.create_task(operation(False))
    await asyncio.sleep(0)
    assert order == [(True, "start")]
    release.set()
    await asyncio.gather(first, second)
    assert order == [(True, "start"), (True, "end"), (False, "start"), (False, "end")]

    opened = asyncio.Future()

    async def run_until_cancelled():
        async with browser_session():
            browser, _ = await _get_browser("managed")
            opened.set_result(browser)
            await asyncio.Event().wait()

    run = asyncio.create_task(run_until_cancelled())
    browser = await opened
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert not browser.is_connected()
