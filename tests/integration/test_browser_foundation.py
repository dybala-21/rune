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


async def test_result_checks_verify_saved_state_without_repeating_input(screen):
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.intent_engine import resolve_intent_contract
    from rune.agent.task_evidence import TaskEvidence

    await screen.set_content('''<label>Quantity<input id="quantity" value="1"></label>
        <button id="save" onclick="window.saves++;document.querySelector('#saved').textContent='Saved '+document.querySelector('#quantity').value">Save</button>
        <output id="saved">Not saved</output><script>window.saves=0</script>''')
    classification = ClassificationResult("browser", .99, 2)
    contract = resolve_intent_contract(classification, .99)
    evidence = TaskEvidence.for_request("Set quantity to 3 and save", classification)
    edit = await browser_act(BrowserActParams(action="type", selector="#quantity", value="3"))
    evidence.observe("browser_act", edit)
    assert evidence.blocker(contract, 0)
    checks = [{"selector":"#quantity", "property":"value", "expected":"3", "requirement":"quantity to 3"},
              {"selector":"#saved", "property":"text", "expected":"Saved 3", "requirement":"save"}]
    before = await browser_observe(BrowserObserveParams(checks=checks))
    evidence.observe("browser_observe", before)
    assert evidence.blocker(contract, 0) and not before.metadata["result_checks"]["passed"]
    saved = await browser_act(BrowserActParams(action="click", selector="#save", checks=checks))
    evidence.observe("browser_act", saved)
    assert evidence.blocker(contract, 0) is None
    resumed = TaskEvidence.for_request("Finish the previous change", classification)
    resumed.request_texts += ("Set quantity to 3 and save",)
    readback = await browser_observe(BrowserObserveParams(checks=checks))
    resumed.observe("browser_observe", readback)
    assert resumed.blocker(contract, 0) is None
    assert await screen.evaluate("window.saves") == 1
    await screen.locator("#quantity").fill("4")
    changed = await browser_observe(BrowserObserveParams(checks=checks))
    resumed.observe("browser_observe", changed)
    assert resumed.blocker(contract, 0)


async def test_partial_readback_rechecks_prior_values_and_corrected_selectors(screen):
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.intent_engine import resolve_intent_contract
    from rune.agent.task_evidence import TaskEvidence

    await screen.set_content('''<input id="quantity" value="3"><input id="title" value="Review">
        <button id="save" onclick="window.saves++;document.querySelector('#saved').textContent='Saved'">Save</button>
        <output id="saved">Not saved</output><script>window.saves=0</script>''')
    classification = ClassificationResult("browser", .99, 2)
    contract = resolve_intent_contract(classification, .99)
    evidence = TaskEvidence.for_request("Set quantity to 3, title to Review and save", classification)
    fields = [{"selector": "#quantity", "property": "value", "expected": "3", "requirement": "quantity to 3"},
              {"selector": "input[type=text]", "property": "value", "expected": "Review", "requirement": "title to Review"}]
    first = await browser_observe(BrowserObserveParams(checks=fields))
    evidence.observe("browser_observe", first)
    assert evidence.blocker(contract, 0)
    await browser_act(BrowserActParams(action="click", selector="#save"))
    params = {"checks": [{**fields[1], "selector": "#title"},
                          {"selector": "#saved", "property": "text", "expected": "Saved", "requirement": "save"}]}
    evidence.prepare("browser_observe", params)
    readback = await browser_observe(BrowserObserveParams(**params))
    evidence.observe("browser_observe", readback)
    assert evidence.blocker(contract, 0) is None
    assert {item["selector"] for item in readback.metadata["result_checks"]["checks"]} == {"#quantity", "#title", "#saved"}
    await screen.locator("#quantity").fill("4")
    params = {"checks": [params["checks"][1]]}
    evidence.prepare("browser_observe", params)
    changed = await browser_observe(BrowserObserveParams(**params))
    evidence.observe("browser_observe", changed)
    assert evidence.blocker(contract, 0)
    assert not changed.metadata["result_checks"]["passed"]
    assert await screen.evaluate("window.saves") == 1


async def test_check_without_a_target_is_rejected_before_input(screen):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="current element ref"):
        await browser_act(BrowserActParams(action="click", selector="#save", checks=[
            {"property": "text", "expected": "Saved", "requirement": "save"},
        ]))
    assert await screen.evaluate("window.saves") == 0


async def test_failed_check_requires_readback_before_another_input(screen):
    from rune.agent.run_control import RunControl, control_scope

    await screen.set_content('''<button id="open" onclick="document.querySelector('dialog').showModal()">Review</button>
        <dialog open><button id="save" onclick="window.saves++;document.querySelector('#saved').textContent='Saved';this.closest('dialog').close()">Save</button></dialog>
        <output id="saved">Not saved</output><script>window.saves=0</script>''')
    with control_scope(RunControl("save")):
        saved = await browser_act(BrowserActParams(action="click", selector="#save", checks=[
            {"selector": "#missing", "property": "text", "expected": "Saved", "requirement": "save"},
        ]))
        assert saved.success and not saved.metadata["result_checks"]["passed"]
        await browser_observe(BrowserObserveParams())
        await browser_observe(BrowserObserveParams(checks=[
            {"selector": "#saved", "property": "text", "expected": "Saved", "requirement": "another task"},
        ]))
        denied = await browser_act(BrowserActParams(action="click", selector="#open"))
        assert not denied.success and denied.metadata["verification_required"]
        assert await screen.evaluate("window.saves") == 1
        verified = await browser_observe(BrowserObserveParams(checks=[
            {"selector": "#saved", "property": "text", "expected": "Saved", "requirement": "save"},
        ]))
        assert verified.metadata["result_checks"]["passed"]
        assert current_session().pending_verification is None


async def test_rejected_save_can_retry_after_reading_the_failure(screen):
    await screen.set_content('''<button id="save" onclick="save(this)">Save</button><output id="status">Not saved</output>
        <script>window.attempts=0;window.saves=0;function save(button){
            window.attempts++;
            if(window.attempts===1){
                document.querySelector('#status').textContent='Service unavailable. Nothing saved.';
                button.replaceWith(button.cloneNode(true));
            }else{window.saves++;document.querySelector('#status').textContent='Saved';}
        }</script>''')
    checks = [{"selector": "#status", "property": "text", "expected": "Saved", "requirement": "save"}]
    first = await browser_act(BrowserActParams(action="click", selector="#save", checks=checks))
    assert first.success and not first.metadata["result_checks"]["passed"]
    assert await screen.evaluate("window.attempts") == 1
    readback = await browser_observe(BrowserObserveParams(checks=checks))
    assert readback.metadata["result_checks"]["checks"][0]["actual"] == "Service unavailable. Nothing saved."
    second = await browser_act(BrowserActParams(action="click", selector=await reference(screen, "Save"), checks=checks))
    assert second.success and second.metadata["result_checks"]["passed"]
    assert await screen.evaluate("[window.attempts,window.saves]") == [2, 1]


async def test_intermediate_mismatch_does_not_block_the_next_save_step(screen):
    await screen.set_content('''<button id="review" onclick="document.querySelector('dialog').showModal()">Review</button>
        <dialog><button id="save" onclick="window.saves++;document.querySelector('#status').textContent='Saved';this.closest('dialog').close()">Save</button></dialog>
        <output id="status">Not saved</output><script>window.saves=0</script>''')
    checks = [{"selector": "#status", "property": "text", "expected": "Saved", "requirement": "save"}]
    review = await browser_act(BrowserActParams(action="click", selector="#review", checks=checks))
    assert review.success and not review.metadata["result_checks"]["passed"]
    saved = await browser_act(BrowserActParams(action="click", selector="#save", checks=checks))
    assert saved.success and saved.metadata["result_checks"]["passed"]
    assert await screen.evaluate("window.saves") == 1


@pytest.mark.parametrize("prop", ["text", "value"])
async def test_select_option_is_not_a_saved_confirmation(screen, prop):
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.intent_engine import resolve_intent_contract
    from rune.agent.task_evidence import TaskEvidence

    await screen.set_content('<select><option id="option">10:00</option></select>')
    goal = "Select 10:00 and save"
    classification = ClassificationResult("browser", .99, 2)
    evidence = TaskEvidence.for_request(goal, classification)
    result = await browser_observe(BrowserObserveParams(checks=[
        {"selector": "#option", "property": prop, "expected": "10:00", "requirement": "Select 10:00"},
    ]))
    evidence.observe("browser_observe", result)
    assert result.metadata["result_checks"]["checks"][0]["kind"] == "control"
    assert evidence.blocker(resolve_intent_contract(classification, .99), 0)


async def test_checks_reject_missing_targets_and_passwords(screen):
    await screen.set_content('<input id="secret" type="password" value="private-value">')
    result = await browser_observe(BrowserObserveParams(checks=[
        {"selector":"#missing", "property":"text", "expected":"Saved"},
        {"selector":"#secret", "property":"value", "expected":"private-value"},
    ]))
    assert not result.metadata["result_checks"]["passed"]
    assert all(check["actual"] is None for check in result.metadata["result_checks"]["checks"])


async def test_action_label_is_not_a_saved_result(screen):
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.intent_engine import resolve_intent_contract
    from rune.agent.task_evidence import TaskEvidence

    goal = "Approve the expense and save"
    classification = ClassificationResult("browser", .99, 2)
    result = await browser_observe(BrowserObserveParams(checks=[
        {"selector": "#save", "property": "text", "expected": "Save", "requirement": "save"},
    ]))
    evidence = TaskEvidence.for_request(goal, classification)
    evidence.observe("browser_observe", result)
    assert evidence.blocker(resolve_intent_contract(classification, .99), 0)
    assert await screen.evaluate("window.saves") == 0


async def test_empty_input_can_be_verified(screen):
    await screen.set_content('<input id="note" value="">')
    result = await browser_observe(BrowserObserveParams(checks=[
        {"selector": "#note", "property": "value", "expected": ""},
    ]))
    assert result.metadata["result_checks"]["passed"]


async def test_successful_readback_skips_snapshot_but_rechecks_live_values(screen, monkeypatch):
    from unittest.mock import AsyncMock

    await screen.set_content('<input id="quantity" value="3"><output id="saved">Saved 3</output>')
    checks = [{"selector": "#quantity", "property": "value", "expected": "3"},
              {"selector": "#saved", "property": "text", "expected": "Saved 3"}]
    with monkeypatch.context() as patch:
        snapshot = AsyncMock(side_effect=AssertionError("A passing readback does not need a snapshot"))
        settle = AsyncMock(side_effect=AssertionError("A passing readback does not need a quiet period"))
        patch.setattr("rune.capabilities.browser.capabilities._accessibility_snapshot", snapshot)
        patch.setattr("rune.capabilities.browser.helpers.wait_for_dom_settle", settle)
        result = await browser_observe(BrowserObserveParams(checks=checks))
        assert result.success and result.metadata["result_checks"]["passed"]
        assert "Interactive Elements" not in result.output
        snapshot.assert_not_called()
        settle.assert_not_called()
    await screen.locator("#quantity").fill("4")
    changed = await browser_observe(BrowserObserveParams(checks=checks))
    assert not changed.metadata["result_checks"]["passed"]
    assert "Interactive Elements" in changed.output
    assert changed.metadata["result_checks"]["checks"][0]["actual"] == "4"


async def test_checks_after_user_control_refresh_references(screen):
    current_session().needs_observation = True
    result = await browser_observe(BrowserObserveParams(checks=[
        {"selector": "#approved", "property": "checked", "expected": "false"},
    ]))
    assert result.success and result.metadata["result_checks"]["passed"]
    assert "Interactive Elements" in result.output
    assert not current_session().needs_observation


async def test_missing_observation_scope_falls_back_without_waiting():
    from unittest.mock import AsyncMock, Mock

    from rune.capabilities.browser.core import _accessibility_snapshot

    missing = Mock(count=AsyncMock(return_value=0), wait_for=AsyncMock())
    root = Mock(aria_snapshot=AsyncMock(return_value='- button "Save"'))
    page = Mock(locator=Mock(side_effect=[missing, root]))
    assert await _accessibility_snapshot(page, "form") == '- button "Save"'
    missing.wait_for.assert_not_called()


async def test_observation_does_not_wait_for_background_network_requests(screen):
    pending = asyncio.Event()
    release = asyncio.Event()

    async def route(request):
        if request.request.url.endswith("/pending"):
            pending.set()
            await release.wait()
            await request.fulfill(body="done")
        else:
            await request.fulfill(content_type="text/html", body=(
                '<label>Count<input value="5"></label><button>Calculate</button>'
                '<script>fetch("/pending")</script>'
            ))

    await screen.route("http://rune.test/**", route)
    try:
        await screen.goto("http://rune.test/fixture", wait_until="domcontentloaded")
        await asyncio.wait_for(pending.wait(), 3)
        result = await asyncio.wait_for(browser_observe(BrowserObserveParams()), 3)
        assert result.success and "Calculate" in result.output
        assert not release.is_set()
    finally:
        release.set()
        await screen.unroute_all(behavior="wait")


async def test_same_url_actions_dispatch_once_and_report_control_state(screen):
    result = await browser_act(BrowserActParams(action="click", selector=await reference(screen, "Approved")))
    assert result.success and result.metadata["control_state"]["checked"] is True
    assert await screen.locator("#approved").is_checked()
    saved = await browser_act(BrowserActParams(action="click", selector=await reference(screen, "Save")))
    assert saved.success and saved.metadata["action_status"] == "dispatched"
    assert not saved.metadata["page_changed"]
    assert await screen.evaluate("window.saves") == 1


async def test_wrong_select_target_can_be_corrected_without_unlocking_uncertain_actions(screen):
    await screen.set_content('<label>City<input id="city"></label><button id="save" onclick="window.saves++">Save</button><script>window.saves=0</script>')
    ref = await reference(screen, "City")
    invalid = await browser_act(BrowserActParams(action="select", selector=ref, value="Gunsan"))
    assert not invalid.success and invalid.metadata["action_status"] == "not_executed"
    assert not current_session().uncertain_action
    assert await screen.locator("#city").input_value() == ""
    assert (await browser_act(BrowserActParams(action="type", selector=ref, value="Gunsan"))).success
    assert (await browser_act(BrowserActParams(action="click", selector=await reference(screen, "Save")))).success
    assert await screen.evaluate("window.saves") == 1


async def test_native_select_and_its_associated_label_remain_supported(screen):
    await screen.set_content('<label id="label" for="city">City</label><select id="city"><option>Busan</option><option>Gunsan</option></select>')
    for selector, value in (("#city", "Gunsan"), ("#label", "Busan")):
        assert (await browser_act(BrowserActParams(action="select", selector=selector, value=value))).success
        assert await screen.locator("#city").input_value() == value


@pytest.mark.parametrize("syntax", ["[{}]", '[ref="{}"]', "[ref='{}']"])
async def test_displayed_brackets_preserve_reference_identity(screen, syntax):
    ref = await reference(screen, "Save")
    result = await browser_act(BrowserActParams(action="click", selector=syntax.format(ref)))
    assert result.success and await screen.evaluate("window.saves") == 1
    await screen.locator("#save").evaluate("el => el.outerHTML = el.outerHTML")
    stale = await browser_act(BrowserActParams(action="click", selector=syntax.format(ref)))
    assert not stale.success and await screen.evaluate("window.saves") == 1


@pytest.mark.parametrize("selector", ['[ref="ordinary"]', '[data-ref="e0"]', '[ref="e0"] button', '[ref="e0"],#save'])
def test_reference_unwrapping_does_not_reinterpret_general_css(selector):
    from rune.capabilities.browser.helpers import element_ref

    assert element_ref(selector) == ""


async def test_successful_input_is_not_repeated_just_to_verify_it(screen):
    from rune.agent.run_control import RunControl, control_scope

    ref = await reference(screen, "Save")
    with control_scope(RunControl("first")):
        assert (await browser_act(BrowserActParams(action="click", selector=ref))).success
        assert (await browser_observe(BrowserObserveParams())).success
        replay = await browser_act(BrowserActParams(action="click", selector=f"[{ref}]"))
        assert replay.success and replay.metadata["replayed"]
        assert await screen.evaluate("window.saves") == 1
        assert (await browser_act(BrowserActParams(action="click", selector=ref, repeat=True))).success
        assert await screen.evaluate("window.saves") == 1
    with control_scope(RunControl("next")):
        assert (await browser_act(BrowserActParams(action="click", selector=ref))).success
        assert await screen.evaluate("window.saves") == 2


async def test_extract_reads_current_refs_and_does_not_hide_empty_matches(screen):
    from rune.capabilities.browser.capabilities import BrowserExtractParams, browser_extract

    await screen.set_content('<label>Count<input value="1"></label>')
    ref = await reference(screen, "Count")
    await screen.get_by_role("textbox").fill("5")
    result = await browser_extract(BrowserExtractParams(selector=f"[{ref}]", attribute="value"))
    assert result.success and "current input value: '5'" in result.output
    assert "HTML value attribute: '1'" in result.output
    missing = await browser_extract(BrowserExtractParams(selector="spinbutton"))
    assert not missing.success and "browser_observe" in missing.error


async def test_extract_distinguishes_missing_attributes_empty_values_and_empty_text(screen):
    from rune.capabilities.browser.capabilities import BrowserExtractParams, browser_extract

    await screen.set_content('<input id="empty" value=""><div id="blank"></div><iframe src="about:blank"></iframe>')
    missing = await browser_extract(BrowserExtractParams(selector="html", attribute="outerHTML"))
    assert not missing.success and missing.metadata["count"] == 0
    assert missing.metadata["matched_count"] == 1 and "HTML attribute" in missing.error
    empty = await browser_extract(BrowserExtractParams(selector="#empty", attribute="value"))
    assert empty.success and empty.output == "" and empty.metadata["count"] == 1
    blank = await browser_extract(BrowserExtractParams(selector="#blank"))
    assert not blank.success and blank.metadata["empty_content"]
    frame = await browser_extract(BrowserExtractParams(selector="iframe", attribute="src"))
    assert frame.success and frame.output == "about:blank"


async def test_page_snapshot_exposes_loaded_frame_urls_without_another_tool_call(screen):
    from rune.capabilities.browser.core import _accessibility_snapshot

    await screen.route("https://frame-fixture.example/**", lambda route: route.fulfill(body="<p>Notice content</p>"))
    await screen.set_content('<main><p>Official site</p><iframe src="https://frame-fixture.example/news"></iframe>'
                             '<iframe srcdoc="inline content"></iframe></main>')
    snapshot = await _accessibility_snapshot(screen)
    assert "Official site" in snapshot and "https://frame-fixture.example/news" in snapshot
    assert "Embedded documents (not read" in snapshot
    assert "about:srcdoc" not in snapshot
    assert "Embedded documents" not in await _accessibility_snapshot(screen, "main p")


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
        if kwargs.get("trial"):
            return
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


@pytest.mark.parametrize("accept_value", [True, False])
async def test_batch_checks_fields_before_saving(screen, monkeypatch, accept_value):
    reg = CapabilityRegistry()
    register_browser_capabilities(reg)
    monkeypatch.setattr("rune.capabilities.registry.get_capability_registry", lambda: reg)
    await screen.set_content('''<input id="quantity" value="1">
        <button id="save" onclick="window.saves++;document.querySelector('#saved').textContent='Saved '+document.querySelector('#quantity').value">Save</button>
        <output id="saved">Not saved</output><script>window.saves=0</script>''')
    if not accept_value:
        await screen.locator("#quantity").evaluate("el => el.oninput = () => el.value = '1'")
    field = {"selector": "#quantity", "property": "value", "expected": "3", "requirement": "quantity to 3"}
    result = await browser_batch(BrowserBatchParams(actions=[
        {"type": "act", "params": {"action": "type", "selector": "#quantity", "value": "3", "checks": [field]}},
        {"type": "act", "params": {"action": "click", "selector": "#save", "checks": [field,
            {"selector": "#saved", "property": "text", "expected": "Saved 3", "requirement": "save"}]}},
    ]))
    assert result.success is accept_value
    assert await screen.evaluate("window.saves") == int(accept_value)
    assert len(result.metadata["browser_steps"]) == (2 if accept_value else 1)
    assert result.metadata["browser_steps"][0]["success"]
    if not accept_value:
        assert "Not executed" in result.output
        assert "do not repeat completed actions" in result.error


async def test_failed_save_check_stops_batch_without_repeating_save(screen, monkeypatch):
    reg = CapabilityRegistry()
    register_browser_capabilities(reg)
    monkeypatch.setattr("rune.capabilities.registry.get_capability_registry", lambda: reg)
    result = await browser_batch(BrowserBatchParams(actions=[
        {"type": "act", "params": {"action": "click", "selector": "#save", "checks": [
            {"selector": "#count", "property": "text", "expected": "Saved", "requirement": "save"}]}},
        {"type": "act", "params": {"action": "click", "selector": "#save"}},
    ]))
    assert not result.success
    assert await screen.evaluate("window.saves") == 1
    step = result.metadata["browser_steps"][0]
    assert step["success"] and step["metadata"]["action_status"] == "dispatched"
    assert not step["metadata"]["result_checks"]["passed"]


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


@pytest.mark.parametrize("entry_point", ["open", "navigate"])
async def test_entry_points_preserve_drafts_until_an_explicit_reload(screen, entry_point):
    from rune.capabilities.browser.core import (
        BrowserNavigateParams,
        BrowserOpenParams,
        browser_navigate,
        browser_open,
    )

    url = "http://rune.test/draft"
    await screen.route("http://rune.test/**", lambda route: route.fulfill(
        body='<title>Draft</title><label>Name<input id="name"></label>', content_type="text/html"))
    await screen.goto(url)
    await screen.locator("#name").fill("unsaved 사용자 입력")
    await screen.context.add_cookies([{"name": "session", "value": "retained", "url": url}])
    original = current_session().browser
    operation, params = ((browser_open, BrowserOpenParams) if entry_point == "open"
                         else (browser_navigate, BrowserNavigateParams))
    reused = await operation(params(url=url))
    assert reused.success and reused.metadata["skipped_navigation"]
    assert original.is_connected() and current_session().page is screen
    assert await screen.locator("#name").input_value() == "unsaved 사용자 입력"
    assert "Interactive Elements" in reused.output

    refreshed = await operation(params(url=url, reload=True))
    assert refreshed.success and not refreshed.metadata["skipped_navigation"]
    assert await screen.locator("#name").input_value() == ""
    moved = await operation(params(url=url + "?next=1"))
    assert moved.success and screen.url.endswith("?next=1")
    assert original.is_connected() and current_session().page is screen
    assert next(cookie for cookie in await screen.context.cookies() if cookie["name"] == "session")["value"] == "retained"


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
