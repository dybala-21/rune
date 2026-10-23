"""Keep save receipts tied to their outcome and submitted values."""

import pytest

from rune.agent.browser_requirements import request_key
from rune.agent.goal_classifier import ClassificationResult
from rune.agent.task_evidence import TaskEvidence
from rune.agent.tool_errors import ToolPreparationError
from rune.capabilities.browser.capabilities import (
    BrowserActParams,
    BrowserObserveParams,
    browser_act,
    browser_observe,
)
from rune.capabilities.browser.checks import format_checks
from tests.integration.test_browser_foundation import screen as screen


def page_status(evidence):
    acceptance = evidence.acceptance()
    if any(item.kind == 'result' for item in evidence.requirements.outcomes):
        assert acceptance['status'] != 'pass', 'A page receipt cannot prove persistence'
    return acceptance['pageStatus']


async def start(goal, items):
    classification = ClassificationResult("browser", .99, 2,
        browser_requirements={"complete": True, "items": items}, browser_requirements_key=request_key((goal,)))
    evidence = TaskEvidence.for_request(goal, classification)
    await evidence.prepare_requirements()
    return evidence


async def act(evidence, **params):
    evidence.prepare_tool("browser_act", params)
    result = await browser_act(BrowserActParams(**params))
    assert result.success, result.error
    evidence.observe("browser_act", result)
    return result


def check(selector, requirement, expected="Saved", prop="text"):
    return dict(selector=selector, requirement=requirement, expected=expected, property=prop)


async def test_one_receipt_cannot_complete_two_independent_saves(screen):
    await screen.set_content('''<button id="saveA" onclick="saved.push('A');receiptA.textContent='Saved'">Save A</button>
        <output id="receiptA">Not saved</output>
        <button id="saveB" onclick="saved.push('B');receiptB.textContent='Saved'">Save B</button>
        <output id="receiptB">Not saved</output><script>window.saved=[]</script>''')
    evidence = await start("Save A and save B", [
        {"requirement": "Save A", "kind": "result"}, {"requirement": "save B", "kind": "result"}])
    ids = [item.id for item in evidence.requirements.items]
    params = dict(action="click", selector="#saveA", checks=[
        dict(selector="#receiptA", property="text", expected="Saved", requirement_ids=ids)])
    with pytest.raises(ToolPreparationError, match="one requirement"):
        evidence.prepare_tool("browser_act", params)
    assert await screen.evaluate("window.saved") == []
    # Old callers may still return a result with several requirement IDs.
    params["checks"][0]["requirement_ids"] = ids[:1]
    result = await browser_act(BrowserActParams(**params))
    result.metadata["result_checks"]["checks"][0]["requirement_ids"] = ids
    evidence.observe("browser_act", result)
    assert await screen.evaluate("window.saved") == ["A"]
    assert page_status(evidence) == "inconclusive"


@pytest.mark.parametrize("mode", ["before_save", "after_save", "script_edit", "resaved", "other_form"])
@pytest.mark.parametrize("external", [False, True], ids=["nested", "associated"])
async def test_receipt_cannot_validate_later_unsaved_values(screen, mode, external):
    html = '''<form id="formA"><input id="quantityA" value="1">
        <button type="button" id="saveA" onclick="saved.A=quantityA.value;receiptA.textContent='Saved'">Save A</button>
        <output id="receiptA">Not saved</output></form>
        <form id="formB"><input id="quantityB" value="1">
        <button type="button" id="saveB" onclick="saved.B=quantityB.value;receiptB.textContent='Saved'">Save B</button>
        <output id="receiptB">Not saved</output></form><script>window.saved={}</script>'''
    if external:
        html = html.replace('<input id="quantityA" value="1">', '') + '<input id="quantityA" form="formA" value="1">'
    await screen.set_content(html)
    evidence = await start("Set quantity A to 2 and save A, and save B", [
        {"requirement": "quantity A to 2", "expected": "2"},
        {"requirement": "save A", "kind": "result"}, {"requirement": "save B", "kind": "result"}])
    if mode in {"before_save", "other_form"}:
        await act(evidence, action="type", selector="#quantityA", value="2")
    await act(evidence, action="click", selector="#saveA", checks=[check("#receiptA", "save A")])
    if mode in {"after_save", "resaved"}:
        await act(evidence, action="type", selector="#quantityA", value="2")
    elif mode == "script_edit":
        await screen.evaluate("quantityA.value='2'")
    elif mode == "other_form":
        await act(evidence, action="type", selector="#quantityB", value="3")
    if mode == "resaved":
        await act(evidence, action="click", selector="#saveA", repeat=True, checks=[check("#receiptA", "save A")])
    result = await act(evidence, action="click", selector="#saveB", checks=[check("#receiptB", "save B"),
        check("#quantityA", "quantity A to 2", "2", "value")])
    complete = mode in {"before_save", "resaved", "other_form"}
    assert await screen.evaluate("window.saved") == {"A": "2" if complete else "1", "B": "3" if mode == "other_form" else "1"}
    assert (page_status(evidence) == "pass") is complete
    payload = format_checks(result.metadata["result_checks"])
    assert '"controls"' not in payload and '"effect"' not in payload


async def test_same_effect_does_not_keep_a_receipt_fresh_after_script_changes(screen):
    await screen.set_content('''<input id="quantity" value="1">
        <button id="save" onclick="saved=quantity.value;receipt.textContent='Saved'">Save</button>
        <output id="receipt">Not saved</output>''')
    evidence = await start("Set quantity to 2 and save", [
        {"requirement": "quantity to 2", "expected": "2"}, {"requirement": "save", "kind": "result"}])
    await act(evidence, action="click", selector="#save", checks=[check("#receipt", "save")])
    await screen.evaluate("quantity.value='2'")
    params = {"checks": [check("#quantity", "quantity to 2", "2", "value")]}
    evidence.prepare_tool("browser_observe", params)
    evidence.observe("browser_observe", await browser_observe(BrowserObserveParams(**params)))
    assert await screen.evaluate("window.saved") == "1"
    assert page_status(evidence) == "inconclusive"


async def test_auto_saved_checkbox_keeps_the_value_from_the_click(screen):
    await screen.set_content('''<form><input id="enabled" type="checkbox"
        onclick="saved=this.checked;receipt.textContent='Saved'"><output id="receipt">Pending</output></form>''')
    evidence = await start("Enable alerts and confirm they were saved", [
        {"requirement": "Enable alerts", "kind": "value"},
        {"requirement": "confirm they were saved", "kind": "result"}])
    await act(evidence, action="click", selector="#enabled", checks=[
        check("#enabled", "Enable alerts", "true", "checked"), check("#receipt", "confirm they were saved")])
    assert await screen.evaluate("window.saved") is True
    assert page_status(evidence) == "pass"
    await screen.evaluate("enabled.checked=false")
    params = {"checks": []}
    evidence.prepare_tool("browser_observe", params)
    evidence.observe("browser_observe", await browser_observe(BrowserObserveParams(**params)))
    assert page_status(evidence) == "inconclusive"


async def test_control_snapshots_exclude_secrets_and_fail_closed_when_truncated(screen):
    from rune.capabilities.browser.effects import control_snapshots, node_key

    await screen.set_content('''<input type="password" value="secret"><input type="hidden" value="hidden">
        <input type="file"><input id="visible" value="public"><button id="save">Save</button>''')
    await node_key(screen.locator("#save"))
    snapshot, submitted = await control_snapshots(screen)
    assert snapshot["complete"] and len(snapshot["values"]) == 1
    assert submitted is None
    assert all(len(value) == 64 for value in snapshot["values"].values())
    await screen.evaluate("document.querySelector('#visible').value='x'.repeat(1001)")
    assert not (await control_snapshots(screen))[0]["complete"]
