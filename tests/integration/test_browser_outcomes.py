"""Check completion against page state and counted side effects."""

import pytest

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.intent_engine import resolve_intent_contract
from rune.agent.run_control import RunControl, control_scope
from rune.agent.task_evidence import TaskEvidence
from rune.capabilities.browser.capabilities import (
    BrowserActParams,
    BrowserObserveParams,
    browser_act,
    browser_observe,
)
from rune.capabilities.browser.session import current_session
from tests.integration.test_browser_foundation import reference
from tests.integration.test_browser_foundation import screen as screen
from tests.integration.test_browser_receipts import page_status


def check(selector, expected, requirement, prop="text"):
    return dict(selector=selector, property=prop, expected=expected, requirement=requirement)


def task(goal):
    classification = ClassificationResult("browser", .99, 2)
    return TaskEvidence.for_request(goal, classification), resolve_intent_contract(classification, .99)


async def observe(evidence, checks):
    params = {"checks": checks}
    evidence.prepare("browser_observe", params)
    result = await browser_observe(BrowserObserveParams(**params))
    evidence.observe("browser_observe", result)
    return result


@pytest.mark.parametrize("mode", ["complete", "unbound", "unbound_field", "missing_second", "wrong_value", "not_saved"])
async def test_page_readback_is_separate_from_request_coverage(screen, mode):
    from rune.agent.browser_requirements import request_key
    from rune.agent.run_outcome import run_outcome

    await screen.set_content('''<input id="count" value="3"><button id="save" onclick="
        if(window.mode==='not_saved')return;
        if(window.mode==='wrong_value')document.querySelector('#count').value='9';
        window.records.push(document.querySelector('#count').value);receipt.textContent='Saved'">Save</button>
        <output id="receipt">Saved</output><script>window.records=[]</script>''')
    await screen.evaluate("mode => window.mode=mode", mode)
    last = "save the second form" if mode == "missing_second" else "confirm the save"
    goal = f"Set count to 3, save once and {last}"
    results = ([{"requirement": "save once", "kind": "result"}, {"requirement": last, "kind": "result"}]
               if mode in {"unbound", "missing_second"} else [{"requirement": f"save once and {last}", "kind": "result"}])
    classification = ClassificationResult("browser", .99, 2, browser_requirements={"complete": True, "items": [
        {"requirement": "count to 3", "expected": "3"}, *results,
    ]}, browser_requirements_key=request_key((goal,)))
    evidence = TaskEvidence.for_request(goal, classification)
    await evidence.prepare_requirements()
    ids = [item.id for item in evidence.requirements.items]
    params = {"action": "click", "selector": "#save", "checks": [
        {"selector": "#count", "property": "value", "expected": "3", "requirement_ids": ids[:1]},
        {"selector": "#receipt", "property": "text", "expected": "Saved",
         "requirement_ids": ids[1:2]},
    ]}
    if mode == "unbound_field":
        params["checks"] = params["checks"][1:]
    evidence.prepare_tool("browser_act", params)
    evidence.observe("browser_act", await browser_act(BrowserActParams(**params)))
    acceptance = evidence.acceptance()
    assert (acceptance["readback"]["status"] == "pass") is (mode not in {"wrong_value", "not_saved"})
    assert bool(evidence.blocker(resolve_intent_contract(classification, .99), 0)) is (mode in {"complete", "unbound_field", "wrong_value", "not_saved"})
    assert (acceptance["pageStatus"] == "pass") is (mode == "complete")
    outcome = run_outcome({"reason": "completed" if mode == "complete" else evidence.stop_reason(),
                           "requirement_acceptance": acceptance})
    assert not outcome.success
    assert await screen.evaluate("window.records") == ([] if mode == "not_saved" else ["9" if mode == "wrong_value" else "3"])


@pytest.mark.parametrize("mode", ["both", "missing", "stale", "same_receipt", "rejected"])
async def test_independent_saves_need_distinct_current_receipts(screen, mode):
    from rune.agent.browser_requirements import request_key

    await screen.set_content('''<button id="first" onclick="records.push('first');receiptA.textContent='Saved'">First</button>
        <output id="receiptA">Not saved</output>
        <button id="second" onclick="if(window.reject)return;records.push('second');receiptB.textContent='Saved'">Second</button>
        <output id="receiptB">Saved</output><script>window.records=[]</script>''')
    goal = "Save the first form and save the second form"
    classification = ClassificationResult("browser", .99, 2, browser_requirements={"complete": True, "items": [
        {"requirement": "Save the first form", "kind": "result"},
        {"requirement": "save the second form", "kind": "result"},
    ]}, browser_requirements_key=request_key((goal,)))
    evidence = TaskEvidence.for_request(goal, classification)
    await evidence.prepare_requirements()

    async def act(selector, checks):
        params = {"action": "click", "selector": selector, "checks": checks}
        evidence.prepare_tool("browser_act", params)
        result = await browser_act(BrowserActParams(**params))
        evidence.observe("browser_act", result)

    first = check("#receiptA", "Saved", "Save the first form")
    second = check("#receiptA" if mode == "same_receipt" else "#receiptB", "Saved", "save the second form")
    await act("#first", [first])
    assert page_status(evidence) == "inconclusive"
    if mode in {"both", "rejected"}:
        await screen.evaluate("reject => window.reject=reject", mode == "rejected")
        await act("#second", [second])
    elif mode in {"stale", "same_receipt"}:
        await observe(evidence, [second])
    assert (page_status(evidence) == "pass") is (mode == "both")
    assert await screen.evaluate("window.records") == (["first", "second"] if mode == "both" else ["first"])


async def test_failed_readback_uses_observed_status_ref_without_repeating_save(screen):
    from rune.agent.browser_requirements import request_key
    from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
    from rune.capabilities.browser.capabilities import register_browser_capabilities
    from rune.capabilities.browser.helpers import (
        extract_interactive_elements,
        format_interactive_elements,
    )
    from rune.capabilities.registry import CapabilityRegistry

    goal = "Save once and confirm the save"
    classification = ClassificationResult("browser", .99, 2, browser_requirements={"complete": True, "items": [
        {"requirement": "Save once and confirm the save", "kind": "result"},
    ]}, browser_requirements_key=request_key((goal,)))
    evidence = TaskEvidence.for_request(goal, classification)
    await evidence.prepare_requirements()
    await screen.set_content('''<button id="save" onclick="window.saves++;receipt.textContent='Saved'">Save</button>
        <p id="receipt" role="status">Not saved</p><script>window.saves=0</script>''')
    registry = CapabilityRegistry()
    register_browser_capabilities(registry)

    async def ended(name, result):
        evidence.observe(name, result)

    options = ToolAdapterOptions(prepare_tool=evidence.prepare_tool, on_tool_end=ended)
    batch = _build_typed_tool(cap_def=registry.get("browser_batch"), reg=registry, cache=None, stall=None, opts=options)
    ids = [item.id for item in evidence.requirements.items]
    await batch.function(actions=[{"type": "act", "params": {"action": "click", "selector": "#save",
        "checks": [{"selector": "div[role='status']", "property": "text", "expected": "Saved", "requirement_ids": ids}]}}])
    assert page_status(evidence) == "inconclusive"
    assert evidence.stop_reason() == "verification_inconclusive"
    elements = await extract_interactive_elements(screen)
    receipt = next(item for item in elements if item.role == "status")
    assert "Readback targets" in format_interactive_elements(elements)
    observer = _build_typed_tool(cap_def=registry.get("browser_observe"), reg=registry, cache=None, stall=None, opts=options)
    await observer.function(checks=[{"selector": receipt.ref, "property": "text", "expected": "Saved", "requirement_ids": ids}])
    assert evidence.blocker(resolve_intent_contract(classification, .99), 0)
    assert page_status(evidence) == "pass"
    assert await screen.evaluate("window.saves") == 1


@pytest.mark.parametrize("second_save", ["saved", "rejected", "wrong_value"])
async def test_followup_preserves_fields_and_rechecks_the_same_save(screen, second_save):
    from rune.agent.browser_requirements import request_key

    await screen.set_content('''<input id="city" value="Oslo"><input id="count" value="3">
        <button id="save" onclick="if(window.reject)return;
          if(window.corrupt)document.querySelector('#count').value='9';
          window.records.push({city:city.value,count:document.querySelector('#count').value});
          receipt.textContent='Saved'">Save</button>
        <output id="receipt">Not saved</output><script>window.records=[]</script>''')

    async def start(goal, items):
        classification = ClassificationResult("browser", .99, 2, browser_requirements={"complete": True, "items": items},
                                             browser_requirements_key=request_key((goal,)))
        evidence = TaskEvidence.for_request(goal, classification)
        await evidence.prepare_requirements()
        return evidence, resolve_intent_contract(classification, .99)

    async def call(evidence, action=None, **params):
        name = "browser_act" if action else "browser_observe"
        if action:
            params["action"] = action
        evidence.prepare_tool(name, params)
        result = (await browser_act(BrowserActParams(**params)) if action
                  else await browser_observe(BrowserObserveParams(**params)))
        evidence.observe(name, result)
        return result

    def bound(evidence, indices, selector, expected, prop="text"):
        return dict(selector=selector, property=prop, expected=expected,
                    requirement_ids=[evidence.requirements.items[index].id for index in indices])

    first, contract = await start("Set city Oslo and count 3, save once and confirm the save", [
        {"requirement": "city Oslo", "expected": "Oslo"}, {"requirement": "count 3", "expected": "3"},
        {"requirement": "save once and confirm the save", "kind": "result"},
    ])
    await call(first, "click", selector="#save", checks=[
        bound(first, [0], "#city", "Oslo", "value"), bound(first, [1], "#count", "3", "value"),
        bound(first, [2], "#receipt", "Saved"),
    ])
    assert first.blocker(contract, 0)
    assert page_status(first) == "pass"
    second, contract = await start("Set count 4, keep the city and save again", [
        {"requirement": "count 4", "expected": "4"},
        {"requirement": "keep the city", "expected": "Oslo", "kind": "preserve"},
        {"requirement": "save again", "kind": "result"},
    ])
    await call(second, checks=[bound(second, [1], "#city", "Oslo", "value")])
    await call(second, "type", selector="#count", value="4")
    await screen.evaluate("mode => {window.reject=mode==='rejected';window.corrupt=mode==='wrong_value'}", second_save)
    await call(second, "click", selector="#save", checks=[bound(second, [2], "#receipt", "Saved")])
    result = await call(second, checks=[bound(second, [0], "#count", "4", "value")])
    assert {item["selector"] for item in result.metadata["result_checks"]["checks"]} == {"#count", "#city", "#receipt"}
    assert second.blocker(contract, 0)
    assert (page_status(second) == "pass") is (second_save == "saved")
    records = await screen.evaluate("window.records")
    assert records[0] == {"city": "Oslo", "count": "3"}
    assert len(records) == (1 if second_save == "rejected" else 2)
    if second_save != "rejected":
        assert records[-1] == {"city": "Oslo", "count": "4" if second_save == "saved" else "9"}


@pytest.mark.parametrize("shared_requirement", [False, True])
@pytest.mark.parametrize("adults", ["3", "9"])
async def test_equal_values_do_not_merge_distinct_fields(screen, shared_requirement, adults):
    await screen.set_content('''<input id="adults" value="3"><input id="children" value="3">
        <button id="save" onclick="window.saved=[document.querySelector('#adults').value,document.querySelector('#children').value];statusText.textContent='Saved'">Save</button>
        <output id="statusText">Not saved</output>''')
    evidence, contract = task("Set adults to 3 and children to 3 and save")
    goal = "Set adults to 3 and children to 3 and save"
    await observe(evidence, [check("#adults", "3", goal if shared_requirement else "adults to 3", "value"),
                             check("#children", "3", goal if shared_requirement else "children to 3", "value")])
    await screen.locator("#adults").fill(adults)
    evidence.observe("browser_act", await browser_act(BrowserActParams(action="click", selector="#save")))
    result = await observe(evidence, [check("#statusText", "Saved", "save")])
    assert await screen.evaluate("window.saved") == [adults, "3"]
    assert (evidence.blocker(contract, 0) is None) is (adults == "3")
    assert {c["selector"] for c in result.metadata["result_checks"]["checks"]} == {"#statusText", "#adults", "#children"}


@pytest.mark.parametrize("case", ["correct", "omitted", "same_control", "wrong_expectation"])
async def test_fixed_request_catches_missing_or_misbound_browser_checks(screen, case):
    from rune.agent.browser_requirements import request_key

    goal = "Adults to 3 and Children to 3, then save"
    classification = ClassificationResult("browser", .99, 2, browser_requirements={"complete": True, "items": [
        {"requirement": "Adults to 3", "expected": "3", "kind": "value"},
        {"requirement": "Children to 3", "expected": "3", "kind": "value"},
        {"requirement": "save", "expected": None, "kind": "result"},
    ]}, browser_requirements_key=request_key((goal,)))
    evidence = TaskEvidence.for_request(goal, classification)
    await evidence.prepare_requirements()
    await screen.set_content('''<input id="adults" value="3"><input id="children" value="3">
        <button id="save" onclick="receipt.textContent='Saved'">Save</button><output id="receipt">Not saved</output>''')
    if case != "correct":
        await screen.locator("#children").fill("9")
    checks = [check("#adults", "3", "Adults to 3", "value")]
    if case != "omitted":
        checks.append(check("#adults" if case == "same_control" else "#children",
                            "9" if case == "wrong_expectation" else "3", "Children to 3", "value"))
    await observe(evidence, checks)
    evidence.observe("browser_act", await browser_act(BrowserActParams(action="click", selector="#save")))
    await observe(evidence, [check("#receipt", "Saved", "save")])
    assert evidence.blocker(resolve_intent_contract(classification, .99), 0)
    assert (page_status(evidence) == "pass") is (case == "correct")


@pytest.mark.parametrize("changed_before_save", [False, True])
async def test_confirmation_can_replace_removed_form_without_repeating_save(screen, changed_before_save):
    await screen.set_content('''<input id="quantity" value="3">
        <button id="save" onclick="window.saved=quantity.value;document.body.innerHTML='<output id=receipt>Saved quantity 3</output>'">Save</button>''')
    evidence, contract = task("Set quantity to 3 and save")
    await observe(evidence, [check("#quantity", "3", "quantity to 3", "value")])
    if changed_before_save:
        await screen.locator("#quantity").fill("9")
    evidence.observe("browser_act", await browser_act(BrowserActParams(action="click", selector="#save")))
    result = await observe(evidence, [check("#receipt", "Saved quantity 3", "save")])
    assert await screen.evaluate("window.saved") == ("9" if changed_before_save else "3")
    assert (evidence.blocker(contract, 0) is None) is not changed_before_save
    if not changed_before_save:
        assert result.metadata["result_checks"]["checks"][1]["source"] == "before_action"


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("repeat", [False, True])
async def test_confirmation_mismatch_does_not_repeat_dispatched_input(screen, alias, repeat):
    await screen.set_content('''<button id="save" onclick="window.saves++;statusText.textContent='Saved quantity 3'">Save</button>
        <output id="statusText">Not saved</output><script>window.saves=0</script>''')
    params = dict(action="click", selector="#save", checks=[check("#statusText", "Saved", "save")])
    with control_scope(RunControl("save")):
        first = await browser_act(BrowserActParams(**params))
        if alias:
            params["selector"] = await reference(screen, "Save")
        params.update(repeat=repeat, checks=[check("#statusText", "Saved quantity 3", "save")])
        second = await browser_act(BrowserActParams(**params))
        assert not first.metadata["result_checks"]["passed"]
        assert second.metadata["input_replayed"]
        assert second.metadata["result_checks"]["passed"]
        assert await screen.evaluate("window.saves") == 1
        assert "repeat" not in BrowserActParams.model_json_schema()["properties"]


async def test_existing_heading_does_not_confirm_an_unsaved_edit(screen):
    await screen.set_content('''<h1 id="heading">Booking draft</h1><input id="quantity" value="1">
        <button id="save" onclick="window.saves++">Save</button><script>window.saves=0</script>''')
    evidence, contract = task("Set quantity to 3 and save")
    evidence.observe("browser_act", await browser_act(BrowserActParams(action="type", selector="#quantity", value="3")))
    result = await observe(evidence, [check("#quantity", "3", "quantity to 3", "value"), check("#heading", "Booking draft", "save")])
    assert result.metadata["result_checks"]["passed"]
    assert not result.metadata["result_checks"]["effect_observed"]
    assert evidence.blocker(contract, 0)
    assert await screen.evaluate("window.saves") == 0


async def test_disabled_click_does_not_poison_the_next_turn(screen):
    await screen.locator("#save").evaluate("el => el.disabled=true")
    with control_scope(RunControl("before")):
        first = await browser_act(BrowserActParams(action="click", selector="#save"))
    assert first.metadata["action_status"] == "not_executed"
    assert not current_session().uncertain_action
    assert await screen.evaluate("window.saves") == 0
    await screen.locator("#save").evaluate("el => el.disabled=false")
    with control_scope(RunControl("next")):
        await browser_observe(BrowserObserveParams())
        second = await browser_act(BrowserActParams(action="click", selector="#save"))
    assert second.metadata["action_status"] == "dispatched"
    assert await screen.evaluate("window.saves") == 1


async def test_click_still_waits_for_a_control_that_becomes_enabled(screen):
    await screen.locator("#save").evaluate("el => {el.disabled=true; setTimeout(() => el.disabled=false, 1800)}")
    result = await browser_act(BrowserActParams(action="click", selector="#save"))
    assert result.success
    assert await screen.evaluate("window.saves") == 1


@pytest.mark.parametrize("heading", ["Draft", "Saved"])
async def test_unrelated_readback_cannot_unlock_an_unread_result(screen, heading):
    await screen.set_content(f'''<h1 id="heading">{heading}</h1><button id="save" onclick="window.saves++">Save</button>
        <script>window.saves=0</script>''')
    with control_scope(RunControl("save")):
        await browser_act(BrowserActParams(action="click", selector="#save", checks=[check("#missing", "Saved", "save")]))
        await browser_observe(BrowserObserveParams(checks=[check("#heading", heading, "save")]))
        assert current_session().pending_verification
        second = await browser_act(BrowserActParams(action="click", selector="#save"))
        assert second.metadata["action_status"] == "not_executed"
        assert await screen.evaluate("window.saves") == 1


async def test_result_observation_cannot_be_reused_after_editing_again(screen):
    await screen.set_content('''<input id="quantity" value="3"><button id="save" onclick="receipt.textContent='Saved 3'">Save</button>
        <output id="receipt">Not saved</output>''')
    evidence, contract = task("Set quantity to 3 and save")
    checks = [check("#quantity", "3", "quantity to 3", "value"), check("#receipt", "Saved 3", "save")]
    evidence.observe("browser_act", await browser_act(BrowserActParams(action="click", selector="#save", checks=checks)))
    assert evidence.blocker(contract, 0) is None
    await browser_act(BrowserActParams(action="type", selector="#quantity", value="4"))
    await browser_act(BrowserActParams(action="type", selector="#quantity", value="3"))
    await observe(evidence, checks)
    assert evidence.blocker(contract, 0)


@pytest.mark.parametrize("quantity", ["3", "9"])
async def test_rerendered_field_is_rechecked_by_identity_without_another_write(screen, quantity):
    await screen.set_content('''<label>Quantity<input id="quantity" value="3"></label>
        <button id="save" onclick="window.saves++;receipt.textContent='Saved'">Save</button>
        <output id="receipt">Not saved</output><script>window.saves=0</script>''')
    evidence, contract = task("Set quantity to 3 and save")
    old_ref = await reference(screen, "Quantity")
    await observe(evidence, [check(old_ref, "3", "quantity to 3", "value")])
    await screen.locator("#quantity").evaluate("(el, value) => { const next=el.cloneNode(); next.value=value; el.replaceWith(next); }", quantity)
    evidence.observe("browser_act", await browser_act(BrowserActParams(action="click", selector="#save")))
    new_ref = await reference(screen, "Quantity")
    result = await observe(evidence, [check(new_ref, "3", "quantity to 3", "value"), check("#receipt", "Saved", "save")])
    checked_field = next(item for item in result.metadata["result_checks"]["checks"] if item["selector"] == old_ref)
    assert checked_field["actual"] == quantity
    assert not any(key[2] == old_ref for key in evidence.expected_checks)
    assert (evidence.blocker(contract, 0) is None) is (quantity == "3")
    assert await screen.evaluate("window.saves") == 1


async def test_retry_requires_a_new_user_turn(screen):
    await screen.set_content('''<button id="save" onclick="window.attempts++;receipt.textContent=window.attempts===1?'Nothing saved':'Saved'">Save</button>
        <output id="receipt">Not saved</output><script>window.attempts=0</script>''')
    with control_scope(RunControl("retry")):
        await browser_act(BrowserActParams(action="click", selector="#save"))
        blocked = await browser_act(BrowserActParams(action="click", selector="#save"))
        assert blocked.metadata["replayed"] and "new turn" in blocked.output
        assert await screen.evaluate("window.attempts") == 1
        await browser_act(BrowserActParams(action="click", selector="#save", repeat=True))
        assert await screen.evaluate("window.attempts") == 1
    with control_scope(RunControl("user-confirmed-retry")):
        retried = await browser_act(BrowserActParams(action="click", selector="#save", repeat=True,
                                                     checks=[check("#receipt", "Saved", "save")]))
        assert retried.metadata["result_checks"]["passed"]
        await browser_act(BrowserActParams(action="click", selector="#save"))
        assert await screen.evaluate("window.attempts") == 2


async def test_duplicate_guard_covers_cli_calls_but_not_manual_clicks(screen):
    from rune.capabilities.browser.session import manual_browser_input

    await browser_act(BrowserActParams(action="click", selector="#save"))
    duplicate = await browser_act(BrowserActParams(action="click", selector="#save", repeat=True))
    assert duplicate.metadata["replayed"]
    assert await screen.evaluate("window.saves") == 1
    with manual_browser_input():
        await browser_act(BrowserActParams(action="click", selector="#save"))
        await browser_act(BrowserActParams(action="click", selector="#save"))
    assert await screen.evaluate("window.saves") == 3


async def test_malformed_batch_returns_recovery_shape_without_dispatch(screen):
    from rune.capabilities.browser.capabilities import register_browser_capabilities
    from rune.capabilities.registry import CapabilityRegistry

    registry = CapabilityRegistry()
    register_browser_capabilities(registry)
    result = await registry.execute("browser_batch", {"actions": [{"type": "act", "selector": "#save"}]})
    assert not result.success and result.metadata["action_status"] == "not_executed"
    assert "standalone browser_act" in result.error and '"params"' in result.error
    assert await screen.evaluate("window.saves") == 0


@pytest.mark.parametrize("time_value,count,complete", [("10:00", "6", True), ("10:00:00", "6.0", True),
                                                       ("22:00", "6", False), ("10:00", "7", False)])
async def test_typed_requirements_check_real_controls_in_the_save_call(screen, time_value, count, complete):
    from rune.agent.browser_requirements import request_key
    from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
    from rune.capabilities.registry import CapabilityRegistry

    goal = "오전 10시에 6명으로 변경하고 저장해"
    plan = {"complete": True, "items": [
        {"requirement": "오전 10시에", "expected": "오전 10시", "value_type": "time", "canonical": "10:00"},
        {"requirement": "6명으로", "expected": "6명", "value_type": "number", "canonical": "6"},
        {"requirement": "저장해", "expected": None, "kind": "result"},
    ]}
    classification = ClassificationResult("browser", .99, 2, browser_requirements=plan,
                                         browser_requirements_key=request_key((goal,)))
    evidence = TaskEvidence.for_request(goal, classification)
    await evidence.prepare_requirements()
    await screen.set_content('''<input id="time"><input id="people">
        <button id="save" onclick="window.saves++;window.saved=[time.value,people.value];receipt.textContent='Saved'">Save</button>
        <output id="receipt">Not saved</output><script>window.saves=0</script>''')
    await screen.locator("#time").fill(time_value)
    await screen.locator("#people").fill(count)
    checks = [check("#time", "10:00:00", "오전 10시에", "value"),
              check("#people", "6.0", "6명으로", "value"), check("#receipt", "Saved", "저장해")]
    registry = CapabilityRegistry()
    from rune.capabilities.browser.capabilities import register_browser_capabilities
    register_browser_capabilities(registry)

    async def ended(name, result):
        evidence.observe(name, result)

    tool = _build_typed_tool(cap_def=registry.get("browser_act"), reg=registry, cache=None, stall=None,
                            opts=ToolAdapterOptions(prepare_tool=evidence.prepare_tool, on_tool_end=ended))
    await tool.function(action="click", selector="#save", checks=checks)
    assert await screen.evaluate("window.saved") == [time_value, count]
    assert await screen.evaluate("window.saves") == 1
    assert evidence.blocker(resolve_intent_contract(classification, .99), 0)
    assert (page_status(evidence) == "pass") is complete


async def test_unclear_outcomes_block_batch_before_any_input(screen):
    from rune.agent.browser_requirements import request_key
    from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
    from rune.capabilities.browser.capabilities import register_browser_capabilities
    from rune.capabilities.registry import CapabilityRegistry

    goal = "Save to the right account"
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2,
        browser_requirements={"complete": False, "items": []}, browser_requirements_key=request_key((goal,))))
    await evidence.prepare_requirements()
    registry = CapabilityRegistry()
    register_browser_capabilities(registry)
    options = ToolAdapterOptions(prepare_tool=evidence.prepare_tool)
    observer = _build_typed_tool(cap_def=registry.get("browser_observe"), reg=registry, cache=None, stall=None, opts=options)
    assert "Expense review" in str(await observer.function())
    batch = _build_typed_tool(cap_def=registry.get("browser_batch"), reg=registry, cache=None, stall=None, opts=options)
    result = await batch.function(actions=[{"type": "observe", "params": {}},
                                          {"type": "act", "params": {"action": "click", "selector": "#save"}}])
    assert "requirements_unclear" in str(result)
    assert await screen.evaluate("window.saves") == 0


async def test_retyping_unchanged_fields_does_not_enable_a_second_save(screen):
    await screen.set_content('''<input id="city" value="군산"><input id="people" value="3">
        <button id="save" onclick="window.saves++;receipt.textContent='Saved'">Save</button>
        <output id="receipt">Not saved</output><script>window.saves=0;window.inputs=0;
        document.addEventListener('input',()=>window.inputs++);</script>''')
    with control_scope(RunControl("draft")):
        await browser_act(BrowserActParams(action="click", selector="#save"))
        for selector, value in (("#city", "군산"), ("#people", "3")):
            result = await browser_act(BrowserActParams(action="type", selector=selector, value=value))
            assert result.metadata["unchanged"] and result.metadata["action_status"] == "not_executed"
        result = await browser_act(BrowserActParams(action="click", selector="#save"))
        assert result.metadata["replayed"]
        assert await screen.evaluate("[window.saves,window.inputs]") == [1, 0]
        fresh = await browser_observe(BrowserObserveParams(checks=[check("#receipt", "Saved", "save")]))
        assert fresh.metadata["result_checks"]["effect_observed"]
        await browser_act(BrowserActParams(action="type", selector="#people", value="4"))
        await browser_act(BrowserActParams(action="click", selector="#save"))
        assert await screen.evaluate("[window.saves,window.inputs]") == [2, 1]


async def test_repeated_save_reads_corrected_checks_without_replaying_old_results(screen):
    await screen.set_content('''<button id="save" onclick="window.saves++;receipt.textContent='Saved draft'">Save</button>
        <output id="receipt">Not saved</output><script>window.saves=0</script>''')
    evidence, contract = task("Save the draft")
    with control_scope(RunControl("draft")):
        first = await browser_act(BrowserActParams(action="click", selector="#save",
                                                  checks=[check("#receipt", "Saved", "Save the draft")]))
        evidence.observe("browser_act", first)
        assert not first.metadata["result_checks"]["passed"]
        second = await browser_act(BrowserActParams(action="click", selector="#save",
                                                   checks=[check("#receipt", "Saved draft", "Save the draft")]))
        evidence.observe("browser_act", second)
        assert second.metadata["input_replayed"] and not second.metadata.get("replayed")
        assert second.metadata["result_checks"]["passed"]
        assert evidence.blocker(contract, 0) is None
        assert await screen.evaluate("window.saves") == 1


@pytest.mark.parametrize("change_city,observe_before", [(False, True), (True, True), (False, False)])
async def test_preservation_uses_the_original_observed_control(screen, change_city, observe_before):
    from rune.agent.browser_requirements import request_key

    goal = "Keep the city and change people to 4 and save"
    classification = ClassificationResult("browser", .99, 2, browser_requirements={"complete": True, "items": [
        {"requirement": "Keep the city", "kind": "preserve"},
        {"requirement": "people to 4", "expected": "4"}, {"requirement": "save", "kind": "result"},
    ]}, browser_requirements_key=request_key((goal,)))
    evidence = TaskEvidence.for_request(goal, classification)
    await evidence.prepare_requirements()
    await screen.set_content('''<label>City<input id="city" value="Gunsan"></label>
        <label>People<input id="people" value="3"></label>
        <button id="save" onclick="window.saves++;receipt.textContent='Saved'">Save</button>
        <output id="receipt">Not saved</output><script>window.saves=0</script>''')
    if observe_before:
        evidence.observe("browser_observe", await browser_observe(BrowserObserveParams()))
    evidence.observe("browser_act", await browser_act(BrowserActParams(action="type", selector="#people", value="4")))
    if change_city:
        await screen.locator("#city").fill("Seoul")
    evidence.observe("browser_act", await browser_act(BrowserActParams(action="click", selector="#save")))
    checks = [check("#city", "Seoul" if change_city else "Gunsan", "Keep the city", "value"),
              check("#people", "4", "people to 4", "value"), check("#receipt", "Saved", "save")]
    values = {"checks": checks}
    evidence.prepare_tool("browser_observe", values)
    evidence.observe("browser_observe", await browser_observe(BrowserObserveParams(**values)))
    assert (page_status(evidence) == "pass") is (observe_before and not change_city)
    assert await screen.evaluate("window.saves") == 1


async def test_missing_type_value_cannot_clear_a_control(screen):
    from rune.capabilities.browser.capabilities import register_browser_capabilities
    from rune.capabilities.registry import CapabilityRegistry

    await screen.set_content('<input id="city" value="Gunsan">')
    registry = CapabilityRegistry()
    register_browser_capabilities(registry)
    result = await registry.execute("browser_act", {"action": "type", "selector": "#city"})
    assert not result.success and result.metadata["action_status"] == "not_executed"
    assert await screen.locator("#city").input_value() == "Gunsan"
    cleared = await registry.execute("browser_act", {"action": "type", "selector": "#city", "value": ""})
    assert cleared.success and await screen.locator("#city").input_value() == ""


async def test_observed_control_baselines_exclude_hidden_and_password_fields(screen):
    await screen.set_content('''<label>City<input value="Gunsan"></label>
        <label>Password<input type="password" value="secret"></label><input type="hidden" value="private">
        <label style="display:none">Hidden<input value="invisible"></label>
        <label>Enabled<input type="checkbox" checked></label>''')
    result = await browser_observe(BrowserObserveParams())
    assert {item["actual"] for item in result.metadata["observed_controls"]} == {"Gunsan", "true"}


@pytest.mark.parametrize("shape", ["action", "type", "act"])
async def test_flat_batch_checks_bind_before_requirement_validation(screen, shape):
    from rune.agent.browser_requirements import request_key
    from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
    from rune.capabilities.browser.capabilities import register_browser_capabilities
    from rune.capabilities.registry import CapabilityRegistry

    goal = "Set quantity to 4"
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2, browser_effect="edit",
        browser_requirements={"complete": True, "items": [{"requirement": goal, "expected": "4"}]},
        browser_requirements_key=request_key((goal,))))
    await evidence.prepare_requirements()
    await screen.set_content('<input id="quantity" value="3">')
    registry = CapabilityRegistry()
    register_browser_capabilities(registry)

    async def ended(name, result):
        evidence.observe(name, result)

    batch = _build_typed_tool(cap_def=registry.get("browser_batch"), reg=registry, cache=None, stall=None,
                            opts=ToolAdapterOptions(prepare_tool=evidence.prepare_tool, on_tool_end=ended))
    step = {"action": "type", "selector": "#quantity", "value": "4",
            "checks": [{"property": "value", "expected": "4", "requirement_ids": [evidence.requirements.items[0].id]}]}
    if shape == "type":
        step["type"] = step.pop("action")
    elif shape == "act":
        step["type"] = "act"
    await batch.function(actions=[step])
    assert await screen.locator("#quantity").input_value() == "4"
    assert page_status(evidence) == "pass"
    assert next(iter(evidence.check_specs.values()))["selector"] == "#quantity"


@pytest.mark.parametrize("checks", [
    [{"property": "value", "expected": "4"}, {"property": "value", "expected": "4"}],
    [{"property": "text", "expected": "Saved"}], [{"property": "value", "expected": "9"}],
])
async def test_ambiguous_inline_checks_stop_the_whole_batch(screen, checks):
    from rune.capabilities.browser.capabilities import register_browser_capabilities
    from rune.capabilities.registry import CapabilityRegistry

    await screen.set_content('<input id="quantity" value="3">')
    registry = CapabilityRegistry()
    register_browser_capabilities(registry)
    result = await registry.execute("browser_batch", {"actions": [
        {"action": "type", "selector": "#quantity", "value": "4"},
        {"action": "type", "selector": "#quantity", "value": "4", "checks": checks},
    ]})
    assert not result.success and await screen.locator("#quantity").input_value() == "3"


@pytest.mark.parametrize("receipt", ["Saved draft", "Still guessed"])
async def test_batch_observes_unspecified_receipt_without_repeating_input(screen, receipt):
    from rune.agent.browser_requirements import request_key
    from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
    from rune.capabilities.browser.capabilities import register_browser_capabilities
    from rune.capabilities.registry import CapabilityRegistry

    goal = "Set quantity to 4 and save"
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2,
        browser_requirements={"complete": True, "items": [
            {"requirement": "quantity to 4", "expected": "4"}, {"requirement": "save", "kind": "result"}]},
        browser_requirements_key=request_key((goal,))))
    await evidence.prepare_requirements()
    await screen.set_content('''<input id="quantity" value="3">
        <button id="save" onclick="window.saves++;statusText.textContent='Saved draft'">Save</button>
        <output id="statusText">Not saved</output><script>window.saves=0;window.inputs=0;
        document.addEventListener('input',()=>window.inputs++);</script>''')
    registry = CapabilityRegistry()
    register_browser_capabilities(registry)

    async def ended(name, result):
        evidence.observe(name, result)

    batch = _build_typed_tool(cap_def=registry.get("browser_batch"), reg=registry, cache=None, stall=None,
                            opts=ToolAdapterOptions(prepare_tool=evidence.prepare_tool, on_tool_end=ended))
    with control_scope(RunControl("draft")):
        for expected in ("Saved", receipt):
            await batch.function(actions=[
                {"action": "type", "selector": "#quantity", "value": "4", "checks": [check("#quantity", "4", "quantity to 4", "value")]},
                {"action": "click", "selector": "#save", "checks": [check("#statusText", expected, "save")]},
            ])
    assert page_status(evidence) == "pass"
    assert evidence.acceptance()["status"] == "inconclusive"
    assert await screen.evaluate("[window.saves,window.inputs]") == [1, 1]


@pytest.mark.parametrize('changed_city', [False, True])
async def test_corrected_control_property_replaces_only_the_invalid_check(screen, changed_city):
    from rune.agent.browser_requirements import request_key

    goal = 'Set people to 4, preserve city, and save once'
    classification = ClassificationResult('browser', .99, 2, browser_requirements={'complete': True, 'items': [
        {'requirement': 'people to 4', 'expected': '4', 'value_type': 'number'},
        {'requirement': 'preserve city', 'kind': 'preserve'}, {'requirement': 'save once', 'kind': 'result'},
    ]}, browser_requirements_key=request_key((goal,)))
    evidence = TaskEvidence.for_request(goal, classification)
    await evidence.prepare_requirements()
    await screen.set_content('''<label>People<input id="people" value="3"></label><label>City<input id="city" value="Gunsan"></label>
        <button id="save" onclick="window.saves++;receipt.textContent='Saved'">Save</button>
        <output id="receipt">Not saved</output><script>window.saves=0</script>''')
    with control_scope(RunControl('correction')):
        evidence.observe('browser_observe', await browser_observe(BrowserObserveParams()))
        assert len(evidence.observed_controls) == 2, evidence.observed_controls
        evidence.observe('browser_act', await browser_act(BrowserActParams(action='type', selector='#people', value='4')))
        checks = [check('#people', '4', 'people to 4', 'value'), check('#city', 'Gunsan', 'preserve city'),
                  check('#receipt', 'Saved', 'save once')]
        params = {'action': 'click', 'selector': '#save', 'checks': checks}
        evidence.prepare_tool('browser_act', params)
        saved = await browser_act(BrowserActParams(**params))
        evidence.observe('browser_act', saved)
        assert not saved.metadata['result_checks']['passed']
        assert saved.metadata['result_checks']['checks'][1]['property_hint'] == 'value'
        if changed_city:
            await screen.locator('#city').fill('Busan')
        checks[1] = check('#city', 'Gunsan', 'preserve city', 'value')
        readback = {'checks': checks}
        evidence.prepare_tool('browser_observe', readback)
        verified = await browser_observe(BrowserObserveParams(**readback))
        evidence.observe('browser_observe', verified)
        assert len(verified.metadata['result_checks']['checks']) == 3
        assert (page_status(evidence) == 'pass') is not changed_city, evidence.acceptance()
        assert await screen.evaluate('window.saves') == 1


@pytest.mark.parametrize('markup,expected,prop', [
    ('<textarea id="field">memo</textarea>', 'memo', 'value'),
    ('<input id="field" type="checkbox" checked>', 'true', 'checked'),
    ('<input id="field" type="radio" checked>', 'true', 'checked'),
])
async def test_control_readback_reports_the_correct_property(screen, markup, expected, prop):
    await screen.set_content(markup)
    evidence, _ = task(f'Set the field to {expected}')
    wrong = await observe(evidence, [check('#field', expected, 'field')])
    item = wrong.metadata['result_checks']['checks'][0]
    assert not item['passed'] and item['property_hint'] == prop
    corrected = await observe(evidence, [check('#field', expected, 'field', prop)])
    checks = corrected.metadata['result_checks']['checks']
    assert len(checks) == 1 and checks[0]['passed'] and checks[0]['property'] == prop
