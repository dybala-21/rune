import json
from unittest.mock import AsyncMock

import pytest

from rune.agent.browser_requirements import BrowserRequirements, request_key, validated
from rune.agent.goal_classifier import ClassificationResult
from rune.agent.intent_engine import resolve_intent_contract
from rune.agent.task_evidence import TaskEvidence
from rune.agent.tool_errors import ToolPreparationError
from rune.types import CapabilityResult

GOAL = "Set Adults to 3 and Children to 3, preserve Reference, and save."
PLAN = {"complete": True, "items": [
    {"requirement": "Adults to 3", "expected": "3", "kind": "value"},
    {"requirement": "Children to 3", "expected": "3", "kind": "value"},
    {"requirement": "preserve Reference", "expected": None, "kind": "preserve"},
    {"requirement": "save", "expected": None, "kind": "result"},
]}


def observation(*, children=True, alias=False, reference="original", result=True):
    checks = [dict(selector="#adults", node_id="adults", property="value", expected="3", actual="3",
                   kind="control", passed=True, requirement="Adults to 3")]
    if children:
        checks.append(dict(checks[0], selector="#adults" if alias else "#children",
                           node_id="adults" if alias else "children", requirement="Children to 3"))
    checks.append(dict(selector="#ref", node_id="ref", property="value", expected=reference, actual=reference,
                       kind="control", passed=True, requirement="preserve Reference"))
    if result:
        checks.append(dict(selector="#receipt", node_id="receipt", property="text", expected="Saved", actual="Saved",
                           kind="result", passed=True, requirement="save"))
    return CapabilityResult(success=True, metadata={"result_checks": {
        "passed": True, "effect_observed": result, "checks": checks}})


@pytest.mark.parametrize("case,complete", [("correct", True), ("missing", False), ("alias", False),
                                         ("changed_reference", False), ("no_baseline", False)])
async def test_checks_cover_fixed_requirements_without_another_model_call(monkeypatch, case, complete):
    extraction = AsyncMock(side_effect=AssertionError("Routing already supplied requirements"))
    monkeypatch.setattr("rune.llm.client.get_llm_client", extraction)
    classification = ClassificationResult("browser", .99, 2, browser_requirements=PLAN,
                                         browser_requirements_key=request_key((GOAL,)))
    evidence = TaskEvidence.for_request(GOAL, classification)
    await evidence.prepare_requirements()
    if case != "no_baseline":
        evidence.observe("browser_observe", observation(result=False))
    evidence.observe("browser_act", CapabilityResult(success=True, metadata={"action_status": "dispatched"}))
    evidence.observe("browser_observe", observation(children=case != "missing", alias=case == "alias",
                                                     reference="changed" if case == "changed_reference" else "original"))
    assert (evidence.acceptance()["pageStatus"] == "pass") is complete
    assert evidence.acceptance()["status"] == "inconclusive"
    extraction.assert_not_called()


@pytest.mark.parametrize("plan", [None, PLAN])
async def test_failed_action_stays_failed_when_coverage_is_unavailable_or_readback_passes(monkeypatch, plan):
    monkeypatch.setattr("rune.llm.client.get_llm_client", AsyncMock(return_value=None))
    classification = ClassificationResult("browser", .99, 2, browser_requirements=plan,
                                         browser_requirements_key=request_key((GOAL,)))
    evidence = TaskEvidence.for_request(GOAL, classification)
    await evidence.prepare_requirements()
    evidence.observe("browser_act", CapabilityResult(success=False, error="Submission rejected"))
    evidence.observe("browser_observe", observation())
    assert evidence.browser_confirmed
    assert evidence.acceptance()["status"] == "fail"
    assert evidence.stop_reason() == "task_incomplete"
    assert "Submission rejected" in evidence.blocker(resolve_intent_contract(classification, .99), 0)


@pytest.mark.parametrize("spec", [None, {"complete": False, "items": []}, {"complete": True, "items": []}])
async def test_unavailable_requirements_cannot_pass(monkeypatch, spec):
    extraction = AsyncMock(return_value=BrowserRequirements.model_validate(spec) if spec else None)
    monkeypatch.setattr("rune.llm.client.get_llm_client", extraction)
    classification = ClassificationResult("browser", .99, 2)
    evidence = TaskEvidence.for_request(GOAL, classification)
    await evidence.prepare_requirements()
    await evidence.prepare_requirements()
    evidence.observe("browser_observe", observation())
    for _ in range(5):
        assert evidence.blocker(resolve_intent_contract(classification, .99), 0)
        assert evidence.acceptance()["status"] == "unavailable"
    extraction.assert_not_called()


def test_expected_value_must_belong_to_its_requirement():
    with pytest.raises(ValueError):
        validated({"complete": True, "items": [{"requirement": "Children to 4", "expected": "3"}]},
                  ("Adults to 3 and Children to 4",))


def test_requirement_feedback_reports_paraphrases_and_invented_values_together():
    with pytest.raises(ValueError) as error:
        validated({"complete": True, "items": [
            {"requirement": "Change city to Oslo", "expected": "Oslo"},
            {"requirement": "save", "kind": "result", "expected": "Successfully saved"},
        ]}, ("Set city to Oslo and save",))
    assert "not an exact user quote" in str(error.value)
    assert "expected=null" in str(error.value)


@pytest.mark.parametrize('confirmation', [
    {'requirement': 'confirm it', 'kind': 'result', 'confirms': 'missing'},
    {'requirement': 'confirm it', 'kind': 'result', 'confirms': 'confirm it'},
    {'requirement': 'confirm it', 'kind': 'preserve', 'confirms': 'save'},
    {'requirement': 'confirm it', 'kind': 'result', 'expected': 'it', 'confirms': 'save'},
])
def test_invalid_confirmation_links_cannot_remove_a_requirement(confirmation):
    with pytest.raises(ValueError):
        validated({'complete': True, 'items': [{'requirement': 'save', 'kind': 'result'}, confirmation]},
                  ('save and confirm it',))


def test_confirmation_links_cannot_form_a_chain():
    with pytest.raises(ValueError):
        BrowserRequirements.model_validate({'complete': True, 'items': [
            {'requirement': 'save', 'kind': 'result'},
            {'requirement': 'confirm', 'kind': 'result', 'confirms': 'save'},
            {'requirement': 'check', 'kind': 'result', 'confirms': 'confirm'},
        ]})

def test_confirmation_belongs_to_one_result_without_copying_its_reference():
    plan = validated({"complete": True, "items": [
        {"requirement": "save", "kind": "result"},
        {"requirement": "confirm it", "kind": "confirmation"},
    ]}, ("save and confirm it",))
    assert len(plan.outcomes) == 1
    assert plan.outcome(plan.items[1]) is plan.items[0]


def test_confirmation_cannot_merge_independent_saved_objects():
    with pytest.raises(ValueError, match="multiple objects"):
        validated({"complete": True, "items": [
            {"requirement": "save A", "kind": "result"},
            {"requirement": "save B", "kind": "result"},
            {"requirement": "confirm", "kind": "confirmation"},
        ]}, ("save A and save B then confirm",))


async def test_confirmation_quote_and_id_bind_to_the_original_result_without_a_model_call(monkeypatch):
    extraction = AsyncMock(side_effect=AssertionError('No additional model classification'))
    monkeypatch.setattr('rune.llm.client.get_llm_client', extraction)
    goal = 'save and confirm it'
    evidence = TaskEvidence.for_request(goal, ClassificationResult('browser', .99, 2,
        browser_requirements={'complete': True, 'items': [
            {'requirement': 'save', 'kind': 'result'},
            {'requirement': 'confirm it', 'kind': 'result', 'confirms': 'save'},
        ]}, browser_requirements_key=request_key((goal,))))
    await evidence.prepare_requirements()
    parent, confirmation = evidence.requirements.items
    params = {'checks': [
        {'selector': '#receipt', 'property': 'text', 'expected': 'Saved', 'requirement': 'save'},
        {'selector': '#receipt', 'property': 'text', 'expected': 'Saved', 'requirement': 'confirm it'},
        {'selector': '#receipt', 'property': 'text', 'expected': 'Saved', 'requirement_ids': [confirmation.id]},
    ]}
    evidence.prepare_tool('browser_observe', params)
    assert len(params['checks']) == 1
    assert params['checks'][0]['requirement_ids'] == [parent.id]
    assert params['checks'][0]['requirement'] == 'save'
    assert evidence.requirements.outcomes == [parent]
    extraction.assert_not_called()


@pytest.mark.parametrize('state', ['pass', 'fail', 'unavailable'])
def test_explicit_readback_is_recorded_even_outside_browser_write_tasks(state):
    evidence = TaskEvidence()
    evidence.observe('browser_observe', CapabilityResult(success=True, metadata={
        'independent_readback': {'status': state, 'scope': 'http_document'},
    }))
    assert evidence.acceptance()['status'] == state
    assert evidence.acceptance()['scope'] == 'browser_result_readback'


@pytest.mark.parametrize('baseline,expected', [('Gunsan', 'pass'), ('Oslo', 'unavailable'), (None, 'unavailable')])
async def test_batched_readback_validates_preservation_after_the_preceding_steps(baseline, expected):
    goal = 'preserve city'
    evidence = TaskEvidence.for_request(goal, ClassificationResult('browser', .99, 2,
        browser_requirements={'complete': True, 'items': [{'requirement': goal, 'kind': 'preserve'}]},
        browser_requirements_key=request_key((goal,))))
    await evidence.prepare_requirements()
    params = {'actions': [{'type': 'observe', 'params': {'readbackUrl': 'https://example.test/record', 'checks': [
        {'selector': '#city', 'property': 'value', 'expected': 'Gunsan', 'requirement': goal},
    ]}}]}
    evidence.prepare_tool('browser_batch', params)
    item = evidence.requirements.items[0]
    if baseline is not None:
        evidence.preserved_values[item.id] = ('value', baseline)
    checks = [{**params['actions'][0]['params']['checks'][0], 'actual': 'Gunsan', 'passed': True, 'error': ''}]
    evidence.observe('browser_observe', CapabilityResult(success=True, metadata={
        'independent_readback': {'status': 'pass', 'scope': 'http_document', 'checks': checks},
    }))
    assert evidence.independent_readback['status'] == expected


def test_preserved_value_is_read_from_the_page_not_inferred_from_history():
    plan = validated({"complete": True, "items": [
        {"requirement": "keep the city", "expected": "Oslo", "kind": "preserve"},
    ]}, ("Set the city to Oslo", "Change count to 4 and keep the city"))
    assert plan.items[0].expected is None
    assert plan.items[0].canonical is None


async def test_unbound_result_keeps_scope_unknown_without_repeating_readback():
    goal = "Save once and confirm the save"
    plan = {"complete": True, "items": [
        {"requirement": "Save once", "kind": "result"},
        {"requirement": "confirm the save", "kind": "result"},
    ]}
    classification = ClassificationResult("browser", .99, 2, browser_requirements=plan,
                                         browser_requirements_key=request_key((goal,)))
    evidence = TaskEvidence.for_request(goal, classification)
    await evidence.prepare_requirements()
    ids = [item.id for item in evidence.requirements.items]
    params = {"checks": [{"selector": "#receipt", "property": "text", "expected": "Saved",
                           "requirement_ids": ids[:1]}]}
    evidence.prepare_tool("browser_observe", params)
    checks = [dict(item, actual="Saved", passed=True, kind="result") for item in params["checks"]]
    evidence.observe("browser_act", CapabilityResult(success=True, metadata={"action_status": "dispatched"}))
    evidence.observe("browser_observe", CapabilityResult(success=True, metadata={"result_checks": {
        "passed": True, "effect_observed": True, "checks": checks}}))
    assert evidence.blocker(resolve_intent_contract(classification, .99), 0) is None
    assert evidence.acceptance()["status"] == "inconclusive"
    assert evidence.acceptance()["readback"] == {"status": "pass", "scope": "observed_page_state"}
    assert evidence.acceptance()["missing"] == ids
    assert evidence.acceptance()["pendingResults"] == ids


async def test_requirement_ids_do_not_allow_equal_values_to_merge_fields():
    evidence = TaskEvidence.for_request(GOAL, ClassificationResult("browser", .99, 2,
        browser_requirements=PLAN, browser_requirements_key=request_key((GOAL,))))
    await evidence.prepare_requirements()
    ids = [item.id for item in evidence.requirements.items[:2]]
    with pytest.raises(ValueError, match="separate"):
        evidence.prepare_tool("browser_observe", {"checks": [{"selector": "#adults", "property": "value",
            "expected": "3", "requirement_ids": ids}]})


def test_value_readback_keeps_the_save_check_for_a_fresh_read():
    evidence = TaskEvidence(request_texts=("Set count to 4 and save",))
    save = {"selector": "#receipt", "property": "text", "requirement": "save", "expected": "Saved"}
    evidence.prepare_tool("browser_act", {"action": "click", "checks": [save]})
    params = {"checks": [{"selector": "#count", "property": "value", "requirement": "count to 4", "expected": "4"}]}
    evidence.prepare_tool("browser_observe", params)
    assert save in params["checks"]
    plain_observation = {}
    evidence.prepare_tool("browser_observe", plain_observation)
    assert len(plain_observation["checks"]) == 2
    assert all(check in plain_observation["checks"] for check in params["checks"])


@pytest.mark.parametrize("expected,accepted", [("original", False), ("3", False), ("Oslo", True)])
async def test_unknown_id_recovers_only_from_an_unambiguous_fixed_value(expected, accepted):
    goal = GOAL + " Set City to Oslo."
    plan = {**PLAN, "items": [*PLAN["items"], {"requirement": "City to Oslo", "expected": "Oslo"}]}
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2,
        browser_requirements=plan, browser_requirements_key=request_key((goal,))))
    await evidence.prepare_requirements()
    params = {"checks": [{"selector": "#field", "property": "value", "expected": expected,
                           "requirement_ids": ["mistyped"]}]}
    if accepted:
        evidence.prepare_tool("browser_observe", params)
        assert params["checks"][0]["requirement_ids"] == [evidence.requirements.items[-1].id]
    else:
        with pytest.raises(ValueError, match="unknown_requirement"):
            evidence.prepare_tool("browser_observe", params)


async def test_bad_expected_value_cannot_poison_a_later_valid_check():
    evidence = TaskEvidence.for_request(GOAL, ClassificationResult("browser", .99, 2,
        browser_requirements=PLAN, browser_requirements_key=request_key((GOAL,))))
    await evidence.prepare_requirements()
    with pytest.raises(ValueError):
        evidence.prepare_tool("browser_observe", {"checks": [{"requirement": "Children to 3", "expected": "9",
                                                             "property": "value", "selector": "#children"}]})
    assert not evidence.expected_checks
    evidence.prepare_tool("browser_observe", {"checks": [{"requirement": "Children to 3", "expected": "3",
                                                         "property": "value", "selector": "#children"}]})
    assert len(evidence.expected_checks) == 1


async def test_missing_plan_is_bound_by_the_executor_without_a_classifier_call(monkeypatch):
    client = AsyncMock(side_effect=AssertionError("No auxiliary model call"))
    monkeypatch.setattr("rune.llm.client.get_llm_client", client)
    evidence = TaskEvidence.for_request(GOAL, ClassificationResult("browser", .99, 2))
    guidance = await evidence.prepare_requirements()
    assert "taskRequirements" in guidance
    params = {"url": "https://example.com", "taskRequirements": PLAN}
    evidence.prepare_tool("browser_navigate", params)
    assert "taskRequirements" not in params
    assert not evidence.requirements_unavailable
    client.assert_not_called()


async def test_executor_contract_accepts_literal_values_without_classifier_fields():
    from rune.agent.browser_plan import executor_schema

    goal = "도시를 군산, 인원을 3으로 바꾸고 임시 저장을 한 번 눌러줘. 저장됐는지도 확인해."
    plan = {"items": [
        {"requirement": "도시를 군산", "expected": "군산", "kind": "value"},
        {"requirement": "인원을 3으로 바꾸고", "expected": "3", "kind": "value", "value_type": "number"},
        {"requirement": "임시 저장을 한 번 눌러줘. 저장됐는지도 확인해.", "expected": None, "kind": "result"},
    ]}
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2))
    await evidence.prepare_requirements()
    evidence.prepare_tool("browser_navigate", {"taskRequirements": plan})
    assert evidence.requirements.complete
    assert len(evidence.requirements.outcomes) == 3
    assert evidence.requirements.items[1].canonical == "3"
    schema = executor_schema()
    assert schema['required'] == ['items']
    assert schema['properties']['items']['items']['required'] == ['requirement', 'kind', 'expected']
    fields = schema['properties']['items']['items']['properties']
    assert fields['kind']['enum'] == ['value', 'preserve', 'result', 'criterion']
    assert 'confirms' not in fields


@pytest.mark.parametrize('change', ['paraphrase', 'canonical', 'expected', 'ambiguous'])
async def test_simpler_contract_keeps_source_and_ambiguity_checks(change):
    goal = 'Set people to 3 and save'
    item = {'requirement': 'people to 3', 'expected': '3', 'kind': 'value'}
    plan = {'items': [item, {'requirement': 'save', 'expected': None, 'kind': 'result'}]}
    evidence = TaskEvidence.for_request(goal, ClassificationResult('browser', .99, 2))
    await evidence.prepare_requirements()
    if change == 'paraphrase':
        item['requirement'] = 'people should be 3'
    elif change == 'canonical':
        item['canonical'] = 'people'
    elif change == 'expected':
        item['expected'] = '4'
    else:
        plan['complete'] = False
        evidence.prepare_tool('browser_observe', {'taskRequirements': plan})
        with pytest.raises(ToolPreparationError, match='requirements_unclear'):
            evidence.prepare_tool('browser_act', {'action': 'type', 'selector': '#people', 'value': '3'})
        return
    with pytest.raises(ValueError):
        evidence.prepare_tool('browser_observe', {'taskRequirements': plan})


async def test_executor_plan_reaches_the_model_and_stays_fixed():
    from rune.agent.browser_plan import configure_tools
    from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
    from rune.agent.tool_errors import ToolPreparationError
    from rune.capabilities.browser.capabilities import BrowserObserveParams
    from rune.capabilities.registry import CapabilityRegistry
    from rune.capabilities.types import CapabilityDefinition

    evidence = TaskEvidence.for_request(GOAL, ClassificationResult('browser', .99, 2))
    await evidence.prepare_requirements()
    execute = AsyncMock(return_value=CapabilityResult(success=True, output='Current page'))
    async def ended(name, result):
        evidence.observe(name, result)
    cap = CapabilityDefinition(name='browser_observe', description='Observe', execute=execute,
                               parameters_model=BrowserObserveParams)
    registry = CapabilityRegistry()
    registry.register(cap)
    tool = _build_typed_tool(cap_def=cap, reg=registry, cache=None, stall=None,
                            opts=ToolAdapterOptions(prepare_tool=evidence.prepare_tool, on_tool_end=ended))
    configure_tools(evidence, {'browser_observe': tool})
    assert 'taskRequirements' in tool.json_schema['properties']
    output = await tool.function(taskRequirements=PLAN)
    assert 'Task outcomes:' in output and evidence.requirements.items[0].id in output
    assert execute.await_count == 1
    replacement = {'complete': True, 'items': [PLAN['items'][0]]}
    with pytest.raises(ToolPreparationError, match='fixed_requirements'):
        evidence.prepare_tool('browser_observe', {'taskRequirements': replacement})
    evidence.browser_actions = 1
    evidence.prepare_tool('browser_observe', {'taskRequirements': PLAN})
    assert not evidence.plan_pending
    with pytest.raises(ToolPreparationError, match='fixed_requirements'):
        evidence.prepare_tool('browser_observe', {'taskRequirements': replacement})


@pytest.mark.parametrize('same_record', [False, True])
async def test_separate_saved_objects_need_separate_record_checks(same_record):
    goal = 'A to 3, B to 4, save A and save B'
    plan = {'complete': True, 'items': [
        {'requirement': 'A to 3', 'expected': '3'}, {'requirement': 'B to 4', 'expected': '4'},
        {'requirement': 'save A', 'kind': 'result'}, {'requirement': 'save B', 'kind': 'result'}]}
    evidence = TaskEvidence.for_request(goal, ClassificationResult('browser', .99, 2,
        browser_requirements=plan, browser_requirements_key=request_key((goal,))))
    await evidence.prepare_requirements()
    evidence.browser_actions = 2
    items = evidence.requirements.items
    with pytest.raises(ValueError, match='ambiguous_result'):
        evidence.prepare_tool('browser_observe', {'readbackUrl': 'https://example.com/record', 'checks': []})
    for index in range(2):
        row = {'selector': '#value', 'property': 'value', 'requirement_ids': [items[index].id],
               'expected': str(index + 3), 'actual': str(index + 3), 'passed': True}
        evidence.observe('browser_observe', CapabilityResult(success=True, metadata={'independent_readback': {
            'status': 'pass', 'scope': 'http_document', 'url': f'https://example.com/{0 if same_record else index}',
            'result': items[index + 2].id, 'checks': [row]}}))
        assert (evidence.acceptance()['status'] == 'pass') is (index == 1 and not same_record)
    evidence.observe('browser_act', CapabilityResult(success=True, metadata={'action_status': 'dispatched', 'action': 'type'}))
    assert evidence.acceptance()['status'] == 'inconclusive'


@pytest.mark.parametrize('same_field', [False, True])
async def test_fetched_fields_cannot_share_a_target(same_field):
    goal = 'adults to 3, children to 3, save'
    plan = {'complete': True, 'items': [
        {'requirement': 'adults to 3', 'expected': '3'}, {'requirement': 'children to 3', 'expected': '3'},
        {'requirement': 'save', 'kind': 'result'}]}
    evidence = TaskEvidence.for_request(goal, ClassificationResult('browser', .99, 2,
        browser_requirements=plan, browser_requirements_key=request_key((goal,))))
    await evidence.prepare_requirements()
    evidence.browser_actions = 1
    items = evidence.requirements.items
    rows = [{'selector': f'#field{0 if same_field else index}', 'property': 'value',
             'requirement_ids': [items[index].id], 'expected': '3', 'actual': '3', 'passed': True} for index in range(2)]
    evidence.observe('browser_observe', CapabilityResult(success=True, metadata={'independent_readback': {
        'status': 'pass', 'url': 'https://example.com/record', 'result': items[-1].id, 'checks': rows}}))
    assert (evidence.acceptance()['status'] == 'pass') is not same_field


@pytest.mark.parametrize('receipt', [None, 'Pending', 'Saved'])
async def test_requested_receipt_wording_is_not_replaced_by_field_readback(receipt):
    goal = 'Set count to 3, save and show Saved'
    plan = {'complete': True, 'items': [
        {'requirement': 'count to 3', 'expected': '3'},
        {'requirement': 'save and show Saved', 'expected': 'Saved', 'kind': 'result'}]}
    evidence = TaskEvidence.for_request(goal, ClassificationResult('browser', .99, 2,
        browser_requirements=plan, browser_requirements_key=request_key((goal,))))
    await evidence.prepare_requirements()
    evidence.browser_actions = 1
    field, result = evidence.requirements.items
    rows = [{'selector': '#count', 'property': 'value', 'requirement_ids': [field.id],
             'expected': '3', 'actual': '3', 'passed': True}]
    if receipt is not None:
        rows.append({'selector': '#receipt', 'property': 'text', 'requirement_ids': [result.id],
                     'expected': 'Saved', 'actual': receipt, 'passed': receipt == 'Saved'})
    evidence.observe('browser_observe', CapabilityResult(success=True, metadata={'independent_readback': {
        'status': 'pass', 'url': 'https://example.com/record', 'result': result.id, 'checks': rows}}))
    assert (evidence.acceptance()['status'] == 'pass') is (receipt == 'Saved')


@pytest.mark.parametrize("goal_type", ["chat", "artifact", "code_modify", "research"])
async def test_other_tasks_do_not_request_browser_requirements(monkeypatch, goal_type):
    extraction = AsyncMock(side_effect=AssertionError("Unrelated verification"))
    monkeypatch.setattr("rune.llm.client.get_llm_client", extraction)
    evidence = TaskEvidence.for_request("Requested work", ClassificationResult(goal_type, .99, 2))
    assert await evidence.prepare_requirements() == ""
    extraction.assert_not_called()


@pytest.mark.parametrize("kind,literal,canonical,actual,wrong", [
    ("time", "오전 10시", "10:00", "10:00:00", "22:00"),
    ("time", "10:00Z", "10:00+00:00", "10:00:00Z", "10:00+01:00"),
    ("number", "6명", "6", "6.0", "7"),
    ("text", "00123", None, "00123", "123"),
])
async def test_fixed_values_accept_equivalent_formats_without_weakening_the_request(kind, literal, canonical, actual, wrong):
    from rune.agent.browser_requirements import matches

    spec = {"complete": True, "items": [{"requirement": literal, "expected": literal,
                                         "value_type": kind, "canonical": canonical}]}
    evidence = TaskEvidence.for_request(literal, ClassificationResult("browser", .99, 2,
        browser_requirements=spec, browser_requirements_key=request_key((literal,))))
    await evidence.prepare_requirements()
    params = {"checks": [{"selector": "#field", "property": "value", "requirement": literal, "expected": actual}]}
    evidence.prepare_tool("browser_observe", params)
    assert params["checks"][0]["value_type"] == kind
    assert matches(evidence.requirements.items[0], params["checks"][0], {})
    with pytest.raises(ValueError):
        evidence.prepare_tool("browser_observe", {"checks": [{**params["checks"][0], "expected": wrong}]})
    assert len(evidence.expected_checks) == 1


@pytest.mark.parametrize("literal,kind,canonical", [("6", "number", "7"), ("10:00", "time", "22:00"),
                                                     ("six", "number", "NaN"), ("morning", "time", "25:00")])
def test_canonical_values_cannot_change_a_parseable_literal(literal, kind, canonical):
    with pytest.raises(ValueError):
        validated({"complete": True, "items": [{"requirement": literal, "expected": literal,
                   "value_type": kind, "canonical": canonical}]}, (literal,))


@pytest.mark.parametrize("literal,kind,canonical,stored", [("군산", "text", "군산", None),
                                                        ("00123", "text", "00123", None), ("3", "number", None, "3")])
def test_redundant_or_parseable_canonical_values_need_no_model_retry(literal, kind, canonical, stored):
    result = validated({"complete": True, "items": [{"requirement": literal, "expected": literal,
                        "value_type": kind, "canonical": canonical}]}, (literal,))
    assert result.items[0].canonical == stored
    assert result.items[0].accepts(literal)


def test_text_canonical_cannot_change_an_identifier():
    with pytest.raises(ValueError):
        validated({"complete": True, "items": [{"requirement": "00123", "expected": "00123",
                   "value_type": "text", "canonical": "123"}]}, ("00123",))


@pytest.mark.parametrize("kind", ["value", "result"])
async def test_page_derived_values_ignore_unused_canonical_without_losing_coverage(monkeypatch, kind):
    goal = "Use the suggested option and save"
    plan = {"complete": True, "items": [
        {"requirement": goal, "kind": kind, "expected": None, "canonical": "click"},
    ]}
    extraction = AsyncMock(side_effect=AssertionError("The existing plan is sufficient"))
    monkeypatch.setattr("rune.llm.client.get_llm_client", extraction)
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2,
        browser_requirements=plan, browser_requirements_key=request_key((goal,))))
    await evidence.prepare_requirements()
    assert not evidence.requirements_unavailable
    assert evidence.requirements.items[0].canonical is None
    assert evidence.acceptance()["status"] == "inconclusive"
    assert evidence.acceptance()["missing"] == [evidence.requirements.items[0].id]
    extraction.assert_not_called()


async def test_checker_failure_does_not_replace_action_permissions_or_claim_completion(monkeypatch):
    extraction = AsyncMock(return_value=None)
    monkeypatch.setattr("rune.llm.client.get_llm_client", extraction)
    evidence = TaskEvidence.for_request(GOAL, ClassificationResult("browser", .99, 2))
    await evidence.prepare_requirements()
    evidence.prepare_tool("browser_navigate", {"url": "https://example.com"})
    evidence.prepare_tool("browser_observe", {})
    evidence.prepare_tool("browser_act", {"action": "scroll", "value": "down"})
    from rune.agent.tool_errors import ToolPreparationError
    with pytest.raises(ToolPreparationError, match="requirements_missing"):
        evidence.prepare_tool("browser_batch", {"actions": [
            {"type": "observe", "params": {}},
            {"type": "act", "params": {"action": "click", "selector": "#save"}},
        ]})
    assert evidence.blocker(resolve_intent_contract(ClassificationResult("browser", .99, 2), .99), 0)
    assert evidence.acceptance()["status"] == "unavailable"
    extraction.assert_not_called()


async def test_fixed_value_error_reaches_the_model_before_dispatch():
    from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
    from rune.capabilities.browser.capabilities import BrowserObserveParams
    from rune.capabilities.registry import CapabilityRegistry
    from rune.capabilities.types import CapabilityDefinition

    evidence = TaskEvidence.for_request(GOAL, ClassificationResult("browser", .99, 2,
        browser_requirements=PLAN, browser_requirements_key=request_key((GOAL,))))
    await evidence.prepare_requirements()
    execute, ended = AsyncMock(), AsyncMock()
    cap = CapabilityDefinition(name="browser_observe", description="Read checks", execute=execute,
                               parameters_model=BrowserObserveParams)
    registry = CapabilityRegistry()
    registry.register(cap)
    tool = _build_typed_tool(cap_def=cap, reg=registry, cache=None, stall=None,
                            opts=ToolAdapterOptions(prepare_tool=evidence.prepare_tool, on_tool_end=ended))
    result = await tool.function(checks=[{"selector": "#children", "property": "value",
                                         "requirement": "Children to 3", "expected": "9"}])
    execute.assert_not_awaited()
    assert "requirement_value_mismatch" in result and "Children to 3" in result
    from rune.agent.tool_output import RejectedToolResult

    assert isinstance(result, RejectedToolResult) and result.code == "requirement_value_mismatch"
    error = ended.call_args.args[1]
    assert error.metadata["action_status"] == "not_executed"
    assert error.metadata["validation_error"]["expected"] == "3"
    assert error.metadata["validation_error"]["supplied"] == "9"


async def test_check_binding_reuses_only_a_unique_fixed_value():
    plan = {"complete": True, "items": [*PLAN["items"], {"requirement": "City to Gunsan", "expected": "Gunsan"}]}
    goal = GOAL + " City to Gunsan"
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2,
        browser_requirements=plan, browser_requirements_key=request_key((goal,))))
    await evidence.prepare_requirements()
    values = {"checks": [{"selector": "#city", "property": "value", "expected": "Gunsan", "requirement": "City"}]}
    evidence.prepare_tool("browser_observe", values)
    assert values["checks"][0]["requirement"] == "City to Gunsan"
    for expected in ("3", "unrequested"):
        with pytest.raises(ValueError, match="unknown_requirement"):
            evidence.prepare_tool("browser_observe", {"checks": [{"selector": "#field", "property": "value",
                                                                  "expected": expected, "requirement": "Field"}]})
    assert len(evidence.expected_checks) == 1


@pytest.mark.parametrize("reference", ["quote", "id"])
async def test_save_reference_on_a_field_binds_only_a_unique_requested_value(reference):
    goal = GOAL + " City to Gunsan"
    plan = {"complete": True, "items": [*PLAN["items"], {"requirement": "City to Gunsan", "expected": "Gunsan"}]}
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2,
        browser_requirements=plan, browser_requirements_key=request_key((goal,))))
    await evidence.prepare_requirements()
    saved = next(item for item in evidence.requirements.items if item.kind == "result")
    source = {"requirement": saved.requirement} if reference == "quote" else {"requirement_ids": [saved.id]}
    params = {"checks": [{"selector": "#city", "property": "value", "expected": "Gunsan", **source}]}
    evidence.prepare_tool("browser_observe", params)
    assert params["checks"][0]["requirement"] == "City to Gunsan"
    for expected in ("3", "unrequested"):
        with pytest.raises(ToolPreparationError, match="ambiguous_requirement"):
            evidence.prepare_tool("browser_observe", {"checks": [
                {"selector": "#field", "property": "value", "expected": expected, **source}]})


async def test_invalid_executor_plan_returns_a_repairable_error_before_input():
    evidence = TaskEvidence.for_request("Set people to 3", ClassificationResult("browser", .99, 2))
    await evidence.prepare_requirements()
    with pytest.raises(ToolPreparationError, match="invalid_requirements") as error:
        evidence.prepare_tool("browser_observe", {"taskRequirements": {"items": [
            {"requirement": "Set people to three", "expected": "three"}]}})
    assert "not an exact user quote" in str(error.value)
    assert evidence.requirements is None and evidence.browser_actions == 0


async def test_quote_repair_is_a_suggestion_and_keeps_every_requested_value():
    goal = "도시를 군산, 인원을 3으로 바꾸고 저장해"
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2))
    await evidence.prepare_requirements()
    with pytest.raises(ToolPreparationError) as error:
        evidence.prepare_tool("browser_navigate", {"taskRequirements": {"items": [
            {"requirement": "도시를 군산으로 변경", "expected": "군산", "kind": "value"},
            {"requirement": "인원을 3으로 변경", "expected": "3", "kind": "value"},
            {"requirement": "저장해", "kind": "result"},
        ]}})
    assert evidence.requirements is None
    proposal = error.value.details["suggestedRequirements"]
    evidence.prepare_tool("browser_navigate", {"taskRequirements": proposal})
    assert [item.expected for item in evidence.requirements.outcomes] == ["군산", "3", None]


@pytest.mark.parametrize("items", [
    [{"requirement": "A should be 3", "expected": "3"}, {"requirement": "B should be 3", "expected": "3"}],
    [{"requirement": "A should be 4", "expected": "4"}],
])
def test_quote_repair_cannot_invent_values_or_disambiguate_equal_values(items):
    from rune.agent.browser_requirements import quote_repair

    assert quote_repair({"complete": True, "items": items}, ("A to 3 and B to 3",)) is None
async def test_unambiguous_page_receipt_binds_without_claiming_storage():
    evidence = TaskEvidence.for_request(GOAL, ClassificationResult("browser", .99, 2,
        browser_requirements=PLAN, browser_requirements_key=request_key((GOAL,))))
    await evidence.prepare_requirements()
    params = {"checks": [{"selector": "#receipt", "property": "text", "expected": "Saved", "requirement": "save confirmation"}]}
    evidence.prepare_tool("browser_observe", params)
    assert params["checks"][0]["requirement"] == "save"
    assert evidence.acceptance()["status"] != "pass"


def test_budget_stop_after_confirmed_storage_returns_facts_without_another_model_call():
    from types import SimpleNamespace

    from rune.agent.browser_result import final_answer
    from rune.agent.run_outcome import run_outcome

    plan = validated({"complete": True, "items": [{"requirement": "save", "kind": "result"}]}, ("save",))
    evidence = TaskEvidence(browser_actions=1, requirements=plan,
                            independent_readback={"status": "pass", "checks": [
                                {"requirement": "People to 3", "expected": "3", "actual": "3"}]})
    trace = SimpleNamespace(reason="request_budget_exhausted", requirement_acceptance={"required": True, "status": "pass"})
    answer = final_answer(evidence, trace, "", "en")
    assert "requested: 3; retrieved: 3" in answer and "budget limit" in answer
    assert run_outcome(trace).completion == "incomplete"
    assert final_answer(evidence, trace, "Existing answer", "en") == "Existing answer"


@pytest.mark.parametrize("scope,reextract", [("step", False), ("task", True)])
async def test_followup_reuses_routed_step_but_whole_tasks_keep_earlier_conditions(monkeypatch, scope, reextract):
    goal = "Change only Children to 4 and save"
    plan = {"complete": True, "items": [{"requirement": "Children to 4", "expected": "4"}]}
    extraction = AsyncMock(return_value=BrowserRequirements.model_validate(plan))
    monkeypatch.setattr("rune.llm.client.get_llm_client", extraction)
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2, task_scope=scope,
        browser_requirements=plan, browser_requirements_key=request_key((goal,))))
    evidence.request_texts = (GOAL, goal)
    await evidence.prepare_requirements()
    await evidence.prepare_requirements()
    extraction.assert_not_called()
    if reextract:
        assert evidence.requirements is None
    else:
        assert evidence.requirements.items[0].expected == "4"


@pytest.mark.parametrize('reason', [
    'request_budget_exhausted', 'cost_budget_exhausted', 'max_iterations', 'timeout',
    'cancelled', 'error:provider unavailable', 'task_blocked', 'verification_inconclusive', 'completed',
])
def test_unconfirmed_save_never_keeps_a_success_answer_on_any_stop_path(reason):
    from types import SimpleNamespace

    from rune.agent.browser_requirements import validated
    from rune.agent.browser_result import final_answer

    evidence = TaskEvidence(browser_actions=2, coverage_required=True, requirements=validated(
        {'complete': True, 'items': [{'requirement': 'save', 'kind': 'result'}]}, ('save',)))
    evidence.independent_readback = {'status': 'fail', 'scope': 'http_document', 'checks': [
        {'requirement': 'people to 3', 'actual': '1', 'expected': '3', 'passed': False}]}
    trace = SimpleNamespace(reason=reason, requirement_acceptance=evidence.acceptance())
    answer = final_answer(evidence, trace, 'Saved successfully.')
    assert 'Saved successfully' not in answer
    assert 'does not match' in answer and 'requested: 3; retrieved: 1' in answer
    assert reason in answer and trace.reason == reason


def test_final_answer_preserves_unrelated_and_verified_work():
    from types import SimpleNamespace

    from rune.agent.browser_result import final_answer

    for status in ('pass', 'inconclusive'):
        trace = SimpleNamespace(reason='cancelled', requirement_acceptance={'required': True, 'status': status})
        assert final_answer(TaskEvidence(), trace, 'Document attached.') == 'Document attached.'


def test_new_document_retires_targets_but_keeps_fixed_outcomes_and_baselines():
    from rune.agent.browser_requirements import validated

    plan = validated({'complete': True, 'items': [{'requirement': 'people to 3', 'expected': '3'}]}, ('people to 3',))
    evidence = TaskEvidence(document_id='old', requirements=plan, preserved_values={'existing': ('value', '2')}, browser_actions=2)
    evidence.prepare('browser_observe', {'checks': [{'selector': '#old', 'property': 'value', 'expected': '3', 'requirement': 'people to 3'}]})
    assert evidence.check_specs
    evidence.observe('browser_navigate', CapabilityResult(success=True, metadata={'document_id': 'new', 'url': 'https://example.test/same'}))
    params = {'checks': [{'selector': '#new', 'property': 'value', 'expected': '3', 'requirement': 'people to 3'}]}
    evidence.prepare('browser_observe', params)
    assert len(params['checks']) == 1
    assert evidence.requirements is plan and evidence.browser_actions == 2
    assert evidence.preserved_values == {'existing': ('value', '2')}


@pytest.mark.parametrize('source', ['network', 'disk', 'memory', 'worker', 'prefetch', 'unknown', 'ambiguous'])
def test_navigation_readback_requires_one_fresh_network_response(source):
    from rune.capabilities.browser.network import NetworkMonitor

    monitor = NetworkMonitor()
    response = {'url': 'https://example.test/record'}
    flags = {'disk': 'fromDiskCache', 'worker': 'fromServiceWorker', 'prefetch': 'fromPrefetchCache'}
    if source in flags:
        response[flags[source]] = True
    if source == 'memory':
        monitor._on_cached({'requestId': 'record'})
    if source != 'unknown':
        monitor._on_response({'type': 'Document', 'requestId': 'record', 'response': response})
    if source == 'ambiguous':
        monitor._on_response({'type': 'Document', 'requestId': 'frame', 'response': response})
    assert monitor.fresh_document(response['url'], 0) is (source == 'network')
    assert not monitor.fresh_document(response['url'], monitor.document_sequence)
    if source == 'network':
        monitor._on_cached({'requestId': 'record'})
        assert not monitor.fresh_document(response['url'], 0)


BOOKING = "Find a room with a projector at 10:00 and book it as Release review."
BOOKING_PLAN = {"complete": True, "items": [
    {"requirement": "a room with a projector", "expected": "a room with a projector", "kind": "criterion"},
    {"requirement": "at 10:00", "expected": "10:00", "kind": "value", "value_type": "time", "canonical": "10:00"},
    {"requirement": "book it as Release review", "expected": "Release review", "kind": "value"},
]}


def test_selection_criterion_has_no_literal_and_is_not_an_outcome():
    plan = validated(BOOKING_PLAN, (BOOKING,))
    criterion = plan.items[0]
    assert criterion.kind == "criterion" and criterion.expected is None and criterion.canonical is None
    assert criterion not in plan.outcomes and len(plan.outcomes) == 2


@pytest.mark.parametrize("binding", ["id", "quote"])
async def test_check_bound_to_a_selection_criterion_does_not_block_the_input(binding):
    evidence = TaskEvidence.for_request(BOOKING, ClassificationResult("browser", .99, 2,
        browser_requirements=BOOKING_PLAN, browser_requirements_key=request_key((BOOKING,))))
    await evidence.prepare_requirements()
    criterion, time, title = evidence.requirements.items
    chosen = {"selector": "#room", "property": "value", "expected": "Cedar"}
    chosen |= {"requirement_ids": [criterion.id]} if binding == "id" else {"requirement": criterion.requirement}
    params = {"actions": [{"type": "act", "params": {"action": "type", "selector": "#title", "value": "Release review", "checks": [
        chosen,
        {"selector": "#time", "property": "value", "expected": "10:00", "requirement_ids": [time.id]},
        {"selector": "#title", "property": "value", "expected": "Release review", "requirement_ids": [title.id]},
    ]}}]}
    evidence.prepare_tool("browser_batch", params)
    checks = params["actions"][0]["params"]["checks"]
    assert [check["requirement"] for check in checks] == [time.requirement, title.requirement]


@pytest.mark.parametrize("status", ["acknowledged", "pass"])
def test_successful_answers_state_what_was_not_checked(status):
    from types import SimpleNamespace

    from rune.agent.browser_result import final_answer

    plan = validated({"complete": True, "items": [
        {"requirement": "a room with a projector", "kind": "criterion"},
        {"requirement": "book it as Release review", "expected": "Release review", "kind": "value"},
        {"requirement": "save once", "kind": "result"},
    ]}, (BOOKING + " save once",))
    evidence = TaskEvidence(browser_actions=2, coverage_required=True, requirements=plan)
    trace = SimpleNamespace(reason="completed", requirement_acceptance={"required": True, "status": status})
    answer = final_answer(evidence, trace, "Booked Cedar.", "en")
    assert answer.startswith("Booked Cedar.")
    assert "“a room with a projector”" in answer
    assert ("stored record was not read back" in answer) is (status == "acknowledged")


def test_answers_without_saved_results_are_left_alone():
    from types import SimpleNamespace

    from rune.agent.browser_result import final_answer

    evidence = TaskEvidence(browser_actions=0)
    trace = SimpleNamespace(reason="completed", requirement_acceptance=None)
    assert final_answer(evidence, trace, "Done.", "ko") == "Done."


async def test_several_saved_results_are_questioned_once_then_accepted():
    goal = "Change people to 7 and save it. Save only once."
    plan = {"items": [
        {"requirement": "Change people to 7", "expected": "7", "kind": "value"},
        {"requirement": "save it", "expected": None, "kind": "result"},
        {"requirement": "Save only once", "expected": None, "kind": "result"},
    ]}
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2))
    await evidence.prepare_requirements()
    with pytest.raises(ToolPreparationError) as asked:
        evidence.prepare_tool("browser_observe", {"taskRequirements": plan})
    assert asked.value.details["code"] == "multiple_saves" and evidence.requirements is None
    # A merged plan is accepted at once.
    merged = {"items": [plan["items"][0], {"requirement": "save it", "expected": None, "kind": "result"}]}
    evidence.prepare_tool("browser_observe", {"taskRequirements": merged})
    assert [item.kind for item in evidence.requirements.outcomes] == ["value", "result"]


async def test_really_separate_saves_stand_when_resubmitted():
    goal = "Save A and save B"
    plan = {"items": [{"requirement": "Save A", "expected": None, "kind": "result"},
                      {"requirement": "save B", "expected": None, "kind": "result"}]}
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2))
    await evidence.prepare_requirements()
    with pytest.raises(ToolPreparationError):
        evidence.prepare_tool("browser_observe", {"taskRequirements": plan})
    evidence.prepare_tool("browser_observe", {"taskRequirements": plan})
    assert len(evidence.requirements.outcomes) == 2


def test_classifier_and_executor_share_one_requirement_wording():
    from rune.agent.browser_plan import executor_schema
    from rune.agent.browser_requirements import EXECUTOR_KINDS, KIND_GUIDE, response_schema

    kind = executor_schema()["properties"]["items"]["items"]["properties"]["kind"]
    assert kind["description"] == KIND_GUIDE and kind["enum"] == list(EXECUTOR_KINDS)
    assert KIND_GUIDE in json.dumps(response_schema())
