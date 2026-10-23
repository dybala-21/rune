"""Structured answers must keep related source fields together."""

import json

import pytest

from rune.agent.source_records import SourceRecords


@pytest.fixture
def records():
    check = SourceRecords()
    check.observe(json.dumps({"unit": "USD", "data": [
        {"symbol": "ACME", "value": 120, "observed_at": "2020-01-01"},
        {"symbol": "ACME", "value": 210, "observed_at": "2026-10-16"},
        {"symbol": "OTHER", "value": 120, "observed_at": "2026-10-16"},
    ]}))
    return check


@pytest.mark.parametrize("changes", [
    {"symbol": "MISSING"}, {"value": 120}, {"unit": "JPY"},
    {"observed_at": "2020-01-01"}, {"value": True}, {"value": "210"},
])
def test_fields_must_match_one_record(records, changes):
    answer = {"symbol": "ACME", "value": 210, "unit": "USD", "observed_at": "2026-10-16"}
    assert records.blocker(json.dumps(answer | changes))
    assert records.blocker(json.dumps(answer)) is None


def test_older_observations_are_valid_when_reported_with_their_own_date(records):
    assert records.blocker('{"symbol":"ACME","value":120,"observed_at":"2020-01-01"}') is None


def test_nested_projections_keep_parent_scope_and_numeric_precision():
    check = SourceRecords()
    check.observe('{"symbol":"ACME","quote":{"value":0.123456789012345678901,"unit":"USD"}}')
    assert check.blocker('```json\n{"symbol":"ACME","value":0.123456789012345678901,"unit":"USD"}\n```') is None
    assert check.blocker('{"symbol":"ACME","quote":{"value":0.123456789012345678902}}')


def test_fields_cannot_be_joined_across_sources():
    check = SourceRecords()
    check.observe('{"symbol":"ACME","value":120,"unit":"USD"}')
    check.observe('{"symbol":"OTHER","value":210,"unit":"JPY"}')
    assert check.blocker('{"symbol":"ACME","value":210,"unit":"JPY"}')


def test_list_answer_checks_each_record(records):
    assert records.blocker('[{"symbol":"ACME","value":210},{"symbol":"OTHER","value":210}]')
    assert records.blocker('[{"symbol":"ACME","value":210},{"symbol":"OTHER","value":120}]') is None


@pytest.mark.parametrize("answer", ['The average is 150.', '{"average":150}', '{"result":{"count":3}}'])
def test_unrelated_output_shapes_are_not_guessed(records, answer):
    assert records.blocker(answer) is None


@pytest.mark.parametrize("source", ['not JSON', '{"value":NaN}', '{"value":1e9999999999999999999}', '[' * 1000,
                                       json.dumps([{"symbol":"X","value":n} for n in range(600)])])
def test_unreadable_or_oversized_sources_cannot_prove_a_contradiction(records, source):
    records.observe(source)
    assert not records.complete
    assert records.blocker('{"symbol":"ACME","value":999}') is None


def test_source_limit_does_not_turn_evicted_evidence_into_a_contradiction():
    check = SourceRecords()
    for value in range(10):
        check.observe(json.dumps({"symbol":"ACME", "value":value}))
    assert len(check.documents) == 8 and not check.complete
    assert check.blocker('{"symbol":"ACME","value":9}') is None


def test_review_keeps_coverage_when_another_source_is_unreadable():
    check = SourceRecords()
    check.observe('{"symbol":"ACME","value":210}', "quote")
    check.observe("HTML commentary", "commentary")
    good = check.review('{"symbol":"ACME","value":210}')
    assert good.status == "supported" and good.checked == 1 and good.source_ids == ("quote",)
    unknown = check.review('[{"symbol":"ACME","value":210},{"symbol":"ACME","value":999}]')
    assert unknown.status == "insufficient" and unknown.checked == 1 and unknown.unchecked == 1
    assert check.review('It is therefore safe.').status == "not_checked"


def test_review_does_not_claim_to_check_latest_selection_or_prose(records):
    review = records.review('{"symbol":"ACME","value":120,"observed_at":"2020-01-01"}')
    assert review.status == "supported"
    assert review.snapshot()["scope"] == "record_fields"
    assert records.review('{"average":165}').status == "not_checked"
