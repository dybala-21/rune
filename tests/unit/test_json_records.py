"""Queries retain record identity and reject ambiguous or lossy calculations."""

import json
from decimal import Decimal

import pytest

from rune.capabilities.json_records import RecordQuery, query_records, record_paths


def test_path_hints_are_bounded_and_do_not_include_source_values():
    body = {"a/b~": [{"station": "private", "value": 17}]}
    hint = record_paths(body)
    assert '"path":"/a~1b~0"' in hint and '"sampleFields":["station","value"]' in hint
    assert "private" not in hint and "17" not in hint
    assert record_paths({str(i): body for i in range(1000)}).count('"path"') <= 4


def test_latest_uses_observation_time_and_keeps_parent_unit():
    source = {"unit": "celsius", "rows": [
        {"station": "North", "value": 120, "observed_at": "2020-01-01T00:00:00Z"},
        {"station": "North", "value": 17.125, "observed_at": "2026-10-16T09:00:00+09:00"},
        {"station": "Other", "value": 999, "observed_at": "2026-10-17T00:00:00Z"},
    ]}
    answer, receipt = query_records(json.dumps(source), RecordQuery(
        path="/rows", where={"station": "North"}, operation="latest", field="observed_at"))
    assert json.loads(answer) == {"unit": "celsius", **source["rows"][1]}
    assert receipt["records"] == 3 and receipt["matched"] == 2
    assert receipt["scope"] == "retrieved_response"


def test_latest_compares_offsets_without_rewriting_the_source():
    body = '[{"n":1,"t":"2026-10-16T08:30:00+09:00"},{"n":2,"t":"2026-10-16T00:00:00Z"}]'
    answer, _ = query_records(body, RecordQuery(operation="latest", field="t"))
    assert json.loads(answer) == {"n": 2, "t": "2026-10-16T00:00:00Z"}


@pytest.mark.parametrize("body", [
    '[{"n":1,"t":"2026-10-16T00:00:00Z"},{"n":2,"t":"2026-10-16T09:00:00+09:00"}]',
    '[{"n":1,"t":"2026-10-16"},{"n":2,"t":"2026-10-16T00:00:00Z"}]',
    '[{"n":1,"t":"2026-10-16T00:00:00"}]', '[{"n":1}]', '[]',
])
def test_latest_rejects_conflicts_missing_fields_and_ambiguous_dates(body):
    with pytest.raises(ValueError):
        query_records(body, RecordQuery(operation="latest", field="t"))


@pytest.mark.parametrize("operation,expected", [("sum", "0.3"), ("mean", "0.15"), ("min", "0.1"), ("max", "0.2")])
def test_aggregates_keep_decimal_precision(operation, expected):
    body, receipt = query_records('[{"v":0.1},{"v":0.2}]', RecordQuery(operation=operation, field="v"))
    assert json.loads(body, parse_float=Decimal)["value"] == Decimal(expected)
    assert receipt["matched"] == 2


@pytest.mark.parametrize("body", ['[{"v":true}]', '[{"v":"1"}]', '[{"v":1},{}]', '[{"v":NaN}]', '[{"v":1e999999}]'])
def test_invalid_numeric_fields_are_not_silently_ignored(body):
    with pytest.raises(ValueError):
        query_records(body, RecordQuery(operation="sum", field="v"))


def test_repeating_mean_requires_explicit_rounding():
    with pytest.raises(ValueError):
        query_records('[{"v":1},{"v":0},{"v":0}]', RecordQuery(operation="mean", field="v"))


def test_filter_does_not_conflate_booleans_with_numbers():
    body, _ = query_records('[{"v":true},{"v":1}]', RecordQuery(where={"v": 1}, operation="count"))
    assert json.loads(body)["value"] == 1
    with pytest.raises(ValueError, match="filter field"):
        query_records('[{"v":1},{}]', RecordQuery(where={"v": 1}, operation="count"))


def test_pointer_escapes_and_missing_paths():
    body, _ = query_records('{"a/b~":[{"v":1}]}', RecordQuery(path="/a~1b~0", operation="count"))
    assert json.loads(body)["value"] == 1
    with pytest.raises(ValueError):
        query_records('[[{"v":1}]]', RecordQuery(path="/-1", operation="count"))
