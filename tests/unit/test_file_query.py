import json
from decimal import Decimal

import pytest

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.intent_engine import resolve_intent_contract
from rune.agent.task_evidence import TaskEvidence
from rune.capabilities.file_query import FileQueryParams, file_query


@pytest.mark.parametrize("operation,expected", [("sum", "0.6"), ("mean", "0.3"), ("min", "0.2"), ("max", "0.4"), ("count", "2")])
async def test_filtered_file_calculations_provide_completion_evidence(tmp_path, operation, expected):
    path = tmp_path / "orders.csv"
    path.write_text("status,amount\npaid,0.2\nrefunded,99\npaid,0.4\n")
    result = await file_query(FileQueryParams(path=str(path), query={"where": {"status": "paid"}, "operation": operation, "field": "amount"}))
    assert result.success, result.error
    body = json.loads(result.output, parse_float=Decimal)
    assert Decimal(str(body["result"]["value"])) == Decimal(expected)
    assert body["source"]["matched"] == 2 and len(body["source"]["sha256"]) == 64
    classification = ClassificationResult("research", .99, 2, intent_categories=frozenset({"calculation"}), calculation_operation=operation)
    evidence = TaskEvidence.for_request("Calculate the requested aggregate", classification)
    evidence.observe("file_query", result)
    assert evidence.blocker(resolve_intent_contract(classification, .99), 0) is None


@pytest.mark.parametrize("text", ["amount,amount\n1,2\n", "amount,status\n1\n", "amount\n1\nunknown\n"])
async def test_malformed_or_nonnumeric_rows_cannot_produce_partial_totals(tmp_path, text):
    path = tmp_path / "orders.csv"
    path.write_text(text)
    result = await file_query(FileQueryParams(path=str(path), query={"operation": "sum", "field": "amount"}))
    assert not result.success and not result.metadata.get("record_query")


async def test_json_preserves_precision_and_missing_files_fail(tmp_path):
    path = tmp_path / "records.json"
    path.write_text('[{"value":9007199254740993.01},{"value":0.01}]')
    result = await file_query(FileQueryParams(path=str(path), query={"operation": "sum", "field": "value"}))
    assert result.success and "9007199254740993.02" in result.output
    path.unlink()
    assert not (await file_query(FileQueryParams(path=str(path), query={"operation": "count"}))).success
