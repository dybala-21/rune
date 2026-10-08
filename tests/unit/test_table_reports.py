"""Reports must agree with source aggregates, including labels and units."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rune.agent.table_acceptance import TableAcceptance
from rune.agent.table_reports import ReportFacts, compare_facts
from rune.capabilities.table_acceptance import TablePlan
from rune.types import CapabilityResult


def plan():
    return TablePlan(applicable=True, requirements=["Save summary.csv and report.docx with every total."],
                     group_by=["team"], aggregates=[{"name": "amount", "column": "amount", "operation": "sum", "unit": "USD"}],
                     grand_total=["TOTAL"], output_names=["summary.csv"],
                     reports=[{"filename": "report.docx", "coverage": "all"}])


def facts(amount="100", unit="USD"):
    return ReportFacts(facts=[{"group": [group], "column": "amount", "value": value,
                              "unit": unit, "quote": f"{group}: {value} {unit}", "label_quotes": [group]}
                             for group, value in [("Engineering", amount), ("Sales", "200"), ("TOTAL", "300")]], complete=True)


EXPECTED = [["engineering", "100"], ["sales", "200"], ["TOTAL", "300"]]


def test_currency_identity_keeps_its_literal_evidence():
    spec = plan()
    spec.aggregates[0].unit = "KRW"
    extracted = facts(unit="KRW")
    for fact in extracted.facts:
        fact.unit_quote = "원"
        fact.quote = fact.quote.replace(" KRW", "원")
    text = "\n".join(f.quote for f in extracted.facts)
    assert not compare_facts(spec, EXPECTED, text, extracted, "all")
    assert compare_facts(spec, EXPECTED, text.replace("원", "달러"), extracted, "all")


def test_source_currency_does_not_invent_a_display_requirement():
    spec = plan()
    extracted = facts()
    for fact in extracted.facts:
        fact.quote = fact.quote.replace(" USD", "")
        fact.unit = None
    text = "\n".join(f.quote for f in extracted.facts)
    assert not compare_facts(spec, EXPECTED, text, extracted, "all")
    spec.aggregates[0].unit_required = True
    assert compare_facts(spec, EXPECTED, text, extracted, "all")


@pytest.mark.parametrize("suffix", [".", ", then continue", "; checked"])
def test_sentence_punctuation_is_not_part_of_the_amount(suffix):
    extracted = facts()
    for fact in extracted.facts:
        fact.quote = fact.quote.replace(" USD", suffix)
        fact.unit = None
    assert not compare_facts(plan(), EXPECTED, "\n".join(f.quote for f in extracted.facts), extracted, "all")


@pytest.mark.parametrize("case", ["correct", "amount", "label", "unit", "missing", "invented", "substring", "incomplete"])
def test_claims_are_compared_numerically_after_grounding(case):
    extracted = facts()
    if case == "amount":
        extracted = facts("99")
    elif case == "unit":
        extracted = facts(unit="KRW")
    elif case == "label":
        extracted.facts[0].group = ["Sales"]
        extracted.facts[0].label_quotes = ["Sales"]
        extracted.facts[0].quote = "Sales: 100 USD"
    elif case == "missing":
        extracted.facts.pop()
    elif case == "incomplete":
        extracted.complete = False
    text = "\n".join(f.quote for f in extracted.facts)
    if case == "invented":
        text = text.replace("100", "99")
    elif case == "substring":
        text = text.replace("100", "1000")
        extracted.facts[0].quote = "Engineering: 1000 USD"
    issues = compare_facts(plan(), EXPECTED, text, extracted, "all")
    assert bool(issues) is (case != "correct")


async def test_saved_report_is_rechecked_only_when_bytes_change(tmp_path, monkeypatch):
    from docx import Document

    guard = SimpleNamespace(validate_file_read_path=lambda _: SimpleNamespace(allowed=True))
    monkeypatch.setattr("rune.safety.guardian.get_guardian", lambda: guard)
    source, output, report = tmp_path / "source.csv", tmp_path / "summary.csv", tmp_path / "report.docx"
    source.write_text("team,amount\nengineering,100\nsales,200\n")
    output.write_text("team,amount\nengineering,100\nsales,200\nTOTAL,300\n")
    current = facts("99")

    def save():
        doc = Document()
        for fact in current.facts:
            doc.add_paragraph(fact.quote)
        doc.save(report)

    save()
    monkeypatch.setattr("rune.agent.table_acceptance.extract_plan", AsyncMock(return_value=plan()))
    extraction = AsyncMock(side_effect=lambda *args: current.model_dump_json())
    monkeypatch.setattr("rune.agent.requirement_gate._completion", extraction)
    state = TableAcceptance(plan().requirements[0], required=True, workspace=str(tmp_path))
    contract = (await state.requirements(str(source), None)).metadata["tableContract"]
    state.observe("document_create", {"path": str(report)}, CapabilityResult(success=True))
    assert (await state.verify(contract["id"], str(output), None, 1)).success
    assert "Incorrect amount" in await state.blocker()
    assert "document_create" in state.recovery_tools()
    assert await state.blocker()
    assert extraction.await_count == 1
    current = facts()
    save()
    state.observe("document_create", {"path": str(report)}, CapabilityResult(success=True))
    assert state.recovery_tools() is None
    assert await state.blocker() is None
    assert extraction.await_count == 2
    assert state.snapshot()["reports"][0]["status"] == "pass"
    assert await state.blocker() is None
    assert extraction.await_count == 2
    payload = json.loads(extraction.call_args.args[1])
    assert "100" not in json.dumps(payload["SCHEMA"])
    source.write_text("team,amount\nengineering,900\nsales,200\n")
    assert await state.blocker()
    assert extraction.await_count == 2
