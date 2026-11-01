import json
from datetime import UTC, datetime, timedelta

import pytest

from rune.agent.browser_context import compact_browser_history
from rune.agent.goal_classifier import ClassificationResult
from rune.agent.intent_engine import resolve_intent_contract
from rune.agent.retrieval_result import accepted_answer
from rune.agent.task_evidence import TaskEvidence
from rune.lodging.evidence import FIELDS, Span, StayEvidence, StayQuery
from rune.types import CapabilityResult
from tests.e2e.lodging_checks import StayRequest, verify_comparison

TEXT = ("Example Sapporo Hotel\nDouble room\nDouble room 90,000 KRW\n"
        "2026.11.15 to 2026.11.16\n1 room, 2 adults, 0 children\nSapporo, Example street\n"
        "Free cancellation until 2026-11-13 00:00; then non-refundable\n"
        "Standard price. No coupon applied.\nPay local accommodation tax at the property: JPY 600\n")


def url(identity="TDP-1", **changes):
    values = {"checkInDate": "2026-11-15", "checkOutDate": "2026-11-16", "rooms%5B0%5D%5Badult%5D": "2", **changes}
    return f"https://nol.yanolja.com/stay/overseas/{identity}/list?" + "&".join(f"{k}={v}" for k, v in values.items())


@pytest.fixture
def ledger():
    return StayEvidence(owner="session-a", run_id="run-a", query=StayQuery(
        city="Sapporo", check_in="2026-11-15", check_out="2026-11-16", adults=2, count=1, max_total_krw=200000))


def fields(source):
    return {key: Span(source=source.id, start=index + 1, end=index + 1) for index, key in enumerate(FIELDS)}


def complete(ledger):
    source = ledger.capture(url(), TEXT, "document-a")
    ledger.record(fields(source), 90000)
    ledger.finished = True
    return source


def test_missing_fields_are_preserved_and_filled_without_recapturing(ledger):
    source = ledger.capture(url(), TEXT, "doc-a")
    spans = fields(source)
    taxes = spans.pop("taxes_quote")
    item = ledger.record(spans, 90000)
    assert ledger.missing(item) == ["taxes_quote"]
    assert not ledger.status()["complete"]
    before = dict(item.fields)
    updated = ledger.record({"taxes_quote": taxes}, candidate_id=item.id)
    assert all(updated.fields[key] == value for key, value in before.items())
    assert not ledger.missing(updated) and len(ledger.sources) == 1
    assert ledger.acceptance()["status"] != "pass"  # finish is explicit
    ledger.finished = True
    result = ledger.result()
    assert result["offers"][0]["taxes_quote"] == TEXT.splitlines()[8]
    assert ledger.acceptance()["status"] == "pass"


@pytest.mark.parametrize("change", [
    {"checkInDate": "2026-11-14"}, {"checkOutDate": "2026-11-17"},
    {"rooms%5B0%5D%5Badult%5D": "3"}, {"rooms%5B1%5D%5Badult%5D": "2"},
])
def test_other_stay_conditions_cannot_enter_the_ledger(ledger, change):
    with pytest.raises(ValueError):
        ledger.capture(url(**change), TEXT, "doc")
    assert not ledger.sources


def test_another_property_cannot_supply_the_missing_tax_policy(ledger):
    source = ledger.capture(url(), TEXT, "a")
    spans = fields(source)
    spans.pop("taxes_quote")
    item = ledger.record(spans, 90000)
    other = ledger.capture(url("TDP-2"), TEXT, "b")
    with pytest.raises(ValueError, match="selected property"):
        ledger.record({"taxes_quote": fields(other)["taxes_quote"]}, candidate_id=item.id)
    assert ledger.missing(item) == ["taxes_quote"]


def test_room_price_changes_cannot_silently_overwrite_a_rate(ledger):
    source = complete(ledger)
    item = next(iter(ledger.candidates.values()))
    with pytest.raises(ValueError, match="price change"):
        ledger.record({"taxes_quote": fields(source)["taxes_quote"]}, 100000, item.id)
    new = ledger.capture(url(), TEXT.replace("Double room", "Twin room"), "a")
    with pytest.raises(ValueError, match="silently replaced"):
        ledger.record({"room": fields(new)["room"]}, candidate_id=item.id)
    assert ledger.result()["offers"][0]["room"] == "Double room"


def test_quotes_are_exact_ranges_and_cannot_be_fabricated(ledger):
    source = ledger.capture(url(), TEXT, "doc")
    with pytest.raises(ValueError, match="outside"):
        ledger.record({"name": Span(source=source.id, start=999, end=1000)}, 90000)
    spans = fields(source)
    spans["price_quote"] = spans["location_quote"]
    with pytest.raises(ValueError, match="price quote"):
        ledger.record(spans, 90000)
    assert not ledger.candidates


def test_property_teaser_cannot_establish_room_rate_conditions(ledger):
    source = ledger.capture(url().replace("/list?", "?"), TEXT, "doc")
    with pytest.raises(ValueError, match="teaser/coupon banner"):
        ledger.record(fields(source), 90000)
    assert not ledger.candidates


def test_literal_find_recovers_only_the_requested_source_region(ledger):
    source = ledger.capture(url(), "Filler\n" * 200 + TEXT, "doc")
    found = ledger.find(source.id, "local accommodation tax")
    assert len(found["matches"]) == 1
    assert "209: Pay local accommodation tax" in found["matches"][0]["lines"]
    assert len(found["matches"][0]["lines"].splitlines()) < 9
    assert not ledger.find(source.id, "nonexistent phrase")["matches"]
    assert len(ledger.sources) == 1


def test_record_boundary_retires_old_ui_bodies_but_keeps_current_controls():
    messages = [{"role": "tool", "name": "browser_extract", "content": "Old body " * 1000},
                {"role": "tool", "name": "browser_observe", "content": "Current control"},
                {"role": "tool", "name": "browser_stay_evidence", "content": '{"recorded":"rate"}'}]
    compacted = compact_browser_history(messages)
    assert len(compacted[0]["content"]) < 800
    assert compacted[1] == messages[1]


async def test_stream_finishes_from_trusted_ledger_without_a_prose_call(monkeypatch, ledger):
    from tests.unit.test_live_streaming import _chunk, _make, _tc

    complete(ledger)
    ledger.finished = False
    evidence = TaskEvidence(browser_effect="view")
    stream = _make("anthropic/claude-opus-5")
    stream._reserve_answer = True
    stream._retrieval_state = lambda: evidence
    requests = []

    async def completion(**kwargs):
        requests.append(kwargs)
        assert len(requests) <= 2 and kwargs.get("tools")

        async def chunks():
            yield _chunk(tool_calls=_tc(), finish="tool_calls")
        return chunks()

    async def execute(batch):
        ledger.finished = len(requests) == 2
        evidence.retrieval_acceptance = ledger.acceptance()
        stream._messages.append({"role": "tool", "tool_call_id": batch[0]["id"], "content": "Saved evidence"})

    monkeypatch.setattr("litellm.acompletion", completion)
    monkeypatch.setattr("rune.agent.timing.needs_answer_reserve", lambda *args, **kwargs: True)
    monkeypatch.setattr(stream, "_execute_tool_batch", execute)
    _ = [text async for text in stream.stream_text()]
    assert len(requests) == 2 and not stream.answer_reserve_reason
    assert json.loads(await stream.get_output()) == ledger.result()


def test_repeated_rooms_at_one_property_cannot_satisfy_two_properties(ledger):
    ledger.query.count = 2
    complete(ledger)
    other = ledger.capture(url(), TEXT.replace("Double room", "Twin room"), "doc")
    ledger.record(fields(other), 90000)
    assert len(ledger.candidates) == 2
    assert ledger.status()["complete_properties"] == 1
    assert not ledger.status()["complete"]


def test_expired_evidence_cannot_be_completed_or_restored_as_fresh(ledger, tmp_path):
    source = complete(ledger)
    ledger.sources[source.id].observed_at = (datetime.now(UTC) - timedelta(minutes=16)).isoformat()
    path = tmp_path / "ledger.json"
    ledger.save(path)
    restored = StayEvidence.restore(path, ledger.owner, ledger.run_id)
    assert not restored.status()["complete"]
    assert restored.result()["limitations"]
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="different owner"):
        StayEvidence.restore(path, "another-user", ledger.run_id)


def test_context_compaction_keeps_sources_recoverable(ledger):
    source = complete(ledger)
    messages = [{"role": "assistant", "tool_calls": [{"id": "capture", "function": {"name": "browser_stay_evidence"}}]},
                {"role": "tool", "tool_call_id": "capture", "content": json.dumps(ledger.read(source.id))},
                {"role": "tool", "name": "browser_stay_evidence", "content": json.dumps({"recorded": "rate", "progress": ledger.status()})}]
    compacted = compact_browser_history(messages)
    assert "JPY 600" not in compacted[1]["content"]
    assert source.id in compacted[1]["content"]
    assert "JPY 600" in ledger.read(source.id)["lines"]
    assert messages[1]["content"] != compacted[1]["content"]


def test_unrecorded_excerpts_survive_reads_and_failed_records():
    messages = [{"role": "tool", "name": "browser_stay_evidence", "content": json.dumps({"source": "s", "lines": lines})}
                for lines in ("1: Double room\n2: 90000", "100: Free cancellation", "200: Tax JPY 600")]
    messages.append({"role": "tool", "name": "browser_stay_evidence", "content": "Error: unknown candidate"})
    assert compact_browser_history(messages) == messages
    messages.append({"role": "tool", "name": "browser_stay_evidence", "content": json.dumps({"recorded": "rate", "missing": []})})
    compacted = compact_browser_history(messages)
    assert all("lines" not in json.loads(item["content"]) for item in compacted[:3])
    assert compacted[-1] == messages[-1]


def test_scoped_multi_page_quotes_pass_independent_capture_checks(ledger):
    source = ledger.capture(url(), TEXT, "doc")
    spans = fields(source)
    spans.pop("taxes_quote")
    item = ledger.record(spans, 90000)
    popup_url = url() + "#policy"
    popup = ledger.capture(popup_url, TEXT + "Modal content\n", "doc")
    ledger.record({"taxes_quote": fields(popup)["taxes_quote"]}, candidate_id=item.id)
    ledger.finished = True
    stay = StayRequest(**ledger.query.model_dump(exclude={"rooms"}), market="overseas")
    pages = [{"url": source.url, "text": source.text}, {"url": popup.url, "text": popup.text}]
    result = ledger.result()
    assert verify_comparison(json.dumps(result), stay, pages)["passed"]
    result["offers"][0]["evidence_refs"]["taxes_quote"]["rate_id"] = "wrong-rate"
    assert not verify_comparison(json.dumps(result), stay, pages)["passed"]


def test_partial_collection_blocks_completion_and_final_answer_is_canonical(ledger):
    evidence = TaskEvidence(browser_effect="view")
    target = resolve_intent_contract(ClassificationResult("browser", .99, 2, browser_effect="view"), .99)
    evidence.observe("browser_stay_evidence", CapabilityResult(success=True, metadata={"retrieval_acceptance": ledger.acceptance()}))
    assert evidence.blocker(target, 0) and not accepted_answer(evidence)
    complete(ledger)
    evidence.observe("browser_stay_evidence", CapabilityResult(success=True, metadata={"retrieval_acceptance": ledger.acceptance()}))
    assert evidence.blocker(target, 0) is None
    assert json.loads(accepted_answer(evidence)) == ledger.result()
    assert evidence.acceptance()["status"] == "pass"


@pytest.mark.parametrize("changes", [
    {"rooms%5B0%5D%5Badult%5D": "3"}, {"rooms%5B1%5D%5Badult%5D": "2"}, {"checkInDate": "2026-11-14"},
])
def test_independent_checker_rejects_a_rehashed_wrong_scope(ledger, changes):
    from rune.lodging.evidence import digest

    source = complete(ledger)
    wrong_url = url(**changes)
    result = ledger.result()
    result["offers"][0]["url"] = wrong_url
    for ref in result["offers"][0]["evidence_refs"].values():
        ref["url"] = wrong_url
        ref["source"] = digest(wrong_url + "\n" + source.text)
    stay = StayRequest(**ledger.query.model_dump(exclude={"rooms"}), market="overseas")
    checked = verify_comparison(json.dumps(result), stay, [{"url": wrong_url, "text": source.text}])
    assert not checked["passed"] and any("scoped citation" in error for error in checked["failures"])


@pytest.mark.parametrize("stop", ["normal", "budget", "partial", "cancelled"])
async def test_real_loop_uses_only_finished_evidence_and_exact_answer(monkeypatch, tmp_path, ledger, stop):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from rune.agent.loop import NativeAgentLoop
    from rune.agent.request_budget import BudgetExceeded
    from rune.agent.run_outcome import run_outcome

    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    complete(ledger)
    ledger.finished = stop != "partial"
    options = []

    class Stream:
        tool_budget_exhausted = False
        tool_round_limit_reached = False
        answer_reserve_reason = ""

        async def stream_text(self, **kwargs):
            await options[-1].on_tool_start("browser_stay_evidence", {"action": "finish"})
            await options[-1].on_tool_end("browser_stay_evidence", CapabilityResult(
                success=True, output=json.dumps(ledger.result()), metadata={"retrieval_acceptance": ledger.acceptance()}))
            if stop in {"budget", "partial", "cancelled"}:
                if stop == "cancelled":
                    loop._cancel_event.set()
                raise BudgetExceeded("No more model calls")
            yield "Model prose is not the canonical result."

        async def get_output(self): return "Model prose is not the canonical result."
        def usage(self): return SimpleNamespace(input_tokens=10, output_tokens=10)
        def get_failure_state(self): return {}, set()
        def all_messages(self): return []

    class Agent:
        def __init__(self, *args, **kwargs): pass

        @asynccontextmanager
        async def run_stream(self, *args, **kwargs):
            yield Stream()

    def build(opts):
        options.append(opts)
        return {}

    monkeypatch.setattr("rune.agent.loop.build_tool_set", build)
    monkeypatch.setattr("rune.agent.loop.LiteLLMAgent", Agent)
    loop = NativeAgentLoop()
    loop._workspace_root = str(tmp_path)
    classification = ClassificationResult("browser", .99, 2, browser_effect="view")
    trace = await loop._execute_loop("Compare one Sapporo hotel", "", [], 1, classification)
    assert run_outcome(trace).success is (stop in {"normal", "budget"})
    if stop in {"normal", "budget"}:
        assert json.loads(loop._last_answer_text) == ledger.result()
        assert trace.requirement_acceptance["status"] == "pass"
    elif stop == "partial":
        partial = json.loads(loop._last_answer_text)
        assert partial["offers"][0]["price_quote"] == TEXT.splitlines()[2]
        assert partial["limitations"]
