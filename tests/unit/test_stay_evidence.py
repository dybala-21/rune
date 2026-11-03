import json
from datetime import UTC, datetime, timedelta

import pytest

from rune.agent.browser_context import compact_browser_history
from rune.agent.goal_classifier import ClassificationResult
from rune.agent.intent_engine import resolve_intent_contract
from rune.agent.retrieval_result import accepted_answer
from rune.agent.task_evidence import TaskEvidence
from rune.lodging.evidence import FIELDS, Span, StayEvidence, StayQuery, digest
from rune.lodging.rates import Lines, RateCard, RateRegion
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


def capture(ledger, observed_url, text, document_id):
    """Synthetic DOM boundaries for ledger unit tests; browser tests use real nodes."""
    if text.splitlines()[0] != TEXT.splitlines()[0]:
        return ledger.capture(observed_url, text, document_id)
    lines = text.splitlines()
    rate_id = digest(observed_url.split("#")[0] + lines[1] + lines[2])
    rate = RateCard(id=rate_id, room_id="room-a", index=0, room=lines[1],
                    room_lines=Lines(start=2,end=2), lines=Lines(start=3,end=3), text=lines[2],
                    current_price=90000, price_text="90,000", price_label="Standard", signature=rate_id)
    regions = [RateRegion(rate_id=rate_id,kind=key,lines=Lines(start=n,end=n),selector="#"+key)
               for key,n in (("cancellation_quote",7),("price_conditions_quote",8),("taxes_quote",9))]
    return ledger.capture(observed_url,text,document_id,rates=[rate],regions=regions)


def complete(ledger):
    source = capture(ledger, url(), TEXT, "document-a")
    ledger.record(fields(source), 90000)
    ledger.finished = True
    return source


def bound_evidence(ledger, request="Compare one Sapporo hotel"):
    from rune.agent.browser_requirements import request_key
    return TaskEvidence(browser_effect="view", request_texts=(request,),
                        stay_query=ledger.query.model_dump(mode="json"), stay_query_key=request_key((request,)))


def test_missing_fields_are_preserved_and_filled_without_recapturing(ledger):
    source = capture(ledger, url(), TEXT, "doc-a")
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
        capture(ledger, url(**change), TEXT, "doc")
    assert not ledger.sources


def test_another_property_cannot_supply_the_missing_tax_policy(ledger):
    source = capture(ledger, url(), TEXT, "a")
    spans = fields(source)
    spans.pop("taxes_quote")
    item = ledger.record(spans, 90000)
    other = capture(ledger, url("TDP-2"), TEXT, "b")
    with pytest.raises(ValueError, match="selected property"):
        ledger.record({"taxes_quote": fields(other)["taxes_quote"]}, candidate_id=item.id)
    assert ledger.missing(item) == ["taxes_quote"]


def test_room_price_changes_cannot_silently_overwrite_a_rate(ledger):
    source = complete(ledger)
    item = next(iter(ledger.candidates.values()))
    with pytest.raises(ValueError, match="price change"):
        ledger.record({"taxes_quote": fields(source)["taxes_quote"]}, 100000, item.id)
    new = capture(ledger, url(), TEXT.replace("Double room", "Twin room"), "a")
    with pytest.raises(ValueError, match="silently replaced"):
        ledger.record({"room": fields(new)["room"]}, candidate_id=item.id)
    assert ledger.result()["offers"][0]["room"] == "Double room"


def test_quotes_are_exact_ranges_and_cannot_be_fabricated(ledger):
    source = capture(ledger, url(), TEXT, "doc")
    with pytest.raises(ValueError, match="outside"):
        ledger.record({"name": Span(source=source.id, start=999, end=1000)}, 90000)
    spans = fields(source)
    spans["price_quote"] = spans["location_quote"]
    with pytest.raises(ValueError, match="price quote"):
        ledger.record(spans, 90000)
    assert not ledger.candidates


def test_property_teaser_cannot_establish_room_rate_conditions(ledger):
    source = capture(ledger, url().replace("/list?", "?"), TEXT, "doc")
    with pytest.raises(ValueError, match="teaser/coupon banner"):
        ledger.record(fields(source), 90000)
    assert not ledger.candidates


def test_literal_find_recovers_only_the_requested_source_region(ledger):
    source = capture(ledger, url(), "Filler\n" * 200 + TEXT, "doc")
    found = ledger.find(source.id, "local accommodation tax")
    assert len(found["matches"]) == 1
    assert "209: Pay local accommodation tax" in found["matches"][0]["lines"]
    assert len(found["matches"][0]["lines"].splitlines()) < 9
    assert not ledger.find(source.id, "nonexistent phrase")["matches"]
    assert len(ledger.sources) == 1


def test_preview_keeps_exact_line_numbers_and_omitted_policy_is_recoverable(ledger):
    text = "Header\n" + "Filler\n" * 249 + "Mandatory policy\n" + "Other rooms\n" * 249 + TEXT
    source = capture(ledger, url(), text, "doc")
    preview = ledger.preview(source.id)
    assert preview["omitted_lines"] > 0
    assert sum(len(item["lines"]) for item in preview["excerpts"]) <= 8000
    for item in preview["excerpts"]:
        for line in item["lines"].splitlines():
            number, value = line.split(": ", 1)
            assert text.splitlines()[int(number) - 1] == value
    assert "Mandatory policy" not in json.dumps(preview)
    assert "251: Mandatory policy" in ledger.read(source.id, 251, 1)["lines"]
    assert source.text == text and not ledger.status()["complete"]


def test_compact_progress_preserves_candidate_identity_and_missing_conditions(ledger):
    source = capture(ledger, url(), TEXT, "doc")
    spans = fields(source)
    spans.pop("taxes_quote")
    candidate = ledger.record(spans, 90000)
    progress = ledger.status(compact=True)
    assert "sources" not in progress and "query" not in progress
    assert progress["candidates"][0]["id"] == candidate.id
    assert progress["candidates"][0]["missing"] == ["taxes_quote"]
    assert not progress["complete"]
    assert ledger.status()["sources"][0]["id"] == source.id


def test_entrypoint_encodes_fixed_request_without_inventing_property_ids(ledger):
    from urllib.parse import parse_qs, urlsplit

    from rune.lodging.nol import entrypoints, source_guidance

    ledger.query.city = "Sapporo & 函館?x=1"
    entry = urlsplit(entrypoints(ledger.query)["overseas_search"])
    assert entry.hostname == "nol.yanolja.com"
    params = parse_qs(entry.query)
    assert params["q"] == [ledger.query.city]
    assert params["checkInDate"] == ["2026-11-15"] and params["checkOutDate"] == ["2026-11-16"]
    assert params["adultCount"] == ["2"] and params["roomCount"] == ["1"]
    assert "TDP-" not in entry.geturl()
    assert not source_guidance(url().replace("/list?", "?"))["rate_source"]
    assert source_guidance(url())["rate_source"]


def test_room_list_entry_preserves_observed_property_and_stay_scope():
    from rune.lodging.nol import source_guidance

    overview = url("TDP-2").replace("/list?", "?")
    assert source_guidance(overview + "#property-modal")["room_list_url"] == url("TDP-2")
    assert "room_list_url" not in source_guidance(url())
    assert "room_list_url" not in source_guidance(overview.replace("nol.yanolja.com", "example.com"))


def test_capture_replaces_current_page_body_but_preserves_every_control():
    controls = '--- Interactive Elements (2/2) ---\n[e0] button "Details"\n[e1] link "Policy" href="https://example.com/policy"'
    view = {"role": "tool", "name": "browser_observe", "content": f"URL: {url()}\n" + "Body " * 1000 + "\n" + controls}
    capture = {"role": "tool", "name": "browser_stay_evidence", "content": json.dumps({
        "source": "original", "captured": True, "url": url(), "excerpts": [{"lines": "1: Body"}]})}
    compacted = compact_browser_history([view, capture])
    assert compacted[0]["content"].endswith(controls)
    assert len(compacted[0]["content"]) < 700 and "source original" in compacted[0]["content"]
    assert compacted[1] == capture and "Body Body" in view["content"]
    assert compact_browser_history(compacted) == compacted


def test_superseded_progress_clears_without_losing_unrecorded_lines_or_errors():
    payload = {"source": "one", "lines": "1: Unrecorded cancellation policy", "progress": {"sources": ["a"]}}
    messages = [{"role": "tool", "name": "browser_stay_evidence", "content": json.dumps(payload)},
                {"role": "tool", "name": "browser_stay_evidence", "content": '{"progress":{"sources":["a","b"]}}'}]
    compacted = compact_browser_history(messages)
    assert "progress" not in json.loads(compacted[0]["content"])
    assert json.loads(compacted[0]["content"])["lines"] == payload["lines"]
    assert compacted[-1] == messages[-1]
    assert compact_browser_history(compacted) == compacted
    failed = {"role": "tool", "name": "browser_act", "content": "[ERROR] Denied " + "details " * 200}
    assert compact_browser_history([failed, *messages, {"role": "tool", "name": "browser_stay_evidence",
        "content": '{"recorded":"rate"}'}, {"role": "tool", "name": "browser_observe", "content": "Current"}])[0] == failed


def test_captured_ui_is_retired_only_after_new_controls_and_same_url():
    old = {"role": "tool", "name": "browser_observe", "content": f"URL: {url()}\n" + "Body " * 500}
    captured = {"role": "tool", "name": "browser_stay_evidence", "content": json.dumps({
        "source": "exact-source", "captured": True, "url": url(), "excerpts": [{"lines": "1: Saved body"}]})}
    assert compact_browser_history([old, captured]) == [old, captured]
    extracted = {"role": "tool", "name": "browser_extract", "content": "Some new facts without a new view"}
    assert compact_browser_history([old, captured, extracted])[0] == old
    latest = {"role": "tool", "name": "browser_observe", "content": "URL: https://example.com/\nCurrent controls"}
    compacted = compact_browser_history([old, captured, latest])
    assert "Source: exact-source" in compacted[0]["content"]
    assert len(compacted[0]["content"]) < 700
    assert compacted[-1] == latest and compacted[1] == captured
    other = {**captured, "content": captured["content"].replace("TDP-1", "TDP-2")}
    assert "Body Body" in compact_browser_history([old, other, latest])[0]["content"]


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
    evidence = bound_evidence(ledger)
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
    other = capture(ledger, url(), TEXT.replace("Double room", "Twin room"), "doc")
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
    source = capture(ledger, url(), TEXT, "doc")
    spans = fields(source)
    spans.pop("taxes_quote")
    item = ledger.record(spans, 90000)
    popup_url = url() + "#policy"
    popup = capture(ledger, popup_url, TEXT + "Modal content\n", "doc")
    ledger.record({"taxes_quote": fields(popup)["taxes_quote"]}, candidate_id=item.id)
    ledger.finished = True
    stay = StayRequest(**ledger.query.model_dump(exclude={"rooms"}), market="overseas")
    pages = [{"url": source.url, "text": source.text}, {"url": popup.url, "text": popup.text}]
    result = ledger.result()
    assert verify_comparison(json.dumps(result), stay, pages, check_dom=False)["passed"]
    result["offers"][0]["evidence_refs"]["taxes_quote"]["rate_id"] = "wrong-rate"
    assert not verify_comparison(json.dumps(result), stay, pages, check_dom=False)["passed"]


def test_partial_collection_blocks_completion_and_final_answer_is_canonical(ledger):
    evidence = bound_evidence(ledger)
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
    from rune.agent.browser_requirements import request_key
    classification = ClassificationResult("browser", .99, 2, browser_effect="view",
        stay_query=ledger.query.model_dump(mode="json"), stay_query_key=request_key(("Compare one Sapporo hotel",)))
    trace = await loop._execute_loop("Compare one Sapporo hotel", "", [], 1, classification)
    assert run_outcome(trace).success is (stop in {"normal", "budget"})
    if stop in {"normal", "budget"}:
        assert json.loads(loop._last_answer_text) == ledger.result()
        assert trace.requirement_acceptance["status"] == "pass"
    elif stop == "partial":
        partial = json.loads(loop._last_answer_text)
        assert partial["offers"][0]["price_quote"] == TEXT.splitlines()[2]
        assert partial["limitations"]
