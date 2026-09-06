"""Numbers in an answer that appear in nothing the run fetched.

A run reporting "SanDisk +22%, Micron +15%" cited nothing; the real figures
were 6.7-11.6% and 4.2-5%. Nothing caught it — the grounding requirement asks
whether a search happened, not whether the answer follows from it, and the
citation checks only inspect claims carrying a URL.

Deterministic comparison is both the cheap and the accurate choice for
numbers: ~1ms against 500ms+ for an LLM that is wrong on 10-30% of simple
numeric comparisons. This flags, never blocks — a derived figure is
legitimate while appearing verbatim nowhere.
"""

from __future__ import annotations

import pytest

from rune.agent.output_integrity import (
    numeric_claims,
    output_integrity_enabled,
    unsourced_numbers,
)

# ---------------------------------------------------------------------------
# Step 1: the URL check runs by default now.
# ---------------------------------------------------------------------------

def test_output_integrity_is_on_by_default(monkeypatch):
    monkeypatch.delenv("RUNE_OUTPUT_INTEGRITY", raising=False)

    assert output_integrity_enabled() is True


def test_output_integrity_can_still_be_turned_off(monkeypatch):
    monkeypatch.setenv("RUNE_OUTPUT_INTEGRITY", "0")

    assert output_integrity_enabled() is False


# ---------------------------------------------------------------------------
# Extracting the claims
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SanDisk 약 +22%, Micron 약 +15%", {"22", "15"}),
        ("closed at $1,016.59", {"1016.59"}),
        # Trailing zeros are normalised away so 6.1% and 6.10% compare equal.
        ("up 6.10% on the day", {"6.1"}),
        ("rose 8 percent", {"8"}),
    ],
)
def test_numbers_are_pulled_out_of_prose(text, expected):
    assert numeric_claims(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "see section 3 for details",          # a reference, not a measurement
        "Micron (MU) and SanDisk (SNDK)",     # ticker text
        "gpt-5.6-sol handled the request",    # a version
        "on 2026-09-04 the market closed",    # a date
        "",
    ],
)
def test_non_measurements_are_left_alone(text):
    assert numeric_claims(text) == set()


# ---------------------------------------------------------------------------
# Matching against what the run actually retrieved
# ---------------------------------------------------------------------------

def _fetched(*chunks):
    return [{"role": "tool", "content": c} for c in chunks]


def test_a_number_the_sources_never_mention_is_flagged():
    answer = "SanDisk rose about +22% today."
    messages = _fetched("SanDisk Rises 8%, Micron Gains 5% as NAND pricing firms")

    assert unsourced_numbers(answer, messages) == ["22"]


def test_a_number_present_in_a_source_passes():
    answer = "SanDisk rose about 8% today."
    messages = _fetched("SanDisk Rises 8%, Micron Gains 5%")

    assert unsourced_numbers(answer, messages) == []


def test_formatting_differences_do_not_count_as_missing():
    """6.1% and 6.10% are the same claim; so are 1,016.59 and 1016.59."""
    answer = "MU closed at $1,016.59, up 6.10%."
    messages = _fetched("MU 1016.59 (+6.1%)")

    assert unsourced_numbers(answer, messages) == []


def test_a_figure_derived_from_the_sources_is_not_flagged():
    """A percentage computed from two quoted prices appears verbatim nowhere."""
    answer = "SNDK went from $1,554.99 to $1,740.00, a gain of 11.90%."
    messages = _fetched("SanDisk previous close 1554.99, last 1740.00")

    assert unsourced_numbers(answer, messages) == []


def test_nothing_retrieved_means_nothing_to_check():
    """Conservative: with no sources there is no basis to call a number wrong."""
    answer = "SanDisk rose about +22% today."

    assert unsourced_numbers(answer, []) == []


def test_search_arguments_count_as_retrieved_text():
    answer = "The figure was 8%."
    messages = [{
        "role": "assistant",
        "tool_calls": [{"function": {"name": "web_search",
                                     "arguments": '{"query":"SanDisk 8% rise"}'}}],
    }]

    assert unsourced_numbers(answer, messages) == []


def test_the_real_failure_is_caught():
    """The answer that started this, against the articles that existed."""
    answer = (
        "**SanDisk:** 약 **+22%**\n"
        "**SK hynix:** 약 **+16%**\n"
        "**Micron:** 약 **+15%**"
    )
    messages = _fetched(
        "SanDisk Rises 8%, Micron Gains 5%: Is the NAND Pricing Cycle Still "
        "Accelerating?",
        "SanDisk Corporation Stock (SNDK) Moved Up by 6.67% on Sep 4",
    )

    flagged = unsourced_numbers(answer, messages)

    assert set(flagged) == {"22", "16", "15"}


def test_a_rounded_quotation_is_not_flagged():
    """"about 6.7%" for a source saying 6.67% is normal quoting, not invention."""
    answer = "SanDisk was up about 6.7% today."
    messages = _fetched("SanDisk Corporation Stock (SNDK) Moved Up by 6.67% on Sep 4")

    assert unsourced_numbers(answer, messages) == []


@pytest.mark.parametrize(
    ("answer", "sources"),
    [
        ("SanDisk rose 8% and Micron gained 5%.", "SanDisk Rises 8%, Micron Gains 5%"),
        ("MU closed at $1,016.59 (+6.1%).", "MU 1016.59 up 6.10 percent"),
        ("Set max_tokens=4096 and retry.", "the API accepts max_tokens 4096"),
        ("gpt-5.6-sol routes to /v1/responses.", "gpt-5.6-sol needs the responses endpoint"),
        ("See section 3 and step 2.", "section 3 covers it"),
        ("Memory stocks rallied on AI demand.", "AI demand lifts memory"),
        ("The fee is $1,200.", "a fee of 1200 dollars"),
        ("On 2026-09-04 the market closed higher.", "Sep 4 2026 close"),
    ],
)
def test_legitimate_answers_are_not_flagged(answer, sources):
    """Over-flagging is the failure mode that would make this worth removing."""
    assert unsourced_numbers(answer, _fetched(sources)) == []


def test_only_the_invented_figure_is_flagged():
    """A partly-grounded answer must not be condemned whole."""
    answer = "SanDisk rose 8%, and SK hynix rose 16%."
    messages = _fetched("SanDisk Rises 8% as NAND pricing firms")

    assert unsourced_numbers(answer, messages) == ["16"]


# ---------------------------------------------------------------------------
# The finding has to reach a person, and it must never fail a run on its own.
# ---------------------------------------------------------------------------

def test_the_trace_carries_the_figures():
    from rune.types import CompletionTrace

    trace = CompletionTrace()

    assert trace.unsourced_numbers == []
    trace.unsourced_numbers = ["22", "15"]
    assert trace.unsourced_numbers == ["22", "15"]


def test_the_trust_payload_carries_them_to_the_app():
    from rune.api.server import build_trust_payload
    from rune.types import CompletionTrace

    trace = CompletionTrace(reason="completed")
    trace.unsourced_numbers = ["22", "15"]

    payload = build_trust_payload(trace)

    assert payload["unsourcedNumbers"] == ["22", "15"]
    assert payload["verified"] is True, (
        "an unsourced figure is a caveat on the answer, not a failed run"
    )


def test_a_clean_answer_carries_nothing():
    from rune.api.server import build_trust_payload
    from rune.types import CompletionTrace

    assert build_trust_payload(CompletionTrace(reason="completed"))["unsourcedNumbers"] == []


def test_a_number_dump_cannot_blow_up_the_check():
    """Scanning is linear in source text and deriving is quadratic in values.

    A run that fetched several large pages must not pay seconds for a check
    that exists to add a caveat.
    """
    import time

    answer = "SanDisk closed at $1,740.00, up 11.9%." * 40
    messages = _fetched(" ".join(f"{i}.{i % 100:02d}%" for i in range(20_000)))
    unsourced_numbers(answer, messages)

    start = time.perf_counter()
    unsourced_numbers(answer, messages)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 100, f"{elapsed_ms:.1f} ms on a number dump"


def test_a_derived_percentage_needs_real_magnitude():
    """With enough values some pair is always near zero; that is not derivation."""
    from rune.agent.output_integrity import _derivable

    known = {"100", "100.05", "200"}

    assert _derivable("100", known) is True      # 100 -> 200
    assert _derivable("0.0000001", known) is False
    assert _derivable("22", known) is False


def test_the_check_stays_on_the_fast_path():
    """This runs at finalize on every answer; CLAUDE.md wants hot paths sub-ms."""
    import time

    answer = (
        "SanDisk rose 8% to $1,740.00 while Micron gained 5% to $1,016.59. "
        "Western Digital added 3.2%, and the sector index moved 2.75%."
    ) * 5
    messages = _fetched(
        "SanDisk Rises 8%, Micron Gains 5%. WDC 3.2%. Index 2.75%. "
        "Prices 1740.00 and 1016.59." * 20
    )
    for _ in range(20):
        unsourced_numbers(answer, messages)

    start = time.perf_counter()
    for _ in range(200):
        unsourced_numbers(answer, messages)
    per_call_ms = (time.perf_counter() - start) / 200 * 1000

    assert per_call_ms < 1.0, f"{per_call_ms:.3f} ms per check"
