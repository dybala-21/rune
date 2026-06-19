"""Unit tests for output-integrity citation checks (deterministic, model-free)."""

from __future__ import annotations

from rune.agent import output_integrity as oi


def test_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("RUNE_OUTPUT_INTEGRITY", raising=False)
    assert oi.output_integrity_enabled() is False
    monkeypatch.setenv("RUNE_OUTPUT_INTEGRITY", "1")
    assert oi.output_integrity_enabled() is True


def test_fabricated_url_flagged():
    # A is in a tool result, B was fetched; output cites A + C (C never retrieved).
    messages = [
        {"role": "tool", "content": "results: https://a.com/x and more"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "web_fetch", "arguments": '{"url": "https://b.com/y"}'}}
        ]},
    ]
    output = "See [A](https://a.com/x), [B](https://b.com/y), [C](https://c.com/z)."
    assert oi.fabricated_citations(output, messages) == ["https://c.com/z"]


def test_all_grounded_passes():
    messages = [{"role": "tool", "content": "https://a.com https://b.com"}]
    output = "Sources: https://a.com https://b.com"
    assert oi.fabricated_citations(output, messages) == []


def test_no_citations_passes():
    messages = [{"role": "tool", "content": "https://a.com"}]
    assert oi.fabricated_citations("a report with no links", messages) == []


def test_no_retrieval_skips():
    # No tool results / fetches -> cannot determine retrieval -> skip (no block).
    messages = [{"role": "assistant", "content": "I think the answer is X"}]
    assert oi.fabricated_citations("cited https://made-up.com", messages) == []


def test_file_write_content_not_counted_as_retrieval():
    # The agent's own file_write (with a citation) must NOT count as retrieval,
    # so a URL only present there + in the output is still flagged.
    messages = [
        {"role": "tool", "content": "search result: https://real.com"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "file_write",
                          "arguments": '{"content": "cite https://fake.com"}'}}
        ]},
    ]
    output = "Refs: https://real.com https://fake.com"
    assert oi.fabricated_citations(output, messages) == ["https://fake.com"]


def test_trailing_punctuation_normalized():
    messages = [{"role": "tool", "content": "https://a.com/page"}]
    output = "see https://a.com/page."
    assert oi.fabricated_citations(output, messages) == []


def test_percent_encoded_retrieval_matches_decoded_citation():
    # A real false-positive from a live run: the page was fetched in
    # percent-encoded form but cited with the decoded non-ASCII path. Same URL,
    # so it must NOT be flagged as fabricated.
    fetched = (
        "https://kr.benzinga.com/news/usa/trading/"
        "420%EC%96%B5-%EB%8B%AC%EB%9F%AC-%EC%9E%AD%ED%8C%9F"
    )
    cited = "https://kr.benzinga.com/news/usa/trading/420억-달러-잭팟"
    messages = [{"role": "assistant", "tool_calls": [
        {"function": {"name": "web_fetch", "arguments": f'{{"url": "{fetched}"}}'}}
    ]}]
    output = f"Source: {cited}"
    assert oi.fabricated_citations(output, messages) == []


def test_nudge_lists_urls():
    msg = oi.build_nudge(["https://x.com", "https://y.com"])
    assert "https://x.com" in msg and "https://y.com" in msg
    assert "cited but never appeared" in msg
