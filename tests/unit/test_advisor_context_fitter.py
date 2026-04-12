"""Unit tests for budget-aware payload construction."""

from __future__ import annotations

from rune.agent.advisor.context_fitter import (
    DEFAULT_TARGET_TOKENS,
    build_payload,
    estimate_tokens,
)
from rune.agent.advisor.protocol import AdvisorRequest


def _req(**overrides) -> AdvisorRequest:
    base = dict(
        trigger="stuck",
        goal="Implement a worker pool in Go",
        classification_summary="code_modify complex_coding",
        activity_phase="implementation",
        step=12,
        token_budget_frac=0.42,
        evidence_snapshot={"reads": 3, "writes": 1, "web_fetches": 0},
        gate_state={
            "outcome": "blocked",
            "missing_requirement_ids": ["R1", "R2"],
            "hard_failures": ["TypeError foo"],
        },
        stall_state={"consecutive": 2, "cumulative": 4},
        recent_messages=[
            {"role": "user", "content": "make it concurrent"},
            {"role": "assistant", "content": "reading main.go now"},
        ],
        files_written=["main.go"],
        last_advisor_note=None,
    )
    base.update(overrides)
    return AdvisorRequest(**base)  # type: ignore[arg-type]


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_roughly_chars_over_3_5(self):
        t = "a" * 70
        # Expect roughly 20 tokens
        assert 15 <= estimate_tokens(t) <= 25


class TestBuildPayload:
    def test_always_contains_p0(self):
        r = _req()
        out = build_payload(r)
        assert "TRIGGER: stuck" in out
        assert "GOAL:" in out
        assert "EVIDENCE:" in out
        assert "GATE:" in out
        assert "main.go" in out

    def test_contains_recent_messages_when_budget_allows(self):
        r = _req()
        out = build_payload(r, target_tokens=DEFAULT_TARGET_TOKENS)
        assert "RECENT_MESSAGES:" in out
        assert "make it concurrent" in out

    def test_tight_budget_drops_lower_priority(self):
        r = _req(
            recent_messages=[
                {"role": "user", "content": "x" * 500}
                for _ in range(5)
            ],
            last_advisor_note="note " * 200,
        )
        out = build_payload(r, target_tokens=200)
        # P0 is always kept
        assert "TRIGGER:" in out
        assert "GOAL:" in out
        # Lower priority may be partially or fully dropped; payload size bounded
        assert len(out) < 2500

    def test_empty_recent_messages(self):
        r = _req(recent_messages=[])
        out = build_payload(r)
        assert "RECENT_MESSAGES:" not in out

    def test_files_written_tail_only(self):
        files = [f"file_{i}.py" for i in range(40)]
        r = _req(files_written=files)
        out = build_payload(r)
        # Only the tail-20 should appear
        assert "file_39.py" in out
        assert "file_0.py" not in out

    def test_stall_state_rendered(self):
        r = _req()
        out = build_payload(r)
        assert "STALL:" in out

    def test_reconcile_note_rendered(self):
        r = _req(last_advisor_note="prior advice: use channels")
        out = build_payload(r)
        assert "LAST_ADVISOR_NOTE" in out
        assert "channels" in out
