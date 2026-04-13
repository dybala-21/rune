"""Unit tests for the strict verb parser."""

from __future__ import annotations

from rune.agent.advisor.parser import (
    MAX_PLAN_STEPS,
    MAX_PLAN_WORDS,
    format_injection,
    parse,
)


class TestParseHappyPath:
    def test_continue_with_steps(self):
        text = "NEXT: continue\n1. read main.py\n2. write helper\n3. run tests"
        d = parse(text)
        assert d.action == "continue"
        assert d.plan_steps == ["read main.py", "write helper", "run tests"]
        assert d.target_tool is None

    def test_retry_tool_with_target(self):
        text = "NEXT: retry_tool:file_read\n1. re-read with bigger offset"
        d = parse(text)
        assert d.action == "retry_tool"
        assert d.target_tool == "file_read"
        assert len(d.plan_steps) == 1

    def test_abort(self):
        d = parse("NEXT: abort\n1. hard failures unrecoverable")
        assert d.action == "abort"

    def test_switch_approach(self):
        d = parse("NEXT: switch_approach\n1. try grep instead of glob")
        assert d.action == "switch_approach"

    def test_need_reconcile(self):
        d = parse("NEXT: need_reconcile\n1. evidence contradicts prior advice")
        assert d.action == "need_reconcile"

    def test_bullet_style_steps(self):
        text = "NEXT: continue\n- first\n- second\n- third"
        d = parse(text)
        assert len(d.plan_steps) == 3

    def test_parenthesis_numbering(self):
        text = "NEXT: continue\n1) one\n2) two"
        d = parse(text)
        assert d.plan_steps == ["one", "two"]


class TestParseDegraded:
    def test_empty_text_returns_continue_noop(self):
        d = parse("")
        assert d.action == "continue"
        assert d.plan_steps == []
        assert d.error_code == "empty_response"

    def test_missing_verb_still_captures_steps(self):
        text = "1. do the first thing\n2. then the second"
        d = parse(text)
        assert d.action == "continue"
        assert d.plan_steps == ["do the first thing", "then the second"]

    def test_invalid_verb_falls_back_to_continue(self):
        text = "NEXT: refactor_world\n1. do nothing"
        d = parse(text)
        assert d.action == "continue"

    def test_verb_case_insensitive(self):
        d = parse("NEXT: ABORT\n1. stop")
        assert d.action == "abort"

    def test_stops_at_max_steps(self):
        text = "NEXT: continue\n" + "\n".join(
            f"{i}. step {i}" for i in range(1, 10)
        )
        d = parse(text)
        assert len(d.plan_steps) == MAX_PLAN_STEPS

    def test_caps_total_words(self):
        word = "word"
        big_step = " ".join([word] * (MAX_PLAN_WORDS + 20))
        text = f"NEXT: continue\n1. {big_step}"
        d = parse(text)
        # Either the step is truncated or dropped — total words bounded
        total = sum(len(s.split()) for s in d.plan_steps)
        assert total <= MAX_PLAN_WORDS


class TestParseMetadata:
    def test_preserves_raw_text(self):
        raw = "NEXT: continue\n1. step"
        d = parse(raw)
        assert d.raw_text == raw

    def test_passes_through_usage(self):
        d = parse(
            "NEXT: continue\n1. x",
            trigger="stuck",
            provider="anthropic",
            model="claude-opus-4-6",
            input_tokens=500,
            output_tokens=120,
            latency_ms=1500,
        )
        assert d.trigger == "stuck"
        assert d.provider == "anthropic"
        assert d.model == "claude-opus-4-6"
        assert d.input_tokens == 500
        assert d.output_tokens == 120
        assert d.latency_ms == 1500


class TestFormatInjection:
    def test_renders_full_injection(self):
        d = parse("NEXT: retry_tool:file_read\n1. read main.py\n2. try smaller offset")
        s = format_injection(d)
        assert "[Advisor]" in s
        assert "RETRY_TOOL" in s
        assert "file_read" in s
        assert "1. read main.py" in s

    def test_continue_noop_is_empty(self):
        d = parse("")
        assert format_injection(d) == ""

    def test_abort_with_no_steps(self):
        d = parse("NEXT: abort")
        assert "ABORT" in format_injection(d)

    def test_stuck_trigger_uses_directive_language(self):
        d = parse(
            "NEXT: retry_tool:web_fetch\n1. fetch the API docs",
            trigger="stuck",
        )
        s = format_injection(d)
        assert "DIRECTIVE" in s
        assert "stalled" in s
        assert "step 1 IMMEDIATELY" in s
        assert "web_fetch" in s

    def test_reconcile_trigger_uses_mandatory_language(self):
        d = parse(
            "NEXT: switch_approach\n1. try a different URL",
            trigger="reconcile",
        )
        s = format_injection(d)
        assert "DIRECTIVE" in s
        assert "not followed" in s
        assert "different approach" in s

    def test_early_trigger_uses_informational(self):
        d = parse(
            "NEXT: continue\n1. read the config file first",
            trigger="early",
        )
        s = format_injection(d)
        assert "[Advisor]" in s
        assert "DIRECTIVE" not in s
