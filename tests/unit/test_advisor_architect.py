"""Unit tests for advisor architect mode (apply_patch action + patch parsing)."""

from __future__ import annotations

from rune.agent.advisor.compliance import build_pending
from rune.agent.advisor.parser import format_injection, parse
from rune.agent.advisor.tiers import resolve_advisor_mode


class TestParsePatch:
    def test_apply_patch_extracts_file_and_code(self):
        raw = """NEXT: apply_patch:fix_error_message
FILE: /tmp/ws/timeparse.py
```python
def parse_time(s: str) -> int:
    if not s:
        raise ValueError('empty')
    return 0
```
"""
        d = parse(raw, trigger="stuck")
        assert d.action == "apply_patch"
        assert d.patch_target_file == "/tmp/ws/timeparse.py"
        assert "def parse_time" in d.patch
        assert "raise ValueError('empty')" in d.patch

    def test_apply_patch_missing_file_falls_back_to_continue(self):
        raw = """NEXT: apply_patch:nofile
```python
def f(): pass
```
"""
        d = parse(raw, trigger="stuck")
        # Missing FILE: marker → degrade gracefully
        assert d.action == "continue"
        assert d.patch is None

    def test_apply_patch_missing_code_block_falls_back(self):
        raw = "NEXT: apply_patch:nocode\nFILE: /tmp/x.py\n(no code here)\n"
        d = parse(raw, trigger="stuck")
        assert d.action == "continue"
        assert d.patch is None

    def test_oversized_patch_is_rejected(self):
        big = "x = 1\n" * 50_000  # ~300KB > 200KB cap
        raw = f"NEXT: apply_patch:big\nFILE: /tmp/big.py\n```python\n{big}```\n"
        d = parse(raw, trigger="stuck")
        assert d.action == "continue"
        assert d.patch is None

    def test_non_patch_action_ignores_code_block(self):
        raw = """NEXT: continue
1. read the file
```python
# example code in advisor output
```
"""
        d = parse(raw, trigger="early")
        assert d.action == "continue"
        assert d.patch is None


class TestFormatInjectionPatch:
    def test_patch_injection_renders_mandatory(self):
        raw = """NEXT: apply_patch:fix
FILE: /ws/x.py
```python
print("ok")
```
"""
        d = parse(raw, trigger="stuck")
        s = format_injection(d)
        assert "MANDATORY" in s
        assert "/ws/x.py" in s
        assert "file_write" in s
        assert "<PATCH>" in s
        assert 'print("ok")' in s

    def test_patch_overrides_stuck_directive_format(self):
        """apply_patch should take precedence over trigger-based formatting."""
        raw = """NEXT: apply_patch:x
FILE: /ws/x.py
```
pass
```
"""
        d = parse(raw, trigger="reconcile")
        s = format_injection(d)
        assert "MANDATORY" in s
        assert "DIRECTIVE" not in s


class TestArchitectMode:
    def test_weak_openai_pair_picks_architect(self):
        mode = resolve_advisor_mode(
            executor_provider="openai",
            executor_tier=45,  # gpt-4o-mini
            advisor_provider="anthropic",
            native_eligible=False,
        )
        assert mode == "architect"

    def test_strong_executor_picks_advice_only(self):
        mode = resolve_advisor_mode(
            executor_provider="openai",
            executor_tier=75,  # gpt-4o
            advisor_provider="anthropic",
            native_eligible=False,
        )
        assert mode == "advice_only"

    def test_anthropic_native_always_preferred(self):
        mode = resolve_advisor_mode(
            executor_provider="anthropic",
            executor_tier=40,
            advisor_provider="anthropic",
            native_eligible=True,
        )
        assert mode == "native"

    def test_weak_ollama_picks_architect(self):
        mode = resolve_advisor_mode(
            executor_provider="ollama",
            executor_tier=30,
            advisor_provider="anthropic",
            native_eligible=False,
        )
        assert mode == "architect"


class TestContextFitterFileContents:
    def test_file_contents_rendered_in_payload(self):
        from rune.agent.advisor.context_fitter import build_payload
        from rune.agent.advisor.protocol import AdvisorRequest

        req = AdvisorRequest(
            trigger="stuck",
            goal="fix parse_time",
            classification_summary="code_modify complex=True",
            activity_phase="implementation",
            step=5,
            token_budget_frac=0.4,
            evidence_snapshot={"writes": 1},
            gate_state=None,
            stall_state={},
            recent_messages=[],
            files_written=["/ws/x.py"],
            file_contents={"/ws/x.py": "def broken():\n    pass\n"},
        )
        payload = build_payload(req)
        assert "FILE_CONTENTS:" in payload
        assert "/ws/x.py" in payload
        assert "def broken" in payload

    def test_empty_file_contents_omitted(self):
        from rune.agent.advisor.context_fitter import build_payload
        from rune.agent.advisor.protocol import AdvisorRequest

        req = AdvisorRequest(
            trigger="early", goal="x", classification_summary="",
            activity_phase="exploration", step=1, token_budget_frac=0.1,
            evidence_snapshot={}, gate_state=None, stall_state={},
            recent_messages=[], files_written=[],
        )
        payload = build_payload(req)
        assert "FILE_CONTENTS:" not in payload

    def test_oversized_file_truncated(self):
        from rune.agent.advisor.context_fitter import build_payload
        from rune.agent.advisor.protocol import AdvisorRequest

        big = "x = 1\n" * 3000  # ~18KB > 10KB per-file cap
        req = AdvisorRequest(
            trigger="stuck", goal="", classification_summary="",
            activity_phase="", step=1, token_budget_frac=0.0,
            evidence_snapshot={}, gate_state=None, stall_state={},
            recent_messages=[], files_written=["/ws/big.py"],
            file_contents={"/ws/big.py": big},
        )
        payload = build_payload(req)
        assert "[truncated]" in payload


class TestPendingAdviceForPatch:
    def test_apply_patch_expects_file_write(self):
        raw = """NEXT: apply_patch:fix
FILE: /ws/x.py
```python
x = 1
```
"""
        d = parse(raw, trigger="stuck")
        pa = build_pending(d, step=5, evidence_total=10, hard_failure_count=0)
        assert pa.expected_tool == "file_write"
        assert pa.expected_action == "apply_patch"
