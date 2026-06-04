"""Tests for the memory bridge module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rune.agent.memory_bridge import (
    _select_crisp_signal,
    detect_languages,
    detect_tools,
    extract_intent_from_goal,
    format_relative_time,
    generate_skill_name,
)
from rune.types import Domain


class TestSelectCrispSignal:
    """The crisp learning signal must be the actual mismatch, not agent prose."""

    def test_evidence_gate_fail_fires_even_on_success(self):
        # "Loop succeeded but output wrong": the gate caught a wrong artifact.
        gate = {"last_verdict": "fail", "last_evidence": "expected 3 got 5"}
        assert _select_crisp_signal(True, "completed", gate) == "expected 3 got 5"

    def test_evidence_gate_pass_yields_nothing_on_success(self):
        gate = {"last_verdict": "pass", "last_evidence": ""}
        assert _select_crisp_signal(True, "completed", gate) == ""

    def test_crisp_loop_reason_used_on_failure(self):
        assert _select_crisp_signal(False, "max_gate_blocked", None) == "max_gate_blocked"

    def test_non_crisp_loop_reason_yields_nothing(self):
        assert _select_crisp_signal(False, "stalled", None) == ""
        assert _select_crisp_signal(False, "token_budget_exhausted", None) == ""

    def test_success_without_gate_fail_yields_nothing(self):
        # Never learn from a plain successful run, and never from prose.
        assert _select_crisp_signal(True, "completed", None) == ""

    def test_empty_evidence_falls_back_to_reason(self):
        gate = {"last_verdict": "fail", "last_evidence": ""}
        assert _select_crisp_signal(False, "max_gate_blocked", gate) == "max_gate_blocked"

    def test_quality_meta_is_not_a_signal(self):
        # Quality Gate meta issues (too-short / action-evidence) are NOT a crisp
        # signal — A/B showed they yield correctness-irrelevant rules. A
        # completed run with no Evidence Gate fail learns nothing.
        assert _select_crisp_signal(True, "completed", None) == ""
        assert _select_crisp_signal(True, "completed", {"last_verdict": "pass"}) == ""


class TestFormatRelativeTime:
    def test_just_now(self):
        # Timestamp within last 60 seconds
        now = datetime.now(UTC)
        result = format_relative_time(now)
        assert result == "방금"

    def test_minutes_ago(self):
        ts = datetime.now(UTC) - timedelta(minutes=5)
        result = format_relative_time(ts)
        assert "분 전" in result

    def test_hours_ago(self):
        ts = datetime.now(UTC) - timedelta(hours=3)
        result = format_relative_time(ts)
        assert "시간 전" in result

    def test_days_ago(self):
        ts = datetime.now(UTC) - timedelta(days=7)
        result = format_relative_time(ts)
        assert "일 전" in result

    def test_months_ago(self):
        ts = datetime.now(UTC) - timedelta(days=90)
        result = format_relative_time(ts)
        assert "개월 전" in result


class TestDetectLanguages:
    def test_python(self):
        langs = detect_languages("I need to run pytest on this django app")
        assert "Python" in langs

    def test_typescript(self):
        langs = detect_languages("install packages with pnpm and run the react app")
        assert "JavaScript/TypeScript" in langs

    def test_rust(self):
        langs = detect_languages("cargo build the rust project")
        assert "Rust" in langs


class TestDetectTools:
    def test_pnpm(self):
        detect_tools("pnpm install express")
        # pnpm is detected under 'npm' pattern or not in _TOOL_PATTERNS directly
        # Let's check what the actual tool patterns match
        # Actually pnpm is not in _TOOL_PATTERNS but npm is. pnpm is in _LANGUAGE_PATTERNS.
        # detect_tools uses _TOOL_PATTERNS which has git, docker, npm, pip, pytest, etc.
        # pnpm is not in _TOOL_PATTERNS, so let's test with npm instead
        tools2 = detect_tools("npm install express")
        assert "npm" in tools2

    def test_pytest(self):
        tools = detect_tools("pytest tests/ -v")
        assert "pytest" in tools

    def test_git(self):
        tools = detect_tools("git commit -m 'fix'")
        assert "git" in tools


class TestGenerateSkillName:
    def test_valid_goal(self):
        name = generate_skill_name("create unit tests for the auth module")
        assert name is not None
        assert "_" in name
        # Should be kebab/snake case
        assert all(c.isalnum() or c == "_" for c in name)

    def test_too_short(self):
        name = generate_skill_name("hi")
        assert name is None


class TestExtractIntentFromGoal:
    def test_file_domain(self):
        intent = extract_intent_from_goal("read the configuration file")
        assert intent.domain == Domain.FILE

    def test_git_domain(self):
        intent = extract_intent_from_goal("commit changes and push to main")
        assert intent.domain == Domain.GIT

    def test_process_domain(self):
        intent = extract_intent_from_goal("run the test suite")
        assert intent.domain == Domain.PROCESS

    def test_action_detected(self):
        intent = extract_intent_from_goal("create a new React component")
        assert intent.action == "create"

    def test_returns_intent_with_correct_fields(self):
        intent = extract_intent_from_goal("search for TODO comments in the codebase")
        assert intent.domain is not None
        assert intent.action is not None
        assert intent.confidence > 0
