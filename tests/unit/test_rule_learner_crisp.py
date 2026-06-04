"""Tests for crisp single-failure instant learning in rule_learner.

Focus on the safety gate ``is_crisp_failure`` (pure, no IO/LLM): only
deterministic/reproducible failures are crisp; environment-dependent ones are
not, because a rule learned from them would not generalize.
"""

from __future__ import annotations

import pytest

from rune.memory.rule_learner import (
    _CRISP_INITIAL_CONFIDENCE,
    _INJECTION_THRESHOLD,
    is_crisp_failure,
    is_crisp_loop_reason,
    learn_from_crisp_failure,
)


class TestIsCrispFailure:
    def test_deterministic_errors_are_crisp(self):
        assert is_crisp_failure("SyntaxError: unexpected EOF while parsing")
        assert is_crisp_failure("bash: foobar: command not found")
        assert is_crisp_failure("E486: Pattern not found")
        assert is_crisp_failure("mismatch expected: X got: Y")

    def test_environment_failures_are_not_crisp(self):
        # These depend on the environment, not the artifact — learning from them
        # does not generalize to the next run.
        assert not is_crisp_failure("Command timed out after 120000ms")
        assert not is_crisp_failure("connection refused")
        assert not is_crisp_failure("ETIMEDOUT: network unreachable")
        assert not is_crisp_failure("permission denied: /etc/hosts")
        assert not is_crisp_failure("HTTP 503 temporarily unavailable")
        assert not is_crisp_failure("rate limit exceeded")

    def test_empty_or_blank_is_not_crisp(self):
        assert not is_crisp_failure("")
        assert not is_crisp_failure("   ")

    def test_case_insensitive_marker_match(self):
        assert not is_crisp_failure("Connection RESET by peer")
        assert not is_crisp_failure("TIMEOUT waiting for service")


class TestIsCrispLoopReason:
    def test_success_reasons_are_not_crisp(self):
        assert not is_crisp_loop_reason("completed")
        assert not is_crisp_loop_reason("verified")
        assert not is_crisp_loop_reason("")

    def test_resource_and_control_reasons_are_not_crisp(self):
        # Running out of steps/budget or being cancelled is not a reproducible
        # mistake — a rule learned from it would not generalize.
        assert not is_crisp_loop_reason("stalled")
        assert not is_crisp_loop_reason("no_progress")
        assert not is_crisp_loop_reason("token_budget_exhausted")
        assert not is_crisp_loop_reason("max_iterations")
        assert not is_crisp_loop_reason("cancelled")
        assert not is_crisp_loop_reason("advisor_abort")

    def test_reproducible_failures_are_crisp(self):
        assert is_crisp_loop_reason("max_gate_blocked")
        assert is_crisp_loop_reason("error: SyntaxError unexpected EOF")

    def test_env_error_reason_is_not_crisp(self):
        # An "error:" reason still flows through the env-marker filter.
        assert not is_crisp_loop_reason("error: connection refused")
        assert not is_crisp_loop_reason("error: command timed out")


def test_crisp_rule_is_demoted_by_negative_outcome(isolated_home, monkeypatch):
    """A wrongly-learned one-shot rule must fall in confidence on a later
    negative outcome — demotion is the safety net for single-failure learning.
    """
    import asyncio

    from rune.memory.rule_learner import update_rules_from_outcome
    from rune.memory.state import load_fact_meta

    _patch_llm(monkeypatch, "close_paren: always close every opened paren before saving")
    asyncio.run(
        learn_from_crisp_failure(
            "code_modify", "SyntaxError: unexpected EOF", domain="code_modify"
        )
    )

    before = next(
        v for v in load_fact_meta().values() if v.get("source") == "crisp_failure"
    )["confidence"]

    # Negative outcome whose context matches the rule's keyword ("paren").
    updated = update_rules_from_outcome(
        "code_modify", task_success=False, goal="fix unbalanced paren in parser"
    )
    assert updated >= 1

    after = next(
        v for v in load_fact_meta().values() if v.get("source") == "crisp_failure"
    )["confidence"]
    assert after < before


def test_crisp_initial_confidence_reaches_injection_threshold():
    # A crisp one-shot rule must be injectable immediately (not stuck below the
    # threshold like the conservative repeated-failure path's 0.40 start).
    assert _CRISP_INITIAL_CONFIDENCE >= _INJECTION_THRESHOLD


@pytest.fixture
def isolated_home(tmp_dir, monkeypatch):
    """Isolate learned.md / fact-meta under a temp RUNE_HOME."""
    monkeypatch.setenv("RUNE_HOME", str(tmp_dir))
    return tmp_dir


def _patch_llm(monkeypatch, content: str):
    class _Client:
        async def completion(self, **_kwargs):
            return {"choices": [{"message": {"content": content}}]}

    import rune.llm.client as client_mod
    monkeypatch.setattr(client_mod, "get_llm_client", lambda: _Client())


@pytest.mark.asyncio
async def test_crisp_failure_creates_injectable_rule(isolated_home, monkeypatch):
    _patch_llm(monkeypatch, "close_brackets: ensure every bracket is closed before saving")
    from rune.memory.state import load_fact_meta

    key = await learn_from_crisp_failure(
        "bash", "SyntaxError: unexpected EOF while parsing", domain="code_modify"
    )
    assert key is not None
    meta = load_fact_meta()
    rule = next((v for k, v in meta.items() if k.startswith("rule:code_modify")), None)
    assert rule is not None
    assert rule["confidence"] >= _INJECTION_THRESHOLD
    assert rule["source"] == "crisp_failure"


@pytest.mark.asyncio
async def test_environment_failure_learns_nothing(isolated_home, monkeypatch):
    _patch_llm(monkeypatch, "x: y")  # should never be called
    from rune.memory.state import load_fact_meta

    key = await learn_from_crisp_failure(
        "bash", "Command timed out after 120000ms", domain="code_modify"
    )
    assert key is None
    assert not any(k.startswith("rule:") for k in load_fact_meta())


@pytest.mark.asyncio
async def test_llm_declines_general_rule_learns_nothing(isolated_home, monkeypatch):
    _patch_llm(monkeypatch, "NONE")
    from rune.memory.state import load_fact_meta

    key = await learn_from_crisp_failure(
        "bash", "SyntaxError: unexpected EOF", domain="code_modify"
    )
    assert key is None
    assert not any(k.startswith("rule:") for k in load_fact_meta())


@pytest.mark.asyncio
async def test_crisp_rule_is_injected_into_next_same_domain_task(isolated_home, monkeypatch):
    # Mechanism A/B (deterministic): a single crisp failure produces a rule that
    # is immediately injectable for the next same-domain task — whereas the
    # conservative repeated-failure path would still be at 0 (needs 2x + bootstrap).
    _patch_llm(monkeypatch, "double_backslash: in setreg double every regex backslash")
    from rune.memory.rule_learner import get_rules_for_domain

    before = get_rules_for_domain("code_modify")
    await learn_from_crisp_failure(
        "code_modify", "E486: Pattern not found", domain="code_modify"
    )
    after = get_rules_for_domain("code_modify")

    assert len(after) == len(before) + 1
    assert any("backslash" in str(r).lower() for r in after)


class _StubManager:
    async def save_episode(self, _episode):
        return None

    async def save(self, *_a, **_k):
        return None


@pytest.mark.asyncio
async def test_evidence_gate_fail_drives_crisp_learning_through_save(
    isolated_home, monkeypatch
):
    """End-to-end wiring: an Evidence Gate 'fail' verdict reaches
    learn_from_crisp_failure via save_agent_result_to_memory even when the loop
    reported success — and the signal learned is the gate evidence, not prose.
    """
    monkeypatch.setenv("RUNE_CRISP_LEARNING", "1")
    _patch_llm(monkeypatch, "match_expected: ensure output equals the expected value")

    captured: dict[str, str] = {}
    import rune.memory.rule_learner as rl

    real_learn = rl.learn_from_crisp_failure

    async def _spy(tool_name, error_message, domain="code_modify"):
        captured["error_message"] = error_message
        return await real_learn(tool_name, error_message, domain=domain)

    monkeypatch.setattr(rl, "learn_from_crisp_failure", _spy)

    from rune.agent.memory_bridge import save_agent_result_to_memory
    from rune.memory.state import load_fact_meta

    await save_agent_result_to_memory(
        goal="write a function that returns 3",
        result={
            "output": "Here is the function. All done!",  # prose — must NOT be learned
            "success": True,  # loop succeeded but the artifact is wrong
            "reason": "completed",
            "evidence_gate": {
                "last_verdict": "fail",
                "last_evidence": "expected return 3 but got 5",
            },
        },
        memory_manager=_StubManager(),
    )

    assert captured.get("error_message") == "expected return 3 but got 5"
    assert any(
        v.get("source") == "crisp_failure" for v in load_fact_meta().values()
    )


@pytest.mark.asyncio
async def test_quality_issue_is_not_a_crisp_signal_through_save(
    isolated_home, monkeypatch
):
    """A completed run with only a Quality Gate meta issue must NOT learn — the
    meta signal was removed because A/B showed it produces harmful, correctness-
    irrelevant rules."""
    monkeypatch.setenv("RUNE_CRISP_LEARNING", "1")
    _patch_llm(monkeypatch, "should_not: be called")

    from rune.agent.memory_bridge import save_agent_result_to_memory
    from rune.memory.state import load_fact_meta

    await save_agent_result_to_memory(
        goal="write a script that prints the first 12 Keith numbers",
        result={
            "output": "done",
            "success": True,
            "reason": "completed",
            "quality_issue": "Response too short — should include concrete results",
        },
        memory_manager=_StubManager(),
        classification_hint="code_modify",
    )
    assert not any(
        v.get("source") == "crisp_failure" for v in load_fact_meta().values()
    )


@pytest.mark.asyncio
async def test_crisp_learning_off_by_default_through_save(isolated_home, monkeypatch):
    """Without RUNE_CRISP_LEARNING the save path performs no crisp learning."""
    monkeypatch.delenv("RUNE_CRISP_LEARNING", raising=False)
    _patch_llm(monkeypatch, "x: y")

    from rune.agent.memory_bridge import save_agent_result_to_memory
    from rune.memory.state import load_fact_meta

    await save_agent_result_to_memory(
        goal="write a function that returns 3",
        result={
            "output": "done",
            "success": True,
            "reason": "completed",
            "evidence_gate": {"last_verdict": "fail", "last_evidence": "expected 3 got 5"},
        },
        memory_manager=_StubManager(),
    )

    assert not any(
        v.get("source") == "crisp_failure" for v in load_fact_meta().values()
    )
