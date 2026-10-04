"""Exercise requirement verdicts with a mocked completion boundary."""

from __future__ import annotations

import pytest

from rune.agent import requirement_gate as rg


def _patch_completion(monkeypatch, replies):
    """Return successive canned LLM replies; None simulates a call failure."""
    calls = {"n": 0}

    async def fake(system, user, max_tokens, judge=None):
        i = calls["n"]
        calls["n"] += 1
        calls.setdefault("judges", []).append(judge)
        return replies[i] if i < len(replies) else replies[-1]

    monkeypatch.setattr(rg, "_completion", fake)
    return calls


def test_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("RUNE_REQUIREMENT_GATE", raising=False)
    assert rg.requirement_gate_enabled() is False
    monkeypatch.setenv("RUNE_REQUIREMENT_GATE", "1")
    assert rg.requirement_gate_enabled() is True


async def test_extract_parses_json_array(monkeypatch):
    _patch_completion(monkeypatch, ['["group by team", "exactly 2 bullets"]'])
    items = await rg.extract_requirements("do the thing with 2 bullets, grouped")
    assert items == ["group by team", "exactly 2 bullets"]


async def test_extract_failsafe_on_unparseable(monkeypatch):
    _patch_completion(monkeypatch, ["not json at all"])
    assert await rg.extract_requirements("x") is None


async def test_extract_failsafe_on_llm_failure(monkeypatch):
    _patch_completion(monkeypatch, [None])
    assert await rg.extract_requirements("x") is None


async def test_check_pass_when_no_unmet(monkeypatch):
    _patch_completion(monkeypatch, ['{"met": [0], "unmet": [], "unknown": []}'])
    state, msg = await rg.check_adherence(["r1"], "output")
    assert state == "pass" and msg is None


async def test_check_fail_lists_unmet(monkeypatch):
    _patch_completion(monkeypatch, ['{"met": [], "unmet": [0], "unknown": []}'])
    state, msg = await rg.check_adherence(["exactly 2 bullets"], "output")
    assert state == "fail"
    assert "exactly 2 bullets" in msg


async def test_check_skip_on_unparseable(monkeypatch):
    _patch_completion(monkeypatch, ["garbage"])
    state, msg = await rg.check_adherence(["r1"], "o")
    assert state == "skip" and msg is None


async def test_check_skip_on_llm_failure(monkeypatch):
    _patch_completion(monkeypatch, [None])
    state, _ = await rg.check_adherence(["r1"], "o")
    assert state == "skip"


async def test_check_skip_on_empty_checklist(monkeypatch):
    # No LLM call should be needed for an empty checklist.
    state, _ = await rg.check_adherence([], "o")
    assert state == "skip"


async def test_gate_extracts_once_and_caches(monkeypatch):
    monkeypatch.setattr(rg, "checker_available", lambda: True)
    calls = _patch_completion(
        monkeypatch, ['["r1"]', '{"met": [0], "unmet": [], "unknown": []}', '{"met": [0], "unmet": [], "unknown": []}']
    )
    gate = rg.RequirementGate("a task")
    s1, _ = await gate.verdict("out1")
    s2, _ = await gate.verdict("out2")
    assert s1 == "pass" and s2 == "pass"
    # 1 extract + 2 checks = 3 completions; extraction did NOT run twice.
    assert calls["n"] == 3


async def test_gate_passes_when_no_checklist(monkeypatch):
    # Empty checklist -> nothing to block on; never calls the checker.
    monkeypatch.setattr(rg, "checker_available", lambda: True)
    calls = _patch_completion(monkeypatch, ["[]"])
    gate = rg.RequirementGate("trivial")
    state, msg = await gate.verdict("anything")
    assert state == "pass" and msg is None
    assert calls["n"] == 1  # only the extraction call


async def test_gate_skips_when_unavailable_and_no_escalation(monkeypatch):
    # Unavailable checker and no configured fallback cannot produce a verdict.
    monkeypatch.setattr(rg, "checker_available", lambda: False)
    monkeypatch.setattr(rg, "escalation_judge", lambda: None)
    calls = _patch_completion(monkeypatch, ['["r1"]', '{"met": [], "unmet": [0], "unknown": []}'])
    gate = rg.RequirementGate("a task")
    state, msg = await gate.verdict("output")
    assert state == "skip" and msg is None
    assert calls["n"] == 0  # never extracts without a configured model


async def test_gate_routes_to_escalation_when_checker_unavailable(monkeypatch):
    # Use the configured fallback when the active checker is unavailable.
    judge = ("anthropic", "claude-sonnet-4-5")
    monkeypatch.setattr(rg, "checker_available", lambda: False)
    monkeypatch.setattr(rg, "escalation_judge", lambda: judge)
    calls = _patch_completion(monkeypatch, ['["r1"]', '{"met": [], "unmet": [0], "unknown": []}'])
    gate = rg.RequirementGate("a task")
    state, msg = await gate.verdict("output")
    assert state == "fail" and "r1" in msg
    # extract + check both ran on the escalation judge, not the active provider.
    assert calls["judges"] == [judge, judge]


async def test_gate_escalates_when_active_check_call_fails(monkeypatch):
    # Retry a failed checker call with the escalation judge, reusing the checklist.
    judge = ("anthropic", "claude-sonnet-4-5")
    monkeypatch.setattr(rg, "checker_available", lambda: True)
    monkeypatch.setattr(rg, "escalation_judge", lambda: judge)
    # extract ok (active), check fails (None), then check on escalation -> fail.
    calls = _patch_completion(monkeypatch, ['["r1"]', None, '{"met": [], "unmet": [0], "unknown": []}'])
    gate = rg.RequirementGate("a task")
    state, msg = await gate.verdict("output")
    assert state == "fail" and "r1" in msg
    # extraction ran once on active; the escalation re-check did NOT re-extract.
    assert calls["judges"] == [None, None, judge]


async def test_gate_skips_when_both_judges_fail(monkeypatch):
    # If both judges fail, skip instead of inventing a failed requirement.
    judge = ("anthropic", "claude-sonnet-4-5")
    monkeypatch.setattr(rg, "checker_available", lambda: True)
    monkeypatch.setattr(rg, "escalation_judge", lambda: judge)
    calls = _patch_completion(monkeypatch, ['["r1"]', None, None])
    gate = rg.RequirementGate("a task")
    state, msg = await gate.verdict("output")
    assert state == "skip" and msg is None
    assert calls["judges"] == [None, None, judge]


def test_checker_available_local_ollama(monkeypatch):
    import rune.config as cfgmod
    import rune.llm.client as clientmod
    monkeypatch.setattr(cfgmod, "get_config", lambda: type("C", (), {
        "llm": type("L", (), {"active_provider": "ollama", "default_provider": "ollama"})()})())
    monkeypatch.setattr(clientmod, "get_llm_client",
                        lambda: type("X", (), {"resolve_model": lambda self, t: "qwen2.5-coder:32b"})())
    assert rg.checker_available() is True


def test_checker_available_ollama_cloud(monkeypatch):
    import rune.config as cfgmod
    import rune.llm.client as clientmod
    monkeypatch.setattr(cfgmod, "get_config", lambda: type("C", (), {
        "llm": type("L", (), {"active_provider": "ollama", "default_provider": "ollama"})()})())
    monkeypatch.setattr(clientmod, "get_llm_client",
                        lambda: type("X", (), {"resolve_model": lambda self, t: "qwen3-coder:480b-cloud"})())
    assert rg.checker_available() is True


def test_checker_available_cloud_provider(monkeypatch):
    import rune.config as cfgmod
    import rune.llm.client as clientmod
    monkeypatch.setattr(cfgmod, "get_config", lambda: type("C", (), {
        "llm": type("L", (), {"active_provider": "anthropic", "default_provider": "anthropic"})()})())
    monkeypatch.setattr(clientmod, "get_llm_client",
                        lambda: type("X", (), {"resolve_model": lambda self, t: "claude-sonnet-4-5"})())
    assert rg.checker_available() is True


async def test_oversized_checklist_is_inconclusive(monkeypatch):
    big = "[" + ",".join(f'"r{i}"' for i in range(50)) + "]"
    _patch_completion(monkeypatch, [big])
    items = await rg.extract_requirements("many")
    assert items is None


@pytest.mark.parametrize("reply", [
    '{"unmet": []}',
    '{"met": [0], "unmet": [], "unknown": []}',
    '{"met": [0, 0], "unmet": [], "unknown": []}',
    '{"met": [0, 2], "unmet": [], "unknown": []}',
    '{"met": [false, true], "unmet": [], "unknown": []}',
    '{"met": [0, 1], "unmet": [], "unknown": [1]}',
])
async def test_partial_or_invalid_review_never_passes(monkeypatch, reply):
    _patch_completion(monkeypatch, [reply])
    assert (await rg.check_adherence(["r1", "r2"], "output"))[0] == "skip"


async def test_unknown_is_reported_without_another_judge_or_repeated_billing(monkeypatch):
    monkeypatch.setattr(rg, "checker_available", lambda: True)
    monkeypatch.setattr(rg, "escalation_judge", lambda: pytest.fail("Missing evidence is not a provider outage"))
    calls = _patch_completion(monkeypatch, [
        '["save file", "check layout"]',
        '{"met": [0], "unmet": [], "unknown": [1]}',
    ])
    gate = rg.RequirementGate("save and check")
    first = await gate.verdict("Saved file; layout not inspected")
    assert first[0] == "skip" and "check layout" in first[1]
    assert await gate.verdict("Saved file; layout not inspected") == first
    assert calls["n"] == 2
    assert gate.summary()["required"] and gate.summary()["status"] == "inconclusive"


async def test_failed_calls_are_bounded_for_unchanged_evidence(monkeypatch):
    monkeypatch.setattr(rg, "checker_available", lambda: True)
    monkeypatch.setattr(rg, "escalation_judge", lambda: None)
    calls = _patch_completion(monkeypatch, [None])
    gate = rg.RequirementGate("task")
    for _ in range(3):
        assert await gate.verdict("output") == ("skip", None)
    assert calls["n"] == 1 and gate.summary()["required"]


async def test_requirement_review_does_not_hide_a_mismatch_behind_unknown(monkeypatch):
    _patch_completion(monkeypatch, ['{"met": [], "unmet": [1], "unknown": [0]}'])
    state, message = await rg.check_adherence(["layout", "preserve owner"], "Changed owner")
    assert state == "fail" and "preserve owner" in message
