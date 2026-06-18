"""Unit tests for the Requirement-Adherence Gate (rune/agent/requirement_gate.py).

The single LLM entry point ``_completion`` is monkeypatched so no network is
needed and each branch (pass / fail / skip / fail-safe) is exercised directly.
"""

from __future__ import annotations

from rune.agent import requirement_gate as rg


def _patch_completion(monkeypatch, replies):
    """Return successive canned LLM replies; None simulates a call failure."""
    calls = {"n": 0}

    async def fake(system, user, max_tokens):
        i = calls["n"]
        calls["n"] += 1
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
    _patch_completion(monkeypatch, ['{"unmet": []}'])
    state, msg = await rg.check_adherence(["r1"], "output")
    assert state == "pass" and msg is None


async def test_check_fail_lists_unmet(monkeypatch):
    _patch_completion(monkeypatch, ['{"unmet": ["exactly 2 bullets"]}'])
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
    monkeypatch.setattr(rg, "checker_capable", lambda: True)
    calls = _patch_completion(
        monkeypatch, ['["r1"]', '{"unmet": []}', '{"unmet": []}']
    )
    gate = rg.RequirementGate("a task")
    s1, _ = await gate.verdict("out1")
    s2, _ = await gate.verdict("out2")
    assert s1 == "pass" and s2 == "pass"
    # 1 extract + 2 checks = 3 completions; extraction did NOT run twice.
    assert calls["n"] == 3


async def test_gate_skips_when_no_checklist(monkeypatch):
    # Extraction yields empty list -> gate never blocks, never calls the checker.
    monkeypatch.setattr(rg, "checker_capable", lambda: True)
    calls = _patch_completion(monkeypatch, ["[]"])
    gate = rg.RequirementGate("trivial")
    state, msg = await gate.verdict("anything")
    assert state == "skip" and msg is None
    assert calls["n"] == 1  # only the extraction call


async def test_gate_skips_when_checker_weak(monkeypatch):
    # A weak checker must make the whole gate skip (it false-blocks correct work).
    monkeypatch.setattr(rg, "checker_capable", lambda: False)
    calls = _patch_completion(monkeypatch, ['["r1"]', '{"unmet": ["r1"]}'])
    gate = rg.RequirementGate("a task")
    state, msg = await gate.verdict("output")
    assert state == "skip" and msg is None
    assert calls["n"] == 0  # never even extracts with a weak checker


def test_checker_capable_local_ollama_weak(monkeypatch):
    import rune.config as cfgmod
    import rune.llm.client as clientmod
    monkeypatch.setattr(cfgmod, "get_config", lambda: type("C", (), {
        "llm": type("L", (), {"active_provider": "ollama", "default_provider": "ollama"})()})())
    monkeypatch.setattr(clientmod, "get_llm_client",
                        lambda: type("X", (), {"resolve_model": lambda self, t: "qwen2.5-coder:32b"})())
    assert rg.checker_capable() is False


def test_checker_capable_ollama_cloud_strong(monkeypatch):
    import rune.config as cfgmod
    import rune.llm.client as clientmod
    monkeypatch.setattr(cfgmod, "get_config", lambda: type("C", (), {
        "llm": type("L", (), {"active_provider": "ollama", "default_provider": "ollama"})()})())
    monkeypatch.setattr(clientmod, "get_llm_client",
                        lambda: type("X", (), {"resolve_model": lambda self, t: "qwen3-coder:480b-cloud"})())
    assert rg.checker_capable() is True


def test_checker_capable_cloud_provider_strong(monkeypatch):
    import rune.config as cfgmod
    import rune.llm.client as clientmod
    monkeypatch.setattr(cfgmod, "get_config", lambda: type("C", (), {
        "llm": type("L", (), {"active_provider": "anthropic", "default_provider": "anthropic"})()})())
    monkeypatch.setattr(clientmod, "get_llm_client",
                        lambda: type("X", (), {"resolve_model": lambda self, t: "claude-sonnet-4-5"})())
    assert rg.checker_capable() is True


async def test_checklist_capped(monkeypatch):
    big = "[" + ",".join(f'"r{i}"' for i in range(50)) + "]"
    _patch_completion(monkeypatch, [big])
    items = await rg.extract_requirements("many")
    assert len(items) == rg._MAX_CHECKLIST_ITEMS
