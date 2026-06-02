"""Tests for the Evidence Gate (output-correctness verification)."""

from __future__ import annotations

import pytest

from rune.agent import evidence_gate as eg
from rune.types import CapabilityResult


class _FakeClient:
    def __init__(self, content: str):
        self._content = content

    async def completion(self, **_kwargs):
        return {"choices": [{"message": {"content": self._content}}]}


def _patch_client(monkeypatch, content: str):
    monkeypatch.setattr(eg, "get_llm_client", lambda: _FakeClient(content), raising=False)
    # get_llm_client is imported lazily inside extract_success_check, so patch
    # the source module too.
    import rune.llm.client as client_mod
    monkeypatch.setattr(client_mod, "get_llm_client", lambda: _FakeClient(content))
    # These tests exercise the legacy LLM-script fallback path; force the
    # preferred spec path OFF so the gate falls back to the script path.
    import rune.agent.evidence_spec as spec_mod

    async def _no_spec(_instruction):
        return None

    monkeypatch.setattr(spec_mod, "extract_spec", _no_spec)


def _patch_registry(monkeypatch, result: CapabilityResult):
    """Stub run_evidence_check from a CapabilityResult-shaped expectation.

    The gate now runs checks via a direct subprocess (not the bash capability),
    so we patch run_evidence_check itself: success → ("pass" path), failure →
    surface output+error as the evidence text (mirrors run_evidence_check).
    """
    state = "pass" if result.success else "fail"
    evidence = "" if result.success else ((result.output or "") + (
        "\n" + result.error if result.error else "")).strip()

    async def _fake_run(script, cwd):
        return state, evidence

    monkeypatch.setattr(eg, "run_evidence_check", _fake_run)


def test_enabled_flag(monkeypatch):
    monkeypatch.delenv("RUNE_BENCH_EVIDENCE_GATE", raising=False)
    assert eg.evidence_gate_enabled() is False
    monkeypatch.setenv("RUNE_BENCH_EVIDENCE_GATE", "1")
    assert eg.evidence_gate_enabled() is True


def test_extract_prompt_mandates_multi_disjoint_sampling():
    # The check must verify large inputs on MULTIPLE DISJOINT samples (first +
    # middle + last), not just the first rows — a first-rows-only check is a
    # Goodhart blind spot an artifact can pass while failing elsewhere.
    assert "MULTIPLE DISJOINT SAMPLES" in eg._EXTRACT_SYSTEM
    assert "MIDDLE" in eg._EXTRACT_SYSTEM and "LAST" in eg._EXTRACT_SYSTEM
    assert "blind spot" in eg._EXTRACT_SYSTEM


def test_default_check_timeout_is_short():
    # Sample checks finish in seconds; the default must stay small so a slow
    # check degrades to "skip" quickly rather than stalling finalize.
    assert eg._DEFAULT_CHECK_TIMEOUT_MS <= 30_000


@pytest.mark.asyncio
async def test_extract_strips_fences(monkeypatch):
    _patch_client(monkeypatch, "```sh\ncmp -s a b || exit 1\n```")
    script = await eg.extract_success_check("verify a == b")
    assert script == "cmp -s a b || exit 1"


@pytest.mark.asyncio
async def test_extract_no_check_returns_none(monkeypatch):
    _patch_client(monkeypatch, "NO_CHECK")
    assert await eg.extract_success_check("vague task") is None


@pytest.mark.asyncio
async def test_check_passes_when_script_succeeds(monkeypatch):
    _patch_client(monkeypatch, "cmp -s out expected")
    _patch_registry(monkeypatch, CapabilityResult(success=True, output="ok"))
    gate = eg.EvidenceGate("task", "/app")
    assert await gate.check() is None  # pass → no block


@pytest.mark.asyncio
async def test_check_blocks_with_evidence_when_script_fails(monkeypatch):
    _patch_client(monkeypatch, "cmp out expected")
    _patch_registry(
        monkeypatch,
        CapabilityResult(success=False, output="line 3 got X exp Y", error="exit 1"),
    )
    gate = eg.EvidenceGate("task", "/app")
    msg = await gate.check()
    assert msg is not None
    assert "[Evidence Gate]" in msg
    assert "line 3 got X exp Y" in msg


@pytest.mark.asyncio
async def test_no_check_never_blocks(monkeypatch):
    # Model declines to produce a check → gate must not block (conservative).
    _patch_client(monkeypatch, "NO_CHECK")
    gate = eg.EvidenceGate("task", "/app")
    assert await gate.check() is None


@pytest.mark.asyncio
async def test_extraction_cached(monkeypatch):
    calls = {"n": 0}

    class _CountingClient:
        async def completion(self, **_kwargs):
            calls["n"] += 1
            return {"choices": [{"message": {"content": "cmp -s out expected"}}]}

    import rune.llm.client as client_mod
    monkeypatch.setattr(client_mod, "get_llm_client", lambda: _CountingClient())
    # Force the spec path off so this cleanly tests script-path extraction caching.
    import rune.agent.evidence_spec as spec_mod

    async def _no_spec(_instruction):
        return None

    monkeypatch.setattr(spec_mod, "extract_spec", _no_spec)
    _patch_registry(monkeypatch, CapabilityResult(success=True, output="ok"))

    gate = eg.EvidenceGate("task", "/app")
    await gate.check()
    await gate.check()
    assert calls["n"] == 1  # script extracted once, reused (spec path disabled)


@pytest.mark.asyncio
async def test_verdict_three_states(monkeypatch):
    # pass: real check ran and succeeded
    _patch_client(monkeypatch, "cmp -s out expected")
    _patch_registry(monkeypatch, CapabilityResult(success=True, output="ok"))
    state, msg = await eg.EvidenceGate("t", "/app").verdict()
    assert state == "pass" and msg is None

    # fail: real check ran and failed → blocking message
    _patch_client(monkeypatch, "cmp out expected")
    _patch_registry(monkeypatch, CapabilityResult(success=False, output="diff", error="exit 1"))
    state, msg = await eg.EvidenceGate("t", "/app").verdict()
    assert state == "fail" and msg is not None

    # skip: model declined to produce a check → neutral
    _patch_client(monkeypatch, "NO_CHECK")
    state, msg = await eg.EvidenceGate("t", "/app").verdict()
    assert state == "skip" and msg is None


@pytest.mark.asyncio
async def test_exec_error_does_not_block(monkeypatch):
    _patch_client(monkeypatch, "cmp out expected")

    async def _boom(script, cwd):
        raise RuntimeError("spawn gone")

    # run_evidence_check itself swallows errors, but verify EvidenceGate also
    # degrades gracefully if the runner ever raises.
    monkeypatch.setattr(eg, "run_evidence_check", _boom)
    gate = eg.EvidenceGate("task", "/app")
    try:
        result = await gate.check()
    except Exception:
        result = "raised"
    assert result in (None, "raised")


# --- real subprocess (no mock): the path that actually runs in benchmarks ---


@pytest.mark.asyncio
async def test_real_subprocess_pass(tmp_path):
    state, out = await eg.run_evidence_check("exit 0", str(tmp_path))
    assert state == "pass"


@pytest.mark.asyncio
async def test_real_subprocess_fail_captures_output(tmp_path):
    state, out = await eg.run_evidence_check(
        "echo 'line 3 mismatch'; exit 1", str(tmp_path)
    )
    assert state == "fail"
    assert "line 3 mismatch" in out


@pytest.mark.asyncio
async def test_real_subprocess_rm_trap_not_blocked(tmp_path):
    # Regression for the v8ev failure: a verifier that cleans up its mktemp dir
    # with `trap 'rm -rf "$tmpdir"' EXIT` was blocked by Guardian when run via
    # the bash capability. The direct-subprocess path must allow it.
    script = (
        'tmpdir=$(mktemp -d); trap \'rm -rf "$tmpdir"\' EXIT; '
        'echo ok > "$tmpdir/x"; cat "$tmpdir/x" >/dev/null; exit 0'
    )
    state, _out = await eg.run_evidence_check(script, str(tmp_path))
    assert state == "pass"


@pytest.mark.asyncio
async def test_real_subprocess_timeout_is_skip_not_pass(tmp_path, monkeypatch):
    # Regression for the v8egr3 false positive: a slow check that exceeds the
    # timeout must be "skip" (inconclusive), NOT "pass". Treating a timed-out
    # 1M-row vim check as success finalized a wrong artifact.
    monkeypatch.setenv("RUNE_BENCH_EVIDENCE_GATE_TIMEOUT_MS", "1000")
    state, _out = await eg.run_evidence_check("sleep 5; exit 0", str(tmp_path))
    assert state == "skip"


@pytest.mark.asyncio
async def test_timeout_skip_does_not_override(tmp_path, monkeypatch):
    # End-to-end: a timed-out check yields verdict "skip" with no block message,
    # so it neither blocks nor (crucially) passes.
    monkeypatch.setenv("RUNE_BENCH_EVIDENCE_GATE_TIMEOUT_MS", "1000")
    _patch_client(monkeypatch, "sleep 5; exit 0")
    gate = eg.EvidenceGate("t", str(tmp_path))
    state, msg = await gate.verdict()
    assert state == "skip" and msg is None
