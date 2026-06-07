"""Tests for verifier-guided rejection sampling (best-of-K)."""

from __future__ import annotations

import pytest

from rune.agent.rejection_sampler import (
    make_evidence_gate_verifier,
    sample_parallel,
    solve_with_rejection,
)


def _runner(values: list[str]):
    async def run_attempt(i: int) -> str:
        return values[i]

    return run_attempt


def _verify_equals(good: str):
    async def verify(candidate: str) -> bool:
        return candidate == good

    return verify


@pytest.mark.asyncio
async def test_selects_first_passing_and_stops() -> None:
    # Candidates: fail, fail, PASS, (4th never sampled due to early stop).
    run = _runner(["bad", "bad", "good", "good"])
    res = await solve_with_rejection(run, _verify_equals("good"), k=4)
    assert res.solved
    assert res.selected == "good"
    assert res.selected_index == 2
    assert len(res.attempts) == 3  # stopped at first pass


@pytest.mark.asyncio
async def test_no_pass_returns_unsolved() -> None:
    run = _runner(["bad", "bad", "bad"])
    res = await solve_with_rejection(run, _verify_equals("good"), k=3)
    assert not res.solved
    assert res.selected is None
    assert res.selected_index is None
    assert res.pass_count == 0
    assert len(res.attempts) == 3


@pytest.mark.asyncio
async def test_sample_all_counts_pass_rate() -> None:
    # stop_on_first_pass=False samples all k to measure the rate.
    run = _runner(["good", "bad", "good", "bad"])
    res = await solve_with_rejection(
        run, _verify_equals("good"), k=4, stop_on_first_pass=False
    )
    assert res.solved
    assert res.selected_index == 0  # first pass still recorded as selected
    assert res.pass_count == 2
    assert len(res.attempts) == 4


@pytest.mark.asyncio
async def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError):
        await solve_with_rejection(_runner([]), _verify_equals("x"), k=0)


@pytest.mark.asyncio
async def test_parallel_samples_all_and_selects_lowest_index() -> None:
    run = _runner(["bad", "good", "bad", "good"])
    res = await sample_parallel(run, _verify_equals("good"), k=4)
    assert res.solved
    assert res.selected_index == 1  # lowest-index pass
    assert res.pass_count == 2
    assert len(res.attempts) == 4  # all sampled (no early stop)


@pytest.mark.asyncio
async def test_parallel_no_pass_unsolved() -> None:
    run = _runner(["bad", "bad"])
    res = await sample_parallel(run, _verify_equals("good"), k=2)
    assert not res.solved
    assert res.pass_count == 0


@pytest.mark.asyncio
async def test_evidence_gate_verifier_selects_only_pass(monkeypatch) -> None:
    import rune.agent.evidence_gate as eg

    async def fake_extract(instruction: str):
        return "echo check"

    async def fake_run(script: str, cwd: str):
        return ("pass" if cwd == "good" else "fail", "")

    monkeypatch.setattr(eg, "extract_success_check", fake_extract)
    monkeypatch.setattr(eg, "run_evidence_check", fake_run)

    verify = await make_evidence_gate_verifier("task")
    assert await verify("good") is True
    assert await verify("bad") is False

    # skip is treated as not-selected (conservative). New verifier picks up the
    # patched run (the closure captures run_evidence_check at build time).
    async def fake_run_skip(script: str, cwd: str):
        return ("skip", "")

    monkeypatch.setattr(eg, "run_evidence_check", fake_run_skip)
    verify_skip = await make_evidence_gate_verifier("task")
    assert await verify_skip("good") is False


@pytest.mark.asyncio
async def test_evidence_gate_verifier_no_check_never_selects(monkeypatch) -> None:
    import rune.agent.evidence_gate as eg

    async def fake_extract(instruction: str):
        return None  # NO_CHECK

    monkeypatch.setattr(eg, "extract_success_check", fake_extract)
    verify = await make_evidence_gate_verifier("task")
    assert await verify("anything") is False
