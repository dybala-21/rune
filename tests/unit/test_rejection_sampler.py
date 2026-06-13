"""Tests for verifier-guided rejection sampling (best-of-K)."""

from __future__ import annotations

import pytest

from rune.agent.rejection_sampler import (
    make_evidence_gate_verifier,
    make_verifier,
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
async def test_evidence_gate_verifier_records_failure_evidence(monkeypatch) -> None:
    import rune.agent.evidence_gate as eg

    async def fake_extract(instruction: str):
        return "echo check"

    async def fake_run(script: str, cwd: str):
        # "bad" fails with mismatch evidence; "good" passes.
        if cwd == "good":
            return ("pass", "")
        return ("fail", f"mismatch at {cwd}")

    monkeypatch.setattr(eg, "extract_success_check", fake_extract)
    monkeypatch.setattr(eg, "run_evidence_check", fake_run)

    verify = await make_evidence_gate_verifier("task")
    assert await verify("bad") is False
    assert await verify("good") is True
    # the failed candidate's evidence is captured (for best-of failure learning),
    # the passing one leaves no evidence
    assert verify.evidence_by_cwd == {"bad": "mismatch at bad"}


@pytest.mark.asyncio
async def test_evidence_gate_verifier_no_check_never_selects(monkeypatch) -> None:
    import rune.agent.evidence_gate as eg

    async def fake_extract(instruction: str):
        return None  # NO_CHECK

    monkeypatch.setattr(eg, "extract_success_check", fake_extract)
    verify = await make_evidence_gate_verifier("task")
    assert await verify("anything") is False


class TestMakeVerifier:
    """Execution-first verifier: repo tests select; Evidence Gate is the fallback."""

    @pytest.mark.asyncio
    async def test_prefers_tests_pass(self, monkeypatch) -> None:
        import rune.agent.rejection_sampler as rs

        monkeypatch.setattr(rs, "make_evidence_gate_verifier", _no_eg)
        import rune.agent.auto_verify as av
        monkeypatch.setattr(av, "detect_test_command", lambda cwd: ["pytest"])

        async def fake_run(cmd, cwd, timeout=60.0):
            return ("pass", "") if cwd == "good" else ("fail", "1 failed")

        monkeypatch.setattr(av, "run_verify", fake_run)
        verify = await make_verifier("task")
        assert await verify("good") is True
        assert await verify("bad") is False
        assert verify.evidence_by_cwd == {"bad": "1 failed"}  # test output kept

    @pytest.mark.asyncio
    async def test_falls_back_to_eg_when_no_tests(self, monkeypatch) -> None:
        import rune.agent.auto_verify as av
        import rune.agent.rejection_sampler as rs

        monkeypatch.setattr(av, "detect_test_command", lambda cwd: None)

        async def fake_eg(instruction):
            async def v(cwd):
                return cwd == "eg_good"
            v.has_check = True
            v.evidence_by_cwd = {}
            return v

        monkeypatch.setattr(rs, "make_evidence_gate_verifier", fake_eg)
        verify = await make_verifier("task")
        assert await verify("eg_good") is True
        assert await verify("eg_bad") is False

    @pytest.mark.asyncio
    async def test_skip_falls_through_to_eg(self, monkeypatch) -> None:
        import rune.agent.auto_verify as av
        import rune.agent.rejection_sampler as rs

        monkeypatch.setattr(av, "detect_test_command", lambda cwd: ["pytest"])

        async def fake_run(cmd, cwd, timeout=60.0):
            return ("skip", "")  # could not run tests

        monkeypatch.setattr(av, "run_verify", fake_run)

        async def fake_eg(instruction):
            async def v(cwd):
                return True  # EG accepts when tests are inconclusive
            v.has_check = True
            v.evidence_by_cwd = {}
            return v

        monkeypatch.setattr(rs, "make_evidence_gate_verifier", fake_eg)
        verify = await make_verifier("task")
        assert await verify("anywhere") is True

    @pytest.mark.asyncio
    async def test_has_check_true_when_seed_has_tests(self, monkeypatch) -> None:
        import rune.agent.auto_verify as av
        import rune.agent.rejection_sampler as rs

        monkeypatch.setattr(rs, "make_evidence_gate_verifier", _no_eg)
        monkeypatch.setattr(av, "detect_test_command", lambda cwd: ["pytest"])
        verify = await make_verifier("task", seed_cwd="/repo")
        assert verify.has_check is True  # tests in the seed = a check exists

    @pytest.mark.asyncio
    async def test_records_which_method_decided(self, monkeypatch) -> None:
        # The UX line "picked #i (passed `pytest -q`)" depends on the verifier
        # recording WHAT decided each candidate: the test command when tests
        # ran (pass or fail), the Evidence Gate when tests were unavailable.
        import rune.agent.auto_verify as av
        import rune.agent.rejection_sampler as rs

        def fake_detect(cwd):
            return ["pytest", "-q"] if cwd in ("good", "bad") else None

        async def fake_run(cmd, cwd, timeout=60.0):
            return ("pass", "") if cwd == "good" else ("fail", "1 failed")

        monkeypatch.setattr(av, "detect_test_command", fake_detect)
        monkeypatch.setattr(av, "run_verify", fake_run)

        async def fake_eg(instruction):
            async def v(cwd):
                return True
            v.has_check = True
            v.evidence_by_cwd = {}
            return v

        monkeypatch.setattr(rs, "make_evidence_gate_verifier", fake_eg)
        verify = await make_verifier("task")
        assert await verify("good") is True
        assert await verify("bad") is False
        assert await verify("no_tests") is True
        assert verify.method_by_cwd == {
            "good": "`pytest -q`",
            "bad": "`pytest -q`",
            "no_tests": "Evidence Gate",
        }


async def _no_eg(instruction):
    """An Evidence Gate that has no check and never selects."""
    async def v(cwd):
        return False
    v.has_check = False
    v.evidence_by_cwd = {}
    return v
