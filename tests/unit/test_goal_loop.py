"""Tests for rune.agent.goal_loop — autonomous Ralph-style outer loop.

The core is dependency-injected, so every case is driven by scripted stub
traces; no LLM / agent stack is touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rune.agent.goal_loop import (
    GoalLoop,
    GoalLoopConfig,
    GoalSpec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class StubTrace:
    reason: str = "completed"
    final_step: int = 1
    total_tokens_used: int = 100
    evidence_score: float = 1.0
    answer: str = "done"


def progress(tokens: int = 100) -> StubTrace:
    return StubTrace(reason="", total_tokens_used=tokens, evidence_score=0.0, answer="wip")


def verified() -> StubTrace:
    return StubTrace(reason="completed", evidence_score=1.0, answer="final answer")


class ScriptedRunner:
    """Returns scripted traces; the last entry repeats for extra iterations."""

    def __init__(self, traces: list[StubTrace]) -> None:
        self._traces = traces
        self.calls: list[tuple[int, str]] = []

    async def __call__(self, prompt: str, iteration: int) -> StubTrace:
        self.calls.append((iteration, prompt))
        idx = min(iteration - 1, len(self._traces) - 1)
        return self._traces[idx]


class PersistSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, str]] = []

    async def __call__(self, success: bool, answer: str) -> None:
        self.calls.append((success, answer))


def validator(passed: bool, detail: str = "ok"):
    async def _v(commands: list[str]) -> tuple[bool, str]:
        return passed, detail

    return _v


def spec(**kw: object) -> GoalSpec:
    base: dict = dict(goal="build X", acceptance_criteria=["ac1", "ac2"])
    base.update(kw)
    return GoalSpec(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Stop-condition matrix
# ---------------------------------------------------------------------------


async def test_verified_immediately(tmp_path: Path) -> None:
    runner = ScriptedRunner([verified()])
    persist = PersistSpy()
    loop = GoalLoop(run_fn=runner, persist_fn=persist, workspace=tmp_path)

    res = await loop.run(spec())

    assert res.success is True
    assert res.stop_cause == "verified"
    assert len(res.iterations) == 1
    assert res.final_answer == "final answer"
    assert persist.calls == [(True, "final answer")]


async def test_progress_then_verified(tmp_path: Path) -> None:
    runner = ScriptedRunner([progress(), progress(), verified()])
    loop = GoalLoop(
        GoalLoopConfig(stagnation_window=0),
        run_fn=runner,
        workspace=tmp_path,
    )

    res = await loop.run(spec())

    assert res.success is True
    assert res.stop_cause == "verified"
    assert [it.n for it in res.iterations] == [1, 2, 3]


async def test_max_iterations(tmp_path: Path) -> None:
    runner = ScriptedRunner([progress()])
    loop = GoalLoop(
        GoalLoopConfig(max_iterations=3, stagnation_window=0),
        run_fn=runner,
        workspace=tmp_path,
    )

    res = await loop.run(spec())

    assert res.success is False
    assert res.stop_cause == "max_iterations"
    assert len(res.iterations) == 3


async def test_stagnation_stops_early(tmp_path: Path) -> None:
    runner = ScriptedRunner([progress()])  # identical signature every call
    loop = GoalLoop(
        GoalLoopConfig(max_iterations=10, stagnation_window=2),
        run_fn=runner,
        workspace=tmp_path,
    )

    res = await loop.run(spec())

    assert res.stop_cause == "stagnation"
    assert len(res.iterations) == 2  # stopped well before max_iterations=10


async def test_budget_cap(tmp_path: Path) -> None:
    runner = ScriptedRunner([progress(tokens=100)])
    loop = GoalLoop(
        GoalLoopConfig(max_total_tokens=50, stagnation_window=0),
        run_fn=runner,
        workspace=tmp_path,
    )

    res = await loop.run(spec())

    assert res.stop_cause == "budget"
    assert res.success is False


async def test_cancelled_before_start(tmp_path: Path) -> None:
    runner = ScriptedRunner([verified()])
    persist = PersistSpy()
    loop = GoalLoop(
        run_fn=runner,
        persist_fn=persist,
        cancelled=lambda: True,
        workspace=tmp_path,
    )

    res = await loop.run(spec())

    assert res.stop_cause == "cancelled"
    assert res.success is False
    assert runner.calls == []  # never invoked the agent
    assert persist.calls == [(False, "")]


async def test_cancelled_midloop(tmp_path: Path) -> None:
    flags = iter([False, True, True])
    runner = ScriptedRunner([progress()])
    loop = GoalLoop(
        run_fn=runner,
        cancelled=lambda: next(flags),
        workspace=tmp_path,
    )

    res = await loop.run(spec())

    assert res.stop_cause == "cancelled"
    assert len(res.iterations) == 1  # one iteration ran before cancel


# ---------------------------------------------------------------------------
# Poisoning-safe: terminal-only persistence (regression)
# ---------------------------------------------------------------------------


async def test_persist_called_once_on_failure(tmp_path: Path) -> None:
    runner = ScriptedRunner([progress()])
    persist = PersistSpy()
    loop = GoalLoop(
        GoalLoopConfig(max_iterations=5, stagnation_window=0),
        run_fn=runner,
        persist_fn=persist,
        workspace=tmp_path,
    )

    res = await loop.run(spec())

    # 5 iterations ran, but episodic memory is written exactly once, at the
    # terminal outcome — no per-iteration anti-example accumulation.
    assert len(res.iterations) == 5
    assert persist.calls == [(False, "wip")]


# ---------------------------------------------------------------------------
# Validation is run by the loop, not trusted to the gate
# ---------------------------------------------------------------------------


async def test_validation_failure_blocks_verified(tmp_path: Path) -> None:
    runner = ScriptedRunner([verified()])  # gate says done every time
    loop = GoalLoop(
        GoalLoopConfig(max_iterations=2, stagnation_window=0),
        run_fn=runner,
        validate_fn=validator(False, "2 failed"),
        workspace=tmp_path,
    )

    res = await loop.run(spec(validation_commands=["pytest -q"]))

    assert res.success is False
    assert res.stop_cause == "max_iterations"
    assert res.iterations[0].validation_passed is False
    assert "Avoid: validation failed" in (tmp_path / "progress.md").read_text()


async def test_validation_pass_allows_verified(tmp_path: Path) -> None:
    runner = ScriptedRunner([verified()])
    loop = GoalLoop(
        run_fn=runner,
        validate_fn=validator(True),
        workspace=tmp_path,
    )

    res = await loop.run(spec(validation_commands=["pytest -q"]))

    assert res.success is True
    assert res.stop_cause == "verified"
    assert res.iterations[-1].validation_passed is True


# ---------------------------------------------------------------------------
# Verdict mapping
# ---------------------------------------------------------------------------


async def test_error_reason_stops_loop(tmp_path: Path) -> None:
    runner = ScriptedRunner([StubTrace(reason="error: boom", evidence_score=0.0)])
    loop = GoalLoop(run_fn=runner, workspace=tmp_path)

    res = await loop.run(spec())

    assert res.stop_cause == "error"
    assert res.success is False


async def test_runner_exception_is_fail_closed(tmp_path: Path) -> None:
    async def boom(prompt: str, iteration: int) -> StubTrace:
        raise RuntimeError("provider down")

    loop = GoalLoop(run_fn=boom, workspace=tmp_path)

    res = await loop.run(spec())

    assert res.stop_cause == "error"
    assert res.success is False
    assert len(res.iterations) == 1


# ---------------------------------------------------------------------------
# File hippocampus
# ---------------------------------------------------------------------------


async def test_files_seeded_and_spec_is_immutable(tmp_path: Path) -> None:
    (tmp_path).mkdir(exist_ok=True)
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text("SENTINEL — human edited", encoding="utf-8")

    runner = ScriptedRunner([verified()])
    loop = GoalLoop(run_fn=runner, workspace=tmp_path)
    await loop.run(spec())

    # SPEC is the immutable drift anchor: never overwritten once present.
    assert spec_path.read_text() == "SENTINEL — human edited"
    assert (tmp_path / "fix_plan.md").exists()
    assert (tmp_path / "progress.md").exists()
    assert "DONE stop_cause=verified" in (tmp_path / "progress.md").read_text()


# ---------------------------------------------------------------------------
# Progress callback (UI hook)
# ---------------------------------------------------------------------------


async def test_on_iteration_fires_per_iteration(tmp_path: Path) -> None:
    runner = ScriptedRunner([progress(), progress(), verified()])
    seen: list[int] = []
    loop = GoalLoop(
        GoalLoopConfig(stagnation_window=0),
        run_fn=runner,
        on_iteration=lambda it: seen.append(it.n),
        workspace=tmp_path,
    )

    res = await loop.run(spec())

    assert seen == [1, 2, 3]
    assert res.success is True


async def test_on_iteration_exception_does_not_break_loop(tmp_path: Path) -> None:
    runner = ScriptedRunner([verified()])

    def _boom(_it: object) -> None:
        raise RuntimeError("render crashed")

    loop = GoalLoop(run_fn=runner, on_iteration=_boom, workspace=tmp_path)

    res = await loop.run(spec())

    assert res.success is True  # render failure is swallowed (logged)


# ---------------------------------------------------------------------------
# Phase 5: adversarial review + SSC self-critique
# ---------------------------------------------------------------------------


def _review(allow: bool, reason: str = "r"):
    async def _r(_rc):
        return allow, reason

    return _r


async def test_review_allows_verified(tmp_path: Path) -> None:
    loop = GoalLoop(
        GoalLoopConfig(adversarial_review=True),
        run_fn=ScriptedRunner([verified()]),
        review_fn=_review(True),
        workspace=tmp_path,
    )

    res = await loop.run(spec())

    assert res.success is True
    assert res.iterations[-1].review_passed is True


async def test_review_block_demotes_to_progress(tmp_path: Path) -> None:
    loop = GoalLoop(
        GoalLoopConfig(adversarial_review=True, max_iterations=2, stagnation_window=0),
        run_fn=ScriptedRunner([verified()]),
        review_fn=_review(False, "hard-coded to tests"),
        workspace=tmp_path,
    )

    res = await loop.run(spec())

    assert res.success is False
    assert res.stop_cause == "max_iterations"
    assert res.iterations[0].review_passed is False
    assert "adversarial review blocked" in (tmp_path / "progress.md").read_text()


async def test_reviewer_exception_is_fail_closed(tmp_path: Path) -> None:
    async def boom(_rc):
        raise RuntimeError("reviewer down")

    loop = GoalLoop(
        GoalLoopConfig(adversarial_review=True, max_iterations=1, stagnation_window=0),
        run_fn=ScriptedRunner([verified()]),
        review_fn=boom,
        workspace=tmp_path,
    )

    res = await loop.run(spec())

    assert res.success is False  # reviewer failure must NOT accept the candidate


async def test_review_disabled_when_no_fn(tmp_path: Path) -> None:
    # Config opts in but no reviewer injected -> no-op (verified stands).
    loop = GoalLoop(
        GoalLoopConfig(adversarial_review=True),
        run_fn=ScriptedRunner([verified()]),
        workspace=tmp_path,
    )

    res = await loop.run(spec())

    assert res.success is True


async def test_ssc_runs_every_interval_and_records(tmp_path: Path) -> None:
    seen: list[int] = []

    async def critique(_spec, _answer, n):
        seen.append(n)
        return f"loophole at {n}"

    loop = GoalLoop(
        GoalLoopConfig(max_iterations=4, stagnation_window=0, ssc_interval=2),
        run_fn=ScriptedRunner([progress()]),
        critique_fn=critique,
        workspace=tmp_path,
    )

    await loop.run(spec())

    assert seen == [2, 4]  # only on multiples of ssc_interval
    assert "SSC (iter 2): loophole at 2" in (tmp_path / "progress.md").read_text()


async def test_ssc_exception_is_non_blocking(tmp_path: Path) -> None:
    async def boom(_spec, _answer, _n):
        raise RuntimeError("ssc down")

    loop = GoalLoop(
        GoalLoopConfig(max_iterations=2, stagnation_window=0, ssc_interval=1),
        run_fn=ScriptedRunner([progress()]),
        critique_fn=boom,
        workspace=tmp_path,
    )

    res = await loop.run(spec())

    assert res.stop_cause == "max_iterations"  # advisory failure swallowed


# ---------------------------------------------------------------------------
# Validation is the arbiter, not the inner evidence score
# (regression: the real NativeAgentLoop returns evidence=0.0 even on success)
# ---------------------------------------------------------------------------


def _completed_no_evidence() -> StubTrace:
    return StubTrace(reason="completed", evidence_score=0.0, answer="did it")


async def test_completed_low_evidence_verified_when_validation_passes(
    tmp_path: Path,
) -> None:
    loop = GoalLoop(
        GoalLoopConfig(stagnation_window=0),
        run_fn=ScriptedRunner([_completed_no_evidence()]),
        validate_fn=validator(True),  # deterministic gate is the arbiter
        workspace=tmp_path,
    )

    res = await loop.run(spec(validation_commands=["go test ./..."]))

    assert res.success is True  # evidence 0.0 must NOT block a tests-pass result
    assert res.iterations[-1].validation_passed is True


async def test_completed_low_evidence_unsubstantiated_without_objective_check(
    tmp_path: Path,
) -> None:
    # No validation commands AND no reviewer -> bare "completed" with no
    # evidence must not be trivially accepted (falls back to evidence floor).
    loop = GoalLoop(
        GoalLoopConfig(max_iterations=2, stagnation_window=0),
        run_fn=ScriptedRunner([_completed_no_evidence()]),
        workspace=tmp_path,
    )

    res = await loop.run(spec())

    assert res.success is False
    assert res.stop_cause == "max_iterations"
    assert "unsubstantiated" in (tmp_path / "progress.md").read_text()


# ---------------------------------------------------------------------------
# Phase 5.1: reviewer judges the changed source (artifact_fn), not just pass/fail
# ---------------------------------------------------------------------------


def _review_capturing():
    """Reviewer that BLOCKs when the artifact looks hollow; records what it saw."""
    seen: dict[str, str] = {}

    async def _r(rc):
        seen["artifact"] = rc.artifact
        seen["evidence"] = rc.validation_output
        # Empty artifact => judge on evidence (no-regression); only an explicit
        # hollow marker in real source blocks.
        hollow = "no assertions" in rc.artifact
        return (not hollow), ("hollow test" if hollow else "genuine")

    return _r, seen


async def test_artifact_fn_output_reaches_reviewer(tmp_path: Path) -> None:
    rv, seen = _review_capturing()

    async def artifact_fn() -> str:
        return "=== x_test.go ===\nfunc TestReal(t){ assert(got==want) }\n"

    loop = GoalLoop(
        GoalLoopConfig(adversarial_review=True, stagnation_window=0),
        run_fn=ScriptedRunner([verified()]),
        review_fn=rv,
        artifact_fn=artifact_fn,
        workspace=tmp_path,
    )

    res = await loop.run(spec())

    assert res.success is True
    assert "TestReal" in seen["artifact"]  # source reached the reviewer


async def test_hollow_artifact_is_blocked(tmp_path: Path) -> None:
    rv, _ = _review_capturing()

    async def artifact_fn() -> str:
        return "=== x_test.go ===\nfunc TestX(t){}  // no assertions\n"

    loop = GoalLoop(
        GoalLoopConfig(adversarial_review=True, max_iterations=2, stagnation_window=0),
        run_fn=ScriptedRunner([verified()]),
        review_fn=rv,
        artifact_fn=artifact_fn,
        workspace=tmp_path,
    )

    res = await loop.run(spec())

    assert res.success is False  # hollow test that "passes" is rejected
    assert "adversarial review blocked" in (tmp_path / "progress.md").read_text()


async def test_no_artifact_fn_is_no_regression(tmp_path: Path) -> None:
    rv, seen = _review_capturing()
    loop = GoalLoop(
        GoalLoopConfig(adversarial_review=True, stagnation_window=0),
        run_fn=ScriptedRunner([verified()]),
        review_fn=rv,  # no artifact_fn injected
        workspace=tmp_path,
    )

    res = await loop.run(spec())

    assert res.success is True
    assert seen["artifact"] == ""  # absent artifact_fn -> empty, prior behavior


async def test_artifact_fn_failure_does_not_block(tmp_path: Path) -> None:
    rv, seen = _review_capturing()

    async def boom_artifact() -> str:
        raise RuntimeError("scan failed")

    loop = GoalLoop(
        GoalLoopConfig(adversarial_review=True, stagnation_window=0),
        run_fn=ScriptedRunner([verified()]),
        review_fn=rv,
        artifact_fn=boom_artifact,
        workspace=tmp_path,
    )

    res = await loop.run(spec())

    # Artifact gathering is best-effort context; its failure must NOT
    # fail-closed the candidate (only the LLM verdict is fail-closed).
    assert res.success is True
    assert seen["artifact"] == ""
