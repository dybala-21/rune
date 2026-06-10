"""Verifier-guided rejection sampling (best-of-K).

Run K *independent* fresh-context attempts at a goal and keep the first one a
verifier accepts. This turns model nondeterminism into a selection signal: if a
single attempt passes with probability p, then K attempts + select succeeds with
probability 1-(1-p)^K, which rises fast even for small p.

This is the counterpart to inline self-fix (``auto_verify``): when a model is too
weak to repair its own output from an injected error, re-running it fresh and
*selecting* a good sample works where in-place repair does not. Empirically (this
repo's gemini calc bench) inline self-fix gave 0/5 while plain sampling gave 1/5,
so the verifier-as-selector path is the one that scales down to weak models.

The sampler is execution-agnostic: callers supply ``run_attempt`` (produce a
candidate) and ``verify`` (accept/reject it). Attempts must be independent —
each a fresh context — so failures don't correlate.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from rune.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Attempt[T]:
    """One sampled candidate and the verifier's verdict on it."""

    index: int
    candidate: T
    passed: bool


@dataclass
class RejectionResult[T]:
    """Outcome of a best-of-K run."""

    selected: T | None  # first candidate the verifier accepted, else None
    selected_index: int | None
    attempts: list[Attempt[T]]

    @property
    def solved(self) -> bool:
        return self.selected is not None

    @property
    def pass_count(self) -> int:
        return sum(1 for a in self.attempts if a.passed)


async def solve_with_rejection[T](
    run_attempt: Callable[[int], Awaitable[T]],
    verify: Callable[[T], Awaitable[bool]],
    k: int,
    *,
    stop_on_first_pass: bool = True,
) -> RejectionResult[T]:
    """Sample up to ``k`` independent candidates; the verifier selects.

    ``run_attempt(i)`` produces candidate i (must be a fresh, independent run).
    ``verify(candidate)`` returns whether it is acceptable. By default we stop at
    the first accepted candidate (cheapest path to a solution); set
    ``stop_on_first_pass=False`` to sample all k (e.g. to measure pass rate).
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    attempts: list[Attempt[T]] = []
    selected: T | None = None
    selected_index: int | None = None

    for i in range(k):
        candidate = await run_attempt(i)
        passed = await verify(candidate)
        attempts.append(Attempt(index=i, candidate=candidate, passed=passed))
        log.info("rejection_attempt", index=i, passed=passed)
        if passed and selected is None:
            selected, selected_index = candidate, i
            if stop_on_first_pass:
                break

    log.info(
        "rejection_result",
        k=k,
        solved=selected is not None,
        selected_index=selected_index,
        sampled=len(attempts),
    )
    return RejectionResult(
        selected=selected, selected_index=selected_index, attempts=attempts
    )


async def sample_parallel[T](
    run_attempt: Callable[[int], Awaitable[T]],
    verify: Callable[[T], Awaitable[bool]],
    k: int,
) -> RejectionResult[T]:
    """Like ``solve_with_rejection`` but run all ``k`` attempts concurrently.

    Faster wall-clock when attempts are independent and I/O-bound (e.g. each is a
    subprocess LLM call). Always samples all k (no early stop); the verifier then
    selects the lowest-index accepted candidate.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    candidates = await asyncio.gather(*(run_attempt(i) for i in range(k)))
    verdicts = await asyncio.gather(*(verify(c) for c in candidates))
    attempts = [
        Attempt(index=i, candidate=c, passed=v)
        for i, (c, v) in enumerate(zip(candidates, verdicts, strict=True))
    ]
    chosen = next((a for a in attempts if a.passed), None)
    log.info(
        "rejection_result_parallel",
        k=k,
        solved=chosen is not None,
        selected_index=chosen.index if chosen else None,
    )
    return RejectionResult(
        selected=chosen.candidate if chosen else None,
        selected_index=chosen.index if chosen else None,
        attempts=attempts,
    )


async def make_evidence_gate_verifier(
    instruction: str,
) -> Callable[[str], Awaitable[bool]]:
    """Build a ``verify(cwd)`` that uses RUNE's Evidence Gate as the selector.

    The success check is extracted ONCE (it depends only on the instruction, not
    the candidate), then each call re-runs it against a candidate's working dir.
    Only an actual ``"pass"`` selects a candidate — ``"fail"``/``"skip"``/no-check
    all return False (conservative: never select an unverified candidate).

    Measured on a hard arithmetic task: FP=0 (never passes a wrong solution) and
    ~87% correct-pass, which makes it a SAFE selector for best-of-K — a wrong pick
    is far costlier than missing one good candidate, and a larger K covers the
    occasional false-negative.
    """
    from rune.agent.evidence_gate import extract_success_check, run_evidence_check

    script = await extract_success_check(instruction)
    if not script:
        # No mechanical check available: the verifier will reject every
        # candidate. Log once here so a best-of-K that selects nothing is
        # explainable (rather than silently failing).
        log.info("rejection_eg_verifier_no_check")

    # Keep each failed candidate's mismatch evidence (keyed by cwd) so callers
    # can learn a correctness rule from it (best-of failure-driven learning).
    evidence_by_cwd: dict[str, str] = {}

    async def verify(cwd: str) -> bool:
        if not script:
            return False
        state, evidence = await run_evidence_check(script, cwd)
        if state == "fail" and evidence:
            evidence_by_cwd[cwd] = evidence
        return state == "pass"

    # Expose whether a mechanical check exists so callers can distinguish
    # "checked but every candidate failed" from "no check could be built, so
    # best-of-K structurally cannot select anything" — two very different
    # outcomes that otherwise both look like 0/K passed.
    verify.has_check = bool(script)  # type: ignore[attr-defined]
    verify.evidence_by_cwd = evidence_by_cwd  # type: ignore[attr-defined]
    return verify
