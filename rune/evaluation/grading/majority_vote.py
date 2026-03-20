"""Majority-vote grading for RUNE evaluation system.

Runs an evaluator function multiple times and aggregates scores via
majority / median voting to mitigate LLM-as-judge non-determinism.
"""

from __future__ import annotations

import asyncio
import statistics
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class MajorityVoteOptions:
    """Configuration for majority-vote execution."""

    rounds: int = 3
    tie_breaker_rounds: int = 1
    consensus_threshold: float = 0.1
    delay_between_rounds: float = 0.0  # seconds


@dataclass(slots=True)
class VotingResult:
    """Outcome of a majority-vote evaluation."""

    final_score: float = 0.0
    votes: list[float] = field(default_factory=list)
    consensus: bool = False
    attempts: int = 0


# Consensus helpers

def _check_consensus(
    votes: list[float], threshold: float,
) -> tuple[bool, float]:
    """Return ``(achieved, score)`` - whether all votes are within *threshold*
    of the average."""
    if not votes:
        return False, 0.0
    avg = sum(votes) / len(votes)
    max_diff = max(abs(v - avg) for v in votes)
    return max_diff <= threshold, avg


def _calculate_median(values: list[float]) -> float:
    if not values:
        return 0.0
    return statistics.median(values)


# Public API

async def run_with_majority_vote(
    evaluator: Callable[[], Awaitable[float]],
    options: MajorityVoteOptions | None = None,
) -> VotingResult:
    """Run *evaluator* multiple times and return a majority-vote result.

    Parameters:
        evaluator: Async callable returning a numeric score.
        options: Voting configuration (defaults are sensible).

    Returns:
        A :class:`VotingResult` with the aggregated score.
    """
    opts = options or MajorityVoteOptions()
    votes: list[float] = []

    # Initial rounds
    for i in range(opts.rounds):
        if i > 0 and opts.delay_between_rounds > 0:
            await asyncio.sleep(opts.delay_between_rounds)
        score = await evaluator()
        votes.append(score)

    achieved, avg_score = _check_consensus(votes, opts.consensus_threshold)
    if achieved:
        return VotingResult(
            final_score=avg_score,
            votes=votes,
            consensus=True,
            attempts=opts.rounds,
        )

    # Tie-breaker rounds
    for _ in range(opts.tie_breaker_rounds):
        if opts.delay_between_rounds > 0:
            await asyncio.sleep(opts.delay_between_rounds)
        score = await evaluator()
        votes.append(score)

    final_score = _calculate_median(votes)
    return VotingResult(
        final_score=final_score,
        votes=votes,
        consensus=False,
        attempts=opts.rounds + opts.tie_breaker_rounds,
    )


async def run_batch_with_majority_vote[T](
    items: list[T],
    evaluator: Callable[[T], Awaitable[float]],
    options: MajorityVoteOptions | None = None,
) -> dict[int, VotingResult]:
    """Run majority-vote evaluations for a batch of items in parallel.

    Parameters:
        items: Items to evaluate.
        evaluator: Async callable that scores a single item.
        options: Voting configuration.

    Returns:
        Mapping of item index to :class:`VotingResult`.
    """

    async def _run(idx: int, item: T) -> tuple[int, VotingResult]:
        result = await run_with_majority_vote(
            lambda: evaluator(item), options,
        )
        return idx, result

    tasks = [_run(i, item) for i, item in enumerate(items)]
    resolved = await asyncio.gather(*tasks)
    return {idx: res for idx, res in resolved}


def get_voting_stats(results: list[VotingResult]) -> dict[str, Any]:
    """Compute summary statistics over a set of voting results.

    Returns:
        Dictionary with ``avg_attempts``, ``consensus_rate``, and
        ``avg_vote_spread``.
    """
    if not results:
        return {"avg_attempts": 0, "consensus_rate": 0.0, "avg_vote_spread": 0.0}

    total_attempts = sum(r.attempts for r in results)
    consensus_count = sum(1 for r in results if r.consensus)
    spreads: list[float] = []
    for r in results:
        if len(r.votes) >= 2:
            spreads.append(max(r.votes) - min(r.votes))

    return {
        "avg_attempts": total_attempts / len(results),
        "consensus_rate": consensus_count / len(results),
        "avg_vote_spread": (
            sum(spreads) / len(spreads) if spreads else 0.0
        ),
    }
