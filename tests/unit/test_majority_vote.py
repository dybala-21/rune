"""Tests for rune.evaluation.grading.majority_vote — majority-vote scoring."""

from __future__ import annotations

import pytest

from rune.evaluation.grading.majority_vote import (
    MajorityVoteOptions,
    VotingResult,
    get_voting_stats,
    run_batch_with_majority_vote,
    run_with_majority_vote,
)

# ---------------------------------------------------------------------------
# run_with_majority_vote
# ---------------------------------------------------------------------------


class TestRunWithMajorityVote:
    @pytest.mark.asyncio
    async def test_consensus_when_votes_are_similar(self):
        call_count = 0
        scores = [0.85, 0.87, 0.84]

        async def evaluator():
            nonlocal call_count
            s = scores[call_count]
            call_count += 1
            return s

        result = await run_with_majority_vote(
            evaluator,
            MajorityVoteOptions(rounds=3, consensus_threshold=0.1),
        )
        assert result.consensus is True
        assert len(result.votes) == 3
        assert abs(result.final_score - 0.8533) < 0.01
        assert result.attempts == 3

    @pytest.mark.asyncio
    async def test_tie_breaker_when_no_consensus(self):
        call_count = 0
        scores = [0.3, 0.9, 0.85, 0.88]

        async def evaluator():
            nonlocal call_count
            s = scores[call_count]
            call_count += 1
            return s

        result = await run_with_majority_vote(
            evaluator,
            MajorityVoteOptions(rounds=3, tie_breaker_rounds=1, consensus_threshold=0.1),
        )
        assert result.consensus is False
        assert len(result.votes) == 4
        assert result.attempts == 4

    @pytest.mark.asyncio
    async def test_median_for_final_score(self):
        call_count = 0
        scores = [0.1, 0.85, 0.9, 0.87]

        async def evaluator():
            nonlocal call_count
            s = scores[call_count]
            call_count += 1
            return s

        result = await run_with_majority_vote(
            evaluator,
            MajorityVoteOptions(rounds=3, tie_breaker_rounds=1, consensus_threshold=0.05),
        )
        # Sorted: [0.1, 0.85, 0.87, 0.9] -> median = (0.85 + 0.87) / 2 = 0.86
        assert abs(result.final_score - 0.86) < 0.01

    @pytest.mark.asyncio
    async def test_evaluator_error_propagates(self):
        async def failing_evaluator():
            raise RuntimeError("LLM error")

        with pytest.raises(RuntimeError, match="LLM error"):
            await run_with_majority_vote(failing_evaluator)


# ---------------------------------------------------------------------------
# run_batch_with_majority_vote
# ---------------------------------------------------------------------------


class TestRunBatchWithMajorityVote:
    @pytest.mark.asyncio
    async def test_processes_multiple_items(self):
        call_count = 0

        async def evaluator(item):
            nonlocal call_count
            call_count += 1
            return 0.8

        results = await run_batch_with_majority_vote(
            ["a", "b", "c"],
            evaluator,
            MajorityVoteOptions(rounds=3),
        )
        assert len(results) == 3
        assert call_count == 9  # 3 items * 3 rounds

    @pytest.mark.asyncio
    async def test_returns_per_item_scores(self):
        async def evaluator(item):
            return item * 0.3

        results = await run_batch_with_majority_vote(
            [1, 2, 3],
            evaluator,
            MajorityVoteOptions(rounds=1),
        )
        assert abs(results[0].final_score - 0.3) < 0.01
        assert abs(results[1].final_score - 0.6) < 0.01
        assert abs(results[2].final_score - 0.9) < 0.01


# ---------------------------------------------------------------------------
# get_voting_stats
# ---------------------------------------------------------------------------


class TestGetVotingStats:
    def test_calculates_correct_statistics(self):
        results = [
            VotingResult(final_score=0.8, votes=[0.8, 0.82, 0.78], consensus=True, attempts=3),
            VotingResult(final_score=0.7, votes=[0.5, 0.7, 0.9, 0.7], consensus=False, attempts=4),
            VotingResult(final_score=0.9, votes=[0.9, 0.91, 0.89], consensus=True, attempts=3),
        ]
        stats = get_voting_stats(results)
        assert abs(stats["avg_attempts"] - 3.33) < 0.1
        assert abs(stats["consensus_rate"] - 0.67) < 0.05
        assert stats["avg_vote_spread"] > 0

    def test_empty_results(self):
        stats = get_voting_stats([])
        assert stats["avg_attempts"] == 0
        assert stats["consensus_rate"] == 0.0
        assert stats["avg_vote_spread"] == 0.0
