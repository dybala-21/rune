"""A reproduction script only counts when it tells the candidates apart.

Observed on two instances: the generated script failed on all 33 candidates
across every run, including the ones that turned out to be correct. It was
written from the example quoted in the issue, and on those instances the
right fix deliberately does not produce that example's output. A script in
that state rejects everyone, so it selects nothing — and feeding its output
to the repair attempt as "this is what went wrong" aims the next attempt at
a requirement the fix was never supposed to meet.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rune.agent.rejection_sampler import ensure_two_graded, repro_discriminates


def _verify(results):
    return SimpleNamespace(repro_results=dict(results))


class TestDiscrimination:
    def test_unknown_until_two_candidates_are_graded(self):
        assert repro_discriminates(_verify({})) is None
        assert repro_discriminates(_verify({"a": False})) is None

    def test_a_split_verdict_discriminates(self):
        assert repro_discriminates(_verify({"a": True, "b": False})) is True

    def test_failing_on_everything_discriminates_nothing(self):
        # The measured case: too strict, rejects the correct fix too.
        assert repro_discriminates(
            _verify({"a": False, "b": False, "c": False})) is False

    def test_passing_everything_discriminates_nothing_either(self):
        # The mirror case: too loose, accepts anything.
        assert repro_discriminates(
            _verify({"a": True, "b": True, "c": True})) is False

    def test_a_verifier_without_the_field_is_handled(self):
        assert repro_discriminates(SimpleNamespace()) is None


class TestReachingAVerdict:
    """The check is worthless if it is asked before it can answer.

    Verification stops at the first candidate that passes, and an attempt
    that times out is never measured, so the decision point is normally
    reached with one verdict — and one verdict can never say whether the
    script separates anything. Three runs in four went that way, letting the
    evidence through by default.
    """

    def _verifier(self, graded, script="assert False"):
        graded = dict(graded)
        asked = []

        async def grade(cwd):
            asked.append(cwd)
            graded[cwd] = False        # this candidate fails it too
            return False

        return SimpleNamespace(repro_results=graded, repro_script=script,
                               grade_repro=grade), asked

    @pytest.mark.asyncio
    async def test_an_ungraded_candidate_is_measured_before_deciding(self):
        v, asked = self._verifier({"a": False})
        assert repro_discriminates(v) is None      # cannot answer yet
        await ensure_two_graded(v, ["a", "b"])
        assert asked == ["b"]
        assert repro_discriminates(v) is False     # now it can

    @pytest.mark.asyncio
    async def test_a_candidate_already_measured_is_not_re_run(self):
        v, asked = self._verifier({"a": True, "b": False})
        await ensure_two_graded(v, ["a", "b"])
        assert asked == []

    @pytest.mark.asyncio
    async def test_it_stops_as_soon_as_two_verdicts_exist(self):
        # Two is enough to answer; grading the rest is wasted script runs.
        v, asked = self._verifier({})
        await ensure_two_graded(v, ["a", "b", "c", "d"])
        assert asked == ["a", "b"]

    @pytest.mark.asyncio
    async def test_a_split_verdict_still_hands_the_evidence_over(self):
        v = SimpleNamespace(repro_results={"a": True, "b": False},
                            repro_script="x", grade_repro=None)
        await ensure_two_graded(v, ["a", "b"])
        assert repro_discriminates(v) is True

    @pytest.mark.asyncio
    async def test_no_repro_script_means_nothing_to_measure(self):
        v, asked = self._verifier({}, script="")
        await ensure_two_graded(v, ["a", "b"])
        assert asked == []

    @pytest.mark.asyncio
    async def test_a_verifier_without_the_hooks_is_left_alone(self):
        await ensure_two_graded(SimpleNamespace(), ["a", "b"])

