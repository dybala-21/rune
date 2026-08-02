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

from rune.agent.rejection_sampler import repro_discriminates


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


class TestRepairEvidence:
    def _harness(self, monkeypatch, verify, evidence):
        """Reproduce best_of's evidence selection against a given verifier."""
        import rune.cli.best_of as bo

        captured = {}

        def fake_log_info(name, **kw):
            captured[name] = kw
        monkeypatch.setattr(bo.log, "info", fake_log_info)

        from rune.agent.rejection_sampler import repro_discriminates as rd
        if rd(verify) is False:
            bo.log.info("repro_non_discriminating")
            return "", captured
        return (list(evidence.values())[-1] if evidence else ""), captured

    def test_evidence_is_withheld_when_the_repro_told_us_nothing(self, monkeypatch):
        ev, captured = self._harness(
            monkeypatch,
            _verify({"a": False, "b": False, "c": False}),
            {"c": "AssertionError: expected array, got NotImplementedError"},
        )
        assert ev == ""
        assert "repro_non_discriminating" in captured

    def test_evidence_is_passed_on_when_the_repro_separated_them(self, monkeypatch):
        ev, _ = self._harness(
            monkeypatch,
            _verify({"a": True, "b": False}),
            {"b": "AssertionError: still wrong"},
        )
        assert "still wrong" in ev
