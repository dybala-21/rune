"""Features come from what a run did, never from how it described itself.

A finished run that failed and one that succeeded both end in a confident
summary, so the summary is the one thing excluded here. What is left is the
tool-call sequence: reading the same things repeatedly and stopping looks
different from writing, checking, and writing again.
"""

from __future__ import annotations

import json

from rune.agent.completion_risk import (
    RiskModel,
    auroc,
    calls_from_log,
    features,
    train,
)

GAVE_UP = ["file_read", "file_read", "file_list", "file_read", "file_read"]
DID_WORK = ["file_read", "file_edit", "bash_execute", "file_edit",
            "bash_execute"]


class TestFeatures:
    def test_no_prose_reaches_the_features(self):
        f = features(DID_WORK)
        assert all(not k.startswith(("msg", "text", "summary")) for k in f)
        assert any(k.startswith("t:") for k in f)
        assert any(k.startswith("b:") for k in f)

    def test_a_run_that_never_wrote_is_marked(self):
        assert features(GAVE_UP)["no_writes"] == 1.0
        assert features(DID_WORK)["no_writes"] == 0.0

    def test_write_then_check_then_write_is_recognised(self):
        assert features(DID_WORK)["write_check_write"] == 1.0
        assert features(GAVE_UP)["write_check_write"] == 0.0

    def test_repeated_calls_are_measured(self):
        assert features(GAVE_UP)["repeat_frac"] > features(DID_WORK)["repeat_frac"]

    def test_an_empty_run_is_handled(self):
        f = features([])
        assert f["empty"] == 1.0

    def test_long_runs_are_bounded(self):
        f = features(["file_read"] * 5000)
        assert f["t:file_read"] <= 400


class TestModel:
    def test_it_separates_the_two_shapes_it_was_taught(self):
        samples = [(GAVE_UP, 1), (DID_WORK, 0)] * 12
        m = RiskModel(weights=train(samples, epochs=300))
        assert m.score(GAVE_UP) > m.score(DID_WORK)

    def test_scores_stay_in_range(self):
        m = RiskModel(weights={"bias": 500.0})
        assert 0.0 <= m.score(DID_WORK) <= 1.0
        m2 = RiskModel(weights={"bias": -500.0})
        assert 0.0 <= m2.score(DID_WORK) <= 1.0

    def test_no_weights_means_no_opinion(self, tmp_path):
        # Silence beats a miscalibrated warning on honest work.
        assert RiskModel.load(tmp_path / "absent.json") is None
        bad = tmp_path / "empty.json"
        bad.write_text(json.dumps({"weights": {}}))
        assert RiskModel.load(bad) is None

    def test_weights_round_trip(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"weights": {"bias": 1.0}, "threshold": 0.7}))
        m = RiskModel.load(p)
        assert m is not None and m.threshold == 0.7


class TestAuroc:
    def test_perfect_and_useless_separation(self):
        assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
        assert auroc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5

    def test_single_class_is_undefined_and_says_so(self):
        assert auroc([0.9, 0.1], [1, 1]) == 0.5


class TestLogParsing:
    def test_tool_names_are_read_in_order(self):
        log = ("info file_read path=a.py\n"
               "info file_edit path=a.py\n"
               "info bash_execute command=pytest\n")
        assert calls_from_log(log) == ["file_read", "file_edit", "bash_execute"]

    def test_unrelated_text_contributes_nothing(self):
        assert calls_from_log("the agent thought about the problem") == []
