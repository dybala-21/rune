import json
from copy import deepcopy

import pytest

from scripts.e2e_report import load_reports, metrics, summarize


def test_failed_trials_count_toward_cost_and_unknown_cost_stays_unknown():
    def trial(outcome, cost):
        return {"provider": "test", "model": "test", "outcome": outcome, "runs": [
            {"seconds": 10, "snapshot": {"usage": {"total": 100, "cost": {"usd": cost}}, "toolCalls": []}},
        ]}
    row = summarize([trial("passed", 1), trial("failed", 3)])[0]
    assert row["success_rate"] == .5 and row["usd_per_success"] == 4
    row = summarize([trial("passed", 1), trial("failed", None)])[0]
    assert row["total_usd"] is None and row["usd_per_success"] is None and row["unpriced_trials"] == 1


def test_known_cost_includes_complete_and_partial_bills():
    runs = [{"seconds": 1, "snapshot": {"usage": {"cost": cost}}} for cost in (
        {"usd": 2}, {"usd": None, "knownUsd": 3}, {"usd": 0, "knownUsd": 9},
    )]
    result = metrics(runs)
    assert result["usd"] is None and result["known_usd"] == 5


@pytest.fixture
def report():
    return {"provider": "test", "model": "model", "outcome": "passed", "runs": [
        {"seconds": 2, "snapshot": {"usage": {"total": 10, "cost": {"usd": .1}}}},
    ], "evaluation": {"version": 1, "batch_id": "batch", "trial_id": "first",
        "source": {"commit": "abc", "content_hash": "tree"}, "source_unchanged": True,
        "scenario_hash": "scenario", "settings": {}, "environment": {}}}


def test_copied_reports_are_deduplicated_but_new_trials_are_counted(tmp_path, report):
    for name in ("original", "copy"):
        (tmp_path / f"{name}.json").write_text(json.dumps(report))
    report["evaluation"]["trial_id"] = "second"
    (tmp_path / "repeat.json").write_text(json.dumps(report))
    reports, audit = load_reports(tmp_path)
    assert len(reports) == 2 and audit["duplicates"] == 1
    assert summarize(reports)[0]["total_usd"] == .2


def test_conflicting_trial_is_excluded_instead_of_choosing_a_result(tmp_path, report):
    (tmp_path / "passed.json").write_text(json.dumps(report))
    report["outcome"] = "failed"
    (tmp_path / "failed.json").write_text(json.dumps(report))
    reports, audit = load_reports(tmp_path)
    assert reports == [] and audit["conflicting_trials"] == 1


@pytest.mark.parametrize("section,key,value", [
    ("source", "commit", "def"), ("source", "content_hash", "edited"),
    ("settings", "reasoning_effort", "high"), ("environment", "python", "3.14"),
])
def test_different_code_or_configuration_is_never_pooled(report, section, key, value):
    changed = deepcopy(report)
    changed["evaluation"][section][key] = value
    assert len(summarize([report, changed])) == 2


def test_legacy_and_interrupted_source_changes_are_visible_in_audit(tmp_path, report):
    legacy = {key: value for key, value in report.items() if key != "evaluation"}
    (tmp_path / "legacy.json").write_text(json.dumps(legacy))
    report["evaluation"]["source_unchanged"] = False
    (tmp_path / "changed.json").write_text(json.dumps(report))
    reports, audit = load_reports(tmp_path)
    assert reports == [] and audit["legacy_excluded"] == audit["source_changed"] == 1
    reports, _ = load_reports(tmp_path, include_legacy=True)
    assert len(reports) == 1 and summarize(reports)[0]["comparison_key"] == "legacy-unversioned"


def test_incomplete_trial_remains_in_success_rate_without_inventing_cost(report):
    incomplete = {**report, "outcome": "incomplete", "runs": []}
    row = summarize([report, incomplete])[0]
    assert row["success_rate"] == .5 and row["unpriced_trials"] == 1
    assert row["total_usd"] is None


def test_nested_phase_spans_are_not_added_to_model_time():
    result = metrics([{"seconds": 3, "snapshot": {"timings": {"spans": [
        {"kind": "phase", "phase": "verification", "durationMs": 2000},
        {"kind": "model", "phase": "verification", "durationMs": 1500},
    ]}}}])
    assert result["model_seconds_by_phase"] == {"verification": 1.5}


def test_interrupted_calls_keep_partial_usage_separate_from_totals(report):
    snapshot = report["runs"][0]["snapshot"]
    snapshot["timings"] = {"usage": {"calls": 2, "reported_calls": 1}}
    snapshot["usage"]["cost"] = {"usd": None, "knownUsd": .1}
    row = summarize([report])[0]
    assert row["total_tokens"] is None and row["known_tokens"] == 10
    assert row["total_usd"] is None and row["known_usd"] == .1
