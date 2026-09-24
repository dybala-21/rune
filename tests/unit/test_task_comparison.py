"""Check comparison results, including failures and missing cost data."""

from scripts.e2e_core import comparison_summary, provider_blocker


def test_code_comparison_requires_a_baseline_before_the_edit(tmp_path):
    import subprocess
    import sys

    from rune.agent.verification_state import VerificationState
    from scripts.e2e_core import prepare, verified_code_comparison

    prepare("code", tmp_path)
    command = "python3 -B -m unittest -v"
    args = [sys.executable, "-B", "-m", "unittest", "-v"]
    before = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True)
    state = VerificationState()
    state.observe_command(command, before.returncode == 0, before.stdout + before.stderr, str(tmp_path))
    state.changed()
    (tmp_path / "stats.py").write_text(
        'def average(values):\n    if not values: raise ValueError("empty")\n    return sum(values) / len(values)\n')
    after = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True)
    assert before.returncode == 1 and after.returncode == 0
    state.observe_command(command, True, after.stdout + after.stderr, str(tmp_path))
    assert verified_code_comparison(state)

    missing_baseline = VerificationState()
    missing_baseline.changed()
    missing_baseline.observe_command(command, True, after.stdout + after.stderr, str(tmp_path))
    assert missing_baseline.passed and not verified_code_comparison(missing_baseline)
    state.changed()
    assert not verified_code_comparison(state)


def test_failed_attempts_are_included_in_latency_and_cost():
    rows = [{"model": "m", "scenario": "csv", "requested_backend": "jev", "passed": passed,
             "seconds": seconds, "usage": {"cost": {"usd": cost, "knownUsd": cost}}}
            for passed, seconds, cost in [(True, 10, .1), (False, 50, .3)]]
    summary, = comparison_summary(rows)
    assert summary["passed"] == 1 and summary["attempts"] == 2
    assert summary["median_seconds"] == 30 and summary["mean_usd"] == .2


def test_missing_usage_is_not_counted_as_free():
    rows = [{"model": "m", "scenario": "code", "requested_backend": "connected", "passed": False,
             "seconds": 190},
            {"model": "m", "scenario": "code", "requested_backend": "connected", "passed": True,
             "seconds": 10, "usage": {"cost": {"usd": .1, "knownUsd": .1}}}]
    summary, = comparison_summary(rows)
    assert summary["mean_usd"] is None and summary["known_usd"] == .1
    assert summary["unpriced_attempts"] == 1 and summary["median_seconds"] == 100


def test_skipped_comparisons_are_not_counted_as_attempts_or_free_successes():
    summary, = comparison_summary([{"model": "m", "scenario": "code", "requested_backend": "jev",
                                    "passed": False, "skipped": "provider_access_denied"}])
    assert summary["attempts"] == 0 and summary["skipped"] == 1
    assert summary["mean_usd"] is None and summary["median_seconds"] is None


def test_provider_failure_stops_further_comparisons_but_transient_recovery_does_not():
    report = {"passed": False, "tools": [], "timings": {"spans": [{"error": {"status": 429}}]}}
    assert provider_blocker(report) is None
    report["timings"]["spans"].append({"error": {"status": 429}})
    assert provider_blocker(report) == "repeated_rate_limit"
    assert provider_blocker({**report, "passed": True}) is None
    for status in (400, 401, 402, 403):
        assert provider_blocker({"passed": False, "tools": [],
                                 "timings": {"spans": [{"error": {"status": status}}]}})
