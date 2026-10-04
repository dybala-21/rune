from scripts.e2e_report import metrics, summarize


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
