"""Summarize repeated workflow reports, including failed and unpriced trials."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def metrics(runs: list[dict]) -> dict:
    snapshots = [r["snapshot"] for r in runs]
    usage = [s.get("usage") or {} for s in snapshots]
    costs = [(u.get("cost") or {}).get("usd") for u in usage]
    tokens = [u.get("total") for u in usage]
    calls = [call for s in snapshots for call in s.get("toolCalls", [])]
    spans = [span for s in snapshots for span in (s.get("timings") or {}).get("spans", [])
             if span.get("kind") == "model"]
    return {
        "seconds": sum(r["seconds"] for r in runs),
        "tokens": sum(tokens) if tokens and all(t is not None for t in tokens) else None,
        "usd": sum(costs) if costs and all(c is not None for c in costs) else None,
        "known_usd": sum((u.get("cost") or {}).get("usd") if (u.get("cost") or {}).get("usd") is not None
                         else (u.get("cost") or {}).get("knownUsd") or 0 for u in usage),
        "model_calls": len(spans), "failed_model_calls": sum(s.get("status") == "interrupted" for s in spans),
        "tool_calls": len(calls), "failed_tool_calls": sum(c.get("success") is False for c in calls),
        "gui_calls": sum(c["toolName"].startswith(("browser", "desktop", "computer")) for c in calls),
        "cache_read_tokens": sum(u.get("cacheRead") or 0 for u in usage),
        "cache_write_tokens": sum(u.get("cacheCreation") or 0 for u in usage),
    }


def summarize(reports: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for report in reports:
        groups[(report["provider"], report["model"], report.get("decision_backend"))].append(report)
    rows = []
    for (provider, model, backend), trials in groups.items():
        values = [metrics(t["runs"]) for t in trials]
        seconds = [v["seconds"] for v in values]
        passed = sum(t.get("outcome") == "passed" for t in trials)
        costs = [v["usd"] for v in values]
        total = sum(costs) if all(c is not None for c in costs) else None
        rows.append({"provider": provider, "model": model, "decision_backend": backend,
                     "trials": len(trials), "passed": passed, "success_rate": passed / len(trials),
                     "median_seconds": statistics.median(seconds), "min_seconds": min(seconds), "max_seconds": max(seconds),
                     "total_usd": total, "usd_per_success": total / passed if total is not None and passed else None,
                     "unpriced_trials": sum(c is None for c in costs),
                     "model_calls": sum(v["model_calls"] for v in values),
                     "tool_failures": sum(v["failed_tool_calls"] for v in values),
                     "total_tokens": sum(v["tokens"] for v in values) if all(v["tokens"] is not None for v in values) else None})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    reports = []
    for path in args.directory.rglob("*.json"):
        item = json.loads(path.read_text())
        if isinstance(item, dict) and {"provider", "model", "outcome", "runs"} <= item.keys():
            item.setdefault("case", path.stem)
            reports.append(item)
    cases = defaultdict(list)
    for report in reports:
        # Parameter suffixes distinguish trials but not scenario names.
        name = report["case"].removeprefix(report["provider"] + "-").split("[", 1)[0]
        cases[name].append(report)
    print(json.dumps({"scope": "Measured model-token cost; excludes machine and external service costs. Small samples are not reliability guarantees. Compare matching scenarios, not unequal provider totals.",
                      "groups": summarize(reports), "by_case": {name: summarize(rows) for name, rows in cases.items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
