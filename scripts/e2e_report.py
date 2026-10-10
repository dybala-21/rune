"""Summarize repeated workflow reports, including failed and unpriced trials."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

if __package__:
    from scripts.e2e_provenance import digest
else:
    from e2e_provenance import digest


def load_reports(directory: Path, *, include_legacy: bool = False) -> tuple[list[dict], dict]:
    trials, conflicts = {}, set()
    audit = {"duplicates": 0, "conflicting_trials": 0, "legacy_excluded": 0,
             "source_changed": 0, "invalid_reports": 0}
    for path in sorted(directory.rglob("*.json")):
        try:
            item = json.loads(path.read_text())
            if not isinstance(item, dict) or not {"provider", "model", "outcome", "runs"} <= item.keys():
                continue
            if not isinstance(item["runs"], list) or any(not isinstance(r, dict) or
                    not isinstance(r.get("seconds"), (int, float)) or
                    not math.isfinite(r["seconds"]) or r["seconds"] < 0 or
                    not isinstance(r.get("snapshot"), dict) for r in item["runs"]):
                raise ValueError("Invalid run record")
            if item["outcome"] not in {"passed", "failed", "incomplete"}:
                raise ValueError("Invalid trial outcome")
            meta = item.get("evaluation")
            fingerprint = digest(item)
            if meta is None:
                if not include_legacy:
                    audit["legacy_excluded"] += 1
                    continue
                identity = "legacy:" + fingerprint
            else:
                if (meta.get("version") != 1 or not meta.get("trial_id") or
                        not meta.get("scenario_hash") or not meta.get("source", {}).get("commit") or
                        not meta.get("source", {}).get("content_hash") or
                        not isinstance(meta.get("settings"), dict) or not isinstance(meta.get("environment"), dict)):
                    raise ValueError("Incomplete provenance")
                identity = str(meta["batch_id"]) + ":" + str(meta["trial_id"])
                if meta.get("source_unchanged") is not True:
                    audit["source_changed"] += 1
                    continue
            if identity in trials:
                if trials[identity][0] != fingerprint:
                    conflicts.add(identity)
                else:
                    audit["duplicates"] += 1
            else:
                trials[identity] = fingerprint, item
        except (ValueError, TypeError, OSError, AttributeError, KeyError):
            audit["invalid_reports"] += 1
    audit["conflicting_trials"] = len(conflicts)
    return [item for identity, (_, item) in trials.items() if identity not in conflicts], audit


def _comparison_key(report: dict) -> str:
    meta = report.get("evaluation")
    if not meta:
        return "legacy-unversioned"
    return digest({key: meta[key] for key in ("source", "settings", "environment")})


def metrics(runs: list[dict]) -> dict:
    snapshots = [r["snapshot"] for r in runs]
    usage = [s.get("usage") or {} for s in snapshots]
    costs = [(u.get("cost") or {}).get("usd") for u in usage]
    tokens = [u.get("total") for u in usage]
    usage_records = [(s.get("timings") or {}).get("usage") or {} for s in snapshots]
    usage_complete = all(record.get("calls", 0) == record.get("reported_calls", 0)
                         for record in usage_records)
    calls = [call for s in snapshots for call in s.get("toolCalls", [])]
    timings = [span for s in snapshots for span in (s.get("timings") or {}).get("spans", [])]
    spans = [s for s in timings if s.get("kind") == "model"]
    phases = defaultdict(float)
    for span in spans:
        phases[span.get("phase", "unknown")] += span.get("durationMs", 0) / 1000
    first_text = [r["first_text_seconds"] for r in runs if r.get("first_text_seconds") is not None]
    return {
        "seconds": sum(r["seconds"] for r in runs),
        "tokens": sum(tokens) if tokens and all(t is not None for t in tokens) and usage_complete else None,
        "known_tokens": sum(t or 0 for t in tokens),
        "usd": sum(costs) if costs and all(c is not None for c in costs) else None,
        "known_usd": sum((u.get("cost") or {}).get("usd") if (u.get("cost") or {}).get("usd") is not None
                         else (u.get("cost") or {}).get("knownUsd") or 0 for u in usage),
        "model_calls": len(spans), "failed_model_calls": sum(s.get("status") == "interrupted" for s in spans),
        "tool_calls": len(calls), "failed_tool_calls": sum(c.get("success") is False for c in calls),
        "gui_calls": sum(c.get("toolName", "").startswith(("browser", "desktop", "computer")) for c in calls),
        "cache_read_tokens": sum(u.get("cacheRead") or 0 for u in usage),
        "cache_write_tokens": sum(u.get("cacheCreation") or 0 for u in usage),
        "model_seconds_by_phase": dict(phases),
        "median_first_text_seconds": statistics.median(first_text) if first_text else None,
    }


def summarize(reports: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for report in reports:
        groups[(report["provider"], report["model"], report.get("decision_backend"), _comparison_key(report))].append(report)
    rows = []
    for (provider, model, backend, comparison), trials in groups.items():
        values = [metrics(t["runs"]) for t in trials]
        seconds = [v["seconds"] for v in values]
        passed = sum(t.get("outcome") == "passed" for t in trials)
        costs = [v["usd"] for v in values]
        total = sum(costs) if all(c is not None for c in costs) else None
        phase_names = {phase for value in values for phase in value["model_seconds_by_phase"]}
        rows.append({"provider": provider, "model": model, "decision_backend": backend,
                     "comparison_key": comparison, "source": trials[0].get("evaluation", {}).get("source"),
                     "settings": trials[0].get("evaluation", {}).get("settings"),
                     "environment": trials[0].get("evaluation", {}).get("environment"),
                     "trials": len(trials), "passed": passed, "success_rate": passed / len(trials),
                     "median_seconds": statistics.median(seconds), "min_seconds": min(seconds), "max_seconds": max(seconds),
                     "total_usd": total, "usd_per_success": total / passed if total is not None and passed else None,
                     "known_usd": sum(v["known_usd"] for v in values),
                     "unpriced_trials": sum(c is None for c in costs),
                     "model_calls": sum(v["model_calls"] for v in values),
                     "tool_failures": sum(v["failed_tool_calls"] for v in values),
                     "gui_calls": sum(v["gui_calls"] for v in values),
                     "cache_read_tokens": sum(v["cache_read_tokens"] for v in values),
                     "cache_write_tokens": sum(v["cache_write_tokens"] for v in values),
                     "model_seconds_by_phase": {p: sum(v["model_seconds_by_phase"].get(p, 0) for v in values) for p in sorted(phase_names)},
                     "known_tokens": sum(v["known_tokens"] for v in values),
                     "total_tokens": sum(v["tokens"] for v in values) if all(v["tokens"] is not None for v in values) else None})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--include-legacy", action="store_true", help="Include old reports separately; their code versions are unknown")
    args = parser.parse_args()
    reports, audit = load_reports(args.directory, include_legacy=args.include_legacy)
    cases = defaultdict(list)
    for report in reports:
        # Group parameterized trials under the same scenario.
        name = report.get("case", "unknown").removeprefix(report["provider"] + "-").split("[", 1)[0]
        if meta := report.get("evaluation"):
            name += ":" + meta["scenario_hash"][:12]
        cases[name].append(report)
    print(json.dumps({"scope": "Measured model-token cost; excludes machine and external service costs. Known usage is a lower bound when calls are unreported. Model durations can overlap; tool failures include expected failing baseline tests. Small samples are not reliability guarantees. Compare matching scenarios, not unequal provider totals.",
                      "audit": audit, "groups": summarize(reports), "by_case": {name: summarize(rows) for name, rows in cases.items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
