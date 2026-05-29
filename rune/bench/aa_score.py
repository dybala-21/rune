"""Artificial Analysis score reporting helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rune.bench.aa_manifest import build_aa_attempt_matrix
from rune.bench.summary import summarize_paths

DEFAULT_COMPONENT = "terminal-bench-v2"
TOKEN_COLUMNS = (
    "total_tokens_used",
    "input_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "output_tokens",
)
TIME_COLUMNS = (
    "duration_ms",
    "environment_setup_ms",
    "agent_setup_ms",
    "agent_execution_ms",
    "verifier_ms",
    "agent_wall_time_ms",
)
INVALID_SAMPLE_LIMIT = 20


def score_paths(
    paths: Iterable[Path],
    *,
    component: str = DEFAULT_COMPONENT,
    require_fingerprint: bool = True,
) -> dict[str, Any]:
    """Summarize artifact paths and calculate an AA component score report."""
    summary = summarize_paths(paths)
    return score_summary(
        summary,
        component=component,
        require_fingerprint=require_fingerprint,
    )


def score_summary(
    summary: dict[str, Any],
    *,
    component: str = DEFAULT_COMPONENT,
    require_fingerprint: bool = True,
) -> dict[str, Any]:
    """Calculate an AA component score report from ``bench summarize`` output."""
    spec = _component_spec(component)
    rows = [row for row in summary.get("rows", []) if isinstance(row, dict)]
    component_rows, ignored_rows = _partition_component_rows(rows, spec)

    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    invalid_reasons: Counter[str] = Counter()

    for row in component_rows:
        reasons = _invalid_reasons(row, spec, require_fingerprint=require_fingerprint)
        if reasons:
            invalid_rows.append(row)
            invalid_reasons.update(reasons)
        else:
            valid_rows.append(row)

    task_counts = Counter(str(row["task_id"]) for row in valid_rows)
    missing_task_ids = [task_id for task_id in spec["task_ids"] if task_counts[task_id] == 0]
    incomplete_task_ids = {
        task_id: count
        for task_id, count in task_counts.items()
        if 0 < count < spec["repeats"]
    }
    extra_attempt_task_ids = {
        task_id: count
        for task_id, count in task_counts.items()
        if count > spec["repeats"]
    }

    scored = _score_block(valid_rows)
    official_blockers = _official_blockers(
        spec=spec,
        component_rows=component_rows,
        valid_rows=valid_rows,
        invalid_rows=invalid_rows,
        missing_task_ids=missing_task_ids,
        incomplete_task_ids=incomplete_task_ids,
        extra_attempt_task_ids=extra_attempt_task_ids,
        require_fingerprint=require_fingerprint,
    )

    return {
        "component": spec["component"],
        "benchmark": spec["benchmark"],
        "score_mode": "official" if require_fingerprint else "exploratory",
        "official_ready": not official_blockers,
        "official_blockers": official_blockers,
        "expected_tasks": len(spec["task_ids"]),
        "expected_repeats": spec["repeats"],
        "expected_attempts": spec["expected_attempts"],
        "found_component_attempts": len(component_rows),
        "ignored_attempts": len(ignored_rows),
        "scored_attempts": len(valid_rows),
        "invalid_attempts": len(invalid_rows),
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "coverage": {
            "scored_tasks": sum(1 for task_id in spec["task_ids"] if task_counts[task_id] > 0),
            "missing_task_count": len(missing_task_ids),
            "missing_task_ids": missing_task_ids,
            "incomplete_task_ids": incomplete_task_ids,
            "extra_attempt_task_ids": extra_attempt_task_ids,
        },
        "score": scored,
        "invalid_samples": _invalid_samples(invalid_rows, spec, require_fingerprint),
    }


def format_score_text(report: dict[str, Any]) -> str:
    """Return a compact human-readable AA score report."""
    raw_score = report.get("score")
    score: dict[str, Any] = raw_score if isinstance(raw_score, dict) else {}
    lines = [
        f"Component: {report.get('component')}",
        f"Benchmark: {report.get('benchmark')}",
        f"Mode: {report.get('score_mode')}",
        f"Official ready: {str(report.get('official_ready')).lower()}",
        (
            f"Score: {score.get('score_percent')}% "
            f"({score.get('passed')}/{score.get('scored_attempts')} passed)"
        ),
        (
            f"Attempts: {report.get('scored_attempts')} scored, "
            f"{report.get('invalid_attempts')} invalid, "
            f"{report.get('expected_attempts')} expected"
        ),
        f"Budget exhausted: {score.get('budget_exhausted_count')}",
    ]
    blockers = report.get("official_blockers")
    if isinstance(blockers, list) and blockers:
        lines.append("Blockers:")
        lines.extend(f"- {blocker}" for blocker in blockers)
    return "\n".join(lines) + "\n"


def _component_spec(component: str) -> dict[str, Any]:
    key = component.lower()
    matrix = [
        row
        for row in build_aa_attempt_matrix()
        if str(row["component"]).lower() == key or str(row["benchmark"]).lower() == key
    ]
    if not matrix:
        raise ValueError(f"unknown AA component or benchmark: {component}")

    task_ids = list(dict.fromkeys(str(row["task_id"]) for row in matrix))
    repeats = max(int(row["attempt_index"]) for row in matrix)
    return {
        "component": str(matrix[0]["component"]),
        "benchmark": str(matrix[0]["benchmark"]),
        "task_ids": task_ids,
        "task_id_set": set(task_ids),
        "repeats": repeats,
        "expected_attempts": len(matrix),
    }


def _partition_component_rows(
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    component_rows: list[dict[str, Any]] = []
    ignored_rows: list[dict[str, Any]] = []
    benchmark = spec["benchmark"]
    task_id_set = spec["task_id_set"]

    for row in rows:
        task_id = row.get("task_id")
        if row.get("benchmark") == benchmark or task_id in task_id_set:
            component_rows.append(row)
        else:
            ignored_rows.append(row)
    return component_rows, ignored_rows


def _invalid_reasons(
    row: dict[str, Any],
    spec: dict[str, Any],
    *,
    require_fingerprint: bool,
) -> list[str]:
    reasons: list[str] = []
    task_id = row.get("task_id")
    benchmark = row.get("benchmark")
    if benchmark is not None and benchmark != spec["benchmark"]:
        reasons.append("benchmark_mismatch")
    if not isinstance(task_id, str) or not task_id:
        reasons.append("missing_task_id")
    elif task_id not in spec["task_id_set"]:
        reasons.append("unknown_task_id")

    if row.get("verifier_disabled") is True:
        reasons.append("verifier_disabled")
    if not isinstance(row.get("reward"), int | float):
        reasons.append("missing_reward")

    fingerprint_valid = row.get("fingerprint_gate_valid")
    if fingerprint_valid is False:
        reasons.append("fingerprint_invalid")
    elif require_fingerprint and fingerprint_valid is not True:
        reasons.append("fingerprint_missing")
    return reasons


def _score_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(row["reward"]) for row in rows if isinstance(row.get("reward"), int | float)]
    passed = sum(1 for reward in rewards if reward == 1.0)
    failed = len(rewards) - passed
    mean_reward = sum(rewards) / len(rewards) if rewards else None

    return {
        "score": mean_reward,
        "score_percent": round(mean_reward * 100, 2) if mean_reward is not None else None,
        "mean_reward": mean_reward,
        "scored_attempts": len(rewards),
        "passed": passed,
        "failed": failed,
        "timeout_count": sum(1 for row in rows if _is_timeout(row)),
        "budget_exhausted_count": sum(1 for row in rows if _is_budget_exhausted(row)),
        "errored_count": sum(1 for row in rows if row.get("exception_type")),
        "tokens": _metric_totals_and_means(rows, TOKEN_COLUMNS),
        "time_ms": _metric_totals_and_means(rows, TIME_COLUMNS),
    }


def _metric_totals_and_means(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in columns:
        values = [int(row[column]) for row in rows if isinstance(row.get(column), int)]
        result[column] = {
            "total": sum(values) if values else None,
            "mean": (sum(values) / len(values)) if values else None,
            "count": len(values),
        }
    return result


def _is_timeout(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("trace_reason", "exception_type", "exception_message")
    ).lower()
    return "timeout" in text or "timed out" in text


def _is_budget_exhausted(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("trace_reason", "exception_type", "exception_message")
    ).lower()
    return "token_budget_exhausted" in text or "token budget" in text


def _official_blockers(
    *,
    spec: dict[str, Any],
    component_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    missing_task_ids: list[str],
    incomplete_task_ids: dict[str, int],
    extra_attempt_task_ids: dict[str, int],
    require_fingerprint: bool,
) -> list[str]:
    blockers: list[str] = []
    if not require_fingerprint:
        blockers.append("fingerprint validation is disabled")
    if len(component_rows) != spec["expected_attempts"]:
        blockers.append(
            f"expected {spec['expected_attempts']} component attempts, "
            f"found {len(component_rows)}"
        )
    if len(valid_rows) != spec["expected_attempts"]:
        blockers.append(
            f"expected {spec['expected_attempts']} valid scored attempts, "
            f"found {len(valid_rows)}"
        )
    if invalid_rows:
        blockers.append(f"{len(invalid_rows)} component attempts are invalid")
    if missing_task_ids:
        blockers.append(f"{len(missing_task_ids)} tasks have no valid scored attempt")
    if incomplete_task_ids:
        blockers.append(f"{len(incomplete_task_ids)} tasks have fewer than expected repeats")
    if extra_attempt_task_ids:
        blockers.append(f"{len(extra_attempt_task_ids)} tasks have extra valid attempts")
    return blockers


def _invalid_samples(
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
    require_fingerprint: bool,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in rows[:INVALID_SAMPLE_LIMIT]:
        samples.append(
            {
                "task_id": row.get("task_id"),
                "trial_name": row.get("trial_name"),
                "reward": row.get("reward"),
                "trace_reason": row.get("trace_reason"),
                "exception_type": row.get("exception_type"),
                "reasons": _invalid_reasons(row, spec, require_fingerprint=require_fingerprint),
            }
        )
    return samples
