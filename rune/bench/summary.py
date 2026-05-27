"""Summarize benchmark trial and RUNE attempt artifacts."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from rune.bench.audit import audit_attempt_dir

CSV_COLUMNS = [
    "benchmark",
    "task_id",
    "trial_name",
    "reward",
    "passed",
    "trace_reason",
    "total_tokens_used",
    "input_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "output_tokens",
    "tool_call_count",
    "audit_finding_count",
    "audit_high_severity_count",
    "fingerprint_gate_valid",
    "agent_variant_id",
    "install_mode",
    "wheelhouse_sha256",
    "source_git_sha",
    "source_diff_sha256",
    "patch_bytes",
    "final_answer_chars",
    "duration_ms",
    "environment_setup_ms",
    "agent_setup_ms",
    "agent_execution_ms",
    "verifier_ms",
    "agent_wall_time_ms",
    "verifier_disabled",
    "exception_type",
    "exception_message",
    "attempt_dir",
    "trial_dir",
    "job_dir",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _jsonl_count(path: Path) -> int:
    count = 0
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    count += 1
    except OSError:
        pass
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        pass
    return rows


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration_ms(section: Any) -> int | None:
    if not isinstance(section, dict):
        return None
    started_at = _parse_datetime(section.get("started_at"))
    finished_at = _parse_datetime(section.get("finished_at"))
    if started_at is None or finished_at is None:
        return None
    return int((finished_at - started_at).total_seconds() * 1000)


def _is_attempt_dir(path: Path) -> bool:
    return (path / "task.json").is_file() and (
        (path / "completion_trace.json").is_file()
        or (path / "fingerprint.json").is_file()
        or (path / "fingerprint_gate.json").is_file()
    )


def _iter_attempt_dirs(path: Path) -> Iterable[Path]:
    if path.is_file():
        path = path.parent
    if _is_attempt_dir(path):
        yield path
        return

    for task_file in sorted(path.rglob("task.json")):
        candidate = task_file.parent
        if _is_attempt_dir(candidate):
            yield candidate


def _is_harbor_trial_result(result: dict[str, Any]) -> bool:
    return bool(result.get("trial_name") and result.get("task_name"))


def _iter_harbor_trial_result_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.name == "result.json" and _is_harbor_trial_result(_read_json(path)):
            yield path
        return

    direct = path / "result.json"
    if direct.is_file() and _is_harbor_trial_result(_read_json(direct)):
        yield direct

    for result_file in sorted(path.rglob("result.json")):
        if result_file == direct:
            continue
        if _is_harbor_trial_result(_read_json(result_file)):
            yield result_file


def _normalise_benchmark(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if value == "terminal-bench":
        return "terminal-bench-v2"
    return value


def _task_id_from_harbor_result(result: dict[str, Any]) -> str | None:
    task_id = result.get("task_id")
    if isinstance(task_id, dict):
        path = task_id.get("path")
        if isinstance(path, str) and path:
            return Path(path).name
    task_name = result.get("task_name")
    return task_name if isinstance(task_name, str) and task_name else None


def _extract_reward(result: dict[str, Any]) -> float | None:
    verifier_result = result.get("verifier_result")
    if not isinstance(verifier_result, dict):
        return None
    rewards = verifier_result.get("rewards")
    if not isinstance(rewards, dict):
        return None
    reward = rewards.get("reward")
    if isinstance(reward, int | float):
        return float(reward)
    return None


def _extract_exception(result: dict[str, Any]) -> tuple[str | None, str | None]:
    exception_info = result.get("exception_info")
    if exception_info is None:
        return None, None
    if isinstance(exception_info, dict):
        exception_type = exception_info.get("type") or exception_info.get("class_name")
        exception_message = exception_info.get("message") or exception_info.get("detail")
        return (
            str(exception_type) if exception_type is not None else None,
            str(exception_message) if exception_message is not None else json.dumps(exception_info),
        )
    return type(exception_info).__name__, str(exception_info)


def _verifier_disabled(result: dict[str, Any]) -> bool | None:
    config = result.get("config")
    if not isinstance(config, dict):
        return None
    verifier = config.get("verifier")
    if not isinstance(verifier, dict):
        return None
    disabled = verifier.get("disable")
    return disabled if isinstance(disabled, bool) else None


def _jsonl_int_sum(rows: list[dict[str, Any]], key: str) -> int | None:
    values: list[int] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, int):
            values.append(value)
    return sum(values) if values else None


def _model_usage_metrics(path: Path) -> dict[str, int | None]:
    rows = [row for row in _read_jsonl(path) if row.get("event") == "model_usage"]
    return {
        "input_tokens": _jsonl_int_sum(rows, "input_tokens"),
        "cached_input_tokens": _jsonl_int_sum(rows, "cached_input_tokens"),
        "cache_write_tokens": _jsonl_int_sum(rows, "cache_write_tokens"),
        "reasoning_tokens": _jsonl_int_sum(rows, "reasoning_tokens"),
        "output_tokens": _jsonl_int_sum(rows, "output_tokens"),
    }


def _attempt_metrics(attempt_dir: Path | None) -> dict[str, Any]:
    if attempt_dir is None:
        return {
            "trace_reason": None,
            "trace_final_step": None,
            "total_tokens_used": None,
            "input_tokens": None,
            "cached_input_tokens": None,
            "cache_write_tokens": None,
            "reasoning_tokens": None,
            "output_tokens": None,
            "agent_wall_time_ms": None,
            "tool_call_count": None,
            "audit_finding_count": None,
            "audit_high_severity_count": None,
            "fingerprint_gate_valid": None,
            "agent_variant_id": None,
            "install_mode": None,
            "wheelhouse_sha256": None,
            "source_git_sha": None,
            "source_diff_sha256": None,
            "patch_bytes": None,
            "final_answer_chars": None,
        }

    trace = _read_json(attempt_dir / "completion_trace.json")
    timing = _read_json(attempt_dir / "timing.json")
    patch = _read_text(attempt_dir / "patch.diff")
    final_answer = _read_text(attempt_dir / "final_answer.txt")
    fingerprint = _read_json(attempt_dir / "fingerprint.json")
    fingerprint_gate = _read_json(attempt_dir / "fingerprint_gate.json")
    install = fingerprint.get("install_fingerprint")
    install = install if isinstance(install, dict) else {}
    audit = audit_attempt_dir(attempt_dir)
    usage = _model_usage_metrics(attempt_dir / "model_usage.jsonl")

    return {
        "trace_reason": trace.get("reason"),
        "trace_final_step": trace.get("final_step"),
        "total_tokens_used": trace.get("total_tokens_used"),
        **usage,
        "agent_wall_time_ms": timing.get("agent_wall_time_ms"),
        "tool_call_count": _jsonl_count(attempt_dir / "tool_calls.jsonl"),
        "audit_finding_count": audit.get("finding_count"),
        "audit_high_severity_count": audit.get("high_severity_count"),
        "fingerprint_gate_valid": fingerprint_gate.get("valid"),
        "agent_variant_id": fingerprint.get("agent_variant_id"),
        "install_mode": install.get("install_mode"),
        "wheelhouse_sha256": install.get("wheelhouse_sha256"),
        "source_git_sha": install.get("source_git_sha"),
        "source_diff_sha256": install.get("source_diff_sha256"),
        "patch_bytes": len(patch.encode("utf-8")),
        "final_answer_chars": len(final_answer),
    }


def _find_attempt_for_trial(trial_dir: Path, task_id: str | None) -> Path | None:
    attempts = list(_iter_attempt_dirs(trial_dir / "agent"))
    if not attempts:
        return None
    if task_id:
        matching = [
            attempt
            for attempt in attempts
            if str(_read_json(attempt / "task.json").get("task_id", "")) == task_id
        ]
        if matching:
            return sorted(matching)[-1]
    return sorted(attempts)[-1]


def _row_from_harbor_result(result_file: Path) -> dict[str, Any]:
    result = _read_json(result_file)
    trial_dir = result_file.parent
    config = result.get("config")
    trials_dir = config.get("trials_dir") if isinstance(config, dict) else None
    task_id = _task_id_from_harbor_result(result)
    attempt_dir = _find_attempt_for_trial(trial_dir, task_id)
    reward = _extract_reward(result)
    exception_type, exception_message = _extract_exception(result)

    row = {
        "benchmark": _normalise_benchmark(result.get("source")),
        "task_id": task_id,
        "trial_name": result.get("trial_name"),
        "reward": reward,
        "passed": reward == 1.0 if reward is not None else None,
        "verifier_disabled": _verifier_disabled(result),
        "exception_type": exception_type,
        "exception_message": exception_message,
        "started_at": result.get("started_at"),
        "finished_at": result.get("finished_at"),
        "duration_ms": _duration_ms(result),
        "environment_setup_ms": _duration_ms(result.get("environment_setup", {})),
        "agent_setup_ms": _duration_ms(result.get("agent_setup", {})),
        "agent_execution_ms": _duration_ms(result.get("agent_execution", {})),
        "verifier_ms": _duration_ms(result.get("verifier", {})),
        "attempt_dir": str(attempt_dir) if attempt_dir else None,
        "trial_dir": str(trial_dir),
        "job_dir": str(Path(trials_dir if isinstance(trials_dir, str) else trial_dir.parent)),
    }
    row.update(_attempt_metrics(attempt_dir))
    return row


def _row_from_attempt_dir(attempt_dir: Path) -> dict[str, Any]:
    task = _read_json(attempt_dir / "task.json")
    row = {
        "benchmark": task.get("benchmark"),
        "task_id": task.get("task_id"),
        "trial_name": None,
        "reward": None,
        "passed": None,
        "verifier_disabled": None,
        "exception_type": None,
        "exception_message": None,
        "started_at": None,
        "finished_at": None,
        "duration_ms": None,
        "environment_setup_ms": None,
        "agent_setup_ms": None,
        "agent_execution_ms": None,
        "verifier_ms": None,
        "attempt_dir": str(attempt_dir),
        "trial_dir": None,
        "job_dir": None,
    }
    row.update(_attempt_metrics(attempt_dir))
    return row


def summarize_paths(paths: Iterable[Path]) -> dict[str, Any]:
    """Summarize Harbor trial results and local RUNE attempt artifacts under paths."""
    rows: list[dict[str, Any]] = []
    seen_trial_results: set[Path] = set()
    seen_attempt_dirs: set[Path] = set()

    input_paths = [Path(path) for path in paths]
    for path in input_paths:
        for result_file in _iter_harbor_trial_result_files(path):
            resolved_result = result_file.resolve()
            if resolved_result in seen_trial_results:
                continue
            seen_trial_results.add(resolved_result)
            row = _row_from_harbor_result(result_file)
            rows.append(row)
            if row.get("attempt_dir"):
                seen_attempt_dirs.add(Path(str(row["attempt_dir"])).resolve())

    for path in input_paths:
        for attempt_dir in _iter_attempt_dirs(path):
            resolved_attempt = attempt_dir.resolve()
            if resolved_attempt in seen_attempt_dirs:
                continue
            seen_attempt_dirs.add(resolved_attempt)
            rows.append(_row_from_attempt_dir(attempt_dir))

    return _aggregate_rows(rows)


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [row["reward"] for row in rows if isinstance(row.get("reward"), int | float)]
    return {
        "count": len(rows),
        "passed": sum(1 for row in rows if row.get("passed") is True),
        "failed": sum(1 for row in rows if row.get("passed") is False),
        "errored": sum(1 for row in rows if row.get("exception_type")),
        "mean_reward": sum(rewards) / len(rewards) if rewards else None,
        "total_tokens_used": sum(
            int(row["total_tokens_used"])
            for row in rows
            if isinstance(row.get("total_tokens_used"), int)
        ),
        "total_input_tokens": sum(
            int(row["input_tokens"])
            for row in rows
            if isinstance(row.get("input_tokens"), int)
        ),
        "total_cached_input_tokens": sum(
            int(row["cached_input_tokens"])
            for row in rows
            if isinstance(row.get("cached_input_tokens"), int)
        ),
        "total_cache_write_tokens": sum(
            int(row["cache_write_tokens"])
            for row in rows
            if isinstance(row.get("cache_write_tokens"), int)
        ),
        "total_reasoning_tokens": sum(
            int(row["reasoning_tokens"])
            for row in rows
            if isinstance(row.get("reasoning_tokens"), int)
        ),
        "total_output_tokens": sum(
            int(row["output_tokens"])
            for row in rows
            if isinstance(row.get("output_tokens"), int)
        ),
        "total_tool_calls": sum(
            int(row["tool_call_count"])
            for row in rows
            if isinstance(row.get("tool_call_count"), int)
        ),
        "audit_finding_count": sum(
            int(row["audit_finding_count"])
            for row in rows
            if isinstance(row.get("audit_finding_count"), int)
        ),
        "audit_high_severity_count": sum(
            int(row["audit_high_severity_count"])
            for row in rows
            if isinstance(row.get("audit_high_severity_count"), int)
        ),
        "fingerprint_invalid_count": sum(
            1 for row in rows if row.get("fingerprint_gate_valid") is False
        ),
        "rows": rows,
    }


def format_summary_csv(summary: dict[str, Any]) -> str:
    """Format summary rows as CSV."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in summary.get("rows", []):
        if isinstance(row, dict):
            writer.writerow(row)
    return output.getvalue()
