"""Artifact audit helpers for benchmark attempts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

Severity = Literal["low", "medium", "high"]

FORBIDDEN_RUNTIME_PATHS = ("/tests", "/oracle")
LEAK_PRONE_FILES = ("AGENTS.md", "CLAUDE.md")
GIT_HISTORY_COMMANDS = ("git log", "git show", "git reflog")
TERMINAL_BENCH_GIT_TASKS = {
    "fix-git",
    "git-leak-recovery",
    "git-multibranch",
    "sanitize-git-repo",
}


@dataclass(slots=True)
class AuditFinding:
    rule_id: str
    severity: Severity
    evidence: str


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
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


def _flatten(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def audit_attempt_dir(attempt_dir: Path) -> dict[str, Any]:
    """Audit one benchmark attempt artifact directory."""
    findings: list[AuditFinding] = []
    task = _read_json(attempt_dir / "task.json")
    benchmark = str(task.get("benchmark", ""))
    task_id = str(task.get("task_id", ""))
    events = _read_jsonl(attempt_dir / "events.jsonl")
    tool_calls = _read_jsonl(attempt_dir / "tool_calls.jsonl")
    event_text = "\n".join(_flatten(row) for row in events + tool_calls)

    for path in FORBIDDEN_RUNTIME_PATHS:
        if path in event_text:
            findings.append(
                AuditFinding(
                    rule_id="forbidden_runtime_path",
                    severity="high",
                    evidence=path,
                )
            )

    for filename in LEAK_PRONE_FILES:
        if filename in event_text:
            findings.append(
                AuditFinding(
                    rule_id="leak_prone_instruction_file",
                    severity="medium",
                    evidence=filename,
                )
            )

    git_history_expected = (
        benchmark.lower() in {"terminal-bench-v2", "terminal-bench"}
        and task_id in TERMINAL_BENCH_GIT_TASKS
    )
    if not git_history_expected:
        for command in GIT_HISTORY_COMMANDS:
            if command in event_text:
                findings.append(
                    AuditFinding(
                        rule_id="git_history_mining",
                        severity="high",
                        evidence=command,
                    )
                )

    patch = _read_text(attempt_dir / "patch.diff").strip()
    if benchmark.lower() in {"swe-atlas-qna", "swe-atlas"} and patch:
        findings.append(
            AuditFinding(
                rule_id="qna_source_modified",
                severity="high",
                evidence="patch.diff is non-empty for a read-only Q&A task",
            )
        )

    result = {
        "attempt_dir": str(attempt_dir),
        "finding_count": len(findings),
        "high_severity_count": sum(1 for finding in findings if finding.severity == "high"),
        "findings": [asdict(finding) for finding in findings],
    }
    return result
