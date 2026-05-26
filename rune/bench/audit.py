"""Artifact audit helpers for benchmark attempts."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

Severity = Literal["low", "medium", "high"]

FORBIDDEN_RUNTIME_PATHS = ("/tests", "/oracle")
LEAK_PRONE_FILES = ("AGENTS.md", "CLAUDE.md")
GIT_HISTORY_SUBCOMMANDS = ("blame", "log", "reflog", "show")
GIT_GLOBAL_OPTIONS_WITH_VALUES = {
    "-C",
    "-c",
    "--config-env",
    "--exec-path",
    "--git-dir",
    "--namespace",
    "--super-prefix",
    "--work-tree",
}
TERMINAL_BENCH_GIT_TASKS = {
    "fix-git",
    "git-leak-recovery",
    "git-multibranch",
    "sanitize-git-repo",
}
FORBIDDEN_RUNTIME_PATH_RE = {
    path: re.compile(rf"(?<![A-Za-z0-9_./-]){re.escape(path)}(?:/|(?=[\s\"'\\,}}\]]|$))")
    for path in FORBIDDEN_RUNTIME_PATHS
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


def _event_command(row: dict[str, Any]) -> str | None:
    for container_key in ("params", "args"):
        container = row.get(container_key)
        if isinstance(container, dict):
            command = container.get("command") or container.get("cmd")
            if isinstance(command, str) and command.strip():
                return command
    command = row.get("command") or row.get("cmd")
    return command if isinstance(command, str) and command.strip() else None


def _git_subcommand(tokens: list[str]) -> str | None:
    if not tokens or Path(tokens[0]).name != "git":
        return None

    idx = 1
    while idx < len(tokens):
        token = tokens[idx]
        if token in GIT_GLOBAL_OPTIONS_WITH_VALUES:
            idx += 2
            continue
        if token.startswith("--") and "=" in token:
            idx += 1
            continue
        if token.startswith("-"):
            idx += 1
            continue
        return token
    return None


def _git_history_command(command: str) -> str | None:
    from rune.agent.bash_parsing import (
        split_shell_segments,
        split_shell_tokens,
        strip_leading_env_assignments,
    )

    for segment in split_shell_segments(command):
        tokens = strip_leading_env_assignments(split_shell_tokens(segment))
        subcommand = _git_subcommand(tokens)
        if subcommand in GIT_HISTORY_SUBCOMMANDS:
            return f"git {subcommand}"
    return None


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
        if FORBIDDEN_RUNTIME_PATH_RE[path].search(event_text):
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
        seen_history_commands: set[str] = set()
        for row in events + tool_calls:
            command = _event_command(row)
            if not command:
                continue
            if history_command := _git_history_command(command):
                seen_history_commands.add(history_command)

        for command in sorted(seen_history_commands):
            findings.append(
                AuditFinding(
                    rule_id="git_history_mining",
                    severity="high",
                    evidence=command,
                )
            )

        if not seen_history_commands:
            for command in GIT_HISTORY_SUBCOMMANDS:
                pattern = re.compile(rf"(?<![A-Za-z0-9_.-])git\s+{command}(?:\s|$)")
                if pattern.search(event_text):
                    findings.append(
                        AuditFinding(
                            rule_id="git_history_mining",
                            severity="high",
                            evidence=f"git {command}",
                        )
                    )
                    break

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
