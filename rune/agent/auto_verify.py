"""Detect project checks and run them in the task's execution environment."""

from __future__ import annotations

import os
import shlex
import sys

from rune.utils.logger import get_logger

log = get_logger(__name__)

# (project marker, fast verify command). First match wins. Fast + read-only.
_VERIFY_COMMANDS: tuple[tuple[str, list[str]], ...] = (
    ("pyproject.toml", ["uv", "run", "ruff", "check", "."]),
    ("package.json", ["npm", "run", "-s", "lint"]),
)

_DEFAULT_TIMEOUT_S = 60.0
_EVIDENCE_TAIL_CHARS = 400


def detect_test_command(cwd: str) -> list[str] | None:
    """Choose the override or detected test runner; return None if neither exists."""
    override = os.environ.get("RUNE_AUTO_VERIFY_CMD", "").strip()
    if override:
        import shlex
        return shlex.split(override)

    # Python: a tests/ dir or top-level test_*.py / *_test.py -> pytest.
    try:
        entries = os.listdir(cwd)
    except OSError:
        return None
    has_pytests = os.path.isdir(os.path.join(cwd, "tests")) or any(
        (e.startswith("test_") or e.endswith("_test.py")) and e.endswith(".py")
        for e in entries
    )
    if has_pytests:
        from rune.safety.execution_environment import execution_config

        python = "python" if execution_config().backend == "container" else sys.executable
        return [python, "-m", "pytest", "-q"]

    # Node: a non-placeholder "test" script in package.json -> npm test.
    pkg = os.path.join(cwd, "package.json")
    if os.path.exists(pkg):
        try:
            import json
            with open(pkg, encoding="utf-8") as fh:
                test_script = (json.load(fh).get("scripts") or {}).get("test", "")
            if test_script and "no test specified" not in test_script:
                return ["npm", "test", "-s"]
        except (OSError, ValueError):
            pass
    return None


def detect_verify_command(cwd: str) -> list[str] | None:
    """Choose the override or a project check; return None if neither exists."""
    override = os.environ.get("RUNE_AUTO_VERIFY_CMD", "").strip()
    if override:
        import shlex
        return shlex.split(override)
    for marker, cmd in _VERIFY_COMMANDS:
        if os.path.exists(os.path.join(cwd, marker)):
            return list(cmd)
    return None


async def run_verify(
    cmd: list[str], cwd: str, timeout: float = _DEFAULT_TIMEOUT_S
) -> tuple[str, str]:
    """Return (pass/fail/skip, evidence); timeouts and spawn errors are inconclusive."""
    from rune.agent.execution_journal import active_journal, record_check
    if active_journal() is not None:
        return await record_check({"command": cmd, "cwd": cwd}, lambda: run_verify(cmd, cwd, timeout))
    from rune.safety.verification import run_check

    result = await run_check(shlex.join(cmd), cwd, timeout)
    text = result.stdout.decode("utf-8", "replace") + result.error
    if result.code is None or result.code in (126, 127):
        log.debug("auto_verify_inconclusive", detail=text[-_EVIDENCE_TAIL_CHARS:])
        return "skip", text[-_EVIDENCE_TAIL_CHARS:]
    if result.code == 0:
        # Keep the summary line so callers can report how many tests passed.
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        return "pass", (lines[-1].strip() if lines else "")
    return "fail", text[-_EVIDENCE_TAIL_CHARS:].strip()


def passed_test_count(summary: str) -> int | None:
    """Parse the passing-test count from a runner summary, or return None."""
    import re
    m = re.search(r"(\d+)\s+passed", summary)
    return int(m.group(1)) if m else None


# An empty test run and an unrecognized summary need different outcomes.
_VACUOUS_SUMMARY_PATTERNS: tuple[str, ...] = (
    r"^(?:#|ℹ)\s*(?:tests|pass)\s+0\b",  # Node TAP/spec
    r"\bno tests ran\b",            # pytest
    r"\bcollected 0 items\b",       # pytest
    r"\b0\s+passed\b",              # pytest / cargo / jest ("0 passed")
    r"\bran 0 tests\b",             # unittest
    r"\b0\s+tests?\b(?!.*\b[1-9]\d*\s+passed)",  # generic "0 tests"
    r"\bno tests to run\b",         # misc runners
    r"\btests?:\s*0\b",             # jest-style "Tests: 0"
)
_ASSERTED_SUMMARY_PATTERNS: tuple[str, ...] = (
    r"^(?:#|ℹ)\s*pass\s+[1-9]\d*\b",  # Node TAP/spec
    r"\b[1-9]\d*\s+passed\b",       # pytest / cargo / jest
    r"\bran\s+[1-9]\d*\s+tests?\b",  # unittest
    r"\bok\b.*\bcoverage:",          # go test with coverage
    r"^ok\s+\S+",                    # go test ("ok  pkg  0.02s")
)


def assertions_ran(summary: str) -> bool | None:
    """Return whether tests ran, or None if the summary is unrecognized."""
    import re

    text = (summary or "").strip()
    if not text:
        return None
    for pat in _ASSERTED_SUMMARY_PATTERNS:
        if re.search(pat, text, re.IGNORECASE | re.MULTILINE):
            return True
    for pat in _VACUOUS_SUMMARY_PATTERNS:
        if re.search(pat, text, re.IGNORECASE | re.MULTILINE):
            return False
    return None


def tests_failed(summary: str) -> bool:
    """Read failure counts even when a shell wrapper hides the runner's exit code."""
    import re

    return any(re.search(pattern, summary, re.IGNORECASE | re.MULTILINE) for pattern in (
        r"\b[1-9]\d*\s+(?:failed|errors?)\b",
        r"^FAILED\s*\((?:failures|errors)=[1-9]\d*",
        r"^(?:FAIL\s|test result: FAILED|not ok\s+\d+)",
        r"^(?:#|ℹ)\s*fail\s+[1-9]\d*\b",
        r"(?:^|: )No module named (?:pytest|unittest|tox)\b",
        r"^(?:ModuleNotFoundError|ImportError|pytest: error):",
    ))
