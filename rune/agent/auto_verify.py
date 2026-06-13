"""Post-edit auto-verification.

After the agent edits code, run the project's fast verifier (lint/typecheck)
so it can self-correct broken edits instead of finalizing them. A full test
suite is intentionally NOT auto-run here (slow, side effects); lint/typecheck
are fast and read-only. Opt-in via the RUNE_AUTO_VERIFY env flag (wired in the
agent loop).

The command runs as a trusted internal subprocess (same trust model as the
Evidence Gate's check), not through the bash capability / Guardian.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
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
    """Pick a CORRECTNESS test command for *cwd*, or None.

    Unlike :func:`detect_verify_command` (fast lint/typecheck — structure, not
    correctness), this runs the project's tests, so it can serve as a best-of-K
    *selection* verifier (execution beats LLM-judge for code; arXiv 2502.14382).

    Precedence: an explicit ``RUNE_AUTO_VERIFY_CMD`` override (the project's own
    test command) wins; otherwise structured marker detection — never NL. Returns
    None when no test runner is evident, so the caller can fall back to the
    Evidence Gate rather than mistake a lint pass for correctness.
    """
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
        # Use the running interpreter, not a bare "python": many machines only
        # have "python3", so "python" fails to spawn and run_verify returns
        # "skip", falling back to the Evidence Gate. sys.executable is the venv
        # interpreter, which has pytest.
        return [sys.executable, "-m", "pytest", "-q"]

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
    """Pick a verify command, or None if unknown.

    An explicit ``RUNE_AUTO_VERIFY_CMD`` env override wins (lets a project use
    its own test/typecheck command). Otherwise fall back to structured
    file-marker detection (not NL matching), so it is safe and deterministic.
    """
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
    """Run *cmd* in *cwd*. Returns ``(state, evidence)`` where state is:

    - ``"pass"`` — exit 0 (no problems).
    - ``"fail"`` — non-zero exit; ``evidence`` is the tail of the output.
    - ``"skip"`` — could not run (spawn error / timeout); inconclusive, never
      treated as failure.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except Exception as exc:
        log.debug("auto_verify_spawn_failed", error=str(exc)[:120])
        return "skip", ""

    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        log.debug("auto_verify_timeout", timeout_s=timeout)
        with contextlib.suppress(Exception):
            proc.kill()
        return "skip", ""

    text = (out or b"").decode("utf-8", "replace")
    if proc.returncode == 0:
        return "pass", ""
    return "fail", text[-_EVIDENCE_TAIL_CHARS:].strip()
