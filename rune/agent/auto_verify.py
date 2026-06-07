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

from rune.utils.logger import get_logger

log = get_logger(__name__)

# (project marker, fast verify command). First match wins. Fast + read-only.
_VERIFY_COMMANDS: tuple[tuple[str, list[str]], ...] = (
    ("pyproject.toml", ["uv", "run", "ruff", "check", "."]),
    ("package.json", ["npm", "run", "-s", "lint"]),
)

_DEFAULT_TIMEOUT_S = 60.0
_EVIDENCE_TAIL_CHARS = 400


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
    except (TimeoutError, asyncio.TimeoutError):
        log.debug("auto_verify_timeout", timeout_s=timeout)
        with contextlib.suppress(Exception):
            proc.kill()
        return "skip", ""

    text = (out or b"").decode("utf-8", "replace")
    if proc.returncode == 0:
        return "pass", ""
    return "fail", text[-_EVIDENCE_TAIL_CHARS:].strip()
