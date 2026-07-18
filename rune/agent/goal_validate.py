"""Deterministic SPEC validation runner for the ``/goal`` loop.

The outer loop runs the SPEC's validation commands itself and checks exit
codes, rather than trusting the inner completion gate or the agent's
self-report. Commands run sequentially and non-interactively with a
per-command timeout; the first failure short-circuits. The shell exec is
injected so this can be unit-tested without spawning processes.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from rune.utils.logger import get_logger

log = get_logger(__name__)

# (command, cwd, timeout_s) -> (exit_code, combined_output)
ExecFn = Callable[[str, str, float], Awaitable[tuple[int, str]]]

# Filenames that mark a buildable project root (structural, not natural
# language). crystallize may bake a "project named X" into the spec and the
# agent then creates an X/ subdir (cargo new does exactly this), leaving the
# manifest one level below the goal working directory.
_MANIFESTS = frozenset({
    "Cargo.toml", "go.mod", "package.json", "pyproject.toml", "setup.py",
    "pom.xml", "build.gradle", "build.gradle.kts", "build.sbt",
    "CMakeLists.txt", "Makefile", "Gemfile", "composer.json", "pubspec.yaml",
})
# Never descend into build output / vcs / deps when locating the root.
_SCAN_EXCLUDE = {
    ".git", ".rune", "node_modules", "vendor", "__pycache__", ".venv",
    "target", "build", "dist", "out", "bin", "obj", ".gradle", ".tox",
}


def _resolve_root(base: str) -> str:
    """Deterministically locate the buildable project root.

    The loop runs the SPEC validation commands at *base* (the goal working
    directory). When *base* itself has no manifest, redirect to the single
    shallowest subdirectory that has one. Ambiguous cases (none, or several
    at the same shallowest depth = a monorepo) keep *base* - never guess a
    wrong location (fail-closed: an honest failure beats validating the
    wrong project). Model-independent; never escapes *base*.
    """
    if not base:
        return base
    basep = Path(base)
    try:
        if any((basep / m).is_file() for m in _MANIFESTS):
            return base  # well-formed: unchanged behavior, zero regression
    except OSError:
        return base
    best_depth: int | None = None
    found: list[str] = []
    seen = 0
    try:
        for dirpath, dirnames, filenames in os.walk(basep):  # no symlinks
            dirnames[:] = [
                d for d in dirnames
                if d not in _SCAN_EXCLUDE and not d.startswith(".")
            ]
            rel = Path(dirpath).relative_to(basep)
            depth = 0 if str(rel) == "." else len(rel.parts)
            if depth >= 3:
                dirnames[:] = []  # bound the scan
            if depth == 0:
                continue  # base already checked above
            seen += 1
            if seen > 5000:
                break
            if any(m in filenames for m in _MANIFESTS):
                if best_depth is None or depth < best_depth:
                    best_depth, found = depth, [dirpath]
                elif depth == best_depth:
                    found.append(dirpath)
    except OSError:
        return base
    if best_depth is not None and len(found) == 1:
        return found[0]
    return base  # 0 or >1 shallowest candidates -> stay put (conservative)


async def _default_exec(command: str, cwd: str, timeout_s: float) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd or None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,  # non-interactive
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, f"timeout after {timeout_s:.0f}s"
    return proc.returncode or 0, (out or b"").decode("utf-8", "replace")


def make_validate_fn(
    *,
    cwd: str = "",
    timeout_s: float = 600.0,
    exec_fn: ExecFn | None = None,
    auto_root: bool = True,
) -> Callable[[list[str]], Awaitable[tuple[bool, str]]]:
    """Build a ``GoalLoop`` ``validate_fn``. Empty command list => pass.

    With ``auto_root`` (default), when *cwd* has no build manifest but a
    single subdirectory does, the commands run there - so an agent that
    created a ``project/`` subfolder is not failed forever by
    "could not find Cargo.toml" at the goal root.
    """
    run = exec_fn or _default_exec

    # Freeze pre-existing test files at goal start: validation must judge the
    # user's own checks, not agent-edited ones (see validation_guard).
    from rune.agent.validation_guard import restoration_note, snapshot_tests

    test_snapshot = snapshot_tests(cwd or ".")

    async def _validate(commands: list[str]) -> tuple[bool, str]:
        if not commands:
            return True, "no validation commands"
        from rune.agent.validation_guard import restore_tests

        restored_note = restoration_note(restore_tests(test_snapshot))
        target = _resolve_root(cwd) if auto_root else cwd
        # Surface a redirect so the reviewer/feedback shows where it ran.
        header = f"# validation cwd: {target}\n" if target != cwd else ""
        if restored_note:
            header = f"# {restored_note}\n{header}"
        transcript: list[str] = []
        for cmd in commands:
            try:
                code, output = await run(cmd, target, timeout_s)
            except Exception as exc:  # treat exec failure as validation failure
                log.debug("goal_validate_exec_error", cmd=cmd[:120], error=str(exc)[:200])
                return False, f"`{cmd}` could not run: {exc}"[:300]
            tail = "\n".join(output.strip().splitlines()[-8:])[:600]
            transcript.append(f"$ {cmd}\n[exit {code}]\n{tail}".rstrip())
            if code != 0:
                # Deterministic evidence the reviewer can trust (not a claim).
                return False, "DETERMINISTIC VALIDATION OUTPUT:\n" + header + "\n\n".join(
                    transcript
                )
        return True, "DETERMINISTIC VALIDATION OUTPUT (all commands exited 0):\n" + header + (
            "\n\n".join(transcript)
        )

    return _validate
