"""Run SPEC validation commands sequentially, stopping at the first failure."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from rune.utils.logger import get_logger

log = get_logger(__name__)

# (command, cwd, timeout_s) -> (exit_code, combined_output)
ExecFn = Callable[[str, str, float], Awaitable[tuple[int, str]]]

# Manifests identify project roots, including projects created below the goal directory.
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
    """Use the single shallowest project root under base; keep base when ambiguous."""
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
    from rune.safety.verification import run_check

    result = await run_check(command, cwd or os.getcwd(), timeout_s)
    return (result.code if result.code is not None else 1), result.stdout.decode("utf-8", "replace") + result.error


def make_validate_fn(
    *,
    cwd: str = "",
    timeout_s: float = 600.0,
    exec_fn: ExecFn | None = None,
    auto_root: bool = True,
) -> Callable[[list[str]], Awaitable[tuple[bool, str]]]:
    """Build a validator with optional project-root detection; an empty command list passes."""
    run = exec_fn or _default_exec

    # Snapshot existing tests so validation cannot rely on agent-weakened checks.
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
