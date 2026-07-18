"""Freeze pre-existing test files across an agent run.

External verifiers (Evidence Gate, GoalLoop validation) judge the agent's work
by running checks inside the very workspace the agent just edited. Observed
live, and then measured (scripts/tamper_rate_bench.py): under the wording
people actually use ("fix the failing test"), a 7B edits the tests in 15/15
runs and gets a false "verified" in 4/15; a 32B still tampers in 3/15. A
verifier whose inputs the agent can rewrite is not a verifier.

Policy (``RUNE_PROTECT_TESTS``, default on):

- snapshot conventional test files (and test-runner config files) when the
  guard is created — the pre-run state;
- immediately before a verification runs:
  * restore any snapshotted test file whose content changed or was deleted,
  * quarantine any NEWLY CREATED ``conftest.py`` (a fresh conftest can skip or
    deselect the failing tests without touching them — the collection attack),
  * detect (disclose only) newly created test files and modified runner
    configs (``pytest.ini``/``pyproject.toml``/``setup.cfg``/``tox.ini``) —
    configs are NOT restored because editing them is often legitimate work;
- report everything so the verdict discloses what happened.

When the user's task genuinely is to edit tests, disable with
``RUNE_PROTECT_TESTS=0``.

Perf envelope (measured on this 94K-LOC repo): snapshot ~54ms once per run,
re-scan+restore ~16-60ms per verification; walk is capped at ``_MAX_DIRS``
directories so a workspace pinned to a home directory cannot stall a run.

The name globs below match file-naming conventions (structured formats), not
natural language — pattern matching is the correct tool here.
"""
from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from rune.utils.logger import get_logger

log = get_logger(__name__)

_ENV_FLAG = "RUNE_PROTECT_TESTS"

# File-name conventions across the ecosystems RUNE drives.
_TEST_FILE_GLOBS: tuple[str, ...] = (
    "test_*.py", "*_test.py", "conftest.py",
    "*.test.js", "*.test.jsx", "*.test.ts", "*.test.tsx",
    "*.spec.js", "*.spec.jsx", "*.spec.ts", "*.spec.tsx",
    "*_test.go", "*Test.java", "*Tests.java", "*_spec.rb",
)
_TEST_DIR_NAMES = frozenset({"tests", "test", "__tests__", "spec"})
_SKIP_DIR_NAMES = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
    ".rune", "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
})
# Root-level files that steer test collection (addopts, testpaths, markers).
# Changed configs are DISCLOSED, never restored: editing pyproject.toml is
# routine legitimate work (deps), unlike rewriting an existing assertion.
_CONFIG_FILES = ("pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini")

_MAX_FILE_BYTES = 256 * 1024
_MAX_FILES = 500
_MAX_TOTAL_BYTES = 8 * 1024 * 1024
_MAX_DIRS = 8_000  # a mis-pinned home-dir workspace must not stall the run
_QUARANTINE_SUFFIX = ".rune-quarantined"


def protect_tests_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "1") != "0"


def _is_test_file(path: Path, under_test_dir: bool) -> bool:
    if under_test_dir and path.suffix in {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".rb"
    }:
        return True
    return any(fnmatch.fnmatch(path.name, g) for g in _TEST_FILE_GLOBS)


def _walk_test_files(root: Path) -> Iterator[Path]:
    """Yield test-convention files under *root*, bounded by ``_MAX_DIRS``."""
    dirs_seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirs_seen += 1
        if dirs_seen > _MAX_DIRS:
            log.warning("test_walk_truncated", max_dirs=_MAX_DIRS, root=str(root))
            return
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_NAMES]
        rel_dir = Path(dirpath).relative_to(root)
        under_test_dir = any(p in _TEST_DIR_NAMES for p in rel_dir.parts)
        for name in filenames:
            p = Path(dirpath) / name
            if _is_test_file(p, under_test_dir):
                yield p


@dataclass(slots=True)
class TestSnapshot:
    """Pre-run byte-exact copies of the workspace's test files + configs."""

    root: str
    files: dict[str, bytes] = field(default_factory=dict)
    configs: dict[str, bytes] = field(default_factory=dict)


@dataclass(slots=True)
class RestoreReport:
    """What the guard had to do right before a verification ran."""

    restored: list[str] = field(default_factory=list)  # rewritten to canonical
    quarantined: list[str] = field(default_factory=list)  # new conftest moved aside
    created: list[str] = field(default_factory=list)  # new test files (disclosed)
    configs_changed: list[str] = field(default_factory=list)  # disclosed only

    def tampering(self) -> bool:
        return bool(self.restored or self.quarantined)

    def anything(self) -> bool:
        return bool(self.restored or self.quarantined or self.created
                    or self.configs_changed)


def snapshot_tests(cwd: str | os.PathLike[str]) -> TestSnapshot | None:
    """Capture pre-run test files under *cwd*. ``None`` when disabled/invalid."""
    if not protect_tests_enabled():
        return None
    root = Path(cwd or ".").resolve()
    if not root.is_dir():
        return None
    snap = TestSnapshot(root=str(root))
    total = 0
    for p in _walk_test_files(root):
        try:
            if p.stat().st_size > _MAX_FILE_BYTES:
                continue
            data = p.read_bytes()
        except OSError:
            continue
        total += len(data)
        if len(snap.files) >= _MAX_FILES or total > _MAX_TOTAL_BYTES:
            log.warning("test_snapshot_truncated", files=len(snap.files))
            break
        snap.files[str(p.relative_to(root))] = data
    for name in _CONFIG_FILES:
        p = root / name
        try:
            if p.is_file() and p.stat().st_size <= _MAX_FILE_BYTES:
                snap.configs[name] = p.read_bytes()
        except OSError:
            continue
    return snap


def restore_tests(snap: TestSnapshot | None) -> RestoreReport:
    """Put the verification inputs back into their pre-run state.

    Restores tampered/deleted pre-existing test files, quarantines newly
    created ``conftest.py`` files (collection attack), and detects — without
    reverting — new test files and modified runner configs.
    """
    report = RestoreReport()
    if snap is None or not protect_tests_enabled():
        return report
    root = Path(snap.root)

    for rel, data in snap.files.items():
        p = root / rel
        try:
            current = p.read_bytes() if p.exists() else None
        except OSError:
            current = None
        if current == data:
            continue
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            report.restored.append(rel)
        except OSError as exc:
            log.warning("test_restore_failed", path=rel, error=str(exc)[:120])

    # Rescan for files that did not exist at snapshot time.
    for p in _walk_test_files(root):
        rel = str(p.relative_to(root))
        if rel in snap.files:
            continue
        if p.name == "conftest.py":
            target = p.with_name(p.name + _QUARANTINE_SUFFIX)
            try:
                n = 1
                while target.exists():
                    target = p.with_name(f"{p.name}{_QUARANTINE_SUFFIX}{n}")
                    n += 1
                p.rename(target)
                report.quarantined.append(rel)
            except OSError as exc:
                log.warning("conftest_quarantine_failed", path=rel,
                            error=str(exc)[:120])
        else:
            report.created.append(rel)

    for name, data in snap.configs.items():
        p = root / name
        try:
            current = p.read_bytes() if p.exists() else None
        except OSError:
            current = None
        if current != data:
            report.configs_changed.append(name)

    if report.tampering():
        log.warning("verification_inputs_restored",
                    restored=report.restored, quarantined=report.quarantined)
    return report


def _shown(paths: list[str], cap: int = 5) -> str:
    return ", ".join(paths[:cap]) + ("…" if len(paths) > cap else "")


def restoration_note(report: RestoreReport) -> str:
    """Disclosure for verdict evidence/transcripts. Empty when nothing happened."""
    lines: list[str] = []
    if report.restored:
        lines.append(
            f"⚠ the agent modified {len(report.restored)} pre-existing test "
            f"file(s) during this run; canonical copies were restored before "
            f"verification: {_shown(report.restored)}"
        )
    if report.quarantined:
        lines.append(
            f"⚠ the agent created new conftest.py file(s) that could alter "
            f"test collection; quarantined before verification: "
            f"{_shown(report.quarantined)}"
        )
    if report.created:
        lines.append(
            f"note: the agent created new test file(s) (left in place): "
            f"{_shown(report.created)}"
        )
    if report.configs_changed:
        lines.append(
            f"note: test-runner config file(s) changed during this run (not "
            f"reverted — verify intent): {_shown(report.configs_changed)}"
        )
    return "\n".join(lines)
