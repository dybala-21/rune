"""Best-of-K (verifier-guided rejection sampling) for the CLI one-shot path.

``rune --message "..." --best-of K`` runs K *independent* fresh-context attempts,
each in an isolated tempdir subprocess, then uses RUNE's Evidence Gate as the
selector (see :mod:`rune.agent.rejection_sampler`) to keep the first attempt that
passes a mechanically-extracted success check. The selected attempt's artifacts
are copied back into the real working directory.

best-of-K lifts a weak model by turning model nondeterminism into a *selection*
signal: if a single attempt passes with probability p, sampling K and keeping the
first that verifies succeeds with probability 1-(1-p)^K.

Gated behind the flag: ``K == 1`` is the unchanged single-attempt path with zero
behavior change.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rune.agent.isolation import ISOLATION_ENV
from rune.agent.rejection_sampler import (
    Attempt,
    RejectionResult,
    make_verifier,
    sample_parallel,
)
from rune.utils.env import env_int
from rune.utils.logger import get_logger
from rune.utils.paths import rune_data

log = get_logger(__name__)

# Set in every attempt subprocess so a nested invocation can never re-enter
# best-of (defense-in-depth; the child command line also omits ``--best-of``).
RECURSION_GUARD_ENV = "RUNE_IN_BEST_OF"

# Per-attempt wall-clock cap: a single hung attempt (model stall) must not block
# the whole gather forever. Overridable for long tasks.
_ATTEMPT_TIMEOUT_MS_ENV = "RUNE_BESTOF_ATTEMPT_TIMEOUT_MS"
_DEFAULT_ATTEMPT_TIMEOUT_MS = 600_000  # 10 min
_TIMEOUT_RETURNCODE = 124  # mirror coreutils `timeout`

# Seeded mode: when nothing verifies, apply the best-effort edits (with
# backup/undo) instead of parking them — a correct fix can fail to verify,
# and withholding it throws away real work. Still exits non-zero, reported
# UNVERIFIED. "0" parks instead.
_APPLY_UNVERIFIED_ENV = "RUNE_BESTOF_APPLY_UNVERIFIED"

# Sampling strategy: stop at the first verified pass instead of always
# paying flat K. auto = sequential for ollama (serial server), race2
# otherwise (2 parallel attempts, then one failure-fed repair attempt);
# "parallel" restores flat K. Larger models profit from seeing the previous
# failure; small local ones resample fresh.
_STRATEGY_ENV = "RUNE_BESTOF_STRATEGY"
_REPAIR_ENV = "RUNE_BESTOF_REPAIR"  # "0" disables failure-fed repair attempts
_REPAIR_EVIDENCE_CAP = 1500

# Rung-0 fast path before the agentic attempts: single-shot localize+edit,
# accepted only when a baseline-failing repro script flips to passing (a
# discriminating check → honest verified). "0" disables.
_FASTPATH_ENV = "RUNE_FASTPATH"


# Attempts used to be near-copies: all at temperature 0, and race2 handed
# the first two the same message, so K samples walked the same path and K
# bought far less than it should. Each attempt now gets a different
# starting point and a different sampling temperature. "0" restores the
# identical-attempt behaviour.
_DIVERSITY_ENV = "RUNE_BESTOF_DIVERSIFY"

# Distinct entry points into the same problem, not instructions to try
# harder. Index 0 keeps the default framing so the cheapest, most direct
# attempt is always among the K.
_ENTRY_POINTS = (
    "",
    "\n\nApproach: reproduce the reported behaviour first and let what you "
    "observe tell you where the cause is, rather than starting from the "
    "file the report names.",
    "\n\nApproach: find the code that already handles the neighbouring "
    "cases correctly and work out why this case does not reach it.",
)

# Attempt 0 stays deterministic; later attempts sample more widely so they
# are not re-drawing the same trajectory.
_TEMPERATURES = (0.0, 0.4, 0.7)


def _diversify(index: int) -> tuple[str, float | None]:
    """Entry-point suffix and sampling temperature for attempt *index*."""
    if os.environ.get(_DIVERSITY_ENV, "1") == "0":
        return "", None
    return (_ENTRY_POINTS[index % len(_ENTRY_POINTS)],
            _TEMPERATURES[index % len(_TEMPERATURES)])


def _repair_suffix(evidence: str) -> str:
    return (
        "\n\nNOTE: a previous independent attempt at this task FAILED "
        "verification with this output:\n```\n"
        + evidence[-_REPAIR_EVIDENCE_CAP:]
        + "\n```\nDiagnose what that attempt likely got wrong, then implement "
        "a correct fix for the root cause. Do not repeat an approach the "
        "failure output already refutes."
    )


@dataclass
class AttemptArtifact:
    """One best-of-K attempt: its isolated workdir and captured output.

    ``produced`` is the snapshot of top-level entries the attempt itself created,
    taken BEFORE the verifier runs — so verifier side-effects (e.g. a
    ``__pycache__`` from importing the candidate) are never restored into the
    real working directory.
    """

    index: int
    workdir: str
    stdout: str
    returncode: int
    produced: list[str]


async def _run_attempt_subprocess(
    index: int,
    message: str,
    model: str | None,
    provider: str | None,
    seed_from: str | None = None,
) -> AttemptArtifact:
    """Run one fresh-context attempt in an isolated tempdir subprocess.

    Each attempt is a separate ``python -m rune.cli.main --message ...`` process
    with its own working directory and a copied env carrying the recursion guard.
    The child command intentionally OMITS ``--best-of`` so it takes the plain
    single-attempt path; the env flag is a second guard in case it ever leaks.

    ``seed_from`` (set in --include-cwd mode) copies that dir into the workdir
    first so the agent can edit existing files; ``produced`` then becomes the set
    of files CHANGED vs the seed, not every top-level entry.
    """
    workdir = tempfile.mkdtemp(prefix=f"rune_bestof_{index}_", dir=_attempt_work_root())

    seed_manifest: dict[str, tuple[float, int]] | None = None
    if seed_from:
        try:
            _seed_workdir(seed_from, workdir)
            seed_manifest = _tree_manifest(workdir)
        except Exception as exc:
            log.warning("bestof_seed_failed", index=index, error=str(exc)[:120])

    def _produced() -> list[str]:
        if seed_manifest is not None:
            changed = _changed_vs_seed(workdir, seed_manifest)
            return _drop_build_metadata(
                _drop_seed_identical(workdir, seed_from, changed)
            )
        return _snapshot_produced(workdir)

    env = dict(os.environ)
    env[RECURSION_GUARD_ENV] = "1"  # recursion guard
    _entry, _temp = _diversify(index)
    if _temp is not None:
        env["RUNE_TEMPERATURE"] = str(_temp)
    # Confine file writes to this attempt's dir via enforce() (same as the
    # parallel-isolated path). enforce() covers the file_write/edit/delete and
    # document_create capabilities; shell/exec and the browser capability are not
    # contained without an OS sandbox.
    env[ISOLATION_ENV] = workdir

    # Verify handover (seeded mode): tell the attempt what this project's test
    # command is — or how to find the nearest tests — so it spends rounds
    # fixing, not rediscovering the verify loop. Detection is structural
    # (detect_test_command); same hint for every attempt.
    child_message = message + _entry
    if seed_from:
        try:
            from rune.agent.auto_verify import detect_test_command
            _tc = detect_test_command(seed_from)
        except Exception:
            _tc = None
        if _tc:
            child_message += (
                f"\n\nThis project's test command: `{' '.join(_tc)}`. After "
                "editing, run the RELEVANT subset (the test file(s) covering "
                "the code you changed) and fix failures before finishing."
            )
        else:
            child_message += (
                "\n\nAfter editing, locate and run the tests covering the "
                "files you changed (look for a tests/ directory or "
                "test_*.py near them); fix failures before finishing."
            )

    cmd = [sys.executable, "-m", "rune.cli.main", "--message", child_message]
    if model:
        cmd += ["--model", model]
    if provider:
        cmd += ["--provider", provider]

    # Seeded (edit-existing-repo) attempts need more wall than greenfield
    # ones: diagnosing a large codebase routinely outlives the 10-min default,
    # and a killed attempt degrades best-of-K to best-of-1-interrupted.
    _default_ms = _DEFAULT_ATTEMPT_TIMEOUT_MS + (300_000 if seed_from else 0)
    timeout_s = max(
        1.0, env_int(_ATTEMPT_TIMEOUT_MS_ENV, _default_ms) / 1000.0
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=workdir,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except Exception as exc:  # spawn failure — treat as a failed attempt
        log.warning("bestof_attempt_spawn_error", index=index, error=str(exc)[:120])
        return AttemptArtifact(
            index=index, workdir=workdir, stdout="", returncode=1, produced=[]
        )

    try:
        stdout_b, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        returncode = proc.returncode or 0
    except TimeoutError:
        # A stalled attempt must not hang the whole best-of gather. Kill it and
        # record a failed attempt (its partial workdir is left for the verifier,
        # which will almost certainly reject it).
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        log.warning("bestof_attempt_timeout", index=index, timeout_s=timeout_s)
        return AttemptArtifact(
            index=index,
            workdir=workdir,
            stdout="",
            returncode=_TIMEOUT_RETURNCODE,
            produced=_produced(),
        )

    stdout = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
    # Snapshot what the attempt produced BEFORE verification runs, so verifier
    # side-effects (a __pycache__ from importing the candidate) aren't restored.
    produced = _produced()
    log.info(
        "bestof_attempt_done",
        index=index,
        returncode=returncode,
        workdir=workdir,
        produced=produced,
    )
    return AttemptArtifact(
        index=index,
        workdir=workdir,
        stdout=stdout,
        returncode=returncode,
        produced=produced,
    )


# Verification/runtime byproducts that should never be restored even if a tool
# created them inside the attempt's workdir.
_RESTORE_DENYLIST = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache"})


def _snapshot_produced(workdir: str) -> list[str]:
    """Top-level names the attempt created, minus known build/cache byproducts."""
    return sorted(
        name for name in os.listdir(workdir) if name not in _RESTORE_DENYLIST
    )


# --- seeded mode (--include-cwd): copy the working tree into each attempt so the
# agent can EDIT existing files, then restore only what it changed (diff vs seed).

# Fallback exclusions, used only outside a git work tree. Inside one,
# _seed_file_list asks git what the project's source is, which covers every
# ecosystem's build dirs without this list needing an entry per toolchain.
_SEED_IGNORE_PATTERNS = (
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
    "*.pyc", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    "target",  # cargo/maven build output — hundreds of MB after one build
    ".rune-bestof-*",
)
_SEED_IGNORE = shutil.ignore_patterns(*_SEED_IGNORE_PATTERNS)


def _seed_file_list(src: str) -> list[str] | None:
    """The project's own view of its source files, or None outside a git tree.

    ``git ls-files -co --exclude-standard`` = tracked + untracked-but-not-
    ignored, honoring the project's .gitignore chain. Deleted-but-tracked
    entries and best-of runtime dirs are dropped; paths are relative to src.
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "-C", src, "ls-files", "-z", "-co", "--exclude-standard"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    out: list[str] = []
    for rel in proc.stdout.split("\0"):
        if not rel or rel.startswith(".rune-bestof-"):
            continue
        if os.path.isfile(os.path.join(src, rel)):
            out.append(rel)
    return out

# Refuse to seed a cwd larger than this (× K copies would otherwise exhaust
# disk). Overridable for genuinely large repos.
_SEED_MAX_MB_ENV = "RUNE_BESTOF_SEED_MAX_MB"
_DEFAULT_SEED_MAX_MB = 200
_SEED_MAX_FILES = 20_000


def _seed_footprint(src: str) -> tuple[int, int]:
    """Count (files, total_bytes) that seeding would copy."""
    listed = _seed_file_list(src)
    if listed is not None:
        total = 0
        for rel in listed:
            try:
                total += os.path.getsize(os.path.join(src, rel))
            except OSError:
                pass
        return len(listed), total

    import fnmatch

    def ignored(name: str) -> bool:
        return any(fnmatch.fnmatch(name, p) for p in _SEED_IGNORE_PATTERNS)

    files = 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if not ignored(d)]
        for fn in filenames:
            if ignored(fn):
                continue
            files += 1
            try:
                total += os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                pass
    return files, total


def _check_seed_size(src: str) -> str | None:
    """Return an error message if seeding ``src`` would be too large, else None."""
    max_mb = env_int(_SEED_MAX_MB_ENV, _DEFAULT_SEED_MAX_MB)
    files, total = _seed_footprint(src)
    if total > max_mb * 1024 * 1024 or files > _SEED_MAX_FILES:
        return (
            f"--include-cwd would copy {files} files / {total / 1024 / 1024:.0f} MB "
            f"into EACH attempt (× K copies). That exceeds the limit "
            f"({max_mb} MB / {_SEED_MAX_FILES} files). Run from a smaller dir, add "
            f"large paths to a .gitignore-style layout, or raise {_SEED_MAX_MB_ENV}."
        )
    return None


def _seed_workdir(src: str, workdir: str) -> None:
    """Copy the project's source files into an attempt's workdir.

    In a git work tree, "source" is what the project itself declares
    (tracked + untracked-unignored — see _seed_file_list); elsewhere the
    static pattern fallback applies.
    """
    listed = _seed_file_list(src)
    if listed is not None:
        for rel in listed:
            dst = os.path.join(workdir, rel)
            os.makedirs(os.path.dirname(dst) or workdir, exist_ok=True)
            try:
                shutil.copy2(os.path.join(src, rel), dst)
            except OSError:
                continue
        return
    shutil.copytree(src, workdir, ignore=_SEED_IGNORE, dirs_exist_ok=True, symlinks=False)


def _drop_build_metadata(rels: list[str]) -> list[str]:
    """Drop packaging byproducts (egg-info/dist-info) from a changed-file list.

    A pip/setuptools invocation inside an attempt regenerates these. They are
    never the agent's intended work, and counting them lets a junk-only
    candidate tie a real edit in the best-effort ranking while burying the
    actual edit in the applied-files report.
    """
    def _is_meta(rel: str) -> bool:
        parts = rel.replace("\\", "/").split("/")
        return any(
            p.endswith(".egg-info") or p.endswith(".dist-info") or p == ".eggs"
            for p in parts
        )

    return [r for r in rels if not _is_meta(r)]


def _drop_seed_identical(
    workdir: str, seed_from: str | None, rels: list[str]
) -> list[str]:
    """Drop rels whose bytes equal the seed original.

    Seeding preserves mtimes, so an edit that was later reverted still shows
    as "changed" by mtime while carrying no delta — it must not be ranked or
    applied as produced work.
    """
    if not seed_from:
        return rels
    real: list[str] = []
    for rel in rels:
        try:
            if (
                Path(workdir, rel).read_bytes()
                == Path(seed_from, rel).read_bytes()
            ):
                continue
        except OSError:
            pass  # new or unreadable file → keep
        real.append(rel)
    return real


def _tree_manifest(root: str) -> dict[str, tuple[float, int]]:
    """Map each file's relpath -> (mtime, size). Used to diff seed vs final."""
    manifest: dict[str, tuple[float, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _RESTORE_DENYLIST]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            try:
                st = os.stat(full)
            except OSError:
                continue
            manifest[os.path.relpath(full, root)] = (st.st_mtime, st.st_size)
    return manifest


def _changed_vs_seed(root: str, seed: dict[str, tuple[float, int]]) -> list[str]:
    """Relpaths that are new or modified (mtime/size) vs the seed manifest.

    Deletions are intentionally NOT reported — best-of never deletes user files.
    """
    changed: list[str] = []
    for rel, (mtime, size) in _tree_manifest(root).items():
        prev = seed.get(rel)
        if prev is None or prev != (mtime, size):
            changed.append(rel)
    # If EVERY seeded file looks changed, the mtime/size diff is likely broken
    # (e.g. mtime not preserved on copy) rather than the agent having rewritten
    # the whole tree. Restore still backs up originals, so this is recoverable —
    # but warn loudly so a whole-tree overwrite is visible.
    if seed and len(changed) >= len(seed) and all(r in changed for r in seed):
        log.warning("bestof_seed_diff_suspicious", changed=len(changed), seeded=len(seed))
    return sorted(changed)


def _attempt_work_root() -> str:
    """Guardian-allowed parent for attempt workdirs.

    The default temp dir is unusable: on macOS ``tempfile.mkdtemp()`` lands under
    ``$TMPDIR`` (``/var/folders/...``), and the Guardian blocks the whole ``/var``
    tree as protected, so every file_write in an attempt fails and best-of reports
    "produced no files". Rooting attempt workdirs under the data dir (in $HOME)
    keeps them writable on every platform.
    """
    root = rune_data() / "bestof-work"
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def _backup_root() -> str:
    """Undo-backup location, under the data dir rather than the user's repo.

    Backing up inside ``dest`` left a ``.rune-bestof-backup-*`` dir in the working
    tree after every run, which accumulated and risked accidental commits. Storing
    it out-of-tree keeps the repo clean; old backups are pruned to bound growth.
    """
    root = rune_data() / "bestof-backups"
    root.mkdir(parents=True, exist_ok=True)
    try:
        existing = sorted(
            (p for p in root.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        for old in existing[:-10]:  # keep the 10 most recent
            shutil.rmtree(old, ignore_errors=True)
    except OSError:
        pass
    return str(root)


def _restore_changed(
    workdir: str, dest: str, relpaths: list[str]
) -> tuple[list[str], str | None]:
    """Copy changed ``relpaths`` from a seeded workdir back into ``dest``.

    Overwriting IS intended here (the agent edited a copy of the user's tree),
    but it's destructive, so every pre-existing target is first backed up into a
    fresh out-of-tree backup dir for undo. Returns ``(restored, backup_dir)``.
    """
    restored: list[str] = []
    backup_dir: str | None = None
    for rel in relpaths:
        src = os.path.join(workdir, rel)
        if not os.path.exists(src):
            continue
        dst = os.path.join(dest, rel)
        if os.path.exists(dst):
            if backup_dir is None:
                backup_dir = tempfile.mkdtemp(prefix="backup-", dir=_backup_root())
            bdst = os.path.join(backup_dir, rel)
            os.makedirs(os.path.dirname(bdst), exist_ok=True)
            shutil.copy2(dst, bdst)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy2(src, dst)
        restored.append(rel)
    return restored, backup_dir


def _restore_artifacts(
    workdir: str, dest: str, names: list[str]
) -> tuple[list[str], list[str]]:
    """Copy the attempt's produced ``names`` from ``workdir`` into ``dest``.

    ``names`` is the pre-verification snapshot (see ``AttemptArtifact.produced``),
    so only what the attempt itself created is restored — never verifier
    byproducts.

    SAFETY: an attempt runs in an isolated temp dir and we copy its output into
    the real cwd, which may already contain user files. We must NOT silently
    clobber them — a name that already exists in ``dest`` is SKIPPED and returned
    as a conflict so the caller can warn. Returns ``(copied, skipped)``.
    """
    copied: list[str] = []
    skipped: list[str] = []
    for name in names:
        src = os.path.join(workdir, name)
        if not os.path.exists(src):  # vanished/never-created — skip defensively
            continue
        dst = os.path.join(dest, name)
        if os.path.exists(dst):  # never overwrite an existing user path
            skipped.append(name)
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        copied.append(name)
    return copied, skipped


def _preserve_skipped(workdir: str, dest: str, skipped: list[str]) -> str | None:
    """Save winner files that couldn't be restored (name collisions) so the K
    attempts aren't wasted.

    Restore never overwrites existing cwd files, but the selected attempt's work
    must not be silently discarded. Copy the skipped (colliding) artifacts into a
    fresh ``.rune-bestof-*`` dir inside ``dest`` (dotfile → ignored by the
    non-empty-cwd warning) and return its path so the caller can point the user
    at it to diff/adopt. Returns ``None`` if nothing was preserved.
    """
    if not skipped:
        return None
    preserve = tempfile.mkdtemp(prefix=".rune-bestof-", dir=dest)
    saved = False
    for name in skipped:
        src = os.path.join(workdir, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(preserve, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        saved = True
    if not saved:
        shutil.rmtree(preserve, ignore_errors=True)
        return None
    return preserve


# Verifier output markers, cheapest first, that a candidate got FURTHER: a suite
# that ran and failed an assertion is closer to correct than one that never
# compiled/collected. Structured runner strings only — not NL matching.
_PROGRESS_MARKERS = ("failed", "assertionerror", "assert", " passed")
_INCONCLUSIVE_MARKERS = (
    "error[e", "could not compile", "cannot find", "unresolved import",
    "modulenotfounderror", "importerror", "collection error", "syntaxerror",
)


_SCRATCH_PREFIXES = ("test", "debug", "tmp", "scratch", "repro", "verify")


def _is_real_change(f: str, seed_from: str | None) -> bool:
    """A produced file that plausibly changes behavior: an edit of a file
    that exists in the seed (whatever its name), or a new file that isn't a
    scratch test/debug script."""
    if seed_from and os.path.isfile(os.path.join(seed_from, f)):
        return True
    return not os.path.basename(f).startswith(_SCRATCH_PREFIXES)


def _best_effort_score(
    a: AttemptArtifact, evidence: str, seed_from: str | None = None
) -> tuple:
    """Rank a FAILED candidate for hand-off. Higher is better.

    Purely a delivery choice — never promotes anything to "verified". Signals,
    cheap and in priority order: produced files at all; made a real change
    (see _is_real_change — a pile of debug scripts is not a fix); the
    verifier ran far enough to fail an assertion (vs never compiling); fewer
    inconclusive errors; then highest index — the repair attempt saw failure
    evidence the others didn't, so on an otherwise silent tie it is the
    best-informed candidate.
    """
    ev = (evidence or "").lower()
    produced = 1 if a.produced else 0
    src = 1 if any(_is_real_change(f, seed_from) for f in a.produced) else 0
    ran = 1 if any(m in ev for m in _PROGRESS_MARKERS) else 0
    inconclusive = sum(ev.count(m) for m in _INCONCLUSIVE_MARKERS)
    return (produced, src, ran, -inconclusive, a.index)


def _rank_best_effort(
    artifacts: list[AttemptArtifact],
    evidence_by_cwd: dict[str, str],
    seed_from: str | None = None,
) -> AttemptArtifact | None:
    """Pick the furthest-along failed candidate to hand off, or None."""
    if not artifacts:
        return None
    return max(
        artifacts,
        key=lambda a: _best_effort_score(
            a, evidence_by_cwd.get(a.workdir, ""), seed_from
        ),
    )


def _preserve_unverified(
    workdir: str, dest: str, produced: list[str]
) -> tuple[str | None, list[str]]:
    """Park an unverified attempt's files beside the project instead of deleting.

    When no attempt passes the verifier we deliberately do not restore anything
    into the working tree — unverified edits must not overwrite the user's
    files. But "we could not verify this" is not "this is wrong", and the
    attempt is about to be wiped by :func:`_cleanup`. Copy it into a fresh
    ``.rune-bestof-unverified-*`` dir inside ``dest`` (dotfile → ignored by the
    non-empty-cwd warning) so the user can diff and adopt it deliberately.

    Returns ``(path, saved_relpaths)``, or ``(None, [])`` when nothing was saved.
    """
    if not produced:
        return None, []
    preserve = tempfile.mkdtemp(prefix=".rune-bestof-unverified-", dir=dest)
    saved: list[str] = []
    for name in produced:
        src = os.path.join(workdir, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(preserve, name)
        try:
            # In seeded mode `produced` holds CHANGED relpaths like
            # "src/lib.rs", so the parent has to exist first. Without this the
            # copy raised and was swallowed, and the parked dir ended up with
            # only the top-level files — the actual edit was dropped while we
            # told the user their work had been kept.
            parent = os.path.dirname(dst)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        except OSError as exc:
            log.warning("bestof_preserve_failed", name=name, error=str(exc)[:120])
            continue
        saved.append(name)
    if not saved:
        shutil.rmtree(preserve, ignore_errors=True)
        return None, []
    return preserve, saved


async def _verifier_discriminates(verify_cwd, seed_from: str) -> bool:
    """Can this verifier fail the untouched baseline?

    Runs the check against a throwaway COPY of the pre-edit tree (never the
    user's own — the check may write files / run a build). Returns:
      True  — baseline FAILS the check (or the check is inconclusive there), so
              the check discriminates and best-of-K can select on it.
      False — baseline PASSES, so the check accepts anything and cannot select.
    On any error, default True: never suppress best-of on a probe failure.
    """
    scratch = None
    try:
        scratch = tempfile.mkdtemp(prefix="rune-probe-")
        probe = os.path.join(scratch, "baseline")
        shutil.copytree(
            seed_from, probe, symlinks=False,
            ignore=shutil.ignore_patterns(*_SEED_IGNORE_PATTERNS),
        )
        passed = await verify_cwd(probe)
        return not passed
    except Exception as exc:  # never let the probe break the run
        log.warning("bestof_probe_failed", error=str(exc)[:120])
        return True
    finally:
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)


# Attempt workdirs are wiped once a winner is chosen. Point this at a
# directory to keep each attempt's changed files, so it stays possible to
# ask later whether a correct fix was generated and not selected. Files
# rather than a patch, because delivery copies files too — an archive
# graded the same way cannot disagree with what shipped.
_KEEP_ATTEMPTS_ENV = "RUNE_BESTOF_KEEP_ATTEMPTS"


def _archive_attempts(artifacts: list[AttemptArtifact], seed_from: str | None) -> None:
    dest_root = os.environ.get(_KEEP_ATTEMPTS_ENV, "").strip()
    if not dest_root or not seed_from:
        return
    root = Path(dest_root).expanduser()
    for a in artifacts:
        out = root / f"attempt_{a.index}"
        try:
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True, exist_ok=True)
            kept = 0
            for rel in a.produced:
                src = Path(a.workdir) / rel
                if not src.is_file():
                    continue
                dst = out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                kept += 1
            (out / "_manifest.txt").write_text(
                "\n".join(a.produced) + f"\ncopied={kept}\n"
            )
        except OSError as exc:  # archiving must never affect the run
            log.debug("attempt_archive_failed", index=a.index, error=str(exc)[:80])


def _cleanup(artifacts: list[AttemptArtifact]) -> None:
    for a in artifacts:
        shutil.rmtree(a.workdir, ignore_errors=True)


async def _record_winner(
    message: str, answer: str, changed_files: list[str] | None = None
) -> bool:
    """Record the selected winner as one success episode.

    Attempt subprocesses are ephemeral (no learning), so without this a
    successful best-of solve teaches the self-improving loop nothing. Record
    exactly one episode — for the verifier-confirmed winner — in the parent
    (non-ephemeral) process. Success here is verifier-gated rather than the
    agent's self-report, so it won't record a wrong "success". Returns True if
    recorded.
    """
    try:
        from rune.agent.agent_context import (
            PostProcessInput,
            PrepareContextOptions,
            post_process_agent_result,
            prepare_agent_context,
        )

        ctx = await prepare_agent_context(
            PrepareContextOptions(goal=message, channel="cli")
        )
        goal_type: str | None = None
        try:
            from rune.agent.goal_classifier import classify_goal
            goal_type = (await classify_goal(message)).goal_type
        except Exception:
            pass

        await post_process_agent_result(
            PostProcessInput(
                context=ctx,
                success=True,  # verifier-confirmed
                answer=answer,
                reason="completed",
                evidence_gate=None,
                classification_hint=goal_type,
                changed_files=list(changed_files or []),
            )
        )
        from rune.memory.manager import get_memory_manager
        await get_memory_manager().promote_memories()
        log.info("bestof_winner_recorded")
        return True
    except Exception as exc:  # best-effort: never fail the run over learning
        log.warning("bestof_winner_record_failed", error=str(exc)[:120])
        return False


# Cap LLM rule-gen calls per best-of run (distinct failures learned from).
_MAX_FAILURE_RULES = 3


def _read_produced(workdir: str, produced: list[str], cap: int = 2) -> str:
    """Concatenate up to ``cap`` produced files from an attempt's workdir."""
    parts: list[str] = []
    for name in produced[:cap]:
        try:
            parts.append(Path(workdir, name).read_text()[:2500])
        except OSError:
            continue
    return "\n\n".join(parts)


async def _learn_from_contrast(
    winner: AttemptArtifact,
    losers: list[AttemptArtifact],
    ev_map: dict[str, str],
) -> str | None:
    """Distill a correctness rule from the winner-vs-losers contrast.

    ``_learn_from_failures`` covers the loser evidence on its own; this adds
    the piece it structurally misses — what the passing solution did
    differently. Best-effort; never breaks the run.
    """
    try:
        winner_code = _read_produced(winner.workdir, winner.produced)
        loser_pairs = [
            (_read_produced(lo.workdir, lo.produced), ev_map.get(lo.workdir, ""))
            for lo in losers
        ]
        loser_pairs = [(c, e) for c, e in loser_pairs if c.strip()]
        if not winner_code.strip() or not loser_pairs:
            return None
        from rune.memory.rule_learner import learn_from_contrast

        key = await learn_from_contrast(winner_code, loser_pairs)
        if key:
            log.info("bestof_contrast_learned", key=key)
        return key
    except Exception as exc:
        log.warning("bestof_contrast_failed", error=str(exc)[:120])
        return None


async def _learn_from_failures(message: str, evidence: list[str]) -> list[str]:
    """Learn correctness rules from the failed attempts' verifier evidence.

    best-of already computed the Evidence Gate verdict for every attempt, so the
    failed candidates' mismatch evidence is a free crisp-failure signal — the
    detection the default path lacks. Recording prevention rules here (in the
    non-ephemeral parent) lets a future task avoid the same mistakes (the rules
    inject via semantic retrieval).

    Learn from all distinct failure evidences (capped), not just the first: the K
    attempts can fail for different reasons — some structurally (missing/broken
    file → "verify files exist") and some on the actual logic (wrong value →
    "division rounding"). Learning from each lets the semantic retriever later
    pick whichever rule is relevant to the future task; crisp-failure signature
    dedup collapses near-duplicates, so this stays bounded. Best-effort; never
    breaks the run.
    """
    seen: set[str] = set()
    distinct: list[str] = []
    for e in evidence:
        e = (e or "").strip()
        if e and e not in seen:
            seen.add(e)
            distinct.append(e)
    if not distinct:
        return []
    try:
        domain = "code_modify"
        try:
            from rune.agent.goal_classifier import classify_goal
            domain = (await classify_goal(message)).goal_type or domain
        except Exception:
            pass
        from rune.memory.rule_learner import learn_from_crisp_failure
        learned: list[str] = []
        for ev in distinct[:_MAX_FAILURE_RULES]:
            try:
                key = await learn_from_crisp_failure("best_of_verifier", ev, domain)
            except Exception as exc:
                log.warning("bestof_learn_one_failed", error=str(exc)[:120])
                continue
            if key:
                learned.append(key)
        if learned:
            log.info("bestof_learned_from_failures", keys=learned)
        return learned
    except Exception as exc:
        log.warning("bestof_learn_failed", error=str(exc)[:120])
        return []


# Reporter: (stdout, solved, selected_index, pass_count, k, copied) -> None
Reporter = Callable[..., None]


async def _best_of_async(
    message: str,
    k: int,
    model: str | None,
    provider: str | None,
    *,
    report: Reporter,
    seed_cwd: bool = False,
) -> int:
    """Core best-of-K flow. Returns a process exit code (0 solved, 1 unsolved).

    ``seed_cwd`` (--include-cwd) copies the working tree into each attempt so the
    agent can edit existing files; restore then writes back only the changed
    files (overwriting, with a backup), instead of the greenfield new-files copy.
    """
    dest = os.getcwd()
    seed_from = dest if seed_cwd else None

    # Execution-first verifier: prefer running the repo's tests over the LLM-judge
    # Evidence Gate (execution selects code better, esp. for weak models
    # arXiv 2502.14382). Falls back to the Evidence Gate when no tests exist.
    verify_cwd = await make_verifier(message, seed_cwd=seed_from)
    has_check = bool(getattr(verify_cwd, "has_check", True))

    # A check that passes the untouched baseline accepts anything — it can
    # neither select nor verify. Probe once; if so, drop to a single attempt
    # and treat its result as unverified.
    check_discriminates = True
    if seed_from and has_check:
        check_discriminates = await _verifier_discriminates(verify_cwd, seed_from)
        if not check_discriminates:
            log.info("bestof_verifier_nondiscriminating_k1", original_k=k)
            k = 1
            # Only the EG check was proven vacuous (targeted tests don't
            # exist on the unchanged baseline) — disable just that component.
            verify_cwd.eg_disabled = True  # type: ignore[attr-defined]

    # Cap concurrent attempt subprocesses: each is a full agent run, so a large
    # K must not spawn K heavyweight processes at once. Mirrors the workflow
    # engine's min(cores-2, ...) policy.
    #
    # A single local model server (Ollama/llama.cpp) serves one request at a
    # time, so K parallel attempts contend for it and starve each other. Cloud
    # APIs parallelize; a local server does not. RUNE_BESTOF_CONCURRENCY=1 forces
    # serial attempts so each gets full model throughput.
    _conc_override = env_int("RUNE_BESTOF_CONCURRENCY", 0)
    if _conc_override > 0:
        cap = max(1, min(k, _conc_override))
    else:
        cap = max(1, min(k, (os.cpu_count() or 4) - 2))
    sem = asyncio.Semaphore(cap)

    async def run_attempt(i: int) -> AttemptArtifact:
        async with sem:
            msg = message + (fastpath_evidence if i == 0 else "")
            return await _run_attempt_subprocess(
                i, msg, model, provider, seed_from=seed_from
            )

    async def verify(artifact: AttemptArtifact) -> bool:
        # Seeded mode: an attempt that changed NOTHING cannot have fixed
        # anything, so a check pass on it is vacuous — and an attempt whose
        # only output is scratch test/debug scripts changed nothing that
        # matters (pre-fix code passes the existing tests just the same).
        # Never select either. A file that exists in the seed is a real
        # edit regardless of its name (django/test/testcases.py is source).
        if seed_cwd and not any(
            _is_real_change(f, seed_from) for f in artifact.produced
        ):
            return False
        # Cap verifier subprocesses too: sample_parallel gathers all K verifies
        # at once, each an Evidence-Gate check subprocess.
        async with sem:
            return await verify_cwd(artifact.workdir)

    # Rung 0: cheap single-shot pass, gated on a discriminating repro check.
    fastpath_evidence = ""
    if seed_from and os.environ.get(_FASTPATH_ENV, "1") != "0":
        from rune.agent.fastpath import run_fastpath

        fp_workdir = tempfile.mkdtemp(
            prefix="rune_fastpath_", dir=_attempt_work_root()
        )
        try:
            _seed_workdir(seed_from, fp_workdir)
            fp = await run_fastpath(message, seed_from, fp_workdir,
                                    model, provider)
        except Exception as exc:  # rung-0 must never sink the run
            log.warning("fastpath_error", error=str(exc)[:160])
            fp = None
        if fp and fp.verified and fp.applied:
            # A model-authored repro flipping to pass selects this fix but
            # cannot verify it — a fix for the reported example can still
            # miss the real requirement. Deliver as provisional.
            copied, backup_dir = _restore_changed(
                fp_workdir, dest, fp.applied
            )
            shutil.rmtree(fp_workdir, ignore_errors=True)
            await _record_winner(message, "", copied)
            report(
                "",
                solved=False,
                selected_index=0,
                pass_count=0,
                k=1,
                copied=[],
                skipped=[],
                has_check=True,
                no_artifact=0,
                applied=copied,
                apply_backup=backup_dir,
                provisional=True,
            )
            return 1
        shutil.rmtree(fp_workdir, ignore_errors=True)
        if fp and fp.repro_script:
            # The discriminating repro also becomes the verifier's first
            # check: flip-to-pass picks the winning candidate early — the
            # delivery label stays provisional.
            verify_cwd.repro_script = fp.repro_script  # type: ignore[attr-defined]
            log.info("repro_attached", chars=len(fp.repro_script))
            # Hand the agentic rung the evidence, not a failed diff. Only
            # attempt 0 gets it: the repro encodes ONE reading of the issue,
            # and broadcasting it to every attempt pins all K samples to
            # that reading — when it's wrong, best-of loses its diversity.
            fastpath_evidence = (
                "\n\nA reproduction script for this issue (currently "
                "FAILING) — make it pass without breaking existing tests:\n"
                "```python\n" + fp.repro_script[:3000] + "\n```\n"
                "Its current output:\n" + fp.repro_output[:800]
            )

    strategy = os.environ.get(_STRATEGY_ENV, "auto").strip().lower()
    if strategy == "auto":
        strategy = "sequential" if (provider or "").lower() == "ollama" else "race2"
    repair_ok = (
        os.environ.get(_REPAIR_ENV, "1") != "0"
        and (provider or "").lower() != "ollama"  # ≥17B repair crossover
    )

    async def run_with(i: int, msg: str) -> AttemptArtifact:
        async with sem:
            return await _run_attempt_subprocess(
                i, msg, model, provider, seed_from=seed_from
            )

    def _last_evidence(arts: list[AttemptArtifact]) -> str:
        for a in reversed(arts):
            ev = ev_map_ref.get(a.workdir, "")
            if ev:
                return ev
        return ""

    # NOTE: must alias the live dict, not `or {}` it — it is still EMPTY here
    # (falsy), and evidence written during verification has to be visible.
    ev_map_ref = getattr(verify_cwd, "evidence_by_cwd", None)
    if ev_map_ref is None:
        ev_map_ref = {}

    if k <= 1 or strategy == "parallel":
        res = await sample_parallel(run_attempt, verify, k)
    elif strategy == "sequential":
        # Sample → verify → exit on first pass; attempt 2 sees the failure.
        attempts: list[Attempt] = []
        selected = selected_index = None
        for i in range(k):
            msg = message + (fastpath_evidence if i == 0 else "")
            if i == 1 and repair_ok:
                ev = _last_evidence([a.candidate for a in attempts])
                if ev:
                    msg = message + _repair_suffix(ev)
        # (attempt 3+ goes back to fresh independent samples for diversity)
            art = await run_with(i, msg)
            passed = await verify(art)
            attempts.append(Attempt(index=i, candidate=art, passed=passed))
            if passed:
                selected, selected_index = art, i
                log.info("bestof_sequential_early_exit", attempts=i + 1)
                break
        res = RejectionResult(
            selected=selected, selected_index=selected_index, attempts=attempts
        )
    else:  # race2: two parallel attempts, early-exit verify, one repair shot
        first_two = await asyncio.gather(
            run_with(0, message + fastpath_evidence), run_with(1, message)
        )
        attempts = []
        selected = selected_index = None
        for i, art in enumerate(first_two):
            passed = False
            if selected is None:  # early-exit: skip verifying after a pass
                passed = await verify(art)
            attempts.append(Attempt(index=i, candidate=art, passed=passed))
            if passed and selected is None:
                selected, selected_index = art, i
        if selected is None and k > 2:
            msg = message
            if repair_ok:
                ev = _last_evidence(first_two)
                if ev:
                    msg = message + _repair_suffix(ev)
                    log.info("bestof_repair_attempt")
            art = await run_with(2, msg)
            passed = await verify(art)
            attempts.append(Attempt(index=2, candidate=art, passed=passed))
            if passed:
                selected, selected_index = art, 2
        res = RejectionResult(
            selected=selected, selected_index=selected_index, attempts=attempts
        )
    artifacts: list[AttemptArtifact] = [a.candidate for a in res.attempts]
    _archive_attempts(artifacts, seed_from)

    # Learn a correctness rule from any failed attempts' verifier evidence
    # (fires whether or not a winner was found — every failed candidate is a
    # detected mistake the default path would have missed).
    ev_map = getattr(verify_cwd, "evidence_by_cwd", {}) or {}
    failed_ev = [
        ev_map.get(a.candidate.workdir, "") for a in res.attempts if not a.passed
    ]
    await _learn_from_failures(message, failed_ev)

    try:
        # Existing-tests pass = PROVISIONAL: pre-fix code passes them too, so
        # it selects the candidate but must deliver as unverified.
        _provisional = bool(
            res.solved
            and res.selected is not None
            and getattr(verify_cwd, "provisional_by_cwd", {}).get(
                res.selected.workdir
            )
        )
        if res.solved and res.selected is not None and _provisional:
            selected = res.selected
            if seed_cwd:
                applied, apply_backup = _restore_changed(
                    selected.workdir, dest, selected.produced
                )
            else:
                applied, _sk = _restore_artifacts(
                    selected.workdir, dest, selected.produced
                )
                apply_backup = None
            log.info(
                "bestof_provisional_selection",
                index=res.selected_index,
                files=len(applied),
            )
            # A provisional pick still separates one candidate that passed a
            # real check from siblings that failed one, which is the contrast
            # worth learning from. Only the verified branch had this, and
            # since a repro flip stopped counting as verified that branch
            # almost never opens on a repo fix — so the distillation was
            # wired to a path that no longer runs.
            if os.environ.get("RUNE_CONTRASTIVE_DISTILL", "1") != "0":
                await _learn_from_contrast(
                    selected,
                    [a.candidate for a in res.attempts if not a.passed],
                    ev_map,
                )
            report(
                selected.stdout,
                solved=False,
                selected_index=res.selected_index,
                pass_count=0,
                k=k,
                copied=[],
                skipped=[],
                has_check=has_check,
                no_artifact=0,
                applied=applied,
                apply_backup=apply_backup,
                provisional=True,
            )
            return 1

        if res.solved and res.selected is not None:
            selected: AttemptArtifact = res.selected
            if seed_cwd:
                # Seeded mode: write back the agent's edits (overwrite intended),
                # backing up originals for undo.
                copied, backup_dir = _restore_changed(
                    selected.workdir, dest, selected.produced
                )
                skipped, preserved_dir = [], None
            else:
                copied, skipped = _restore_artifacts(
                    selected.workdir, dest, selected.produced
                )
                # Don't discard the winner on collision — save it for the user.
                preserved_dir = _preserve_skipped(selected.workdir, dest, skipped)
                backup_dir = None
            # Learn from the verifier-confirmed winner (1 episode).
            await _record_winner(message, selected.stdout, selected.produced)
            # With a verified winner in hand, distill what separated it from
            # the failed attempts into a learned rule (failure evidence alone
            # only ever yields process advice). RUNE_CONTRASTIVE_DISTILL=0
            # opts out.
            if os.environ.get("RUNE_CONTRASTIVE_DISTILL", "1") != "0":
                await _learn_from_contrast(
                    selected, [a.candidate for a in res.attempts if not a.passed], ev_map
                )
            method_map = getattr(verify_cwd, "method_by_cwd", {}) or {}
            report(
                selected.stdout,
                solved=True,
                selected_index=res.selected_index,
                pass_count=res.pass_count,
                k=k,
                copied=copied,
                skipped=skipped,
                preserved_dir=preserved_dir,
                backup_dir=backup_dir,
                has_check=has_check,
                no_artifact=0,
                verify_method=method_map.get(selected.workdir),
            )
            return 0

        # No attempt passed. Break the 0/K down so the user can tell WHY:
        #  - no mechanical check could be built  → best-of-K cannot select at all
        #  - attempts produced no files          → generator didn't write artifacts
        #  - attempts wrote files but failed      → generator produced wrong output
        # When nothing verifies, hand off the BEST-EFFORT candidate rather than
        # always attempt #0: #0 may have produced nothing while a sibling wrote a
        # real (if unverified) patch, which is how a would-be delivery became an
        # empty one. Ranking picks the furthest-along candidate; it only chooses
        # what to hand off — a passing verification is still the ONLY thing that
        # flips "done", so this cannot manufacture a fake success.
        no_artifact = sum(1 for a in artifacts if not a.produced)
        best = (
            _rank_best_effort(artifacts, ev_map, seed_from)
            if artifacts else None
        )
        # Unverified is not wrong: seeded mode applies the best effort
        # (with backup); otherwise park it beside the project.
        applied: list[str] = []
        apply_backup: str | None = None
        unverified_dir: str | None = None
        unverified_files: list[str] = []
        if best and best.produced:
            if seed_cwd and os.environ.get(_APPLY_UNVERIFIED_ENV, "1") != "0":
                applied, apply_backup = _restore_changed(
                    best.workdir, dest, best.produced
                )
                log.info(
                    "bestof_unverified_applied",
                    files=len(applied),
                    backup=apply_backup,
                )
            if not applied:
                unverified_dir, unverified_files = _preserve_unverified(
                    best.workdir, dest, best.produced
                )
        report(
            best.stdout if best else "",
            solved=False,
            selected_index=None,
            pass_count=0,
            k=k,
            copied=[],
            skipped=[],
            has_check=has_check,
            no_artifact=no_artifact,
            unverified_dir=unverified_dir,
            unverified_files=unverified_files,
            applied=applied,
            apply_backup=apply_backup,
            check_discriminates=check_discriminates,
        )
        return 1
    finally:
        _cleanup(artifacts)


def run_best_of(
    message: str,
    k: int,
    model: str | None = None,
    provider: str | None = None,
    seed_cwd: bool = False,
) -> None:
    """Synchronous entry point for the CLI: run best-of-K and print the outcome.

    Raises ``typer.Exit(1)`` when no attempt passes the verifier so the one-shot
    command exits non-zero (mirrors a failed single run). ``seed_cwd`` enables
    --include-cwd mode (edit existing files instead of greenfield new files).
    """
    from rich.console import Console

    console = Console(stderr=True)

    if seed_cwd:
        # Seeding copies the cwd into each of K attempts and all K workdirs
        # persist until cleanup, so a large cwd can exhaust disk. Refuse before
        # doing any work.
        _err = _check_seed_size(os.getcwd())
        if _err:
            console.print(f"[red]{_err}[/red]")
            import typer

            raise typer.Exit(2)
        # Seeded mode edits a copy of the working tree and writes changes back
        # (overwriting, with a backup). Tell the user it's destructive-by-design.
        console.print(
            "[dim]best-of --include-cwd: each attempt edits a copy of the working "
            "tree; the winner's changes are written back (originals backed up).[/dim]"
        )
    else:
        # B-warn: greenfield attempts run in isolated EMPTY temp dirs — they do
        # NOT see the working tree. Warn when run from a non-empty dir so this
        # isn't a silent failure; suggest --include-cwd for edit tasks.
        try:
            if any(not n.startswith(".") for n in os.listdir(os.getcwd())):
                console.print(
                    "[dim]best-of: each attempt runs in an isolated empty temp dir "
                    "(working tree NOT copied in). Suited to new-file tasks; use "
                    "--include-cwd to edit existing files.[/dim]"
                )
        except OSError:
            pass

    def report(
        stdout: str,
        *,
        solved: bool,
        selected_index: int | None,
        pass_count: int,
        k: int,
        copied: list[str],
        skipped: list[str] | None = None,
        preserved_dir: str | None = None,
        backup_dir: str | None = None,
        has_check: bool = True,
        no_artifact: int = 0,
        verify_method: str | None = None,
        unverified_dir: str | None = None,
        unverified_files: list[str] | None = None,
        applied: list[str] | None = None,
        apply_backup: str | None = None,
        check_discriminates: bool = True,
        provisional: bool = False,
    ) -> None:
        def _applied_note() -> str:
            """Describe the applied-but-unverified delivery and its undo path."""
            if not applied:
                return ""
            listed = ", ".join(applied[:5]) + ("…" if len(applied) > 5 else "")
            undo = ""
            if apply_backup:
                try:
                    shown = os.path.relpath(apply_backup)
                except ValueError:
                    shown = apply_backup
                undo = f"\n  undo:  cp -R {shown}/. ."
            return (
                f"\nApplied the best-effort attempt to {len(applied)} file(s) "
                f"(UNVERIFIED — review before trusting): {listed}.{undo}"
            )

        def _kept(path: str | None) -> str:
            """Hand the unverified work over: what it is, and how to take it.

            Deliberately NOT a bare "UNVERIFIED" banner. A blanket
            low-confidence label was measured to cut trust and drive
            under-reliance without improving decisions (arXiv:2402.07632 Exp2),
            so name the files and give the exact apply command — the reader can
            then size the review against the actual change instead of against a
            warning.
            """
            if not path:
                return ""
            try:
                shown = os.path.relpath(path)
            except ValueError:  # different drive/root
                shown = path
            files = unverified_files or []
            listed = ", ".join(files[:5]) + ("…" if len(files) > 5 else "")
            what = f" {len(files)} file(s): {listed}." if files else ""
            return (
                f"\nUnverified result kept — nothing was written to your files."
                f"{what}"
                f"\n  see:   diff -ru . {shown} | head"
                f"\n  apply: cp -R {shown}/. ."
            )

        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n", flush=True)
        if solved:
            names = ", ".join(copied) if copied else "—"
            # Report what the winner passed (the repo's test command when known,
            # else a generic "verifier").
            passed = f"passed {verify_method}" if verify_method else "passed verifier"
            console.print(
                f"[dim]best-of-{k}: tried {k} · {pass_count} verified · "
                f"picked #{selected_index} ({passed}); "
                f"restored {len(copied)} item(s): {names}[/dim]"
            )
            if backup_dir:
                console.print(
                    f"[dim]best-of: originals backed up to "
                    f"{backup_dir}/ before overwrite.[/dim]"
                )
            if skipped:
                where = (
                    f" Winner saved to {os.path.relpath(preserved_dir)}/ — "
                    f"diff against your version."
                    if preserved_dir
                    else ""
                )
                console.print(
                    f"[yellow]best-of: NOT overwritten (already exist in cwd): "
                    f"{', '.join(skipped)}.{where}[/yellow]"
                )
        elif provisional:
            console.print(
                f"[yellow]best-of-{k}: picked the candidate that passes the "
                f"repo's EXISTING tests — but those tests also pass the "
                f"pre-fix code, so the change itself is NOT verified."
                f"{_applied_note()}[/yellow]"
            )
        elif not has_check:
            console.print(
                f"[yellow]best-of-{k}: no mechanical success check could be built "
                f"for this task, so the verifier cannot select a candidate "
                f"(best-of-K only helps verifiable tasks). Showing the best "
                f"attempt unverified.{_applied_note()}{_kept(unverified_dir)}"
                f"[/yellow]"
            )
        elif not check_discriminates:
            console.print(
                f"[yellow]best-of-{k}: the success check also passes the "
                f"UNFIXED baseline, so it cannot verify this task — a pass "
                f"from it would be meaningless. Showing the best attempt "
                f"unverified.{_applied_note()}{_kept(unverified_dir)}[/yellow]"
            )
        else:
            wrote = k - no_artifact
            console.print(
                f"[yellow]best-of-{k}: no attempt passed the verifier (0/{k}); "
                f"{no_artifact}/{k} produced no files (generator didn't write "
                f"artifacts), {wrote}/{k} wrote files but failed the check. "
                f"Showing the best attempt unverified."
                f"{_applied_note()}{_kept(unverified_dir)}[/yellow]"
            )

    exit_code = asyncio.run(
        _best_of_async(message, k, model, provider, report=report, seed_cwd=seed_cwd)
    )
    if exit_code != 0:
        import typer

        raise typer.Exit(exit_code)
