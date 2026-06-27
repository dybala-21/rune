"""Merge worker change-sets back to the main tree, atomically.

Two-phase: resolve/validate every change first; if anything conflicts or won't
apply, apply nothing. A pre-merge snapshot of the touched paths is returned so
the caller can roll back. Same file from several workers is resolved per policy:
auto_3way (line-level git merge-file; identical edits are no-ops), fail_closed,
or last_write. Binary/symlink/delete-vs-modify overlaps are hard conflicts.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

from rune.agent.worktree import ChangeSet, FileChange
from rune.utils.logger import get_logger

log = get_logger(__name__)

AUTO_3WAY = "auto_3way"
FAIL_CLOSED = "fail_closed"
LAST_WRITE = "last_write"


@dataclass(slots=True)
class MergeResult:
    ok: bool
    applied: list[str] = field(default_factory=list)
    conflicts: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)
    # pre-merge state of touched paths for rollback: path -> bytes | "__ABSENT__"
    snapshot: dict[str, bytes | str] = field(default_factory=dict)
    reason: str = ""


_ABSENT = "__ABSENT__"


def analyze_overlaps(changesets: list[ChangeSet]) -> dict[str, list[str]]:
    """path -> [worker_id, ...] for paths touched by more than one worker."""
    by_path: dict[str, list[str]] = {}
    for cs in changesets:
        for path in cs.changes:
            by_path.setdefault(path, []).append(cs.worker_id)
    return {p: w for p, w in by_path.items() if len(w) > 1}


def _git_merge_file(base: bytes, a: bytes, b: bytes) -> tuple[bytes, bool]:
    """3-way line merge via `git merge-file`. Returns (merged, had_conflict)."""
    d = tempfile.mkdtemp(prefix="rune-merge-")
    try:
        fa, fbase, fb = (os.path.join(d, n) for n in ("a", "base", "b"))
        for fn, data in ((fa, a), (fbase, base), (fb, b)):
            with open(fn, "wb") as fh:
                fh.write(data)
        # merges fb's changes (vs base) into fa, writing result to fa.
        r = subprocess.run(["git", "merge-file", "-p", fa, fbase, fb],
                           capture_output=True, timeout=30)
        return r.stdout, r.returncode != 0
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _resolve_path(
    repo: str, path: str, writers: list[tuple[str, FileChange]],
    policy: str,
) -> tuple[FileChange | None, str | None]:
    """Resolve possibly-multiple changes to one. Returns (change, conflict_reason)."""
    if len(writers) == 1:
        return writers[0][1], None

    # Multiple workers touched this path.
    ops = {fc.op for _, fc in writers}
    # Identical content → no real conflict.
    contents = {(fc.op, fc.content, fc.is_symlink, fc.symlink_target)
                for _, fc in writers}
    if len(contents) == 1:
        return writers[0][1], None

    if policy == FAIL_CLOSED:
        return None, f"{len(writers)} workers modified this file (fail_closed)"
    if policy == LAST_WRITE:
        return writers[-1][1], None

    # auto_3way: only line-mergeable text modifications.
    if "deleted" in ops:
        return None, "delete vs modify conflict"
    if any(fc.is_symlink or fc.content is None for _, fc in writers):
        return None, "symlink/binary conflict (not line-mergeable)"
    # base = current main content (workers started from it); empty if new file.
    main_path = os.path.join(repo, path)
    base = b""
    if os.path.isfile(main_path) and not os.path.islink(main_path):
        with open(main_path, "rb") as fh:
            base = fh.read()
    merged = writers[0][1].content or b""
    for _, fc in writers[1:]:
        merged, conflict = _git_merge_file(base, merged, fc.content or b"")
        if conflict:
            return None, "overlapping edits could not be auto-merged"
    return FileChange(op="modified", content=merged,
                      mode=writers[0][1].mode), None


def _snapshot_path(repo: str, path: str) -> bytes | str:
    ap = os.path.join(repo, path)
    if os.path.islink(ap):
        return "__SYMLINK__:" + os.readlink(ap)
    if os.path.isfile(ap):
        with open(ap, "rb") as fh:
            return fh.read()
    return _ABSENT


def _within_repo(repo: str, path: str) -> bool:
    real = os.path.realpath(os.path.join(repo, path))
    rroot = os.path.realpath(repo)
    return real == rroot or real.startswith(rroot + os.sep)


def merge_changesets(
    repo: str, changesets: list[ChangeSet], *, policy: str = AUTO_3WAY,
) -> MergeResult:
    """Two-phase atomic merge. Applies all-or-nothing; returns rollback snapshot."""
    changesets = [cs for cs in changesets if cs is not None]
    # --- Phase 1: resolve + validate EVERYTHING (apply nothing yet) ---
    by_path: dict[str, list[tuple[str, FileChange]]] = {}
    for cs in changesets:
        for path, fc in cs.changes.items():
            by_path.setdefault(path, []).append((cs.worker_id, fc))

    resolved: dict[str, FileChange] = {}
    conflicts: list[tuple[str, str]] = []
    for path, writers in by_path.items():
        if not _within_repo(repo, path):
            conflicts.append((path, "path escapes the workspace"))
            continue
        change, reason = _resolve_path(repo, path, writers, policy)
        if reason:
            conflicts.append((path, reason))
        elif change is not None:
            resolved[path] = change

    if conflicts:
        return MergeResult(ok=False, conflicts=conflicts,
                           reason=f"{len(conflicts)} conflict(s); nothing applied")

    # snapshot touched paths so the caller can roll back
    snapshot = {path: _snapshot_path(repo, path) for path in resolved}

    # --- Phase 2: apply all ---
    applied: list[str] = []
    try:
        for path, fc in resolved.items():
            ap = os.path.join(repo, path)
            if fc.op == "deleted":
                if os.path.lexists(ap):
                    os.remove(ap)
            elif fc.is_symlink:
                if os.path.lexists(ap):
                    os.remove(ap)
                os.makedirs(os.path.dirname(ap), exist_ok=True)
                os.symlink(fc.symlink_target, ap)
            else:
                os.makedirs(os.path.dirname(ap), exist_ok=True)
                with open(ap, "wb") as fh:
                    fh.write(fc.content or b"")
                if fc.mode:
                    os.chmod(ap, fc.mode)
            applied.append(path)
    except Exception as exc:
        # Mid-apply failure → roll back what we applied (best-effort), report.
        log.error("merge_apply_failed", error=str(exc)[:200])
        rollback(repo, {p: snapshot[p] for p in applied})
        return MergeResult(ok=False, snapshot=snapshot,
                           reason=f"apply failed, rolled back: {exc}")

    log.info("merge_applied", count=len(applied), policy=policy)
    return MergeResult(ok=True, applied=applied, snapshot=snapshot)


def rollback(repo: str, snapshot: dict[str, bytes | str]) -> None:
    """Restore touched paths to their pre-merge state."""
    for path, state in snapshot.items():
        ap = os.path.join(repo, path)
        try:
            if state == _ABSENT:
                if os.path.lexists(ap):
                    os.remove(ap)
            elif isinstance(state, str) and state.startswith("__SYMLINK__:"):
                if os.path.lexists(ap):
                    os.remove(ap)
                os.symlink(state[len("__SYMLINK__:"):], ap)
            else:
                os.makedirs(os.path.dirname(ap), exist_ok=True)
                with open(ap, "wb") as fh:
                    fh.write(state if isinstance(state, bytes) else state.encode())
        except Exception as exc:
            log.error("rollback_failed", path=path, error=str(exc)[:120])
