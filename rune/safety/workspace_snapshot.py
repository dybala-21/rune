"""Keep a restorable copy of the workspace before a shell command runs.

The file capabilities route deletes through a trash directory and re-read
what they wrote, but a shell command reaches the same files without going
near them: `rm`, `mv`, a `>` redirect, a script that rewrites a tree. That
is the same gap Claude Code documents in its own checkpoints — they cover
its file tools and not Bash — and it is how an agent asked to tidy a
directory can remove work nobody can regenerate.

Guessing which paths a command will touch is not worth attempting: flags,
variables, redirects and `xargs` defeat any pattern, and a miss means lost
data. Instead the whole workspace is cloned first. On APFS and on Linux
filesystems with reflink support the clone shares blocks with the
original, so a 138 MB repository takes well under a second and no extra
disk until something diverges.

Cheap enough to be unconditional, but not free, so it is skipped for
commands whose every segment is a known read-only tool. The list is
deliberately short: anything unrecognised is treated as capable of
writing, because over-snapshotting costs a moment and under-snapshotting
costs the user's files.

Snapshots live outside the workspace so they can never be swept up by the
next command, and only the most recent few are kept.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from rune.utils.logger import get_logger
from rune.utils.paths import rune_data

log = get_logger(__name__)

_ENV_FLAG = "RUNE_SHELL_SNAPSHOT"
_KEEP = 3
_CLONE_TIMEOUT_S = 30.0
# A clone is metadata-only, but a huge tree still costs wall time.
_MAX_ENTRIES = 200_000

# Commands that cannot change the tree. Everything else is assumed to be
# able to, including interpreters — `python -c` writes files as easily as
# `rm` does.
_READ_ONLY = frozenset({
    "ls", "cat", "head", "tail", "wc", "file", "stat", "du", "df", "pwd",
    "echo", "printf", "grep", "egrep", "fgrep", "rg", "ag", "which", "type",
    "basename", "dirname", "realpath", "readlink", "sort", "uniq", "cut",
    "tr", "diff", "cmp", "env", "date", "uname", "id", "whoami", "true",
    "false", "test", "sleep", "seq", "tree", "jq", "column", "nl", "less",
    "more", "man", "history", "hostname", "ps", "top", "uptime",
})
_READ_ONLY_SUB = {
    "git": frozenset({"status", "log", "diff", "show", "blame", "branch",
                      "tag", "remote", "ls-files", "rev-parse", "describe",
                      "cat-file", "for-each-ref", "shortlog", "reflog"}),
}
# `find` and `xargs` are read-only only without their execute/delete forms.
_FIND_WRITES = re.compile(r"-(delete|exec|execdir|ok|okdir|fprint\w*|fls)\b")


def snapshot_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "1") != "0"


def looks_read_only(command: str) -> bool:
    """True when every segment of *command* is a known read-only tool."""
    if not command or not command.strip():
        return True
    if ">" in command or ">>" in command:
        return False
    for seg in re.split(r"[;&|]+", command):
        words = seg.strip().split()
        while words and "=" in words[0] and not words[0].startswith("-"):
            words = words[1:]  # leading VAR=value assignments
        if not words:
            continue
        head = os.path.basename(words[0])
        if head in ("find", "xargs"):
            if _FIND_WRITES.search(seg):
                return False
            continue
        if head in _READ_ONLY_SUB:
            sub = next((w for w in words[1:] if not w.startswith("-")), "")
            if sub not in _READ_ONLY_SUB[head]:
                return False
            continue
        if head not in _READ_ONLY:
            return False
    return True


def _snapshot_root(workspace: Path) -> Path:
    tag = re.sub(r"[^\w.-]", "_", str(workspace))[-120:]
    return Path(rune_data()) / "snapshots" / tag


def _clone_cmd(src: Path, dst: Path) -> list[str]:
    if sys.platform == "darwin":
        return ["cp", "-c", "-R", str(src), str(dst)]
    return ["cp", "-r", "--reflink=auto", str(src), str(dst)]


def _too_big(workspace: Path) -> bool:
    seen = 0
    for _root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d != ".rune-trash"]
        seen += len(files)
        if seen > _MAX_ENTRIES:
            return True
    return False


def _prune(root: Path) -> None:
    try:
        kept = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
    except OSError:
        return
    for old in kept[_KEEP:]:
        shutil.rmtree(old, ignore_errors=True)


def take(workspace: str | Path) -> Path | None:
    """Clone *workspace* and return the copy, or None if it was skipped."""
    if not snapshot_enabled():
        return None
    ws = Path(workspace).expanduser().resolve()
    if not ws.is_dir():
        return None
    try:
        if _too_big(ws):
            log.debug("snapshot_skipped_large", workspace=str(ws))
            return None
        root = _snapshot_root(ws)
        root.mkdir(parents=True, exist_ok=True)
        dest = root / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        r = subprocess.run(_clone_cmd(ws, dest), capture_output=True,
                           text=True, timeout=_CLONE_TIMEOUT_S)
        if r.returncode != 0 or not dest.exists():
            log.debug("snapshot_failed", error=r.stderr.strip()[:120])
            return None
        _prune(root)
        log.info("workspace_snapshot", path=str(dest))
        return dest
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("snapshot_error", error=str(exc)[:120])
        return None


def snapshots(workspace: str | Path) -> list[Path]:
    """Kept snapshots of *workspace*, newest first."""
    root = _snapshot_root(Path(workspace).expanduser().resolve())
    try:
        return sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
    except OSError:
        return []


def latest(workspace: str | Path) -> Path | None:
    """Most recent snapshot of *workspace*, if there is one."""
    snaps = snapshots(workspace)
    return snaps[0] if snaps else None


def restore(workspace: str | Path, rel: str) -> bool:
    """Put one file back, from the newest snapshot that still holds it.

    A command that removes something is followed by a snapshot taken for
    the next command, and that one no longer contains what was just
    deleted. Looking only at the newest copy would therefore fail to
    recover anything except what the most recent command removed, so this
    walks back through the kept snapshots.
    """
    dst = Path(workspace).expanduser().resolve() / rel
    for snap in snapshots(workspace):
        src = snap / rel
        if not src.is_file():
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        except OSError:
            return False
        log.info("workspace_restored", path=rel, snapshot=snap.name)
        return True
    return False


def missing_since_snapshot(workspace: str | Path) -> list[str]:
    """Files present in the newest snapshot and gone from the workspace."""
    snap = latest(workspace)
    if snap is None:
        return []
    ws = Path(workspace).expanduser().resolve()
    gone: list[str] = []
    for src in snap.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(snap)
        if rel.parts and rel.parts[0] in (".git", ".rune-trash"):
            continue
        if not (ws / rel).exists():
            gone.append(str(rel))
            if len(gone) >= 200:
                break
    return gone
