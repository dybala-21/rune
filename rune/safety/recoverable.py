"""Make destructive file operations recoverable, and verify they happened.

Two rules, both learned the hard way by other agents:

1. Never destroy bytes we cannot get back. A vague "clean this up" is a
   request the agent has to interpret, and interpreting it wrongly costs
   the user files they may not be able to recreate. Deleting through a
   trash directory turns that from data loss into an undo. Asking the
   user first does not work as a control — approval prompts get approved
   almost every time — so recoverability, not permission, is the gate.

2. Never infer success from a call that returned. An agent that assumes a
   directory was created, or a file removed, and then acts on that belief
   will corrupt state in ways that are hard to trace. Every mutation
   re-reads what it just did and reports the observed result.

The trash lives beside the workspace so an undo does not depend on RUNE
being reachable, and entries are timestamped so repeated deletes of the
same name do not collide.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rune.utils.logger import get_logger

log = get_logger(__name__)

TRASH_DIRNAME = ".rune-trash"
_ENV_TRASH = "RUNE_TRASH"
_ENV_VERIFY = "RUNE_VERIFY_MUTATIONS"

# Regenerable by a build, an install, or the OS. These are the only things
# worth deleting outright; everything else goes to the trash.
_JUNK_NAMES = frozenset({
    ".DS_Store", "Thumbs.db", "desktop.ini", ".pytest_cache",
    "__pycache__", ".mypy_cache", ".ruff_cache", ".tox", ".coverage",
})
_JUNK_SUFFIXES = (".pyc", ".pyo", ".class", ".o", ".obj", ".log.gz")
_JUNK_DIR_PARTS = frozenset({
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", ".gradle",
})


def trash_enabled() -> bool:
    return os.environ.get(_ENV_TRASH, "1") != "0"


def verify_enabled() -> bool:
    return os.environ.get(_ENV_VERIFY, "1") != "0"


def is_regenerable(path: Path) -> bool:
    """True when losing *path* costs a rebuild, not the user's work."""
    if path.name in _JUNK_NAMES:
        return True
    if path.suffix in _JUNK_SUFFIXES:
        return True
    return any(part in _JUNK_DIR_PARTS for part in path.parts)


def _trash_root(target: Path) -> Path:
    """Trash beside the workspace root, or beside the file as a fallback."""
    cwd = Path.cwd().resolve()
    try:
        target.relative_to(cwd)
        base = cwd
    except ValueError:
        base = target.parent
    return base / TRASH_DIRNAME


@dataclass
class TrashEntry:
    original: str
    stored: str


def move_to_trash(target: Path) -> TrashEntry:
    """Move *target* into the workspace trash and return where it landed."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    root = _trash_root(target) / stamp
    root.mkdir(parents=True, exist_ok=True)
    dest = root / target.name
    shutil.move(str(target), str(dest))
    log.info("trashed", original=str(target), stored=str(dest))
    return TrashEntry(original=str(target), stored=str(dest))


@dataclass
class MutationCheck:
    ok: bool
    detail: str


def verify_written(path: Path, expected_size: int | None = None) -> MutationCheck:
    """Confirm a write landed, by reading the file back off disk."""
    if not verify_enabled():
        return MutationCheck(True, "verification disabled")
    if not path.is_file():
        return MutationCheck(False, f"{path} does not exist after the write")
    size = path.stat().st_size
    if expected_size is not None and size == 0 and expected_size > 0:
        return MutationCheck(False, f"{path} is empty after writing {expected_size} bytes")
    return MutationCheck(True, f"{size} bytes on disk")


def verify_gone(path: Path) -> MutationCheck:
    """Confirm a delete actually removed the path."""
    if not verify_enabled():
        return MutationCheck(True, "verification disabled")
    if path.exists():
        return MutationCheck(False, f"{path} still exists after the delete")
    return MutationCheck(True, "removed")


def verify_dir(path: Path) -> MutationCheck:
    """Confirm a directory exists. A silently failed mkdir followed by
    moves into the missing destination is how agents overwrite files."""
    if not verify_enabled():
        return MutationCheck(True, "verification disabled")
    if not path.is_dir():
        return MutationCheck(False, f"{path} is not a directory")
    return MutationCheck(True, "directory present")
