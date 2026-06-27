"""Keep an isolated worker's file writes inside its worktree.

cwd only confines relative paths and Guardian only blocks system paths, so a
worker could still write an absolute / ``..`` / ``~`` path outside its
workspace. When RUNE_ISOLATION_ROOT is set, file capabilities call enforce()
before writing. No root set (normal single-agent runs) -> no-op.
"""

from __future__ import annotations

import os

ISOLATION_ENV = "RUNE_ISOLATION_ROOT"


def isolation_root() -> str | None:
    """Return the active isolation root (realpath), or None if not isolating."""
    raw = os.environ.get(ISOLATION_ENV)
    if not raw:
        return None
    try:
        return os.path.realpath(raw)
    except OSError:
        return raw


def _resolve(path: str, root: str) -> str:
    # expanduser first: `~/foo` means home (outside the worktree), not a literal
    # `~` subdir. realpath collapses symlinks and `..`.
    p = os.path.expanduser(path)
    p = p if os.path.isabs(p) else os.path.join(root, p)
    return os.path.realpath(p)


def is_within(path: str) -> bool:
    """Whether *path* is inside the isolation root (True if not isolating)."""
    root = isolation_root()
    if root is None:
        return True
    resolved = _resolve(path, root)
    # os.sep guard avoids /work matching /work2
    return resolved == root or resolved.startswith(root + os.sep)


def enforce(path: str) -> str | None:
    """Error string if *path* escapes the isolation root, else None (deny on non-None)."""
    root = isolation_root()
    if root is None:
        return None
    if is_within(path):
        return None
    return (
        f"Isolation violation: '{path}' resolves outside the worker's isolation "
        f"root ({root}). Workers may only modify files within their workspace."
    )
