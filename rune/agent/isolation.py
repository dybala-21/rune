"""Confine worker file writes to RUNE_ISOLATION_ROOT when set."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar

ISOLATION_ENV = "RUNE_ISOLATION_ROOT"
_root: ContextVar[str | None] = ContextVar("workspace_isolation", default=None)


def isolation_root() -> str | None:
    """Return the active isolation root (realpath), or None if not isolating."""
    raw = _root.get() or os.environ.get(ISOLATION_ENV)
    if not raw:
        return None
    try:
        return os.path.realpath(raw)
    except OSError:
        return raw


@contextmanager
def isolation_scope(root: str):
    parent = isolation_root()
    resolved = os.path.realpath(root)
    if parent and not is_within(resolved):
        raise ValueError("A nested workspace cannot expand the active isolation boundary")
    token = _root.set(resolved)
    try:
        yield
    finally:
        _root.reset(token)


def _resolve(path: str, root: str) -> str:
    # Expand ~ before resolving symlinks and .. so home paths cannot appear workspace-relative.
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
