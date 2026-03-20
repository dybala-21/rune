"""Path utilities for RUNE.

Ported from src/utils/paths.ts - path expansion, normalization,
glob matching, and system/hidden path detection.
"""

from __future__ import annotations

import os
import re
from fnmatch import fnmatch
from pathlib import Path

_HOME = Path.home()


def expand_path(p: str) -> Path:
    """Expand ~ and $HOME/$USER env vars in a path string."""
    p = p.replace("~", str(_HOME), 1) if p.startswith("~") else p
    p = os.path.expandvars(p)
    return Path(p)


def normalize_path(p: str | Path) -> Path:
    """Resolve a path to its canonical absolute form."""
    return Path(os.path.realpath(os.path.expanduser(str(p))))


def to_absolute(p: str | Path, base: str | Path | None = None) -> Path:
    """Convert a path to absolute, relative to *base* (default: cwd)."""
    path = Path(p)
    if path.is_absolute():
        return path
    base_dir = Path(base) if base else Path.cwd()
    return (base_dir / path).resolve()


def matches_pattern(path: str | Path, pattern: str) -> bool:
    """Check if *path* matches a glob *pattern*."""
    return fnmatch(str(path), pattern)


# System / Hidden detection

_SYSTEM_PREFIXES = (
    "/bin", "/sbin", "/usr/bin", "/usr/sbin",
    "/System", "/Library",
    "/etc", "/private/etc",
)

_HIDDEN_RE = re.compile(r"(^|/)\.(?!\.)[^/]*")


def is_system_path(p: str | Path) -> bool:
    """Return True if the path is inside a system directory."""
    s = str(normalize_path(p))
    return any(s.startswith(prefix) for prefix in _SYSTEM_PREFIXES)


def is_hidden_path(p: str | Path) -> bool:
    """Return True if any component of the path is a dotfile/dotdir."""
    return bool(_HIDDEN_RE.search(str(p)))


# RUNE-specific paths

def rune_home() -> Path:
    """Return ~/.rune, the user-level RUNE configuration directory."""
    d = _HOME / ".rune"
    d.mkdir(parents=True, exist_ok=True)
    return d


def rune_data() -> Path:
    """Return ~/.rune/data, persistent data (DB, embeddings, etc.)."""
    d = rune_home() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Alias used by UI modules
rune_data_dir = rune_data


def rune_logs() -> Path:
    """Return ~/.rune/logs, the log files directory."""
    d = rune_home() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def project_rune_dir(project_root: Path | None = None) -> Path:
    """Return <project>/.rune, the project-level RUNE directory."""
    root = project_root or Path.cwd()
    d = root / ".rune"
    d.mkdir(parents=True, exist_ok=True)
    return d
