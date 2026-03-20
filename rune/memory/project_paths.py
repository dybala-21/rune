"""Project path detection and workspace root resolution.

Ported from src/memory/project-paths.ts - encodes workspace paths
into project keys, resolves per-project memory/session directories.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from rune.utils.paths import expand_path, normalize_path, rune_home

# Types


@dataclass(slots=True)
class ProjectMemoryPaths:
    """All paths associated with a project's memory storage."""

    workspace_path: Path
    project_key: str
    base_dir: Path
    sessions_dir: Path
    sessions_index_file: Path
    memory_dir: Path
    memory_file: Path
    daily_dir: Path


# Key encoding

_CLEAN_RE = re.compile(r"[^A-Za-z0-9._-]")
_MULTI_DASH_RE = re.compile(r"-+")


def to_project_memory_key(workspace_path: str | Path) -> str:
    """Encode an absolute path into a filesystem-safe project key.

    Example::

        /Users/alice/workspace/rune -> -Users-alice-workspace-rune
    """
    normalized = str(normalize_path(workspace_path)).replace("\\", "/")
    key = normalized.replace(":", "").replace("/", "-")
    key = _CLEAN_RE.sub("-", key)
    key = _MULTI_DASH_RE.sub("-", key)

    if normalized.startswith("/") and not key.startswith("-"):
        key = f"-{key}"

    return key or "workspace"


# Path resolution

def get_project_memory_paths(
    workspace_path: str | Path | None = None,
    rune_config_dir: str | Path | None = None,
) -> ProjectMemoryPaths:
    """Resolve all per-project memory paths.

    Parameters
    ----------
    workspace_path:
        Root of the project workspace. Defaults to ``os.getcwd()``.
    rune_config_dir:
        Override for ``~/.rune`` config root.
    """
    resolved_ws = normalize_path(workspace_path or os.getcwd())
    project_key = to_project_memory_key(resolved_ws)

    if rune_config_dir is not None:
        config_root = normalize_path(expand_path(str(rune_config_dir)))
    else:
        config_root = rune_home()

    base_dir = config_root / "projects" / project_key
    memory_dir = base_dir / "memory"

    return ProjectMemoryPaths(
        workspace_path=resolved_ws,
        project_key=project_key,
        base_dir=base_dir,
        sessions_dir=base_dir / "sessions",
        sessions_index_file=base_dir / "sessions-index.json",
        memory_dir=memory_dir,
        memory_file=memory_dir / "MEMORY.md",
        daily_dir=memory_dir / "daily",
    )


# Workspace detection

_WORKSPACE_MARKERS = (
    ".git",
    "package.json",
    "pyproject.toml",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    ".rune",
)


def detect_workspace_root(start: str | Path | None = None) -> Path | None:
    """Walk up from *start* to find the workspace root.

    Returns the first ancestor that contains a known project marker
    (e.g. ``.git``, ``pyproject.toml``), or ``None`` if none is found.
    """
    current = Path(start or os.getcwd()).resolve()

    for _ in range(50):  # safety limit
        for marker in _WORKSPACE_MARKERS:
            if (current / marker).exists():
                return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    return None
