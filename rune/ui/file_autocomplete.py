"""File path autocomplete provider for RUNE TUI.

Provides completions for ``@file`` references and slash-command
arguments that expect file paths. Respects ``.gitignore``-style
patterns and common ignore lists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Directories to always skip when scanning
_IGNORED_DIRS: frozenset[str] = frozenset({
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".next",
    ".rune",
})

# Maximum number of results to return per query
MAX_COMPLETIONS = 50

# Maximum directory depth for scanning
MAX_DEPTH = 5


@dataclass(slots=True, frozen=True)
class CompletionItem:
    """A single file-path completion suggestion."""

    text: str
    display: str
    is_directory: bool


def complete_path(
    partial: str,
    cwd: str | Path | None = None,
) -> list[CompletionItem]:
    """Return completion items for *partial* relative to *cwd*.

    *partial* may be empty (list cwd), a directory prefix, or a
    partial file name. Tilde (``~``) is expanded to the home directory.
    """
    working = Path(cwd) if cwd else Path.cwd()

    # Expand tilde
    if partial.startswith("~"):
        expanded = Path(partial).expanduser()
    elif os.path.isabs(partial):
        expanded = Path(partial)
    else:
        expanded = working / partial

    # If the partial ends with a separator, list the directory contents
    if partial.endswith(os.sep) or partial.endswith("/"):
        parent = expanded
        prefix = ""
    else:
        parent = expanded.parent
        prefix = expanded.name

    if not parent.is_dir():
        return []

    items: list[CompletionItem] = []
    try:
        entries = sorted(parent.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return []

    for entry in entries:
        if entry.name.startswith(".") and not prefix.startswith("."):
            continue
        if entry.is_dir() and entry.name in _IGNORED_DIRS:
            continue
        if prefix and not entry.name.lower().startswith(prefix.lower()):
            continue

        # Build the completion text relative to cwd when possible
        try:
            rel = entry.relative_to(working)
            text = str(rel)
        except ValueError:
            text = str(entry)

        if entry.is_dir():
            text += "/"

        items.append(
            CompletionItem(
                text=text,
                display=entry.name + ("/" if entry.is_dir() else ""),
                is_directory=entry.is_dir(),
            )
        )

        if len(items) >= MAX_COMPLETIONS:
            break

    return items


def complete_at_reference(
    partial: str,
    cwd: str | Path | None = None,
) -> list[CompletionItem]:
    """Complete an ``@``-prefixed file reference.

    Strips the leading ``@`` before delegating to :func:`complete_path`.
    """
    cleaned = partial.lstrip("@")
    return complete_path(cleaned, cwd=cwd)
