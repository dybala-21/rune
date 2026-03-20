"""Project-specific persistent memory via MEMORY.md files.

Ported from src/memory/project-memory.ts - reads, writes, and appends
to a MEMORY.md file at the project root, enabling per-project context
that survives across sessions.
"""

from __future__ import annotations

import fcntl
import tempfile
from pathlib import Path
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

PROJECT_MEMORY_FILENAME = "MEMORY.md"


# Public API

def read_project_memory_head(
    workspace_path: str | Path,
    options: dict[str, Any] | None = None,
) -> str:
    """Read the first *max_lines* lines of the project MEMORY.md.

    ``options`` may contain:
    - ``max_lines`` (int): maximum lines to return (default 200).
    - ``max_chars`` (int): maximum characters to return (default 8000).
    """
    opts = options or {}
    max_lines: int = opts.get("max_lines", 200)
    max_chars: int = opts.get("max_chars", 8000)

    mem_file = find_project_memory_file(workspace_path)
    if mem_file is None:
        return ""

    try:
        text = mem_file.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("project_memory_read_failed", path=str(mem_file), error=str(exc))
        return ""

    lines = text.splitlines(keepends=True)
    result_lines = lines[:max_lines]
    result = "".join(result_lines)

    if len(result) > max_chars:
        result = result[:max_chars]

    return result


def write_project_memory(workspace_path: str | Path, content: str) -> None:
    """Atomically overwrite the project MEMORY.md.

    Uses a temporary file + rename for crash safety, and an advisory
    file lock to prevent concurrent writes.
    """
    ws = Path(workspace_path)
    ws.mkdir(parents=True, exist_ok=True)
    target = ws / PROJECT_MEMORY_FILENAME
    lock_path = ws / f".{PROJECT_MEMORY_FILENAME}.lock"

    try:
        with open(lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                # Write to temp file in the same directory (same filesystem for rename)
                fd, tmp_path_str = tempfile.mkstemp(
                    dir=str(ws), prefix=".memory_", suffix=".md.tmp",
                )
                tmp_path = Path(tmp_path_str)
                try:
                    tmp_path.write_text(content, encoding="utf-8")
                    tmp_path.replace(target)
                    log.debug("project_memory_written", path=str(target), chars=len(content))
                except Exception:
                    tmp_path.unlink(missing_ok=True)
                    raise
                finally:
                    import os
                    os.close(fd)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
    except OSError as exc:
        log.error("project_memory_write_failed", path=str(target), error=str(exc))
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def append_project_memory(
    workspace_path: str | Path,
    section: str,
    content: str,
) -> None:
    """Append *content* under *section* in the project MEMORY.md.

    If the section heading already exists, content is appended after the
    last line of that section. Otherwise a new section is added at the
    end of the file.
    """
    ws = Path(workspace_path)
    target = ws / PROJECT_MEMORY_FILENAME

    existing = ""
    if target.exists():
        try:
            existing = target.read_text(encoding="utf-8")
        except OSError:
            existing = ""

    section_heading = f"## {section}"
    new_block = f"\n{content.rstrip()}\n"

    if section_heading in existing:
        # Find the end of the section (next heading or EOF)
        lines = existing.split("\n")
        insert_idx: int | None = None
        in_section = False

        for i, line in enumerate(lines):
            if line.strip() == section_heading:
                in_section = True
                continue
            if in_section and line.startswith("## "):
                insert_idx = i
                break

        if insert_idx is not None:
            lines.insert(insert_idx, content.rstrip() + "\n")
        else:
            # Section runs to end of file
            lines.append(content.rstrip())

        updated = "\n".join(lines)
    else:
        # Append new section
        if existing and not existing.endswith("\n"):
            existing += "\n"
        updated = f"{existing}\n{section_heading}{new_block}"

    write_project_memory(workspace_path, updated)
    log.debug("project_memory_appended", section=section, path=str(target))


def find_project_memory_file(workspace_path: str | Path) -> Path | None:
    """Search up the directory tree from *workspace_path* for MEMORY.md.

    Returns the Path if found, None otherwise.
    """
    current = Path(workspace_path).resolve()

    # Walk up to filesystem root
    while True:
        candidate = current / PROJECT_MEMORY_FILENAME
        if candidate.is_file():
            return candidate

        parent = current.parent
        if parent == current:
            # Reached filesystem root
            break
        current = parent

    return None
