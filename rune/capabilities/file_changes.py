"""Bounded text diffs captured when a file operation completes."""

from __future__ import annotations

from difflib import unified_diff
from pathlib import Path
from typing import Any
from uuid import uuid4

from rune.utils.logger import get_logger

log = get_logger(__name__)
_MAX_TEXT = 256_000
_MAX_LINES = 8_000
_MAX_PATCH = 32_000


def read_before(path: Path, encoding: str = "utf-8") -> str | None:
    try:
        if not path.exists():
            return ""
        if not path.is_file() or path.stat().st_size > _MAX_TEXT:
            return None
        return path.read_text(encoding=encoding)
    except (OSError, UnicodeError) as exc:
        log.debug("file_diff_read_failed", path=str(path), error=str(exc))
        return None


def file_change(path: Path, before: str | None, *, existed: bool = True,
                encoding: str = "utf-8") -> dict[str, Any]:
    present = path.exists()
    after = read_before(path, encoding)
    change: dict[str, Any] = {
        "id": uuid4().hex, "path": str(path),
        "kind": "deleted" if not present else "modified" if existed else "created",
        "patch": "",
    }
    if before is None or after is None or max(len(before), len(after)) > _MAX_TEXT:
        change["notice"] = "Text preview unavailable for this file's size or encoding."
        return change
    old, new = before.splitlines(), after.splitlines()
    if max(len(old), len(new)) > _MAX_LINES or "\0" in before + after:
        change["notice"] = "Text preview unavailable for this file's size or format."
        return change
    lines = unified_diff(old, new, fromfile=str(path) if existed else "/dev/null",
                         tofile=str(path) if present else "/dev/null", lineterm="")
    patch = ""
    for line in lines:
        if len(patch) + len(line) + 1 > _MAX_PATCH:
            change["notice"] = "Diff preview truncated. Open the file to inspect the rest."
            break
        patch += line + "\n"
    change["patch"] = patch
    if not patch and "notice" not in change:
        change["notice"] = "No line changes." if before == after else "Only line endings changed."
    return change
