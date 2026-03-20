"""Parse @file, @url, @symbol references in user input.

Ported from src/utils/at-reference.ts - extracts ``@path/to/file``
references, resolves them to absolute paths, and reads their content
for agent context injection.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Types


@dataclass(slots=True)
class AtReference:
    """A parsed but not yet resolved ``@`` reference."""

    raw: str          # "@src/agent/loop.py"
    path: str         # "src/agent/loop.py"
    full_path: str    # absolute path
    type: Literal["file", "directory", "not_found"]


@dataclass(slots=True)
class ResolvedReference(AtReference):
    """An ``@`` reference whose content has been loaded."""

    content: str = ""
    truncated: bool = False
    error: str = ""
    attachment_type: Literal["image", "document"] | None = None


@dataclass(slots=True, frozen=True)
class ResolveOptions:
    max_file_size: int = 100 * 1024   # 100 KB
    max_dir_depth: int = 3
    max_files: int = 100


# Constants

_DEFAULT_OPTIONS = ResolveOptions()

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})
_DOCUMENT_EXTENSIONS = frozenset({".pdf"})
_ATTACHMENT_EXTENSIONS = _IMAGE_EXTENSIONS | _DOCUMENT_EXTENSIONS

_BINARY_EXTENSIONS = frozenset({
    ".bmp", ".ico", ".svg", ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".woff", ".woff2", ".ttf", ".eot",
    ".pyc", ".class", ".o", ".obj",
    ".sqlite", ".db",
})

_KNOWN_EXT = (
    r"pdf|txt|md|ts|tsx|js|jsx|json|yaml|yml|py|rs|go|java|sh|css|html|svg"
    r"|toml|sql|csv|log|xml|conf|cfg|ini|env|lock|mod"
)

_AT_REFERENCE_RE = re.compile(
    rf'(?:^|(?<=\s))@(?:"([^"]+)"'
    rf"|(.+?\.(?:{_KNOWN_EXT}))(?=\s|$)"
    rf"|(\.\.?/[^\s]*|~/[^\s]*|/[^\s]*/[^\s]*|[a-zA-Z][^\s]*/[^\s]*|[a-zA-Z][^\s]*\.[a-zA-Z][^\s]*))"
)


# Workspace directive

def extract_at_workspace_directive(text: str, cwd: str | None = None) -> str | None:
    """Extract a workspace directory directive from ``@path`` in *text*.

    Returns the resolved directory path, or *None* if no directive found.
    """
    m = re.search(r"@((?:/|\.\.?/|~/)\S*)", text)
    if not m:
        return None
    raw = m.group(1).rstrip("/")
    expanded = raw.replace("~", str(Path.home()), 1) if raw.startswith("~") else raw
    resolved = str(Path(cwd, expanded).resolve()) if cwd else str(Path(expanded).resolve())
    try:
        p = Path(resolved)
        if p.is_dir():
            return resolved
        if p.exists():
            return str(p.parent)
    except OSError:
        pass
    return None


# Parser

def parse_at_references(text: str, cwd: str) -> list[AtReference]:
    """Parse all ``@`` references from *text*."""
    refs: list[AtReference] = []
    seen: set[str] = set()

    for m in _AT_REFERENCE_RE.finditer(text):
        raw_path = m.group(1) or m.group(2) or m.group(3)
        if not raw_path or raw_path in seen:
            continue
        seen.add(raw_path)

        if raw_path.startswith("~/"):
            resolved = str(Path.home() / raw_path[2:])
        elif os.path.isabs(raw_path):
            resolved = raw_path
        else:
            resolved = str(Path(cwd, raw_path).resolve())

        clean = resolved.rstrip("/")
        p = Path(clean)
        try:
            if p.is_file():
                ref_type: Literal["file", "directory", "not_found"] = "file"
            elif p.is_dir():
                ref_type = "directory"
            else:
                ref_type = "not_found"
        except OSError:
            ref_type = "not_found"

        refs.append(AtReference(raw=f"@{raw_path}", path=raw_path, full_path=clean, type=ref_type))

    return refs


# Resolver

def _list_directory(dir_path: Path, max_depth: int, max_files: int) -> str:
    """Build a tree-like listing of a directory."""
    lines: list[str] = []

    def _walk(p: Path, depth: int) -> None:
        if depth > max_depth or len(lines) >= max_files:
            return
        try:
            entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return
        for entry in entries:
            if entry.name.startswith("."):
                continue
            indent = "  " * depth
            if entry.is_dir():
                lines.append(f"{indent}{entry.name}/")
                _walk(entry, depth + 1)
            else:
                lines.append(f"{indent}{entry.name}")
            if len(lines) >= max_files:
                return

    _walk(dir_path, 0)
    return "\n".join(lines)


def resolve_references(
    refs: list[AtReference],
    options: ResolveOptions | None = None,
) -> list[ResolvedReference]:
    """Resolve parsed references by reading their content from disk."""
    opts = options or _DEFAULT_OPTIONS
    results: list[ResolvedReference] = []

    for ref in refs:
        resolved = ResolvedReference(
            raw=ref.raw,
            path=ref.path,
            full_path=ref.full_path,
            type=ref.type,
        )

        if ref.type == "not_found":
            resolved.error = f"Path not found: {ref.full_path}"
            results.append(resolved)
            continue

        p = Path(ref.full_path)
        ext = p.suffix.lower()

        if ref.type == "directory":
            resolved.content = _list_directory(p, opts.max_dir_depth, opts.max_files)
            results.append(resolved)
            continue

        # File
        if ext in _ATTACHMENT_EXTENSIONS:
            resolved.attachment_type = "image" if ext in _IMAGE_EXTENSIONS else "document"
            resolved.content = f"[Attachment: {p.name}]"
            results.append(resolved)
            continue

        if ext in _BINARY_EXTENSIONS:
            resolved.error = f"Binary file skipped: {p.name}"
            results.append(resolved)
            continue

        try:
            size = p.stat().st_size
            if size > opts.max_file_size:
                content = p.read_text(encoding="utf-8", errors="replace")[: opts.max_file_size]
                resolved.truncated = True
            else:
                content = p.read_text(encoding="utf-8", errors="replace")
            resolved.content = content
        except OSError as exc:
            resolved.error = str(exc)

        results.append(resolved)

    return results


def parse_and_resolve(
    text: str,
    cwd: str,
    options: ResolveOptions | None = None,
) -> list[ResolvedReference]:
    """Convenience function: parse + resolve in one call."""
    refs = parse_at_references(text, cwd)
    return resolve_references(refs, options)
