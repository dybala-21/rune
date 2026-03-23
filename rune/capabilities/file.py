"""File capabilities for RUNE.

Ported from src/capabilities/file.ts - read, write, edit, delete, list, search.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.config.defaults import DEFAULT_MAX_FILE_SIZE, DEFAULT_MAX_LINE_COUNT
from rune.safety.guardian import get_guardian
from rune.types import CapabilityResult, Domain, RiskLevel

# Parameter schemas (Zod to Pydantic)

class FileReadParams(BaseModel):
    path: str = Field(description="Absolute or relative file path")
    encoding: str = Field(default="utf-8")
    offset: int | None = Field(default=None, description="1-based line number to start from")
    limit: int | None = Field(default=None, description="Number of lines to read")
    max_size: int = Field(default=DEFAULT_MAX_FILE_SIZE, alias="maxSize")


class FileWriteParams(BaseModel):
    path: str = Field(description="Absolute or relative file path")
    content: str = Field(description="Content to write")
    encoding: str = Field(default="utf-8")
    create_dirs: bool = Field(default=True, alias="createDirs")


class FileEditParams(BaseModel):
    path: str = Field(description="File path to edit")
    search: str = Field(description="Text to search for")
    replace: str = Field(description="Replacement text")
    all: bool = Field(default=False, description="Replace all occurrences")


class FileDeleteParams(BaseModel):
    path: str = Field(description="File path to delete")
    recursive: bool = Field(default=False)


class FileListParams(BaseModel):
    path: str = Field(description="Directory path")
    pattern: str | None = Field(default=None, description="Glob pattern")
    recursive: bool = Field(default=False)
    max_depth: int = Field(default=10, alias="maxDepth")
    max_files: int = Field(default=1000, alias="maxFiles")
    include_directories: bool = Field(default=True, alias="includeDirectories")


class FileSearchParams(BaseModel):
    path: str = Field(description="Directory to search in")
    pattern: str = Field(description="Regex or text pattern")
    file_pattern: str | None = Field(default=None, alias="filePattern",
                                      description="Glob to filter files")
    max_results: int = Field(default=50, alias="maxResults")
    max_depth: int = Field(default=10, description="Maximum directory depth to search")
    regex: bool = Field(default=False, description="Treat pattern as regex")
    ignore_case: bool = Field(default=False, description="Case-insensitive matching")
    context: int = Field(default=0, description="Number of context lines around matches")


# Implementations

async def file_read(params: FileReadParams) -> CapabilityResult:
    """Read a file with optional line offset/limit."""
    guardian = get_guardian()
    validation = guardian.validate_file_read_path(params.path)
    if not validation.allowed:
        return CapabilityResult(success=False, error=validation.reason)

    file_path = Path(params.path).resolve()
    if not file_path.is_file():
        return CapabilityResult(
            success=False,
            error=(
                f"File not found: {params.path}. "
                f"Use the full path from project_map or file_list "
                f"(e.g., rune/agent/loop.py, not loop.py)."
            ),
        )

    size = file_path.stat().st_size
    if size > params.max_size:
        return CapabilityResult(
            success=False,
            error=f"File too large ({size} bytes, max {params.max_size})",
        )

    try:
        text = file_path.read_text(encoding=params.encoding)
    except UnicodeDecodeError:
        return CapabilityResult(success=False, error=f"Binary or unreadable file: {params.path}")

    lines = text.splitlines(keepends=True)

    # Apply offset and limit
    offset = (params.offset or 1) - 1  # 1-based to 0-based
    limit = params.limit or DEFAULT_MAX_LINE_COUNT

    if offset > 0 or limit < len(lines):
        lines = lines[offset:offset + limit]

    # Number lines (cat -n style)
    start_num = offset + 1
    numbered = ""
    for i, line in enumerate(lines):
        numbered += f"{start_num + i:6d}\t{line}"

    return CapabilityResult(
        success=True,
        output=numbered,
        metadata={"path": str(file_path), "lines": len(lines), "total_size": size},
    )


async def file_write(params: FileWriteParams) -> CapabilityResult:
    """Write content to a file."""
    if not params.path or not params.path.strip():
        return CapabilityResult(success=False, error="Empty file path")

    guardian = get_guardian()
    validation = guardian.validate_file_path(params.path)
    if not validation.allowed:
        return CapabilityResult(success=False, error=validation.reason)

    file_path = Path(params.path).resolve()

    # Defense-in-depth: block writes near filesystem root
    _home = os.environ.get("HOME", str(Path.home()))
    resolved_str = str(file_path)
    if resolved_str == "/" or resolved_str == _home or len(file_path.parts) < 3:
        return CapabilityResult(
            success=False,
            error=f"BLOCKED: refusing to write to critical path: {resolved_str}",
        )

    # Check for idempotent write
    if file_path.is_file():
        existing = file_path.read_text(encoding=params.encoding)
        if existing == params.content:
            return CapabilityResult(
                success=True,
                output=f"No changes needed: {params.path}",
                metadata={"changed": False},
            )

    if params.create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    # Syntax guard: validate before writing to disk
    from rune.agent.syntax_guard import validate as _syntax_validate
    _syn_err = _syntax_validate(str(file_path), params.content)
    if _syn_err:
        return CapabilityResult(
            success=False,
            output=f"Syntax error in {file_path.name}: {_syn_err}. Fix the content and retry.",
        )

    file_path.write_text(params.content, encoding=params.encoding)

    return CapabilityResult(
        success=True,
        output=f"Written {len(params.content)} bytes to {params.path}",
        metadata={"path": str(file_path), "size": len(params.content), "changed": True},
    )


async def file_edit(params: FileEditParams) -> CapabilityResult:
    """Edit a file by search-and-replace."""
    if not params.path or not params.path.strip():
        return CapabilityResult(success=False, error="Empty file path")

    guardian = get_guardian()
    validation = guardian.validate_file_path(params.path)
    if not validation.allowed:
        return CapabilityResult(success=False, error=f"Guardian blocked: {validation.reason}")

    file_path = Path(params.path).resolve()

    # Defense-in-depth: block edits near filesystem root
    _home = os.environ.get("HOME", str(Path.home()))
    resolved_str = str(file_path)
    if resolved_str == "/" or resolved_str == _home or len(file_path.parts) < 3:
        return CapabilityResult(
            success=False,
            error=f"BLOCKED: refusing to edit critical path: {resolved_str}",
        )
    if not file_path.is_file():
        return CapabilityResult(
            success=False,
            error=(
                f"File not found: {params.path}. "
                f"Use the full path from project_map or file_list "
                f"(e.g., rune/agent/loop.py, not loop.py)."
            ),
        )

    content = file_path.read_text()

    if params.search not in content:
        return CapabilityResult(
            success=False,
            error=f"Search string not found in {params.path}",
        )

    if params.all:
        new_content = content.replace(params.search, params.replace)
        count = content.count(params.search)
    else:
        new_content = content.replace(params.search, params.replace, 1)
        count = 1

    if new_content == content:
        return CapabilityResult(success=True, output="No changes made")

    # Syntax guard: validate replacement before writing to disk
    from rune.agent.syntax_guard import validate as _syntax_validate
    _syn_err = _syntax_validate(str(file_path), new_content)
    if _syn_err:
        return CapabilityResult(
            success=False,
            output=f"Edit would create syntax error in {file_path.name}: {_syn_err}. Fix and retry.",
        )

    file_path.write_text(new_content)

    return CapabilityResult(
        success=True,
        output=f"Replaced {count} occurrence(s) in {params.path}",
        metadata={"path": str(file_path), "replacements": count},
    )


async def file_delete(params: FileDeleteParams) -> CapabilityResult:
    """Delete a file or directory."""
    # -- empty path guard (defense-in-depth) -----------------------------------
    if not params.path or not params.path.strip():
        return CapabilityResult(success=False, error="Empty file path")

    guardian = get_guardian()
    validation = guardian.validate_file_path(params.path)
    if not validation.allowed:
        return CapabilityResult(success=False, error=validation.reason)

    file_path = Path(params.path).resolve()

    # -- hard safety net (defense-in-depth, independent of Guardian) -----------
    _home = os.environ.get("HOME", str(Path.home()))
    resolved_str = str(file_path)
    if resolved_str == "/" or resolved_str == _home:
        return CapabilityResult(
            success=False,
            error=f"BLOCKED: refusing to delete critical path: {resolved_str}",
        )
    if len(file_path.parts) < 3:
        return CapabilityResult(
            success=False,
            error=f"BLOCKED: path too close to filesystem root: {resolved_str}",
        )

    if not file_path.exists():
        return CapabilityResult(success=False, error=f"Path not found: {params.path}")

    if file_path.is_dir():
        if not params.recursive:
            return CapabilityResult(
                success=False,
                error=f"'{params.path}' is a directory. Use recursive=true to delete.",
            )
        import shutil
        shutil.rmtree(file_path)
    else:
        file_path.unlink()

    return CapabilityResult(
        success=True,
        output=f"Deleted: {params.path}",
    )


async def file_list(params: FileListParams) -> CapabilityResult:
    """List files in a directory with optional glob pattern."""
    dir_path = Path(params.path).resolve()
    if not dir_path.is_dir():
        return CapabilityResult(success=False, error=f"Not a directory: {params.path}")

    entries: list[str] = []
    count = 0

    if params.recursive and params.pattern:
        for p in dir_path.rglob(params.pattern):
            if count >= params.max_files:
                break
            if p.is_dir() and not params.include_directories:
                continue
            entries.append(str(p.relative_to(dir_path)))
            count += 1
    elif params.recursive:
        for p in dir_path.rglob("*"):
            if count >= params.max_files:
                break
            if p.is_dir() and not params.include_directories:
                continue
            entries.append(str(p.relative_to(dir_path)))
            count += 1
    else:
        for p in sorted(dir_path.iterdir()):
            if count >= params.max_files:
                break
            if p.is_dir() and not params.include_directories:
                continue
            if params.pattern and not p.match(params.pattern):
                continue
            entries.append(p.name)
            count += 1

    return CapabilityResult(
        success=True,
        output="\n".join(entries),
        metadata={"count": len(entries), "path": str(dir_path)},
    )


async def file_search(params: FileSearchParams) -> CapabilityResult:
    """Search for a pattern across files in a directory."""
    import re

    dir_path = Path(params.path).resolve()
    if not dir_path.is_dir():
        return CapabilityResult(success=False, error=f"Not a directory: {params.path}")

    # Build the pattern matcher
    flags = re.IGNORECASE if params.ignore_case else 0
    if params.regex:
        try:
            regex = re.compile(params.pattern, flags)
        except re.error as e:
            return CapabilityResult(success=False, error=f"Invalid regex: {e}")
    else:
        try:
            regex = re.compile(re.escape(params.pattern), flags)
        except re.error as e:
            return CapabilityResult(success=False, error=f"Invalid pattern: {e}")

    results: list[str] = []
    glob_pattern = params.file_pattern or "*"

    for file_path in dir_path.rglob(glob_pattern):
        if len(results) >= params.max_results:
            break
        if not file_path.is_file():
            continue
        # Check max_depth
        try:
            rel = file_path.relative_to(dir_path)
            if len(rel.parts) - 1 > params.max_depth:
                continue
        except ValueError:
            continue
        try:
            text = file_path.read_text(errors="ignore")
            all_lines = text.splitlines()
            for i, line in enumerate(all_lines):
                if regex.search(line):
                    line_num = i + 1
                    match_lines: list[str] = []
                    # Add context lines before
                    for ctx in range(max(0, i - params.context), i):
                        match_lines.append(f"{rel}:{ctx + 1}: {all_lines[ctx].strip()}")
                    # Add the matching line
                    match_lines.append(f"{rel}:{line_num}: {line.strip()}")
                    # Add context lines after
                    for ctx in range(i + 1, min(len(all_lines), i + 1 + params.context)):
                        match_lines.append(f"{rel}:{ctx + 1}: {all_lines[ctx].strip()}")
                    results.extend(match_lines)
                    if len(results) >= params.max_results:
                        break
        except (OSError, UnicodeDecodeError):
            continue

    return CapabilityResult(
        success=True,
        output="\n".join(results),
        metadata={"matches": len(results)},
    )


# Registration

def register_file_capabilities(registry: CapabilityRegistry) -> None:
    """Register all file capabilities."""
    registry.register(CapabilityDefinition(
        name="file_read", description="Read a file",
        domain=Domain.FILE, risk_level=RiskLevel.LOW,
        group="read", parameters_model=FileReadParams, execute=file_read,
    ))
    registry.register(CapabilityDefinition(
        name="file_write", description="Write to a file",
        domain=Domain.FILE, risk_level=RiskLevel.MEDIUM,
        group="write", parameters_model=FileWriteParams, execute=file_write,
    ))
    registry.register(CapabilityDefinition(
        name="file_edit", description="Edit a file (search and replace)",
        domain=Domain.FILE, risk_level=RiskLevel.MEDIUM,
        group="write", parameters_model=FileEditParams, execute=file_edit,
    ))
    registry.register(CapabilityDefinition(
        name="file_delete", description="Delete a file or directory",
        domain=Domain.FILE, risk_level=RiskLevel.HIGH,
        group="write", parameters_model=FileDeleteParams, execute=file_delete,
    ))
    registry.register(CapabilityDefinition(
        name="file_list", description="List files in a directory",
        domain=Domain.FILE, risk_level=RiskLevel.LOW,
        group="read", parameters_model=FileListParams, execute=file_list,
    ))
    registry.register(CapabilityDefinition(
        name="file_search", description="Search for patterns in files",
        domain=Domain.FILE, risk_level=RiskLevel.LOW,
        group="read", parameters_model=FileSearchParams, execute=file_search,
    ))
