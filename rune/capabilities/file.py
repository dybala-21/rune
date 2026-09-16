"""File capabilities for RUNE.

Ported from src/capabilities/file.ts - read, write, edit, delete, list, search.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from rune.agent.isolation import enforce as _enforce_isolation
from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.config.defaults import DEFAULT_MAX_FILE_SIZE, DEFAULT_MAX_LINE_COUNT
from rune.safety.approval_context import was_approved
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

def _authorize_mutation(path: str) -> CapabilityResult | None:
    check = get_guardian().validate_file_path(path)
    if not check.allowed or (check.requires_approval and not was_approved()):
        return CapabilityResult(
            success=False, error=check.reason,
            metadata={"action_status": "not_executed",
                      "requires_approval": check.allowed and check.requires_approval},
        )
    if error := _enforce_isolation(path):
        return CapabilityResult(success=False, error=error,
                                metadata={"action_status": "not_executed"})
    return None

async def file_read(params: FileReadParams) -> CapabilityResult:
    """Read a file with optional line offset/limit."""
    guardian = get_guardian()
    validation = guardian.validate_file_read_path(params.path)
    if not validation.allowed:
        return CapabilityResult(success=False, error=validation.reason)

    file_path = Path(params.path).expanduser().resolve()
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

    # Tabular files carry a whole-file profile regardless of the window
    # read: duplicate rows the model would silently double-count live
    # outside whatever offset/limit it asked for.
    from rune.capabilities.table_profile import profile_table
    profile = profile_table(text, file_path.name)
    if profile:
        if not numbered.endswith("\n"):
            numbered += "\n"
        numbered += profile

    return CapabilityResult(
        success=True,
        output=numbered,
        metadata={"path": str(file_path), "lines": len(lines), "total_size": size},
    )


def _reject_test_overwrite(file_path: Path) -> str | None:
    """Protect existing tests from being rewritten to accept a broken change.

    Tests created in this run can be corrected while their contents still
    match the recorded revision. Set RUNE_PROTECT_TESTS=0 for tasks that
    intentionally change existing tests.
    """
    from rune.agent.validation_guard import (
        _is_test_file,
        authored_test_unchanged,
        protect_tests_enabled,
    )

    if not protect_tests_enabled() or not file_path.is_file():
        return None
    under_test_dir = any(
        part in {"tests", "test", "__tests__", "spec"} for part in file_path.parts
    )
    if not _is_test_file(file_path, under_test_dir):
        return None
    if authored_test_unchanged(file_path):
        return None
    return (
        f"BLOCKED: {file_path.name} is an existing test. Editing a test so it "
        f"agrees with the code under test removes the check the user relies "
        f"on. If the code is wrong, fix the code. If the test and the "
        f"project's own spec genuinely contradict each other, that is not "
        f"something to work around — call task_blocked with both sides of "
        f"the conflict and stop. Set RUNE_PROTECT_TESTS=0 only when editing "
        f"tests IS the task."
    )


async def file_write(params: FileWriteParams) -> CapabilityResult:
    """Write content to a file."""
    if not params.path or not params.path.strip():
        return CapabilityResult(success=False, error="Empty file path")

    if blocked := _authorize_mutation(params.path):
        return blocked

    file_path = Path(params.path).expanduser().resolve()

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

    # Syntax guard: validate before writing to disk
    from rune.agent.syntax_guard import validate as _syntax_validate
    _syn_err = _syntax_validate(str(file_path), params.content)
    if _syn_err:
        return CapabilityResult(
            success=False,
            output=f"Syntax error in {file_path.name}: {_syn_err}. Fix the content and retry.",
            metadata={"action_status": "not_executed"},
        )

    _tamper = _reject_test_overwrite(file_path)
    if _tamper:
        return CapabilityResult(success=False, error=_tamper,
                                metadata={"action_status": "not_executed"})

    if params.create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    from rune.capabilities.file_changes import file_change, read_before
    existed = file_path.exists()
    before = read_before(file_path, params.encoding)
    file_path.write_text(params.content, encoding=params.encoding)

    from rune.safety.recoverable import verify_written
    check = verify_written(file_path, len(params.content))
    if not check.ok:
        return CapabilityResult(success=False, error=f"Write failed: {check.detail}")

    from rune.agent.validation_guard import record_test_write
    record_test_write(file_path, existed=existed)

    return CapabilityResult(
        success=True,
        output=f"Written {len(params.content)} bytes to {params.path}",
        metadata={"path": str(file_path), "size": len(params.content),
                  "changed": True, "verified": check.detail,
                  "fileChange": file_change(file_path, before, existed=existed, encoding=params.encoding)},
    )


async def file_edit(params: FileEditParams) -> CapabilityResult:
    """Edit a file by search-and-replace."""
    if not params.path or not params.path.strip():
        return CapabilityResult(success=False, error="Empty file path")

    if blocked := _authorize_mutation(params.path):
        return blocked

    file_path = Path(params.path).expanduser().resolve()

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
                f"File not found: {params.path}. To create a new file, call "
                f"file_write (not file_edit). To edit an existing file, check "
                f"the path: use the full path from project_map or file_list "
                f"(e.g., rune/agent/loop.py, not loop.py)."
            ),
            metadata={"action_status": "not_executed"},
        )

    content = file_path.read_text()

    from rune.capabilities.edit_matching import (
        apply_block,
        closest_section_hint,
        escalation_hint,
        find_block,
        record_edit_failure,
        record_edit_success,
    )

    matched_via = "exact"
    if params.search in content:
        if params.all:
            new_content = content.replace(params.search, params.replace)
            count = content.count(params.search)
        else:
            new_content = content.replace(params.search, params.replace, 1)
            count = 1
    else:
        # Fuzzy ladder: a near-miss search (whitespace/indent drift) is the
        # most common weak-model edit failure; recover the UNIQUE-match cases
        # instead of bouncing the model into a retry spiral. `all` implies
        # multiple occurrences — fuzzy handles single-block edits only.
        block = None if params.all else find_block(content, params.search)
        if block is None:
            failures = record_edit_failure(str(file_path))
            hint = closest_section_hint(content, params.search)
            return CapabilityResult(
                success=False,
                error=(
                    f"Search string not found in {params.path}."
                    + (f"\n{hint}" if hint else "")
                    + escalation_hint(params.path, failures)
                ),
                metadata={"action_status": "not_executed"},
            )
        new_content = apply_block(content, block, params.replace)
        matched_via = block.strategy
        count = 1

    if new_content == content:
        record_edit_success(str(file_path))
        return CapabilityResult(success=True, output="No changes made", metadata={"changed": False})

    # Syntax guard: validate replacement before writing to disk
    from rune.agent.syntax_guard import validate as _syntax_validate
    _syn_err = _syntax_validate(str(file_path), new_content)
    if _syn_err:
        failures = record_edit_failure(str(file_path))
        return CapabilityResult(
            success=False,
            output=(
                f"Edit would create syntax error in {file_path.name}: "
                f"{_syn_err}. Fix and retry."
                + escalation_hint(params.path, failures)
            ),
            metadata={"action_status": "not_executed"},
        )

    _tamper = _reject_test_overwrite(file_path)
    if _tamper:
        return CapabilityResult(success=False, error=_tamper,
                                metadata={"action_status": "not_executed"})

    file_path.write_text(new_content)
    record_edit_success(str(file_path))

    from rune.safety.recoverable import verify_written
    _check = verify_written(file_path, len(new_content))
    if not _check.ok:
        return CapabilityResult(success=False, error=f"Edit failed: {_check.detail}")

    from rune.agent.validation_guard import record_test_write
    record_test_write(file_path, existed=True)

    note = "" if matched_via == "exact" else (
        f" (matched via {matched_via} fuzzy match — verify the edit landed "
        f"where intended with file_read if unsure)"
    )
    from rune.capabilities.file_changes import file_change
    return CapabilityResult(
        success=True,
        output=f"Replaced {count} occurrence(s) in {params.path}{note}",
        metadata={"path": str(file_path), "replacements": count,
                  "matched_via": matched_via, "fileChange": file_change(file_path, content)},
    )


async def file_delete(params: FileDeleteParams) -> CapabilityResult:
    """Delete a file or directory."""
    # -- empty path guard (defense-in-depth) -----------------------------------
    if not params.path or not params.path.strip():
        return CapabilityResult(success=False, error="Empty file path")

    if blocked := _authorize_mutation(params.path):
        return blocked

    file_path = Path(params.path).expanduser().resolve()

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
        return CapabilityResult(success=False, error=f"Path not found: {params.path}",
                                metadata={"action_status": "not_executed"})

    if file_path.is_dir() and not params.recursive:
        return CapabilityResult(
            success=False,
            error=f"'{params.path}' is a directory. Use recursive=true to delete.",
            metadata={"action_status": "not_executed"},
        )

    from rune.agent.validation_guard import _walk_test_files, protect_tests_enabled
    if protect_tests_enabled():
        candidates = _walk_test_files(file_path, strict=True) if file_path.is_dir() else (file_path,)
        try:
            for candidate in candidates:
                if reason := _reject_test_overwrite(candidate):
                    return CapabilityResult(success=False, error=reason, metadata={"action_status": "not_executed"})
        except (OSError, ValueError) as exc:
            return CapabilityResult(success=False, error=f"Cannot check protected tests before deletion: {exc}",
                                    metadata={"action_status": "not_executed"})

    # Delete regenerable output directly; keep other files in the workspace trash.
    from rune.capabilities.file_changes import file_change, read_before
    from rune.safety.recoverable import (
        is_regenerable,
        move_to_trash,
        trash_enabled,
        verify_gone,
    )
    before = read_before(file_path)
    recycled = is_regenerable(file_path) or not trash_enabled()
    stored: str | None = None
    if recycled:
        if file_path.is_dir():
            import shutil
            shutil.rmtree(file_path)
        else:
            file_path.unlink()
    else:
        stored = move_to_trash(file_path).stored

    check = verify_gone(file_path)
    if not check.ok:
        return CapabilityResult(success=False, error=f"Delete failed: {check.detail}")

    if stored:
        return CapabilityResult(
            success=True,
            output=(f"Moved to trash (recoverable): {params.path}\n"
                    f"Restore from: {stored}"),
            metadata={"path": str(file_path), "trashed": stored, "fileChange": file_change(file_path, before)},
        )
    return CapabilityResult(
        success=True,
        output=f"Deleted: {params.path}",
        metadata={"path": str(file_path), "trashed": None, "fileChange": file_change(file_path, before)},
    )


async def file_list(params: FileListParams) -> CapabilityResult:
    """List files in a directory with optional glob pattern."""
    dir_path = Path(params.path).expanduser().resolve()
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

    dir_path = Path(params.path).expanduser().resolve()
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
