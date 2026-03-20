"""Project map capability for RUNE.

Ported from src/capabilities/project.ts - generates Aider-style repository
maps with directory tree and symbol summaries.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Default ignore patterns (aligned with common .gitignore)

_DEFAULT_IGNORE = {
    ".git", ".svn", ".hg", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "node_modules", ".tox", ".venv", "venv", "env",
    ".env", "dist", "build", ".next", ".nuxt", "coverage", ".coverage",
    ".idea", ".vscode", ".DS_Store", "*.pyc", "*.pyo", "*.egg-info",
}


# Parameter schema

class ProjectMapParams(BaseModel):
    path: str = Field(default=".", description="Root directory to map")
    max_depth: int = Field(default=5, alias="maxDepth")
    compact: bool = Field(default=True, description="Compact mode (symbols only)")


# Gitignore loading

def _load_gitignore_patterns(root: Path) -> set[str]:
    """Load patterns from .gitignore if it exists, merged with defaults."""
    patterns = set(_DEFAULT_IGNORE)
    gitignore = root / ".gitignore"
    if gitignore.is_file():
        try:
            for line in gitignore.read_text(errors="ignore").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    # Normalise trailing slashes
                    patterns.add(line.rstrip("/"))
        except OSError:
            pass
    return patterns


def _is_ignored(path: Path, root: Path, patterns: set[str]) -> bool:
    """Check if a path should be ignored based on gitignore-style patterns."""
    name = path.name
    if name in patterns:
        return True
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = str(path)
    for pat in patterns:
        if pat.startswith("*"):
            if name.endswith(pat[1:]):
                return True
        elif "/" in pat and (rel.startswith(pat) or rel == pat):
            return True
    return False


# File collection

def _collect_files(
    root: Path,
    max_depth: int,
    ignore_patterns: set[str],
) -> list[dict]:
    """Recursively collect file info dicts up to max_depth."""
    files: list[dict] = []

    def _walk(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except PermissionError:
            return

        for entry in entries:
            if _is_ignored(entry, root, ignore_patterns):
                continue

            rel = str(entry.relative_to(root))

            if entry.is_dir():
                files.append({
                    "path": rel,
                    "type": "directory",
                    "depth": depth,
                })
                _walk(entry, depth + 1)
            elif entry.is_file():
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                files.append({
                    "path": rel,
                    "type": "file",
                    "depth": depth,
                    "size": size,
                    "ext": entry.suffix.lower(),
                })

    _walk(root, 0)
    return files


# File analysis

_SOURCE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
    ".rb", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt",
    ".lua", ".sh", ".bash",
}


def _analyze_file(path: Path) -> dict:
    """Analyse a source file for symbols using tree-sitter AST or line count fallback."""
    result: dict = {"path": str(path), "symbols": [], "lines": 0}

    try:
        text = path.read_text(errors="ignore")
        result["lines"] = len(text.splitlines())
    except OSError:
        return result

    if path.suffix.lower() not in _SOURCE_EXTENSIONS:
        return result

    # Try tree-sitter analysis
    try:
        from rune.intelligence.ast_analyzer import get_ast_analyzer

        analyzer = get_ast_analyzer()
        analysis = analyzer.analyze_file(path)
        if analysis is not None:
            for sym in analysis.symbols:
                result["symbols"].append({
                    "name": sym.name,
                    "kind": sym.kind,
                    "line": sym.start_line,
                })
            return result
    except Exception as exc:
        log.debug("ast_fallback", path=str(path), error=str(exc))

    # Fallback: basic regex extraction for common patterns
    import re

    for line_num, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        # Python: def / class
        m = re.match(r"^(def|class)\s+(\w+)", stripped)
        if m:
            result["symbols"].append({
                "name": m.group(2),
                "kind": "function" if m.group(1) == "def" else "class",
                "line": line_num,
            })
            continue
        # JS/TS: function / export function / const X =
        m = re.match(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)", stripped)
        if m:
            result["symbols"].append({
                "name": m.group(1),
                "kind": "function",
                "line": line_num,
            })

    return result


# Formatting

def _format_map(files: list[dict], analyses: dict[str, dict], compact: bool) -> str:
    """Format file list and symbol summaries into an Aider-style map."""
    lines: list[str] = []

    for f in files:
        depth = f.get("depth", 0)
        indent = "  " * depth

        if f["type"] == "directory":
            lines.append(f"{indent}{f['path']}/")
        else:
            path = f["path"]
            analysis = analyses.get(path, {})
            symbols = analysis.get("symbols", [])
            file_lines = analysis.get("lines", 0)

            if compact:
                if symbols:
                    sym_names = [s["name"] for s in symbols[:10]]
                    sym_str = ", ".join(sym_names)
                    if len(symbols) > 10:
                        sym_str += f" (+{len(symbols) - 10} more)"
                    lines.append(f"{indent}{path}  [{sym_str}]")
                else:
                    lines.append(f"{indent}{path}  ({file_lines} lines)")
            else:
                lines.append(f"{indent}{path}  ({file_lines} lines)")
                for sym in symbols:
                    lines.append(
                        f"{indent}  {sym['kind']} {sym['name']}  L{sym['line']}"
                    )

    return "\n".join(lines)


# Capability implementation

async def project_map(params: ProjectMapParams) -> CapabilityResult:
    """Generate an Aider-style repository map."""
    root = Path(params.path).resolve()
    if not root.is_dir():
        return CapabilityResult(
            success=False,
            error=f"Not a directory: {params.path}",
        )

    log.debug("project_map", root=str(root), max_depth=params.max_depth)

    ignore_patterns = _load_gitignore_patterns(root)
    files = _collect_files(root, params.max_depth, ignore_patterns)

    # Analyse source files
    analyses: dict[str, dict] = {}
    for f in files:
        if f["type"] == "file" and f.get("ext") in _SOURCE_EXTENSIONS:
            file_path = root / f["path"]
            analyses[f["path"]] = _analyze_file(file_path)

    output = _format_map(files, analyses, params.compact)

    total_files = sum(1 for f in files if f["type"] == "file")
    total_dirs = sum(1 for f in files if f["type"] == "directory")
    total_symbols = sum(
        len(a.get("symbols", [])) for a in analyses.values()
    )

    return CapabilityResult(
        success=True,
        output=output,
        metadata={
            "root": str(root),
            "files": total_files,
            "directories": total_dirs,
            "symbols": total_symbols,
        },
    )


# Registration

def register_project_capabilities(registry: CapabilityRegistry) -> None:
    """Register project map capability."""
    registry.register(CapabilityDefinition(
        name="project_map",
        description="Generate repository map with directory tree and symbol summaries",
        domain=Domain.FILE,
        risk_level=RiskLevel.LOW,
        group="read",
        parameters_model=ProjectMapParams,
        execute=project_map,
    ))
