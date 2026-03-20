"""Repo map: graph-ranked code context for agent prompts.

Inspired by Aider's repo map. Parses source files with tree-sitter,
builds a reference graph, ranks symbols by importance, and selects
the most relevant definitions within a token budget.

Usage:
    map_text = await build_repo_map(Path("."), max_tokens=2048)
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path

from rune.intelligence.ast_analyzer import ASTAnalyzer, FileAnalysis, Symbol
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Module-level cache: {(file_path, mtime) -> FileAnalysis}
_analysis_cache: dict[tuple[str, float], FileAnalysis] = {}
_last_result_cache: dict[tuple[str, int, int], str] = {}  # (root, max_tokens, max_files) -> rendered

# Files/directories to skip
_SKIP_DIRS = {
    ".git", ".svn", ".hg", "node_modules", "__pycache__", ".venv", "venv",
    ".env", "dist", "build", ".rune", ".next", ".nuxt", "target",
    "vendor", ".tox", ".mypy_cache", ".pytest_cache", "coverage",
    ".eggs", "egg-info",
}

_SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".o", ".a",
    ".jpg", ".jpeg", ".png", ".gif", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".woff", ".woff2", ".ttf", ".eot",
    ".lock", ".min.js", ".min.css",
}

# Rough token estimate: ~4 chars per token for code
_CHARS_PER_TOKEN = 4

# Max files to scan (prevent slowdown on huge repos)
_MAX_FILES = 500


def _collect_source_files(root: Path, max_files: int = _MAX_FILES) -> list[Path]:
    """Walk the directory tree and collect parseable source files."""
    files: list[Path] = []
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune skip dirs in-place
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]

            for fname in filenames:
                if len(files) >= max_files:
                    return files
                fp = Path(dirpath) / fname
                if fp.suffix.lower() in _SKIP_EXTENSIONS:
                    continue
                if fp.stat().st_size > 500_000:  # skip files > 500KB
                    continue
                files.append(fp)
    except OSError:
        pass
    return files


def _build_reference_graph(
    analyses: list[FileAnalysis],
) -> tuple[dict[str, int], dict[str, list[Symbol]]]:
    """Build a reference frequency graph from file analyses.

    Returns:
        ref_counts: {symbol_name: total_reference_count}
        definitions: {symbol_name: [Symbol definitions]}
    """
    # Collect all defined symbol names and their locations
    definitions: dict[str, list[Symbol]] = defaultdict(list)
    defined_in_file: dict[str, set[str]] = defaultdict(set)  # file -> set of symbol names

    for analysis in analyses:
        for sym in analysis.symbols:
            definitions[sym.name].append(sym)
            defined_in_file[analysis.path].add(sym.name)

    # Count cross-file references: if file A imports/uses a name defined in file B
    ref_counts: Counter[str] = Counter()

    for analysis in analyses:
        file_defs = defined_in_file.get(analysis.path, set())
        # Imports reference symbols from other files
        for imp in analysis.imports:
            # Extract the imported name (last part of dotted path)
            name = imp.rsplit(".", 1)[-1] if "." in imp else imp
            if name in definitions and name not in file_defs:
                ref_counts[name] += 1

        # Symbols that match names defined elsewhere
        for sym in analysis.symbols:
            if sym.kind in ("variable", "import") and sym.name in definitions:
                # If this file doesn't define this symbol, it's a reference
                if sym.name not in file_defs:
                    ref_counts[sym.name] += 1

    return dict(ref_counts), dict(definitions)


def _rank_symbols(
    ref_counts: dict[str, int],
    definitions: dict[str, list[Symbol]],
) -> list[tuple[str, float, Symbol]]:
    """Rank symbols by importance score.

    Score = reference_count * kind_weight.
    Higher = more important to include in context.
    """
    kind_weights = {
        "class": 3.0,
        "function": 2.0,
        "method": 1.5,
        "variable": 0.5,
        "import": 0.1,
    }

    ranked: list[tuple[str, float, Symbol]] = []
    for name, syms in definitions.items():
        refs = ref_counts.get(name, 0)
        for sym in syms:
            weight = kind_weights.get(sym.kind, 1.0)
            # Boost: longer names are more specific/important
            name_bonus = 1.5 if len(name) >= 8 else 1.0
            # Penalize: underscore-prefixed (private)
            private_penalty = 0.3 if name.startswith("_") and not name.startswith("__") else 1.0
            score = (refs + 1) * weight * name_bonus * private_penalty
            ranked.append((name, score, sym))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _render_symbol(sym: Symbol, source_lines: list[str] | None = None) -> str:
    """Render a symbol as a compact definition line."""
    prefix = f"  {sym.parent}." if sym.parent else "  "
    kind_label = sym.kind[0].upper()  # F=function, C=class, M=method, V=variable
    line_info = f"L{sym.start_line}"
    return f"{prefix}{sym.name} ({kind_label}) {line_info}"


def _render_repo_map(
    ranked: list[tuple[str, float, Symbol]],
    max_tokens: int,
) -> str:
    """Render ranked symbols into a token-budgeted text block."""
    lines: list[str] = []
    current_file = ""
    estimated_tokens = 0

    for _name, _score, sym in ranked:
        file_path = sym.file_path
        # Group by file
        if file_path != current_file:
            header = f"\n{file_path}:"
            header_tokens = len(header) // _CHARS_PER_TOKEN
            if estimated_tokens + header_tokens > max_tokens:
                break
            lines.append(header)
            estimated_tokens += header_tokens
            current_file = file_path

        line = _render_symbol(sym)
        line_tokens = len(line) // _CHARS_PER_TOKEN
        if estimated_tokens + line_tokens > max_tokens:
            break
        lines.append(line)
        estimated_tokens += line_tokens

    return "\n".join(lines)


def build_repo_map_sync(
    root: Path | str,
    max_tokens: int = 2048,
    max_files: int = _MAX_FILES,
) -> str:
    """Build a ranked repo map for agent context injection.

    Scans source files, extracts symbols via tree-sitter, ranks by
    reference frequency, and renders within the token budget.

    Uses mtime-based caching: unchanged files reuse previous analysis.
    Full result is cached until any file changes.

    Returns empty string if no parseable files found.
    """
    root = Path(root)
    if not root.is_dir():
        return ""

    # 1. Collect source files
    files = _collect_source_files(root, max_files=max_files)
    if not files:
        return ""

    # 2. Check if any file changed (fast mtime scan)
    current_mtimes: dict[str, float] = {}
    changed = False
    for fp in files:
        try:
            mtime = fp.stat().st_mtime
            key = str(fp)
            current_mtimes[key] = mtime
            if (key, mtime) not in _analysis_cache:
                changed = True
        except OSError:
            changed = True

    # Return cached result if nothing changed
    cache_key = (str(root), max_tokens, max_files)
    if not changed and cache_key in _last_result_cache:
        return _last_result_cache[cache_key]

    # 3. Parse files (reuse cached analyses for unchanged files)
    analyzer = ASTAnalyzer()
    analyses: list[FileAnalysis] = []
    for fp in files:
        key = str(fp)
        mtime = current_mtimes.get(key, 0.0)

        # Use cache if file unchanged
        cached = _analysis_cache.get((key, mtime))
        if cached is not None:
            analyses.append(cached)
            continue

        # Parse and cache
        try:
            result = analyzer.analyze_file(fp)
            if result and result.symbols:
                result.path = str(fp.relative_to(root))
                for sym in result.symbols:
                    sym.file_path = result.path
                analyses.append(result)
                _analysis_cache[(key, mtime)] = result
                # Evict old mtime entries for this file
                for old_key in [k for k in _analysis_cache if k[0] == key and k[1] != mtime]:
                    del _analysis_cache[old_key]
        except Exception:
            continue

    if not analyses:
        return ""

    # 3. Build reference graph
    ref_counts, definitions = _build_reference_graph(analyses)

    # 4. Rank symbols
    ranked = _rank_symbols(ref_counts, definitions)

    if not ranked:
        return ""

    # 5. Render within token budget
    total_symbols = len(ranked)
    total_files = len(analyses)
    header = f"# Repo Map ({total_files} files, {total_symbols} symbols)\n"
    header_tokens = len(header) // _CHARS_PER_TOKEN

    body = _render_repo_map(ranked, max_tokens=max_tokens - header_tokens)
    if not body.strip():
        return ""

    result = header + body
    _last_result_cache[cache_key] = result
    return result
