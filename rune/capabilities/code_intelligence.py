"""Code intelligence capabilities for RUNE.

Ported from src/capabilities/code-intelligence.ts - file analysis,
symbol definition lookup, reference finding, and impact analysis
using the AST analyzer and code graph.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Parameter schemas

class CodeAnalyzeParams(BaseModel):
    file_path: str = Field(description="Path to the file to analyse")
    index_to_graph: bool = Field(default=False, description="Also index the file into the code graph for cross-reference analysis")


class CodeFindDefParams(BaseModel):
    name: str = Field(description="Symbol name to look up")
    kind: str = Field(default="", description="Filter by kind (function/class/method)")


class CodeFindRefsParams(BaseModel):
    name: str = Field(description="Symbol name to find references for")


class CodeImpactParams(BaseModel):
    file_path: str = Field(description="File to analyse impact for")
    symbol_name: str = Field(default="", description="Optional symbol to scope analysis")


# Implementations

async def code_analyze(params: CodeAnalyzeParams) -> CapabilityResult:
    """Analyse a source file and extract its structure (symbols, imports)."""
    file_path = Path(params.file_path).resolve()
    if not file_path.is_file():
        return CapabilityResult(
            success=False,
            error=f"File not found: {params.file_path}",
        )

    log.debug("code_analyze", file=str(file_path))

    try:
        from rune.intelligence.ast_analyzer import get_ast_analyzer

        analyzer = get_ast_analyzer()
        analysis = analyzer.analyze_file(file_path)

        if analysis is None:
            # Fallback: basic line count for unsupported languages
            try:
                text = file_path.read_text(errors="ignore")
                line_count = len(text.splitlines())
                return CapabilityResult(
                    success=True,
                    output=f"File: {params.file_path}\nLines: {line_count}\n"
                           f"Language: unsupported (no AST parser available)",
                    metadata={
                        "path": str(file_path),
                        "lines": line_count,
                        "language": "unknown",
                    },
                )
            except OSError as exc:
                return CapabilityResult(success=False, error=str(exc))

        # Format the analysis
        lines: list[str] = [
            f"File: {analysis.path}",
            f"Language: {analysis.language}",
            f"Symbols: {len(analysis.symbols)}",
            f"Imports: {len(analysis.imports)}",
            "",
        ]

        if analysis.symbols:
            lines.append("Symbols:")
            for sym in analysis.symbols:
                parent_info = f" (in {sym.parent})" if sym.parent else ""
                lines.append(
                    f"  {sym.kind} {sym.name}{parent_info}  "
                    f"L{sym.start_line}-{sym.end_line}"
                )

        if analysis.imports:
            lines.append("")
            lines.append("Imports:")
            for imp in analysis.imports:
                lines.append(f"  {imp}")

        # Index into code graph if requested
        indexed_to_graph = False
        if params.index_to_graph:
            try:
                from rune.intelligence.code_graph import CodeGraph
                from rune.utils.paths import rune_data

                db_path = rune_data() / "code_graph.db"
                graph = CodeGraph(db_path)
                try:
                    for sym in analysis.symbols:
                        symbol_id = f"{file_path}:{sym.name}:{sym.start_line}"
                        graph.upsert_symbol(
                            symbol_id=symbol_id,
                            name=sym.name,
                            kind=sym.kind,
                            file_path=str(file_path),
                            start_line=sym.start_line,
                            end_line=sym.end_line,
                            parent=sym.parent or "",
                        )
                    for imp in analysis.imports:
                        graph.add_file_dep(
                            source=str(file_path),
                            target=imp,
                        )
                    indexed_to_graph = True
                    lines.append("")
                    lines.append(f"Indexed to code graph: {len(analysis.symbols)} symbols, {len(analysis.imports)} deps")
                finally:
                    graph.close()
            except Exception as idx_exc:
                log.warning("code_graph_index_failed", error=str(idx_exc))
                lines.append(f"\nCode graph indexing failed: {idx_exc}")

        return CapabilityResult(
            success=True,
            output="\n".join(lines),
            metadata={
                "path": str(file_path),
                "language": analysis.language,
                "symbol_count": len(analysis.symbols),
                "import_count": len(analysis.imports),
                "indexed_to_graph": indexed_to_graph,
                "symbols": [
                    {
                        "name": s.name,
                        "kind": s.kind,
                        "start_line": s.start_line,
                        "end_line": s.end_line,
                        "parent": s.parent,
                    }
                    for s in analysis.symbols
                ],
            },
        )

    except ImportError:
        return CapabilityResult(
            success=False,
            error="AST analyzer unavailable (tree-sitter not installed)",
        )
    except Exception as exc:
        return CapabilityResult(
            success=False,
            error=f"Analysis failed: {exc}",
        )


async def code_find_def(params: CodeFindDefParams) -> CapabilityResult:
    """Find symbol definitions by name using the code graph."""
    log.debug("code_find_def", name=params.name, kind=params.kind)

    try:
        from rune.intelligence.code_graph import CodeGraph
        from rune.utils.paths import rune_data

        db_path = rune_data() / "code_graph.db"
        if not db_path.is_file():
            return CapabilityResult(
                success=True,
                output="No code graph available. Index the project first.",
                metadata={"found": 0},
            )

        graph = CodeGraph(db_path)
        try:
            # Query symbols by name
            rows = graph._conn.execute(
                "SELECT id, name, kind, file_path, start_line, end_line, parent "
                "FROM symbols WHERE name = ?",
                (params.name,),
            )
            results = [
                {
                    "id": r[0], "name": r[1], "kind": r[2],
                    "file": r[3], "start_line": r[4], "end_line": r[5],
                    "parent": r[6],
                }
                for r in rows
            ]

            # Filter by kind if specified
            if params.kind:
                results = [r for r in results if r["kind"] == params.kind]

        finally:
            graph.close()

        if not results:
            return CapabilityResult(
                success=True,
                output=f"No definitions found for '{params.name}'",
                metadata={"found": 0},
            )

        lines: list[str] = [f"Definitions of '{params.name}':"]
        for r in results:
            parent_info = f" in {r['parent']}" if r["parent"] else ""
            lines.append(
                f"  {r['kind']} {r['name']}{parent_info}"
                f"  {r['file']}:L{r['start_line']}-{r['end_line']}"
            )

        return CapabilityResult(
            success=True,
            output="\n".join(lines),
            metadata={"found": len(results), "definitions": results},
        )

    except Exception as exc:
        return CapabilityResult(
            success=False,
            error=f"Definition lookup failed: {exc}",
        )


async def code_find_refs(params: CodeFindRefsParams) -> CapabilityResult:
    """Find all references to a symbol using the code graph."""
    log.debug("code_find_refs", name=params.name)

    try:
        from rune.intelligence.code_graph import CodeGraph
        from rune.utils.paths import rune_data

        db_path = rune_data() / "code_graph.db"
        if not db_path.is_file():
            return CapabilityResult(
                success=True,
                output="No code graph available. Index the project first.",
                metadata={"found": 0},
            )

        graph = CodeGraph(db_path)
        try:
            refs = graph.find_references(params.name)
        finally:
            graph.close()

        if not refs:
            return CapabilityResult(
                success=True,
                output=f"No references found for '{params.name}'",
                metadata={"found": 0},
            )

        lines: list[str] = [f"References to '{params.name}' ({len(refs)} found):"]
        for ref in refs:
            lines.append(
                f"  {ref['file']}:L{ref['line']}  "
                f"({ref['kind']}) from {ref['source']}"
            )

        return CapabilityResult(
            success=True,
            output="\n".join(lines),
            metadata={"found": len(refs), "references": refs},
        )

    except Exception as exc:
        return CapabilityResult(
            success=False,
            error=f"Reference lookup failed: {exc}",
        )


async def code_impact(params: CodeImpactParams) -> CapabilityResult:
    """Analyse the impact of changes to a file or symbol."""
    file_path = Path(params.file_path).resolve()
    log.debug("code_impact", file=str(file_path), symbol=params.symbol_name)

    try:
        from rune.intelligence.code_graph import CodeGraph
        from rune.utils.paths import rune_data

        db_path = rune_data() / "code_graph.db"
        if not db_path.is_file():
            return CapabilityResult(
                success=True,
                output="No code graph available. Index the project first.",
                metadata={"dependents": 0},
            )

        graph = CodeGraph(db_path)
        try:
            # Find files that depend on this file
            dependents = graph.find_dependents(str(file_path))

            # Get symbols in the file for context
            symbols = graph.get_symbols_in_file(str(file_path))

            # If a specific symbol is given, find its references too
            symbol_refs: list[dict] = []
            if params.symbol_name:
                symbol_refs = graph.find_references(params.symbol_name)
        finally:
            graph.close()

        lines: list[str] = [f"Impact analysis for: {params.file_path}"]
        lines.append("")

        if symbols:
            lines.append(f"Symbols in file ({len(symbols)}):")
            for sym in symbols:
                lines.append(f"  {sym['kind']} {sym['name']}  L{sym['start_line']}")

        lines.append("")
        lines.append(f"Dependent files ({len(dependents)}):")
        if dependents:
            for dep in dependents:
                lines.append(f"  {dep}")
        else:
            lines.append("  (none found)")

        if params.symbol_name and symbol_refs:
            lines.append("")
            lines.append(
                f"References to '{params.symbol_name}' ({len(symbol_refs)}):"
            )
            for ref in symbol_refs:
                lines.append(
                    f"  {ref['file']}:L{ref['line']} ({ref['kind']})"
                )

        return CapabilityResult(
            success=True,
            output="\n".join(lines),
            metadata={
                "file": str(file_path),
                "dependents": len(dependents),
                "symbols": len(symbols),
                "symbol_refs": len(symbol_refs) if params.symbol_name else 0,
            },
        )

    except Exception as exc:
        return CapabilityResult(
            success=False,
            error=f"Impact analysis failed: {exc}",
        )


# Registration

def register_code_intelligence_capabilities(registry: CapabilityRegistry) -> None:
    """Register all code intelligence capabilities."""
    registry.register(CapabilityDefinition(
        name="code_analyze",
        description="Analyse a source file structure (symbols, imports)",
        domain=Domain.FILE,
        risk_level=RiskLevel.LOW,
        group="read",
        parameters_model=CodeAnalyzeParams,
        execute=code_analyze,
    ))
    registry.register(CapabilityDefinition(
        name="code_find_def",
        description="Find symbol definitions by name",
        domain=Domain.FILE,
        risk_level=RiskLevel.LOW,
        group="read",
        parameters_model=CodeFindDefParams,
        execute=code_find_def,
    ))
    registry.register(CapabilityDefinition(
        name="code_find_refs",
        description="Find all references to a symbol",
        domain=Domain.FILE,
        risk_level=RiskLevel.LOW,
        group="read",
        parameters_model=CodeFindRefsParams,
        execute=code_find_refs,
    ))
    registry.register(CapabilityDefinition(
        name="code_impact",
        description="Analyse impact of changes to a file or symbol",
        domain=Domain.FILE,
        risk_level=RiskLevel.LOW,
        group="read",
        parameters_model=CodeImpactParams,
        execute=code_impact,
    ))
