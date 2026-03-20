"""Code graph - symbol-level dependency tracking.

Ported from src/intelligence/code-graph.ts - APSW-backed graph of
symbols, references, and file relationships.
"""

from __future__ import annotations

import re
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import apsw

from rune.utils.logger import get_logger
from rune.utils.paths import rune_data

log = get_logger(__name__)

_GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS symbols (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    file_path TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    parent TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS references_ (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_symbol TEXT NOT NULL,
    target_symbol TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line INTEGER,
    kind TEXT DEFAULT 'usage',
    FOREIGN KEY (source_symbol) REFERENCES symbols(id),
    FOREIGN KEY (target_symbol) REFERENCES symbols(id)
);

CREATE TABLE IF NOT EXISTS file_deps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT NOT NULL,
    target_file TEXT NOT NULL,
    import_name TEXT,
    UNIQUE(source_file, target_file, import_name)
);

CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_refs_source ON references_(source_symbol);
CREATE INDEX IF NOT EXISTS idx_refs_target ON references_(target_symbol);
CREATE INDEX IF NOT EXISTS idx_file_deps_source ON file_deps(source_file);
"""


class CodeGraph:
    """SQLite-backed code symbol and dependency graph."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = rune_data() / "code_graph.db"
        self._conn = apsw.Connection(str(db_path))
        self._conn.pragma("journal_mode", "wal")
        self._conn.pragma("foreign_keys", True)
        self._conn.pragma("busy_timeout", 5000)
        self._conn.execute(_GRAPH_SCHEMA)

    def upsert_symbol(
        self,
        symbol_id: str,
        name: str,
        kind: str,
        file_path: str,
        start_line: int = 0,
        end_line: int = 0,
        parent: str = "",
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """INSERT INTO symbols (id, name, kind, file_path, start_line, end_line, parent, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 name=excluded.name, kind=excluded.kind,
                 file_path=excluded.file_path, start_line=excluded.start_line,
                 end_line=excluded.end_line, parent=excluded.parent,
                 updated_at=excluded.updated_at""",
            (symbol_id, name, kind, file_path, start_line, end_line, parent, now),
        )

    def add_reference(
        self, source: str, target: str, file_path: str, line: int = 0, kind: str = "usage",
    ) -> None:
        self._conn.execute(
            """INSERT INTO references_ (source_symbol, target_symbol, file_path, line, kind)
               VALUES (?, ?, ?, ?, ?)""",
            (source, target, file_path, line, kind),
        )

    def add_file_dep(self, source: str, target: str, import_name: str = "") -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO file_deps (source_file, target_file, import_name)
               VALUES (?, ?, ?)""",
            (source, target, import_name),
        )

    def find_references(self, symbol_name: str) -> list[dict]:
        rows = self._conn.execute(
            """SELECT r.file_path, r.line, r.kind, s.name as source_name
               FROM references_ r
               JOIN symbols s ON s.id = r.source_symbol
               JOIN symbols t ON t.id = r.target_symbol
               WHERE t.name = ?""",
            (symbol_name,),
        )
        return [
            {"file": r[0], "line": r[1], "kind": r[2], "source": r[3]}
            for r in rows
        ]

    def find_dependents(self, file_path: str) -> list[str]:
        """Find all files that depend on *file_path*."""
        rows = self._conn.execute(
            "SELECT source_file FROM file_deps WHERE target_file = ?",
            (file_path,),
        )
        return [r[0] for r in rows]

    def get_symbols_in_file(self, file_path: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, name, kind, start_line, end_line, parent FROM symbols WHERE file_path = ?",
            (file_path,),
        )
        return [
            {"id": r[0], "name": r[1], "kind": r[2],
             "start_line": r[3], "end_line": r[4], "parent": r[5]}
            for r in rows
        ]

    def clear_file(self, file_path: str) -> None:
        """Remove all symbols and references for a file (before re-indexing)."""
        symbol_ids = [
            r[0] for r in self._conn.execute(
                "SELECT id FROM symbols WHERE file_path = ?", (file_path,)
            )
        ]
        if symbol_ids:
            placeholders = ",".join("?" * len(symbol_ids))
            self._conn.execute(
                f"DELETE FROM references_ WHERE source_symbol IN ({placeholders})"
                f" OR target_symbol IN ({placeholders})",
                symbol_ids + symbol_ids,
            )
        self._conn.execute("DELETE FROM symbols WHERE file_path = ?", (file_path,))
        self._conn.execute("DELETE FROM file_deps WHERE source_file = ?", (file_path,))

    # AST-integrated indexing

    def index_file(self, file_path: str) -> dict:
        """Read *file_path*, run AST analysis, and store symbols/references/deps.

        Returns a summary dict with counts of symbols, imports, and references.
        """
        from rune.intelligence.ast_analyzer import get_ast_analyzer

        analyzer = get_ast_analyzer()
        analysis = analyzer.analyze_file(file_path)
        if analysis is None:
            return {"file": file_path, "symbols": 0, "imports": 0, "references": 0, "skipped": True}

        # Clear stale data first
        self.clear_file(file_path)

        abs_path = str(Path(file_path).resolve())
        sym_count = 0
        ref_count = 0

        for sym in analysis.symbols:
            symbol_id = f"{abs_path}:{sym.name}:{sym.start_line}"
            self.upsert_symbol(
                symbol_id=symbol_id,
                name=sym.name,
                kind=sym.kind,
                file_path=abs_path,
                start_line=sym.start_line,
                end_line=sym.end_line,
                parent=sym.parent,
            )
            sym_count += 1

        # Parse imports to extract file dependencies and symbol references
        imp_count = 0
        for imp_text in analysis.imports:
            target_module = self._extract_import_target(imp_text, analysis.language)
            if target_module:
                self.add_file_dep(abs_path, target_module, imp_text)
                imp_count += 1

        log.debug(
            "index_file_done",
            file=abs_path,
            symbols=sym_count,
            imports=imp_count,
        )
        return {
            "file": abs_path,
            "symbols": sym_count,
            "imports": imp_count,
            "references": ref_count,
            "skipped": False,
        }

    def index_files(self, file_paths: list[str]) -> dict:
        """Batch index multiple files.  Returns aggregate summary."""
        total_symbols = 0
        total_imports = 0
        total_references = 0
        indexed = 0
        skipped = 0

        for fp in file_paths:
            try:
                result = self.index_file(fp)
                if result.get("skipped"):
                    skipped += 1
                else:
                    indexed += 1
                    total_symbols += result["symbols"]
                    total_imports += result["imports"]
                    total_references += result["references"]
            except Exception as exc:
                log.debug("index_file_skipped", file=fp, error=str(exc))
                skipped += 1

        return {
            "indexed": indexed,
            "skipped": skipped,
            "total_symbols": total_symbols,
            "total_imports": total_imports,
            "total_references": total_references,
        }

    # Advanced queries

    def analyze_impact(self, symbol_name: str, max_depth: int = 5) -> dict:
        """BFS impact analysis - find all symbols/files affected by *symbol_name*.

        Returns dict with affected_files, affected_symbols, and total_impact_score.
        """
        affected_files: set[str] = set()
        affected_symbols: list[dict] = []
        visited: set[str] = set()

        # Seed: find all symbols with this name
        seed_rows = list(self._conn.execute(
            "SELECT id, name, kind, file_path, start_line FROM symbols WHERE name = ?",
            (symbol_name,),
        ))

        if not seed_rows:
            return {
                "symbol": symbol_name,
                "affected_files": set(),
                "affected_symbols": [],
                "total_impact_score": 0,
            }

        # BFS queue: (symbol_id, depth)
        queue: deque[tuple[str, int]] = deque()
        for row in seed_rows:
            sid = row[0]
            visited.add(sid)
            queue.append((sid, 0))

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            # Find symbols that reference this symbol (via references_ table)
            referrers = list(self._conn.execute(
                """SELECT DISTINCT s.id, s.name, s.kind, s.file_path, s.start_line
                   FROM references_ r
                   JOIN symbols s ON s.id = r.source_symbol
                   WHERE r.target_symbol = ?""",
                (current_id,),
            ))

            for ref_row in referrers:
                ref_id, ref_name, ref_kind, ref_file, ref_line = ref_row
                affected_files.add(ref_file)
                affected_symbols.append({
                    "id": ref_id,
                    "name": ref_name,
                    "kind": ref_kind,
                    "file": ref_file,
                    "line": ref_line,
                    "depth": depth + 1,
                })
                if ref_id not in visited:
                    visited.add(ref_id)
                    queue.append((ref_id, depth + 1))

            # Also find files that depend on files containing the current symbol
            current_file_rows = list(self._conn.execute(
                "SELECT file_path FROM symbols WHERE id = ?", (current_id,),
            ))
            if current_file_rows:
                current_file = current_file_rows[0][0]
                dependent_files = self.find_dependents(current_file)
                for dep_file in dependent_files:
                    affected_files.add(dep_file)

        return {
            "symbol": symbol_name,
            "affected_files": affected_files,
            "affected_symbols": affected_symbols,
            "total_impact_score": len(affected_files) + len(affected_symbols),
        }

    def search_symbols(
        self,
        query: str,
        kind: str | None = None,
        file_path: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Search symbols by name pattern with optional filters."""
        conditions: list[str] = []
        params: list[str | int] = []

        # Use LIKE for pattern matching; wrap in % if no wildcards already present
        like_query = query if "%" in query else f"%{query}%"
        conditions.append("name LIKE ?")
        params.append(like_query)

        if kind is not None:
            conditions.append("kind = ?")
            params.append(kind)

        if file_path is not None:
            conditions.append("file_path = ?")
            params.append(file_path)

        where = " AND ".join(conditions)
        sql = (
            f"SELECT id, name, kind, file_path, start_line, end_line, parent"
            f" FROM symbols WHERE {where}"
            f" ORDER BY name LIMIT ?"
        )
        params.append(limit)

        rows = self._conn.execute(sql, params)
        return [
            {
                "id": r[0],
                "name": r[1],
                "kind": r[2],
                "file_path": r[3],
                "start_line": r[4],
                "end_line": r[5],
                "parent": r[6],
            }
            for r in rows
        ]

    def get_file_dependencies(self, file_path: str, depth: int = 1) -> dict:
        """Get import dependencies and dependents of *file_path*.

        With *depth* > 1, follows transitive relationships.
        """
        imports: set[str] = set()
        dependents: set[str] = set()

        # BFS for imports (files this file depends on)
        visited_imports: set[str] = {file_path}
        queue_imports: deque[tuple[str, int]] = deque([(file_path, 0)])
        while queue_imports:
            current, d = queue_imports.popleft()
            if d >= depth:
                continue
            rows = list(self._conn.execute(
                "SELECT target_file FROM file_deps WHERE source_file = ?",
                (current,),
            ))
            for (target,) in rows:
                imports.add(target)
                if target not in visited_imports:
                    visited_imports.add(target)
                    queue_imports.append((target, d + 1))

        # BFS for dependents (files that depend on this file)
        visited_deps: set[str] = {file_path}
        queue_deps: deque[tuple[str, int]] = deque([(file_path, 0)])
        while queue_deps:
            current, d = queue_deps.popleft()
            if d >= depth:
                continue
            dep_files = self.find_dependents(current)
            for dep in dep_files:
                dependents.add(dep)
                if dep not in visited_deps:
                    visited_deps.add(dep)
                    queue_deps.append((dep, d + 1))

        return {
            "file": file_path,
            "imports": sorted(imports),
            "dependents": sorted(dependents),
            "depth": depth,
        }

    # Internal helpers

    @staticmethod
    def _extract_import_target(import_text: str, language: str) -> str:
        """Extract module/file target from an import statement string."""
        if language == "python":
            # "from foo.bar import baz" -> "foo.bar" (skip relative-only like "from . import")
            m = re.match(r"from\s+([\w][\w.]*)\s+import", import_text)
            if m:
                return m.group(1)
            # "import foo.bar" -> "foo.bar"
            m = re.match(r"import\s+([\w][\w.]*)", import_text)
            if m:
                return m.group(1)
        elif language in ("javascript", "typescript", "tsx"):
            # import ... from 'module'  or  import 'module'
            m = re.search(r"""from\s+['"](.*?)['"]""", import_text)
            if m:
                return m.group(1)
            m = re.search(r"""import\s+['"](.*?)['"]""", import_text)
            if m:
                return m.group(1)
        else:
            # Generic: try to grab a quoted string or dotted name
            m = re.search(r"""['"](.*?)['"]""", import_text)
            if m:
                return m.group(1)
        return ""

    def close(self) -> None:
        self._conn.close()
