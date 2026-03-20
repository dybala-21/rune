"""Tests for CodeGraph advanced methods (index_file, search_symbols, etc.)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from rune.intelligence.code_graph import CodeGraph


@pytest.fixture()
def graph(tmp_path: Path) -> CodeGraph:
    """In-memory code graph for testing."""
    g = CodeGraph(db_path=tmp_path / "test_graph.db")
    yield g
    g.close()


# ---------------------------------------------------------------------------
# index_file / index_files
# ---------------------------------------------------------------------------


def test_index_file_python(graph: CodeGraph, tmp_path: Path) -> None:
    """index_file should extract symbols from a Python file."""
    src = tmp_path / "example.py"
    src.write_text(
        textwrap.dedent("""\
        import os

        class Greeter:
            def greet(self, name: str) -> str:
                return f"Hello, {name}"

        def helper():
            pass
        """)
    )

    result = graph.index_file(str(src))
    assert result["skipped"] is False
    assert result["symbols"] >= 3  # Greeter, greet, helper
    assert result["imports"] >= 1  # import os

    # Verify symbols are in the DB
    symbols = graph.get_symbols_in_file(str(src.resolve()))
    names = {s["name"] for s in symbols}
    assert "Greeter" in names
    assert "greet" in names
    assert "helper" in names


def test_index_file_unsupported_extension(graph: CodeGraph, tmp_path: Path) -> None:
    """index_file should return skipped=True for unsupported files."""
    src = tmp_path / "data.csv"
    src.write_text("a,b,c\n1,2,3")
    result = graph.index_file(str(src))
    assert result["skipped"] is True
    assert result["symbols"] == 0


def test_index_files_batch(graph: CodeGraph, tmp_path: Path) -> None:
    """index_files should aggregate results across multiple files."""
    f1 = tmp_path / "a.py"
    f1.write_text("def foo(): pass\ndef bar(): pass\n")
    f2 = tmp_path / "b.py"
    f2.write_text("class Baz:\n    def method(self): pass\n")
    f3 = tmp_path / "c.txt"
    f3.write_text("not code")

    result = graph.index_files([str(f1), str(f2), str(f3)])
    assert result["indexed"] == 2
    assert result["skipped"] == 1
    assert result["total_symbols"] >= 4  # foo, bar, Baz, method


def test_index_files_clears_stale(graph: CodeGraph, tmp_path: Path) -> None:
    """Re-indexing a file should clear stale symbols."""
    src = tmp_path / "mod.py"
    src.write_text("def old_func(): pass\n")
    graph.index_file(str(src))

    symbols_before = graph.get_symbols_in_file(str(src.resolve()))
    assert any(s["name"] == "old_func" for s in symbols_before)

    # Overwrite file
    src.write_text("def new_func(): pass\n")
    graph.index_file(str(src))

    symbols_after = graph.get_symbols_in_file(str(src.resolve()))
    names = {s["name"] for s in symbols_after}
    assert "new_func" in names
    assert "old_func" not in names


# ---------------------------------------------------------------------------
# search_symbols
# ---------------------------------------------------------------------------


def test_search_symbols_by_name(graph: CodeGraph, tmp_path: Path) -> None:
    src = tmp_path / "search.py"
    src.write_text("def find_user(): pass\ndef find_order(): pass\ndef create(): pass\n")
    graph.index_file(str(src))

    results = graph.search_symbols("find")
    assert len(results) == 2
    names = {r["name"] for r in results}
    assert names == {"find_user", "find_order"}


def test_search_symbols_by_kind(graph: CodeGraph, tmp_path: Path) -> None:
    src = tmp_path / "kinds.py"
    src.write_text("class MyClass:\n    def my_method(self): pass\ndef my_func(): pass\n")
    graph.index_file(str(src))

    classes = graph.search_symbols("%", kind="class")
    assert all(r["kind"] == "class" for r in classes)
    assert any(r["name"] == "MyClass" for r in classes)


def test_search_symbols_by_file(graph: CodeGraph, tmp_path: Path) -> None:
    f1 = tmp_path / "x.py"
    f1.write_text("def alpha(): pass\n")
    f2 = tmp_path / "y.py"
    f2.write_text("def beta(): pass\n")
    graph.index_files([str(f1), str(f2)])

    results = graph.search_symbols("%", file_path=str(f1.resolve()))
    assert len(results) >= 1
    assert all(r["file_path"] == str(f1.resolve()) for r in results)


def test_search_symbols_limit(graph: CodeGraph, tmp_path: Path) -> None:
    src = tmp_path / "many.py"
    funcs = "\n".join(f"def fn_{i}(): pass" for i in range(20))
    src.write_text(funcs + "\n")
    graph.index_file(str(src))

    results = graph.search_symbols("fn_", limit=5)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# analyze_impact
# ---------------------------------------------------------------------------


def test_analyze_impact_no_symbol(graph: CodeGraph) -> None:
    """Impact analysis on non-existent symbol returns empty result."""
    result = graph.analyze_impact("nonexistent")
    assert result["affected_files"] == set()
    assert result["affected_symbols"] == []
    assert result["total_impact_score"] == 0


def test_analyze_impact_with_references(graph: CodeGraph) -> None:
    """Impact analysis follows references through the graph."""
    # Manually set up symbols and references
    graph.upsert_symbol("a:foo:1", "foo", "function", "/a.py", 1, 5)
    graph.upsert_symbol("b:bar:1", "bar", "function", "/b.py", 1, 5)
    graph.upsert_symbol("c:baz:1", "baz", "function", "/c.py", 1, 5)

    # bar references foo, baz references bar (transitive chain)
    graph.add_reference("b:bar:1", "a:foo:1", "/b.py", 2)
    graph.add_reference("c:baz:1", "b:bar:1", "/c.py", 2)

    result = graph.analyze_impact("foo")
    assert "/b.py" in result["affected_files"]
    assert result["total_impact_score"] >= 1

    # Check that transitive impact is found (baz -> bar -> foo)
    sym_names = {s["name"] for s in result["affected_symbols"]}
    assert "bar" in sym_names
    assert "baz" in sym_names


def test_analyze_impact_respects_max_depth(graph: CodeGraph) -> None:
    """Impact analysis should stop at max_depth."""
    graph.upsert_symbol("a:s0:1", "s0", "function", "/a.py", 1, 1)
    graph.upsert_symbol("b:s1:1", "s1", "function", "/b.py", 1, 1)
    graph.upsert_symbol("c:s2:1", "s2", "function", "/c.py", 1, 1)
    graph.upsert_symbol("d:s3:1", "s3", "function", "/d.py", 1, 1)

    graph.add_reference("b:s1:1", "a:s0:1", "/b.py", 1)
    graph.add_reference("c:s2:1", "b:s1:1", "/c.py", 1)
    graph.add_reference("d:s3:1", "c:s2:1", "/d.py", 1)

    result = graph.analyze_impact("s0", max_depth=1)
    sym_names = {s["name"] for s in result["affected_symbols"]}
    # Only depth 1: s1 references s0 directly
    assert "s1" in sym_names
    # s2 is at depth 2, should NOT be found
    assert "s2" not in sym_names


# ---------------------------------------------------------------------------
# get_file_dependencies
# ---------------------------------------------------------------------------


def test_get_file_dependencies_basic(graph: CodeGraph) -> None:
    """get_file_dependencies returns imports and dependents."""
    graph.add_file_dep("/app/main.py", "/app/utils.py", "import utils")
    graph.add_file_dep("/app/main.py", "/app/config.py", "import config")
    graph.add_file_dep("/app/tests.py", "/app/main.py", "import main")

    result = graph.get_file_dependencies("/app/main.py")
    assert "/app/utils.py" in result["imports"]
    assert "/app/config.py" in result["imports"]
    assert "/app/tests.py" in result["dependents"]


def test_get_file_dependencies_transitive(graph: CodeGraph) -> None:
    """Transitive deps with depth > 1."""
    graph.add_file_dep("/a.py", "/b.py", "import b")
    graph.add_file_dep("/b.py", "/c.py", "import c")

    # Depth 1: only direct
    result1 = graph.get_file_dependencies("/a.py", depth=1)
    assert "/b.py" in result1["imports"]
    assert "/c.py" not in result1["imports"]

    # Depth 2: transitive
    result2 = graph.get_file_dependencies("/a.py", depth=2)
    assert "/b.py" in result2["imports"]
    assert "/c.py" in result2["imports"]


def test_get_file_dependencies_no_deps(graph: CodeGraph) -> None:
    """File with no dependencies returns empty lists."""
    result = graph.get_file_dependencies("/lonely.py")
    assert result["imports"] == []
    assert result["dependents"] == []


# ---------------------------------------------------------------------------
# _extract_import_target
# ---------------------------------------------------------------------------


def test_extract_import_python() -> None:
    assert CodeGraph._extract_import_target("import os", "python") == "os"
    assert CodeGraph._extract_import_target("from os.path import join", "python") == "os.path"
    assert CodeGraph._extract_import_target("import json", "python") == "json"
    assert CodeGraph._extract_import_target("from . import foo", "python") == ""


def test_extract_import_javascript() -> None:
    assert CodeGraph._extract_import_target("import { foo } from './utils'", "javascript") == "./utils"
    assert CodeGraph._extract_import_target('import React from "react"', "javascript") == "react"


def test_extract_import_unknown_language() -> None:
    # Falls back to quoted string extraction
    assert CodeGraph._extract_import_target('require("lodash")', "unknown") == "lodash"
