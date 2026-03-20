"""Tests for code intelligence parameter models."""

from __future__ import annotations

from rune.capabilities.code_intelligence import (
    CodeAnalyzeParams,
    CodeFindDefParams,
    CodeFindRefsParams,
)


def test_code_analyze_params():
    """CodeAnalyzeParams requires file_path."""
    params = CodeAnalyzeParams(file_path="/src/main.py")
    assert params.file_path == "/src/main.py"


def test_code_find_def_params():
    """CodeFindDefParams requires name, kind defaults to empty."""
    params = CodeFindDefParams(name="my_func")
    assert params.name == "my_func"
    assert params.kind == ""

    params2 = CodeFindDefParams(name="MyClass", kind="class")
    assert params2.kind == "class"


def test_code_find_refs_params():
    """CodeFindRefsParams requires name."""
    params = CodeFindRefsParams(name="some_symbol")
    assert params.name == "some_symbol"
