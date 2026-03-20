"""Tests for rune.utils.at_reference — ported from at-reference.test.ts."""

from pathlib import Path

import pytest

from rune.utils.at_reference import (
    extract_at_workspace_directive,
    parse_at_references,
    resolve_references,
)


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temp workspace mirroring the TS test fixtures."""
    (tmp_path / "hello.ts").write_text('console.log("hello");')
    (tmp_path / "readme.md").write_text("# Hello World")
    src = tmp_path / "src"
    src.mkdir()
    (src / "index.ts").write_text("export default {};")
    (src / "utils.ts").write_text("export const add = (a, b) => a + b;")
    nested = src / "nested"
    nested.mkdir()
    (nested / "deep.ts").write_text("export const deep = true;")
    return tmp_path


class TestExtractAtWorkspaceDirective:
    """Tests for extract_at_workspace_directive()."""

    def test_resolves_relative_path_with_cwd(self, tmp_workspace):
        sub_dir = str(tmp_workspace / "src" / "nested")
        result = extract_at_workspace_directive("@../../ analyze", sub_dir)
        assert result == str(tmp_workspace)

    def test_resolves_dot_slash_relative_path(self, tmp_workspace):
        result = extract_at_workspace_directive("@./src/ show structure", str(tmp_workspace))
        assert result == str(tmp_workspace / "src")

    def test_resolves_dot_dot_relative_path(self, tmp_workspace):
        sub_dir = str(tmp_workspace / "src")
        result = extract_at_workspace_directive("@../ analyze", sub_dir)
        assert result == str(tmp_workspace)

    def test_resolves_absolute_path(self, tmp_workspace):
        result = extract_at_workspace_directive(f"@{tmp_workspace}/ analyze")
        assert result == str(tmp_workspace)

    def test_resolves_home_relative_path(self):
        result = extract_at_workspace_directive("@~/ analyze")
        assert result == str(Path.home())

    def test_returns_parent_dir_for_file_paths(self, tmp_workspace):
        file_path = str(tmp_workspace / "hello.ts")
        result = extract_at_workspace_directive(f"@{file_path} analyze")
        assert result == str(tmp_workspace)

    def test_returns_none_for_nonexistent_paths(self):
        result = extract_at_workspace_directive("@/nonexistent/path/xyz analyze")
        assert result is None

    def test_returns_none_for_plain_text_without_at(self):
        result = extract_at_workspace_directive("analyze the project")
        assert result is None

    def test_returns_none_for_email_addresses(self):
        result = extract_at_workspace_directive("user@domain.com")
        assert result is None


class TestParseAtReferences:
    """Tests for parse_at_references()."""

    def test_parses_basic_file_reference_with_slash(self, tmp_workspace):
        refs = parse_at_references("@src/index.ts analyze", str(tmp_workspace))
        assert len(refs) == 1
        assert refs[0].path == "src/index.ts"
        assert refs[0].raw == "@src/index.ts"
        assert refs[0].type == "file"

    def test_parses_relative_path_dot_slash(self, tmp_workspace):
        refs = parse_at_references("@./hello.ts read", str(tmp_workspace))
        assert len(refs) == 1
        assert refs[0].type == "file"

    def test_parses_directory_reference(self, tmp_workspace):
        refs = parse_at_references("@src/ show structure", str(tmp_workspace))
        assert len(refs) == 1
        assert refs[0].type == "directory"

    def test_parses_multiple_references(self, tmp_workspace):
        refs = parse_at_references(
            "@src/index.ts @src/utils.ts compare these",
            str(tmp_workspace),
        )
        assert len(refs) == 2

    def test_does_not_match_email_addresses(self, tmp_workspace):
        refs = parse_at_references("user@domain.com send", str(tmp_workspace))
        assert len(refs) == 0

    def test_handles_not_found_paths(self, tmp_workspace):
        refs = parse_at_references("@nonexistent/file.ts read", str(tmp_workspace))
        assert len(refs) == 1
        assert refs[0].type == "not_found"

    def test_deduplicates_same_references(self, tmp_workspace):
        refs = parse_at_references(
            "@src/index.ts and @src/index.ts compare",
            str(tmp_workspace),
        )
        assert len(refs) == 1

    def test_parses_file_with_extension_no_slash(self, tmp_workspace):
        refs = parse_at_references("@hello.ts read", str(tmp_workspace))
        assert len(refs) == 1
        assert refs[0].path == "hello.ts"
        assert refs[0].type == "file"


class TestResolveReferences:
    """Tests for resolve_references()."""

    def test_resolves_file_content(self, tmp_workspace):
        refs = parse_at_references("@hello.ts read", str(tmp_workspace))
        resolved = resolve_references(refs)
        assert len(resolved) == 1
        assert resolved[0].content == 'console.log("hello");'

    def test_resolves_directory_listing(self, tmp_workspace):
        refs = parse_at_references("@src/ list", str(tmp_workspace))
        resolved = resolve_references(refs)
        assert len(resolved) == 1
        assert "index.ts" in resolved[0].content

    def test_reports_error_for_not_found(self, tmp_workspace):
        refs = parse_at_references("@missing/file.ts read", str(tmp_workspace))
        resolved = resolve_references(refs)
        assert len(resolved) == 1
        assert resolved[0].error != ""
