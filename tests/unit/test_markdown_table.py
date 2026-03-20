"""Tests for rune.utils.markdown_table — markdown table generation."""

from __future__ import annotations

from rune.utils.markdown_table import Align, markdown_table


class TestMarkdownTable:
    def test_basic_table(self):
        result = markdown_table(
            headers=["Name", "Score"],
            rows=[["Alice", "100"], ["Bob", "9"]],
        )
        assert "| Name" in result
        assert "| Alice" in result
        assert "| Bob" in result
        # Should contain separator
        assert "---" in result

    def test_right_aligned_column(self):
        result = markdown_table(
            headers=["Item", "Value"],
            rows=[["A", "1"], ["B", "22"]],
            alignments=[Align.LEFT, Align.RIGHT],
        )
        # Right-aligned separator ends with ':'
        lines = result.split("\n")
        sep_line = lines[1]
        assert sep_line.rstrip("|").rstrip().endswith(":")

    def test_center_aligned_column(self):
        result = markdown_table(
            headers=["X"],
            rows=[["val"]],
            alignments=[Align.CENTER],
        )
        lines = result.split("\n")
        sep_line = lines[1]
        # Center-aligned separator starts and ends with ':'
        inner = sep_line.strip("|")
        assert inner.startswith(":")
        assert inner.endswith(":")

    def test_rows_padded_to_header_count(self):
        result = markdown_table(
            headers=["A", "B", "C"],
            rows=[["only_one"]],
        )
        # Should not crash; row gets padded with empty strings
        assert "| only_one" in result

    def test_empty_rows(self):
        result = markdown_table(headers=["Col"], rows=[])
        lines = result.split("\n")
        # Header + separator, no data rows
        assert len(lines) == 2

    def test_minimum_column_width(self):
        result = markdown_table(headers=["X"], rows=[["a"]])
        lines = result.split("\n")
        sep_line = lines[1]
        # Separator should have at least 3 dashes
        dashes = sep_line.replace("|", "").replace(":", "").strip()
        assert len(dashes) >= 3

    def test_custom_padding(self):
        result = markdown_table(
            headers=["H"],
            rows=[["data"]],
            pad=2,
        )
        # With pad=2, there should be 2 spaces around cell content
        assert "  H  " in result or "  data  " in result
