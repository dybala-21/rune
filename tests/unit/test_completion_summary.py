"""Tests for rune.ui.completion_summary — narrative builder and file tracking."""

from __future__ import annotations

from rune.ui.completion_summary import (
    ToolCallBlock,
    build_completion_narrative,
    get_touched_files,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _block(**kwargs) -> ToolCallBlock:
    defaults = dict(
        id="tool-1",
        action="file.write",
        observation="ok",
        success=True,
        timestamp="",
        capability="",
        params={},
    )
    defaults.update(kwargs)
    return ToolCallBlock(**defaults)


# ---------------------------------------------------------------------------
# get_touched_files
# ---------------------------------------------------------------------------


class TestGetTouchedFiles:
    def test_returns_unique_short_paths_for_file_writes(self):
        blocks = [
            _block(capability="file.write", params={"path": "/a/b/docs/report.md"}),
            _block(id="t2", capability="file.edit", params={"filePath": "/a/b/src/ui/App.tsx"}),
            _block(id="t3", capability="file.write", params={"path": "/a/b/docs/report.md"}),
            _block(id="t4", capability="web.search"),
            _block(id="t5", capability="file.delete", success=False, params={"path": "/a/b/docs/ignored.md"}),
        ]
        files = get_touched_files(blocks)
        assert files == ["docs/report.md", "ui/App.tsx"]

    def test_empty_blocks(self):
        assert get_touched_files([]) == []

    def test_ignores_blocks_without_capability(self):
        blocks = [_block(capability="", params={"path": "/foo/bar.py"})]
        assert get_touched_files(blocks) == []


# ---------------------------------------------------------------------------
# build_completion_narrative
# ---------------------------------------------------------------------------


class TestBuildCompletionNarrative:
    def test_success_with_tool_calls_and_file_updates(self):
        blocks = [
            _block(capability="file.write", params={"path": "/a/b/docs/report.md"}),
            _block(id="t2", capability="web.search"),
        ]
        text = build_completion_narrative(
            success=True, iterations=12, tool_call_blocks=blocks,
        )
        assert text == "I wrapped up after 2 tool calls and 1 file update."

    def test_success_without_tool_calls(self):
        text = build_completion_narrative(
            success=True, iterations=3, tool_call_blocks=[],
        )
        assert text == "I wrapped up after 3 steps."

    def test_failure_with_file_updates(self):
        blocks = [
            _block(capability="file.write", params={"path": "/a/b/docs/report.md"}),
            _block(id="t2", capability="bash", success=False),
        ]
        text = build_completion_narrative(
            success=False, iterations=5, tool_call_blocks=blocks,
        )
        assert text == "I hit a stopping point after 2 tool calls and 1 file update."

    def test_failure_with_no_file_updates(self):
        blocks = [_block(capability="web.search", success=False)]
        text = build_completion_narrative(
            success=False, iterations=4, tool_call_blocks=blocks,
        )
        assert text == "I hit a stopping point after 1 tool call, with 1 failure."

    def test_success_no_iterations_no_tools(self):
        text = build_completion_narrative(
            success=True, iterations=0, tool_call_blocks=[],
        )
        assert text == "I wrapped up the run."

    def test_failure_no_iterations_no_tools(self):
        text = build_completion_narrative(
            success=False, iterations=0, tool_call_blocks=[],
        )
        assert text == "I hit a stopping point before the run could finish cleanly."
