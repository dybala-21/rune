"""Tests for rune.tools.file — ported from file-tool.test.ts."""

from pathlib import Path

import pytest

from rune.tools.file import FileTool


@pytest.fixture
def tool():
    return FileTool()


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace with some files."""
    (tmp_path / "hello.txt").write_text("hello world")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.txt").write_text("nested content")
    return tmp_path


class TestFileToolProperties:
    """Tests for FileTool properties."""

    def test_has_name_file(self, tool):
        assert tool.name == "file"

    def test_has_domain_file(self, tool):
        assert tool.domain == "file"

    def test_includes_expected_actions(self, tool):
        expected = ["scan", "list", "read", "create", "create_dirs", "move", "copy", "delete"]
        assert tool.actions == expected
        assert len(tool.actions) == 8

    def test_has_risk_level_medium(self, tool):
        assert tool.risk_level == "medium"


class TestFileToolValidate:
    """Tests for FileTool.validate()."""

    @pytest.mark.asyncio
    async def test_fails_when_action_is_missing(self, tool):
        valid, msg = await tool.validate({})
        assert valid is False
        assert msg == "Missing action parameter"

    @pytest.mark.asyncio
    async def test_fails_for_unknown_action(self, tool):
        valid, msg = await tool.validate({"action": "explode"})
        assert valid is False
        assert msg == "Unknown action: explode"

    @pytest.mark.asyncio
    async def test_succeeds_for_valid_read_action(self, tool):
        valid, msg = await tool.validate({"action": "read", "path": "/tmp/file.txt"})
        assert valid is True
        assert msg == ""

    @pytest.mark.asyncio
    async def test_succeeds_for_every_valid_action(self, tool):
        for action in tool.actions:
            valid, _ = await tool.validate({"action": action})
            assert valid is True

    @pytest.mark.asyncio
    async def test_blocks_system_paths(self, tool):
        valid, msg = await tool.validate({"action": "read", "path": "/usr/bin/ls"})
        assert valid is False
        assert "System path access denied" in msg

    @pytest.mark.asyncio
    async def test_blocks_deny_pattern_matches(self, tool):
        valid, msg = await tool.validate({"action": "read", "path": "/tmp/.env"})
        assert valid is False
        assert "Sensitive path access denied" in msg

    @pytest.mark.asyncio
    async def test_ignores_non_string_path_params(self, tool):
        valid, msg = await tool.validate({"action": "read", "path": 123})
        assert valid is True
        assert msg == ""


class TestFileToolExecute:
    """Tests for FileTool.execute()."""

    @pytest.mark.asyncio
    async def test_read_returns_file_content(self, tool, tmp_workspace):
        result = await tool.execute({
            "action": "read",
            "path": str(tmp_workspace / "hello.txt"),
        })
        assert result.success is True
        assert result.data["content"] == "hello world"

    @pytest.mark.asyncio
    async def test_read_fails_for_missing_file(self, tool, tmp_workspace):
        result = await tool.execute({
            "action": "read",
            "path": str(tmp_workspace / "nonexistent.txt"),
        })
        assert result.success is False

    @pytest.mark.asyncio
    async def test_list_returns_entries(self, tool, tmp_workspace):
        result = await tool.execute({
            "action": "list",
            "path": str(tmp_workspace),
        })
        assert result.success is True
        names = [e["name"] for e in result.data["entries"]]
        assert "hello.txt" in names
        assert "sub" in names

    @pytest.mark.asyncio
    async def test_create_writes_file(self, tool, tmp_workspace):
        new_file = str(tmp_workspace / "new.txt")
        result = await tool.execute({
            "action": "create",
            "path": new_file,
            "content": "brand new",
        })
        assert result.success is True
        assert Path(new_file).read_text() == "brand new"

    @pytest.mark.asyncio
    async def test_move_renames_file(self, tool, tmp_workspace):
        src = str(tmp_workspace / "hello.txt")
        dst = str(tmp_workspace / "renamed.txt")
        result = await tool.execute({
            "action": "move",
            "source": src,
            "destination": dst,
        })
        assert result.success is True
        assert Path(dst).exists()
        assert not Path(src).exists()

    @pytest.mark.asyncio
    async def test_copy_duplicates_file(self, tool, tmp_workspace):
        src = str(tmp_workspace / "hello.txt")
        dst = str(tmp_workspace / "hello_copy.txt")
        result = await tool.execute({
            "action": "copy",
            "source": src,
            "destination": dst,
        })
        assert result.success is True
        assert Path(dst).read_text() == "hello world"
        assert Path(src).exists()
