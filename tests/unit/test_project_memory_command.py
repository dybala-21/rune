"""Tests for rune.memory.project_memory_command — slash command parsing and execution."""

import pytest

from rune.memory.project_memory_command import (
    ProjectMemoryCommand,
    execute_project_memory_command,
    parse_project_memory_command,
    parse_project_memory_slash_args,
)

# ---------------------------------------------------------------------------
# Tests: parse_project_memory_slash_args
# ---------------------------------------------------------------------------

class TestParseSlashArgs:
    def test_empty_args_returns_show(self):
        cmd = parse_project_memory_slash_args("")
        assert cmd.action == "show"
        assert cmd.source == "slash"

    def test_show_keyword(self):
        cmd = parse_project_memory_slash_args("show")
        assert cmd.action == "show"

    def test_list_keyword(self):
        cmd = parse_project_memory_slash_args("list")
        assert cmd.action == "show"

    def test_help_keyword(self):
        cmd = parse_project_memory_slash_args("help")
        assert cmd.action == "help"

    def test_add_with_pipe_separator(self):
        cmd = parse_project_memory_slash_args("add Decisions | rollback to previous image")
        assert cmd.action == "add"
        assert cmd.section == "Decisions"
        assert cmd.content == "rollback to previous image"

    def test_add_with_colon_separator(self):
        cmd = parse_project_memory_slash_args("add Decisions: rollback strategy")
        assert cmd.action == "add"
        assert cmd.section == "Decisions"
        assert cmd.content == "rollback strategy"

    def test_add_defaults_to_notes_section(self):
        cmd = parse_project_memory_slash_args("add some general note")
        assert cmd.action == "add"
        assert cmd.section == "Notes"
        assert cmd.content == "some general note"

    def test_section_alias_pref_to_preferences(self):
        cmd = parse_project_memory_slash_args("add pref | use dark mode")
        assert cmd.section == "Preferences"

    def test_section_alias_env_to_environment(self):
        cmd = parse_project_memory_slash_args("add env | Node v20.11")
        assert cmd.section == "Environment"

    def test_bare_text_becomes_add(self):
        cmd = parse_project_memory_slash_args("just some bare text note")
        assert cmd.action == "add"
        assert cmd.section == "Notes"

    def test_append_keyword(self):
        cmd = parse_project_memory_slash_args("append Notes | extra info")
        assert cmd.action == "add"
        assert cmd.section == "Notes"

    def test_save_keyword(self):
        cmd = parse_project_memory_slash_args("save Patterns | use factories for test data")
        assert cmd.action == "add"
        assert cmd.section == "Patterns"


# ---------------------------------------------------------------------------
# Tests: parse_project_memory_command
# ---------------------------------------------------------------------------

class TestParseProjectMemoryCommand:
    def test_parses_slash_show(self):
        cmd = parse_project_memory_command("/memory show")
        assert cmd is not None
        assert cmd.action == "show"
        assert cmd.source == "slash"

    def test_parses_slash_add(self):
        cmd = parse_project_memory_command("/memory add Notes | test note")
        assert cmd is not None
        assert cmd.action == "add"

    def test_returns_none_for_non_memory_text(self):
        cmd = parse_project_memory_command("just some text")
        assert cmd is None

    def test_natural_show_when_enabled(self):
        cmd = parse_project_memory_command("memory show", allow_natural=True)
        assert cmd is not None
        assert cmd.action == "show"
        assert cmd.source == "natural"

    def test_natural_not_parsed_when_disabled(self):
        cmd = parse_project_memory_command("memory show", allow_natural=False)
        assert cmd is None

    def test_empty_string_returns_none(self):
        cmd = parse_project_memory_command("")
        assert cmd is None


# ---------------------------------------------------------------------------
# Tests: execute_project_memory_command
# ---------------------------------------------------------------------------

class TestExecuteProjectMemoryCommand:
    @pytest.mark.asyncio
    async def test_help_command(self):
        cmd = ProjectMemoryCommand(action="help", source="slash")
        result = await execute_project_memory_command(cmd)
        assert result["success"] is True
        assert "/memory" in str(result["message"])

    @pytest.mark.asyncio
    async def test_show_empty_project(self, tmp_path):
        cmd = ProjectMemoryCommand(action="show", source="slash")
        result = await execute_project_memory_command(
            cmd,
            workspace_path="/Users/test/workspace/rune",
            rune_config_dir=str(tmp_path / ".rune"),
        )
        assert result["success"] is True
        assert "empty" in str(result["message"]).lower() or "Project Memory" in str(result["message"])

    @pytest.mark.asyncio
    async def test_add_then_show(self, tmp_path):
        config_dir = str(tmp_path / ".rune")
        workspace = "/Users/test/workspace/rune"

        # Add
        add_cmd = ProjectMemoryCommand(
            action="add", source="slash",
            section="Notes", content="release checklist updated",
        )
        add_result = await execute_project_memory_command(
            add_cmd, workspace_path=workspace, rune_config_dir=config_dir,
        )
        assert add_result["success"] is True
        assert "Saved" in str(add_result["message"])

        # Show
        show_cmd = ProjectMemoryCommand(action="show", source="slash")
        show_result = await execute_project_memory_command(
            show_cmd, workspace_path=workspace, rune_config_dir=config_dir,
        )
        assert show_result["success"] is True
        assert "release checklist updated" in str(show_result["message"])

    @pytest.mark.asyncio
    async def test_add_creates_section_header(self, tmp_path):
        config_dir = str(tmp_path / ".rune")
        workspace = "/Users/test/workspace/rune"

        cmd = ProjectMemoryCommand(
            action="add", source="slash",
            section="Decisions", content="use blue-green deploys",
        )
        await execute_project_memory_command(
            cmd, workspace_path=workspace, rune_config_dir=config_dir,
        )

        show_cmd = ProjectMemoryCommand(action="show", source="slash")
        result = await execute_project_memory_command(
            show_cmd, workspace_path=workspace, rune_config_dir=config_dir,
        )
        assert "## Decisions" in str(result["message"])

    @pytest.mark.asyncio
    async def test_add_empty_content_fails(self, tmp_path):
        cmd = ProjectMemoryCommand(
            action="add", source="slash",
            section="Notes", content="",
        )
        result = await execute_project_memory_command(
            cmd,
            workspace_path="/test",
            rune_config_dir=str(tmp_path / ".rune"),
        )
        assert result["success"] is False
