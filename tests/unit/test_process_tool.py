"""Tests for rune.tools.process — ported from process-tool.test.ts."""

import pytest

from rune.tools.process import ProcessTool


@pytest.fixture
def tool():
    return ProcessTool()


class TestProcessToolProperties:
    """Tests for ProcessTool properties."""

    def test_has_name_process(self, tool):
        assert tool.name == "process"

    def test_has_domain_process(self, tool):
        assert tool.domain == "process"

    def test_has_description(self, tool):
        assert tool.description == "System process management (list, run, kill, find, monitor)"

    def test_has_risk_level_high(self, tool):
        assert tool.risk_level == "high"

    def test_includes_all_5_actions(self, tool):
        assert tool.actions == ["list", "run", "kill", "find", "monitor"]
        assert len(tool.actions) == 5


class TestProcessToolValidate:
    """Tests for ProcessTool.validate()."""

    @pytest.mark.asyncio
    async def test_fails_when_action_missing(self, tool):
        valid, msg = await tool.validate({})
        assert valid is False
        assert msg == "Missing action parameter"

    @pytest.mark.asyncio
    async def test_fails_when_action_is_empty(self, tool):
        valid, msg = await tool.validate({"action": ""})
        assert valid is False
        assert msg == "Missing action parameter"

    @pytest.mark.asyncio
    async def test_fails_for_unknown_action(self, tool):
        valid, msg = await tool.validate({"action": "restart"})
        assert valid is False
        assert msg == "Unknown action: restart"

    @pytest.mark.asyncio
    async def test_succeeds_for_list_action(self, tool):
        valid, msg = await tool.validate({"action": "list"})
        assert valid is True
        assert msg == ""

    @pytest.mark.asyncio
    async def test_succeeds_for_find_action(self, tool):
        valid, msg = await tool.validate({"action": "find"})
        assert valid is True

    @pytest.mark.asyncio
    async def test_succeeds_for_monitor_action(self, tool):
        valid, msg = await tool.validate({"action": "monitor"})
        assert valid is True

    @pytest.mark.asyncio
    async def test_succeeds_for_kill_with_non_protected_name(self, tool):
        valid, msg = await tool.validate({"action": "kill", "name": "myapp"})
        assert valid is True

    @pytest.mark.asyncio
    async def test_succeeds_for_kill_without_name(self, tool):
        valid, msg = await tool.validate({"action": "kill"})
        assert valid is True

    @pytest.mark.asyncio
    async def test_run_fails_when_command_missing(self, tool):
        valid, msg = await tool.validate({"action": "run"})
        assert valid is False
        assert msg == "Missing command parameter"

    @pytest.mark.asyncio
    async def test_run_fails_when_command_empty(self, tool):
        valid, msg = await tool.validate({"action": "run", "command": ""})
        assert valid is False
        assert msg == "Missing command parameter"

    @pytest.mark.asyncio
    async def test_run_succeeds_for_valid_command(self, tool):
        valid, msg = await tool.validate({"action": "run", "command": "ls -la"})
        assert valid is True
        assert msg == ""

    @pytest.mark.asyncio
    async def test_run_denies_rm_rf_slash(self, tool):
        valid, msg = await tool.validate({"action": "run", "command": "rm -rf /etc"})
        assert valid is False
        assert "Command denied by policy" in msg

    @pytest.mark.asyncio
    async def test_kill_denies_protected_process(self, tool):
        valid, msg = await tool.validate({"action": "kill", "name": "systemd"})
        assert valid is False
        assert "Cannot kill protected process: systemd" in msg

    @pytest.mark.asyncio
    async def test_kill_denies_init(self, tool):
        valid, msg = await tool.validate({"action": "kill", "name": "init"})
        assert valid is False
        assert "Cannot kill protected process" in msg
