"""Tests for rune.tools.registry — ported from registry.test.ts."""


import pytest

from rune.tools.base import Tool
from rune.types import Domain, RiskLevel


class FakeTool(Tool):
    """Minimal concrete Tool for testing."""

    def __init__(self, *, tool_name: str = "fake", domain: Domain = Domain.GENERAL,
                 desc: str = "A fake tool", risk: RiskLevel = RiskLevel.LOW,
                 tool_actions: list[str] | None = None, healthy: bool = True):
        self._name = tool_name
        self._domain = domain
        self._desc = desc
        self._risk = risk
        self._actions = tool_actions or ["action1", "action2"]
        self._healthy = healthy

    @property
    def name(self) -> str:
        return self._name

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def description(self) -> str:
        return self._desc

    @property
    def risk_level(self) -> RiskLevel:
        return self._risk

    @property
    def actions(self) -> list[str]:
        return self._actions

    async def validate(self, params):
        return True, ""

    async def simulate(self, params):
        return self.success()

    async def execute(self, params):
        return self.success()

    async def rollback(self, rollback_data):
        return self.success()

    async def health_check(self) -> bool:
        return self._healthy


class TestToolRegistryConstruction:
    """Tests for ToolRegistry construction with auto_register_defaults=False."""

    def test_starts_empty_without_defaults(self):
        from rune.tools.registry import ToolRegistry
        registry = ToolRegistry(auto_register_defaults=False)
        assert registry.list() == []

    def test_register_makes_tool_retrievable(self):
        from rune.tools.registry import ToolRegistry
        registry = ToolRegistry(auto_register_defaults=False)
        tool = FakeTool(tool_name="custom")
        registry.register(tool)
        assert registry.get("custom") is tool

    def test_register_overwrites_existing_tool(self):
        from rune.tools.registry import ToolRegistry
        registry = ToolRegistry(auto_register_defaults=False)
        tool_a = FakeTool(tool_name="dup", desc="first")
        tool_b = FakeTool(tool_name="dup", desc="second")
        registry.register(tool_a)
        registry.register(tool_b)
        assert registry.get("dup") is tool_b

    def test_unregister_removes_tool(self):
        from rune.tools.registry import ToolRegistry
        registry = ToolRegistry(auto_register_defaults=False)
        tool = FakeTool(tool_name="removable")
        registry.register(tool)
        assert registry.unregister("removable") is True
        assert registry.get("removable") is None

    def test_unregister_returns_false_for_unknown(self):
        from rune.tools.registry import ToolRegistry
        registry = ToolRegistry(auto_register_defaults=False)
        assert registry.unregister("nope") is False

    def test_list_returns_all_tools(self):
        from rune.tools.registry import ToolRegistry
        registry = ToolRegistry(auto_register_defaults=False)
        registry.register(FakeTool(tool_name="a"))
        registry.register(FakeTool(tool_name="b"))
        assert len(registry.list()) == 2

    def test_get_by_domain(self):
        from rune.tools.registry import ToolRegistry
        registry = ToolRegistry(auto_register_defaults=False)
        registry.register(FakeTool(tool_name="f1", domain=Domain.FILE))
        registry.register(FakeTool(tool_name="p1", domain=Domain.PROCESS))
        registry.register(FakeTool(tool_name="f2", domain=Domain.FILE))
        file_tools = registry.get_by_domain("file")
        assert len(file_tools) == 2

    def test_find_by_action(self):
        from rune.tools.registry import ToolRegistry
        registry = ToolRegistry(auto_register_defaults=False)
        registry.register(FakeTool(tool_name="scanner", tool_actions=["scan", "list"]))
        registry.register(FakeTool(tool_name="runner", tool_actions=["run", "kill"]))
        found = registry.find_by_action("scan")
        assert found is not None
        assert found.name == "scanner"

    def test_find_by_action_returns_none_when_not_found(self):
        from rune.tools.registry import ToolRegistry
        registry = ToolRegistry(auto_register_defaults=False)
        registry.register(FakeTool(tool_name="runner", tool_actions=["run"]))
        assert registry.find_by_action("fly") is None

    @pytest.mark.asyncio
    async def test_health_check_returns_dict(self):
        from rune.tools.registry import ToolRegistry
        registry = ToolRegistry(auto_register_defaults=False)
        registry.register(FakeTool(tool_name="ok", healthy=True))
        registry.register(FakeTool(tool_name="bad", healthy=False))
        results = await registry.health_check()
        assert results == {"ok": True, "bad": False}

    def test_list_info_returns_tool_info_objects(self):
        from rune.tools.registry import ToolRegistry
        registry = ToolRegistry(auto_register_defaults=False)
        registry.register(FakeTool(tool_name="info-test"))
        infos = registry.list_info()
        assert len(infos) == 1
        assert infos[0].name == "info-test"
