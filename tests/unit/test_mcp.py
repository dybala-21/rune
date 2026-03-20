"""Tests for MCP client manager, bridge, and config modules."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import TOOL_GROUPS
from rune.mcp.bridge import (
    _ensure_mcp_group,
    _mcp_tool_to_capability,
    connect_single_server,
    get_mcp_status,
    initialize_mcp_bridge,
    shutdown_mcp_bridge,
)
from rune.mcp.client import (
    ConnectedServer,
    MCPClient,
    MCPClientManager,
    MCPTool,
    MCPToolResult,
    filter_sensitive_env,
)
from rune.mcp.config import MCPServerConfig, load_mcp_config, save_mcp_config
from rune.types import CapabilityResult, Domain, RiskLevel

# ============================================================================
# Config tests
# ============================================================================


class TestMCPServerConfig:
    def test_defaults(self):
        cfg = MCPServerConfig(name="test")
        assert cfg.name == "test"
        assert cfg.command is None
        assert cfg.args == []
        assert cfg.env == {}
        assert cfg.transport == "stdio"
        assert cfg.url is None
        assert cfg.headers == {}
        assert cfg.disabled is False

    def test_streamable_http_config(self):
        cfg = MCPServerConfig(
            name="remote",
            transport="streamable-http",
            url="https://example.com/mcp",
            headers={"Authorization": "Bearer tok"},
        )
        assert cfg.transport == "streamable-http"
        assert cfg.url == "https://example.com/mcp"
        assert cfg.headers["Authorization"] == "Bearer tok"

    def test_disabled_config(self):
        cfg = MCPServerConfig(name="off", disabled=True)
        assert cfg.disabled is True


class TestLoadSaveConfig:
    def test_load_empty(self, tmp_path):
        with patch("rune.mcp.config._config_path", return_value=tmp_path / "nope.json"):
            result = load_mcp_config()
        assert result == {}

    def test_round_trip(self, tmp_path):
        cfg_path = tmp_path / "mcp_servers.json"
        with patch("rune.mcp.config._config_path", return_value=cfg_path):
            configs = {
                "s1": MCPServerConfig(name="s1", command="node", args=["server.js"]),
                "s2": MCPServerConfig(
                    name="s2",
                    transport="streamable-http",
                    url="https://example.com/mcp",
                ),
            }
            save_mcp_config(configs)
            loaded = load_mcp_config()

        assert "s1" in loaded
        assert loaded["s1"].command == "node"
        assert loaded["s1"].args == ["server.js"]
        assert "s2" in loaded
        assert loaded["s2"].transport == "streamable-http"
        assert loaded["s2"].url == "https://example.com/mcp"

    def test_load_legacy_format(self, tmp_path):
        cfg_path = tmp_path / "mcp_servers.json"
        cfg_path.write_text(json.dumps({
            "servers": [
                {"name": "legacy", "command": "python", "args": ["-m", "server"]}
            ]
        }))
        with patch("rune.mcp.config._config_path", return_value=cfg_path):
            loaded = load_mcp_config()
        assert "legacy" in loaded
        assert loaded["legacy"].command == "python"


# ============================================================================
# Sensitive env filtering
# ============================================================================


class TestFilterSensitiveEnv:
    def test_filters_api_keys(self):
        test_env = {
            "HOME": "/home/user",
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": "sk-xxx",
            "ANTHROPIC_API_KEY": "ak-xxx",
            "MY_SECRET": "hidden",
            "DATABASE_URL": "postgres://...",
            "GITHUB_TOKEN": "ghp_xxx",
            "NORMAL_VAR": "visible",
            "MY_PASSWORD": "hunter2",
        }
        with patch.dict("os.environ", test_env, clear=True):
            filtered = filter_sensitive_env()

        assert "HOME" in filtered
        assert "PATH" in filtered
        assert "NORMAL_VAR" in filtered
        assert "OPENAI_API_KEY" not in filtered
        assert "ANTHROPIC_API_KEY" not in filtered
        assert "MY_SECRET" not in filtered
        assert "DATABASE_URL" not in filtered
        assert "GITHUB_TOKEN" not in filtered
        assert "MY_PASSWORD" not in filtered


# ============================================================================
# MCPTool / MCPToolResult
# ============================================================================


class TestMCPTypes:
    def test_mcp_tool(self):
        tool = MCPTool(
            name="search",
            description="Search stuff",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        assert tool.name == "search"
        assert tool.description == "Search stuff"

    def test_mcp_tool_result(self):
        result = MCPToolResult(
            content=[{"type": "text", "text": "hello"}],
            is_error=False,
        )
        assert not result.is_error
        assert result.content[0]["text"] == "hello"

    def test_mcp_tool_result_error(self):
        result = MCPToolResult(content=[], is_error=True)
        assert result.is_error


# ============================================================================
# MCPClientManager
# ============================================================================


class TestMCPClientManager:
    @pytest.fixture
    def manager(self):
        return MCPClientManager(close_timeout=1.0)

    def test_initial_state(self, manager: MCPClientManager):
        assert manager.get_connected_count() == 0
        assert manager.get_all_tools() == []
        assert manager.get_status() == []
        assert manager.find_server_for_tool("anything") is None

    @pytest.mark.asyncio
    async def test_connect_disabled_server(self, manager: MCPClientManager):
        cfg = MCPServerConfig(name="off", disabled=True)
        tools = await manager.connect("off", cfg)
        assert tools == []
        assert manager.get_connected_count() == 0

    @pytest.mark.asyncio
    async def test_connect_caches_result(self, manager: MCPClientManager):
        """After a successful connect, calling connect again returns cached tools."""
        cfg = MCPServerConfig(name="test", command="echo")
        fake_tools = [MCPTool(name="t1", description="", input_schema={})]

        with patch.object(
            manager,
            "_connect_internal",
            new_callable=AsyncMock,
            return_value=fake_tools,
        ) as mock_connect:
            # First call triggers _connect_internal
            tools1 = await manager.connect("test", cfg)
            assert tools1 == fake_tools
            mock_connect.assert_awaited_once()

        # Manually mark as connected (since we mocked the internal method,
        # the server entry was set by _connect_internal's side effects)
        # We need to set up the server entry manually
        manager._servers["test"] = ConnectedServer(
            name="test",
            config=cfg,
            client=MagicMock(),
            tools=fake_tools,
            connected=True,
        )

        # Second call returns cached
        tools2 = await manager.connect("test", cfg)
        assert tools2 == fake_tools

    @pytest.mark.asyncio
    async def test_connect_deduplication(self, manager: MCPClientManager):
        """Concurrent connect calls for the same server share one task."""
        cfg = MCPServerConfig(name="dedup", command="echo")
        call_count = 0

        async def slow_connect(name, config, timeout):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            tools = [MCPTool(name="t1", description="", input_schema={})]
            manager._servers[name] = ConnectedServer(
                name=name, config=config, client=MagicMock(),
                tools=tools, connected=True,
            )
            return tools

        with patch.object(manager, "_connect_internal", side_effect=slow_connect):
            results = await asyncio.gather(
                manager.connect("dedup", cfg),
                manager.connect("dedup", cfg),
                manager.connect("dedup", cfg),
            )

        # All three should get the same tools
        for tools in results:
            assert len(tools) == 1
            assert tools[0].name == "t1"

        # But _connect_internal was only called once
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_connect_all(self, manager: MCPClientManager):
        """connect_all connects multiple servers in parallel."""
        configs = {
            "a": MCPServerConfig(name="a", command="echo"),
            "b": MCPServerConfig(name="b", command="echo"),
            "c": MCPServerConfig(name="c", disabled=True),
        }

        async def fake_connect(name, config, timeout=None):
            if config.disabled:
                return []
            tool = MCPTool(name=f"{name}_tool", description="", input_schema={})
            manager._servers[name] = ConnectedServer(
                name=name, config=config, client=MagicMock(),
                tools=[tool], connected=True,
            )
            return [tool]

        with patch.object(manager, "_connect_internal", side_effect=fake_connect):
            results = await manager.connect_all(configs)

        assert "a" in results
        assert len(results["a"]) == 1
        assert "b" in results
        assert len(results["b"]) == 1
        assert "c" in results
        assert results["c"] == []

    @pytest.mark.asyncio
    async def test_disconnect(self, manager: MCPClientManager):
        cfg = MCPServerConfig(name="test", command="echo")
        mock_client = MagicMock()
        mock_client.disconnect = AsyncMock()

        manager._servers["test"] = ConnectedServer(
            name="test",
            config=cfg,
            client=mock_client,
            tools=[],
            connected=True,
        )
        assert manager.get_connected_count() == 1

        await manager.disconnect("test")
        assert manager.get_connected_count() == 0
        mock_client.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_all(self, manager: MCPClientManager):
        for name in ("a", "b"):
            cfg = MCPServerConfig(name=name, command="echo")
            mock_client = MagicMock()
            mock_client.disconnect = AsyncMock()
            manager._servers[name] = ConnectedServer(
                name=name, config=cfg, client=mock_client,
                tools=[], connected=True,
            )

        assert manager.get_connected_count() == 2
        await manager.disconnect_all()
        assert manager.get_connected_count() == 0

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self, manager: MCPClientManager):
        with pytest.raises(ConnectionError, match="not connected"):
            await manager.call_tool("nope", "tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_success(self, manager: MCPClientManager):
        cfg = MCPServerConfig(name="srv", command="echo")
        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(
            return_value=MCPToolResult(
                content=[{"type": "text", "text": "result"}],
                is_error=False,
            )
        )
        manager._servers["srv"] = ConnectedServer(
            name="srv", config=cfg, client=mock_client,
            tools=[], connected=True,
        )

        result = await manager.call_tool("srv", "my_tool", {"x": 1})
        assert result.content[0]["text"] == "result"
        assert not result.is_error
        mock_client.call_tool.assert_awaited_once_with("my_tool", {"x": 1})

    def test_list_tools(self, manager: MCPClientManager):
        tools = [MCPTool(name="t1", description="", input_schema={})]
        cfg = MCPServerConfig(name="srv", command="echo")
        manager._servers["srv"] = ConnectedServer(
            name="srv", config=cfg, client=MagicMock(),
            tools=tools, connected=True,
        )
        assert manager.list_tools("srv") == tools
        assert manager.list_tools("nonexistent") == []

    def test_get_all_tools(self, manager: MCPClientManager):
        tools_a = [MCPTool(name="a1", description="", input_schema={})]
        tools_b = [MCPTool(name="b1", description="", input_schema={})]
        for name, tools in [("a", tools_a), ("b", tools_b)]:
            cfg = MCPServerConfig(name=name, command="echo")
            manager._servers[name] = ConnectedServer(
                name=name, config=cfg, client=MagicMock(),
                tools=tools, connected=True,
            )

        all_tools = manager.get_all_tools()
        assert len(all_tools) == 2
        names = {t.name for _, t in all_tools}
        assert names == {"a1", "b1"}

    def test_find_server_for_tool(self, manager: MCPClientManager):
        tools = [MCPTool(name="search", description="", input_schema={})]
        cfg = MCPServerConfig(name="srv", command="echo")
        manager._servers["srv"] = ConnectedServer(
            name="srv", config=cfg, client=MagicMock(),
            tools=tools, connected=True,
        )
        assert manager.find_server_for_tool("search") == "srv"
        assert manager.find_server_for_tool("nonexistent") is None

    def test_get_status(self, manager: MCPClientManager):
        cfg = MCPServerConfig(name="srv", command="echo")
        manager._servers["srv"] = ConnectedServer(
            name="srv", config=cfg, client=MagicMock(),
            tools=[MCPTool(name="t1", description="", input_schema={})],
            connected=True,
        )
        status = manager.get_status()
        assert len(status) == 1
        assert status[0]["name"] == "srv"
        assert status[0]["connected"] is True
        assert status[0]["tool_count"] == 1


# ============================================================================
# Bridge tests
# ============================================================================


class TestMCPToolToCapability:
    def test_basic_conversion(self):
        tool = MCPTool(
            name="search",
            description="Search the web",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        manager = MagicMock()
        cap = _mcp_tool_to_capability("brave", tool, manager)

        assert cap.name == "mcp.brave.search"
        assert cap.description == "Search the web"
        assert cap.risk_level == RiskLevel.MEDIUM
        assert cap.domain == Domain.NETWORK
        assert cap.group == "mcp"
        assert cap.execute is not None

    def test_missing_description_fallback(self):
        tool = MCPTool(name="run", description="", input_schema={})
        manager = MagicMock()
        cap = _mcp_tool_to_capability("server", tool, manager)
        assert cap.description == "[MCP:server] run"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        tool = MCPTool(name="ping", description="Ping", input_schema={})
        manager = MagicMock()
        manager.call_tool = AsyncMock(
            return_value=MCPToolResult(
                content=[{"type": "text", "text": "pong"}],
                is_error=False,
            )
        )
        cap = _mcp_tool_to_capability("srv", tool, manager)
        result = await cap.execute({"host": "example.com"})

        assert isinstance(result, CapabilityResult)
        assert result.success is True
        assert result.output == "pong"
        assert result.metadata["mcp_server"] == "srv"
        assert result.metadata["mcp_tool"] == "ping"

    @pytest.mark.asyncio
    async def test_execute_error_from_server(self):
        tool = MCPTool(name="fail", description="", input_schema={})
        manager = MagicMock()
        manager.call_tool = AsyncMock(
            return_value=MCPToolResult(
                content=[{"type": "text", "text": "bad input"}],
                is_error=True,
            )
        )
        cap = _mcp_tool_to_capability("srv", tool, manager)
        result = await cap.execute({})

        assert result.success is False
        assert "MCP tool error" in (result.error or "")

    @pytest.mark.asyncio
    async def test_execute_exception(self):
        tool = MCPTool(name="boom", description="", input_schema={})
        manager = MagicMock()
        manager.call_tool = AsyncMock(side_effect=ConnectionError("offline"))
        cap = _mcp_tool_to_capability("srv", tool, manager)
        result = await cap.execute({})

        assert result.success is False
        assert "offline" in (result.error or "")


class TestEnsureMCPGroup:
    def test_adds_to_group(self):
        # Clean up after test
        original = TOOL_GROUPS.get("mcp", []).copy()
        try:
            _ensure_mcp_group("mcp.test.tool1")
            assert "mcp.test.tool1" in TOOL_GROUPS["mcp"]
            # Duplicate add is idempotent
            _ensure_mcp_group("mcp.test.tool1")
            assert TOOL_GROUPS["mcp"].count("mcp.test.tool1") == 1
        finally:
            if original:
                TOOL_GROUPS["mcp"] = original
            else:
                TOOL_GROUPS.pop("mcp", None)


class TestInitializeMCPBridge:
    @pytest.mark.asyncio
    async def test_no_configs(self):
        registry = CapabilityRegistry()
        result = await initialize_mcp_bridge(configs={}, registry=registry)
        assert result.registered_count == 0
        assert result.connected_servers == 0

    @pytest.mark.asyncio
    async def test_all_disabled(self):
        registry = CapabilityRegistry()
        configs = {
            "off1": MCPServerConfig(name="off1", disabled=True),
            "off2": MCPServerConfig(name="off2", disabled=True),
        }
        result = await initialize_mcp_bridge(configs=configs, registry=registry)
        assert result.registered_count == 0

    @pytest.mark.asyncio
    async def test_successful_bridge(self):
        registry = CapabilityRegistry()
        configs = {
            "brave": MCPServerConfig(name="brave", command="node", args=["brave-server.js"]),
        }

        fake_tools = [
            MCPTool(name="search", description="Web search", input_schema={}),
            MCPTool(name="news", description="News search", input_schema={}),
        ]

        with patch("rune.mcp.bridge.get_mcp_client_manager") as mock_get:
            mock_manager = MagicMock()
            mock_manager.connect_all = AsyncMock(return_value={"brave": fake_tools})
            mock_manager.get_connected_count.return_value = 1
            mock_get.return_value = mock_manager

            # Clean up TOOL_GROUPS after test
            original_mcp = TOOL_GROUPS.get("mcp", []).copy()
            try:
                result = await initialize_mcp_bridge(configs=configs, registry=registry)
            finally:
                if original_mcp:
                    TOOL_GROUPS["mcp"] = original_mcp
                else:
                    TOOL_GROUPS.pop("mcp", None)

        assert result.registered_count == 2
        assert result.connected_servers == 1
        assert result.failed_servers == []
        assert "mcp.brave.search" in result.capabilities
        assert "mcp.brave.news" in result.capabilities

        # Check registry
        assert registry.get("mcp.brave.search") is not None
        assert registry.get("mcp.brave.news") is not None

    @pytest.mark.asyncio
    async def test_failed_server(self):
        registry = CapabilityRegistry()
        configs = {
            "good": MCPServerConfig(name="good", command="echo"),
            "bad": MCPServerConfig(name="bad", command="nonexistent"),
        }

        with patch("rune.mcp.bridge.get_mcp_client_manager") as mock_get:
            mock_manager = MagicMock()
            mock_manager.connect_all = AsyncMock(return_value={
                "good": [MCPTool(name="t1", description="", input_schema={})],
                "bad": [],
            })
            mock_manager.get_connected_count.return_value = 1
            mock_get.return_value = mock_manager

            original_mcp = TOOL_GROUPS.get("mcp", []).copy()
            try:
                result = await initialize_mcp_bridge(configs=configs, registry=registry)
            finally:
                if original_mcp:
                    TOOL_GROUPS["mcp"] = original_mcp
                else:
                    TOOL_GROUPS.pop("mcp", None)

        assert result.registered_count == 1
        assert "bad" in result.failed_servers
        assert "mcp.good.t1" in result.capabilities


class TestConnectSingleServer:
    @pytest.mark.asyncio
    async def test_success(self):
        registry = CapabilityRegistry()
        config = MCPServerConfig(name="new_srv", command="echo")

        with patch("rune.mcp.bridge.get_mcp_client_manager") as mock_get:
            mock_manager = MagicMock()
            mock_manager.connect = AsyncMock(
                return_value=[MCPTool(name="do_thing", description="Does a thing", input_schema={})]
            )
            mock_get.return_value = mock_manager

            original_mcp = TOOL_GROUPS.get("mcp", []).copy()
            try:
                result = await connect_single_server("new_srv", config, registry=registry)
            finally:
                if original_mcp:
                    TOOL_GROUPS["mcp"] = original_mcp
                else:
                    TOOL_GROUPS.pop("mcp", None)

        assert result.registered_count == 1
        assert result.connected_servers == 1
        assert "mcp.new_srv.do_thing" in result.capabilities

    @pytest.mark.asyncio
    async def test_failure(self):
        registry = CapabilityRegistry()
        config = MCPServerConfig(name="bad", command="nope")

        with patch("rune.mcp.bridge.get_mcp_client_manager") as mock_get:
            mock_manager = MagicMock()
            mock_manager.connect = AsyncMock(return_value=[])
            mock_get.return_value = mock_manager

            result = await connect_single_server("bad", config, registry=registry)

        assert result.registered_count == 0
        assert "bad" in result.failed_servers


class TestShutdownMCPBridge:
    @pytest.mark.asyncio
    async def test_shutdown(self):
        with patch("rune.mcp.bridge.get_mcp_client_manager") as mock_get:
            mock_manager = MagicMock()
            mock_manager.disconnect_all = AsyncMock()
            mock_get.return_value = mock_manager

            await shutdown_mcp_bridge()
            mock_manager.disconnect_all.assert_awaited_once()


class TestGetMCPStatus:
    def test_status(self):
        with patch("rune.mcp.bridge.get_mcp_client_manager") as mock_get:
            mock_manager = MagicMock()
            mock_manager.get_status.return_value = [
                {"name": "a", "connected": True, "tool_count": 3},
                {"name": "b", "connected": False, "tool_count": 0},
            ]
            mock_get.return_value = mock_manager

            status = get_mcp_status()

        assert len(status["servers"]) == 2
        assert status["total_tools"] == 3


# ============================================================================
# MCPClient transport tests
# ============================================================================


class TestMCPClientStreamableHTTP:
    """Test streamable-http transport parsing logic."""

    @pytest.mark.asyncio
    async def test_connect_requires_url(self):
        cfg = MCPServerConfig(name="nourl", transport="streamable-http")
        client = MCPClient("nourl", cfg)
        with pytest.raises(ValueError, match="missing url"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_connect_requires_httpx(self):
        cfg = MCPServerConfig(
            name="http", transport="streamable-http", url="https://example.com/mcp"
        )
        client = MCPClient("http", cfg)
        # This will either succeed (httpx installed) or raise ImportError
        # In test env httpx should be available
        try:
            await client.connect()
            assert client.connected
        except ImportError:
            pytest.skip("httpx not installed")
        finally:
            await client.disconnect()


class TestMCPClientSSE:
    @pytest.mark.asyncio
    async def test_connect_requires_url(self):
        cfg = MCPServerConfig(name="nourl", transport="sse")
        client = MCPClient("nourl", cfg)
        with pytest.raises(ValueError, match="missing url"):
            await client.connect()
