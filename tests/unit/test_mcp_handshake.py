"""The MCP initialize handshake, proven against a real stdio subprocess.

The client spoke tools/list before any initialize, which a conformant
server rejects or hangs on — so "connect any MCP server" did not hold. The
whole mock-based MCP suite could not catch it because nothing spoke real
JSON-RPC to a process. This does: a tiny stdio server that refuses to list
tools until it has seen initialize + notifications/initialized.
"""
from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path

import pytest

from rune.mcp.client import MCPClient
from rune.mcp.config import MCPServerConfig

# A minimal, spec-shaped stdio MCP server. It records whether the handshake
# happened and fails tools/list otherwise — the exact behaviour the missing
# handshake would trip over.
_SERVER = textwrap.dedent('''
    import json, sys
    initialized = False
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        method = msg.get("method")
        mid = msg.get("id")
        if method == "initialize":
            sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":mid,
                "result":{"protocolVersion":"2024-11-05","capabilities":{},
                "serverInfo":{"name":"fake","version":"1"}}})+"\\n")
            sys.stdout.flush()
        elif method == "notifications/initialized":
            initialized = True                       # notification: no reply
        elif method == "tools/list":
            if not initialized:
                sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":mid,
                    "error":{"code":-32002,"message":"not initialized"}})+"\\n")
            else:
                sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":mid,
                    "result":{"tools":[{"name":"ping","description":"p",
                    "inputSchema":{"type":"object"}}]}})+"\\n")
            sys.stdout.flush()
''')


@pytest.fixture
def server_path(tmp_path) -> Path:
    p = tmp_path / "fake_mcp_server.py"
    p.write_text(_SERVER)
    return p


@pytest.mark.asyncio
async def test_connect_handshakes_then_lists_tools(server_path):
    cfg = MCPServerConfig(name="fake", transport="stdio",
                          command=sys.executable, args=[str(server_path)])
    client = MCPClient("fake", cfg)
    try:
        await client.connect(timeout=10)
        assert client.connected
        tools = await client.list_tools()
        # tools/list only succeeds because initialize + initialized ran
        # first; without the handshake the server returns -32002 and this
        # raises instead.
        assert [t.name for t in tools] == ["ping"]
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_cancelled_handshake_reaps_its_child(tmp_path, monkeypatch):
    from rune.mcp import client as module

    server = tmp_path / "held.py"
    server.write_text("import time\ntime.sleep(60)\n")
    cfg = MCPServerConfig(name="held", transport="stdio", command=sys.executable, args=[str(server)])
    client = MCPClient("held", cfg)
    manager = module.MCPClientManager()
    monkeypatch.setattr(module, "MCPClient", lambda *_: client)
    task = asyncio.create_task(manager.connect("held", cfg))
    try:
        async with asyncio.timeout(2):
            while client._process is None:
                await asyncio.sleep(.01)
        process = client._process
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert process.returncode is not None
        assert not manager._connecting and not manager._servers
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await client.disconnect()
