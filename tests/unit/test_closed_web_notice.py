"""A hosted machine whose web is closed says why when a network tool fails, instead of a bare error."""

from unittest.mock import AsyncMock

import pytest

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.connectors import web_access
from rune.connectors.store import authority_key
from rune.types import CapabilityResult

CLOSED = {"closed": True, "untrusted": True, "open_until": None, "approval_url": "https://approve.example.com/"}


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_BROKER_HOME", str(tmp_path / "broker"))
    monkeypatch.setenv("RUNE_CONNECTOR_URL", "http://10.207.0.1:4100")
    authority_key(tmp_path / "broker", create=True)
    monkeypatch.setattr(web_access, "_cache", (0.0, None))
    broker = AsyncMock(return_value=CLOSED)
    monkeypatch.setattr("rune.connectors.client.call_broker", broker)
    return broker


def registry(*tools):
    reg = CapabilityRegistry()
    for name, result in tools:
        reg.register(CapabilityDefinition(name=name, description=name, execute=AsyncMock(return_value=result)))
    return reg


async def test_a_failed_fetch_says_the_web_is_closed_and_where_to_open_it(hosted):
    reg = registry(("web_fetch", CapabilityResult(success=False, error="Connection closed")))
    result = await reg.execute("web_fetch", {})
    assert result.error.startswith("Connection closed")
    assert "closed to this machine" in result.error and "https://approve.example.com/" in result.error
    assert result.metadata["web_access"] == "closed"
    hosted.assert_awaited_once_with("status", {})


async def test_the_broker_is_asked_once_for_a_burst_of_failures(hosted):
    reg = registry(("browser_navigate", CapabilityResult(success=False, error="net::ERR_CONNECTION_CLOSED")),
                   ("bash_execute", CapabilityResult(success=False, error="curl: (35) SSL connect error")))
    for name in ("browser_navigate", "bash_execute", "browser_navigate"):
        assert (await reg.execute(name, {})).metadata["web_access"] == "closed"
    assert hosted.await_count == 1


async def test_open_web_other_tools_and_successes_are_left_alone(hosted):
    reg = registry(("web_fetch", CapabilityResult(success=True, output="ok")),
                   ("file_read", CapabilityResult(success=False, error="No such file")))
    assert (await reg.execute("web_fetch", {})).output == "ok"
    assert (await reg.execute("file_read", {})).error == "No such file"
    hosted.assert_not_awaited()
    hosted.return_value = {"closed": False}
    failing = registry(("web_fetch", CapabilityResult(success=False, error="Timed out")))
    assert (await failing.execute("web_fetch", {})).error == "Timed out"


async def test_an_older_broker_without_status_leaves_the_error_as_it_was(hosted):
    hosted.side_effect = RuntimeError("404")
    reg = registry(("web_fetch", CapabilityResult(success=False, error="Connection closed")))
    result = await reg.execute("web_fetch", {})
    assert result.error == "Connection closed" and "web_access" not in result.metadata


async def test_a_local_install_never_asks(monkeypatch, tmp_path):
    monkeypatch.delenv("RUNE_CONNECTOR_URL", raising=False)
    monkeypatch.setattr(web_access, "_cache", (0.0, None))
    broker = AsyncMock()
    monkeypatch.setattr("rune.connectors.client.call_broker", broker)
    reg = registry(("web_fetch", CapabilityResult(success=False, error="Connection closed")))
    assert (await reg.execute("web_fetch", {})).error == "Connection closed"
    broker.assert_not_awaited()
