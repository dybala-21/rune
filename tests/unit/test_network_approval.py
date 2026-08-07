"""Network approval gate: writes need sign-off, reads don't, bypass is honored."""

import pytest

from rune.agent.tool_adapter import _network_approval_request


@pytest.fixture
def approval_cfg(monkeypatch):
    """Drive the gate through the real config object."""
    from rune.config.loader import get_config

    # Writes are only gated when they can actually happen (same flag the
    # executor checks), so the write tests need the path enabled.
    monkeypatch.setenv("RUNE_HYBRID_API", "1")
    cfg = get_config().approval
    monkeypatch.setattr(cfg, "network_approval", "writes", raising=False)
    monkeypatch.setattr(cfg, "profile", "general", raising=False)
    return cfg


def test_post_fetch_requires_approval(approval_cfg):
    req = _network_approval_request(
        "web_fetch",
        {"url": "https://shop.example.com/api/order.do", "method": "POST", "body": "id=1"},
    )
    assert req is not None
    display, reason, key = req.display, req.reason, req.cache_key
    assert display.startswith("POST https://shop.example.com")
    assert "cannot be undone" in reason
    assert key == "POST|shop.example.com"
    assert req.is_write is True


def test_get_fetch_and_reads_are_not_gated_by_default(approval_cfg):
    assert _network_approval_request(
        "web_fetch", {"url": "https://example.com/page"}
    ) is None
    assert _network_approval_request("web_search", {"query": "showtimes"}) is None
    assert _network_approval_request(
        "browser_navigate", {"url": "https://example.com"}
    ) is None
    assert _network_approval_request(
        "browser_act", {"action": "click", "selector": "e5"}
    ) is None


def test_bypass_modes_skip_every_gate(approval_cfg, monkeypatch):
    post = {"url": "https://x.test/a.do", "method": "POST"}

    monkeypatch.setattr(approval_cfg, "network_approval", "off", raising=False)
    assert _network_approval_request("web_fetch", post) is None

    # Automation profile is the headless bypass and overrides the mode.
    monkeypatch.setattr(approval_cfg, "network_approval", "all", raising=False)
    monkeypatch.setattr(approval_cfg, "profile", "automation", raising=False)
    assert _network_approval_request("web_fetch", post) is None
    assert _network_approval_request("web_search", {"query": "q"}) is None


def test_all_mode_gates_reads_and_clicks(approval_cfg, monkeypatch):
    monkeypatch.setattr(approval_cfg, "network_approval", "all", raising=False)

    get_req = _network_approval_request("web_fetch", {"url": "https://example.com/p"})
    assert get_req is not None and get_req.cache_key == "GET|example.com"

    search_req = _network_approval_request("web_search", {"query": "showtimes"})
    assert search_req is not None and "showtimes" in search_req.display

    click_req = _network_approval_request(
        "browser_act", {"action": "click", "selector": "e5"}
    )
    assert click_req is not None and "click" in click_req.display


def test_unknown_capabilities_are_never_gated(approval_cfg, monkeypatch):
    for mode in ("writes", "all"):
        monkeypatch.setattr(approval_cfg, "network_approval", mode, raising=False)
        assert _network_approval_request("file_read", {"path": "a.py"}) is None
        assert _network_approval_request("bash_execute", {"command": "ls"}) is None


def test_config_failure_keeps_the_write_gate(monkeypatch):
    # Fail-closed: an unreadable config must not silently open the gate.
    import rune.config.loader as loader

    monkeypatch.setenv("RUNE_HYBRID_API", "1")

    def _boom():
        raise RuntimeError("config unavailable")

    monkeypatch.setattr(loader, "get_config", _boom)
    req = _network_approval_request(
        "web_fetch", {"url": "https://x.test/a.do", "method": "POST"}
    )
    assert req is not None


def test_denied_post_is_blocked_and_approved_host_is_cached(approval_cfg):
    """End-to-end through the tool wrapper: deny blocks, approve caches by host."""
    import asyncio

    from rune.agent.tool_adapter import ToolAdapterOptions, build_tool_set

    asked: list[tuple[str, str]] = []
    verdict = {"approve": False}

    async def _cb(command: str, reason: str) -> bool:
        asked.append((command, reason))
        return verdict["approve"]

    tools = build_tool_set(
        ToolAdapterOptions(approval_callback=_cb, allowed_tools={"web_fetch"})
    )
    fetch = tools["web_fetch"].function

    # Denied → the capability never runs, and the model is told not to retry.
    out = asyncio.run(fetch(url="https://x.test/order.do", method="POST"))
    assert "User declined" in out and "Do NOT retry" in out
    assert len(asked) == 1

    # Approved once → a second POST to the same host doesn't re-prompt.
    verdict["approve"] = True
    asyncio.run(fetch(url="https://x.test/order.do", method="POST"))
    assert len(asked) == 2
    asyncio.run(fetch(url="https://x.test/other.do", method="POST"))
    assert len(asked) == 2, "same host should reuse the granted approval"

    # A different host is a fresh decision.
    asyncio.run(fetch(url="https://other.test/a.do", method="POST"))
    assert len(asked) == 3

    # GET never prompts.
    asyncio.run(fetch(url="https://third.test/page"))
    assert len(asked) == 3


def test_write_without_approval_channel_fails_closed(approval_cfg):
    """No channel to ask on: a read proceeds, an irreversible write does not."""
    import asyncio

    from rune.agent.tool_adapter import ToolAdapterOptions, build_tool_set

    tools = build_tool_set(ToolAdapterOptions(allowed_tools={"web_fetch"}))
    fetch = tools["web_fetch"].function

    out = asyncio.run(fetch(url="https://x.test/order.do", method="POST"))
    assert "no approval channel" in out

    # A GET is not a hazard — it still runs (and fails on DNS, not on the gate).
    read = asyncio.run(fetch(url="https://x.test/page"))
    assert "no approval channel" not in read


def test_approval_decisions_from_the_ui_are_honored():
    """The card offers approve_once/approve_always; both must mean yes.

    Regression: the server compared against a plain "approve" that no client
    ever sends, so every web-UI approval resolved as a denial.
    """
    from rune.api.server import approval_granted

    assert approval_granted({"decision": "approve_once"}) is True
    assert approval_granted({"decision": "approve_always"}) is True
    assert approval_granted({"decision": "approve"}) is True
    assert approval_granted({"decision": "deny"}) is False
    assert approval_granted({}) is False
    assert approval_granted(None) is False



def test_write_gate_is_silent_when_writes_are_disabled(approval_cfg, monkeypatch):
    """No sign-off for a call the executor will refuse anyway."""
    monkeypatch.setenv("RUNE_HYBRID_API", "0")
    assert _network_approval_request(
        "web_fetch", {"url": "https://x.test/a.do", "method": "POST"}
    ) is None


def test_post_is_refused_rather_than_downgraded_to_get(monkeypatch):
    """A disabled write must fail loudly, never silently become a read."""
    import asyncio

    from rune.capabilities.web import WebFetchParams, web_fetch

    monkeypatch.setenv("RUNE_HYBRID_API", "0")
    res = asyncio.run(web_fetch(WebFetchParams(
        url="https://write-disabled.test/order.do", method="POST", body="a=1",
    )))
    assert res.success is False
    assert "disabled" in (res.error or "")
