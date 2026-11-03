"""Opt-in browser state survives a relaunch; it stays off by default and owner-only on disk."""

import asyncio
import stat

import pytest

pytest.importorskip("playwright")


async def _serve():
    async def handle(reader, writer):
        request = await reader.readuntil(b"\r\n\r\n")
        cookie = b"session=" in request
        body = b"<title>%s</title>" % (b"signed-in" if cookie else b"anonymous")
        writer.write(b"HTTP/1.1 200 OK\r\nSet-Cookie: session=abc; Path=/; Max-Age=3600\r\n"
                     b"Content-Type: text/html\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s" % (len(body), body))
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/"


async def _visit(url):
    from rune.capabilities.browser.core import _get_browser
    from rune.capabilities.browser.session import browser_session

    async with browser_session() as session:
        try:
            _browser, page = await _get_browser("managed")
        except Exception as exc:
            # Skip only when the browser binary is missing; any other launch error is a failure.
            if "Executable doesn't exist" not in str(exc) and "playwright install" not in str(exc):
                raise
            pytest.skip("Chromium is not installed")
        await page.goto(url)
        title = await page.title()
        await session.release_resources()
    return title


@pytest.mark.parametrize("enabled", [True, False])
async def test_login_cookie_survives_relaunch_only_when_enabled(enabled, tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    monkeypatch.delenv("RUNE_CLOUD_WORKER", raising=False)
    if enabled:
        monkeypatch.setenv("RUNE_BROWSER_STATE", "1")
    else:
        monkeypatch.delenv("RUNE_BROWSER_STATE", raising=False)
    server, url = await _serve()
    try:
        assert await _visit(url) == "anonymous"
        second = await _visit(url)
    finally:
        server.close()
        await server.wait_closed()
    saved = tmp_path / "browser" / "storage-state.json"
    if enabled:
        assert second == "signed-in"
        assert stat.S_IMODE(saved.stat().st_mode) == 0o600
    else:
        assert second == "anonymous" and not saved.exists()


def test_unreadable_state_is_set_aside_not_deleted(tmp_path, monkeypatch):
    from rune.capabilities.browser.persistence import context_options

    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    monkeypatch.setenv("RUNE_BROWSER_STATE", "1")
    saved = tmp_path / "browser" / "storage-state.json"
    saved.parent.mkdir()
    saved.write_text("{not json")
    assert context_options() == {}
    assert (tmp_path / "browser" / "storage-state.json.unreadable").read_text() == "{not json"


@pytest.mark.parametrize("state", ['{"cookies": [1]}', '{"cookies": [{"name": "a"}]}', '{"origins": "x"}'])
async def test_state_playwright_rejects_is_set_aside_and_browser_still_opens(state, tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    monkeypatch.setenv("RUNE_BROWSER_STATE", "1")
    saved = tmp_path / "browser" / "storage-state.json"
    saved.parent.mkdir()
    saved.write_text(state)
    server, url = await _serve()
    try:
        assert await _visit(url) == "anonymous"
    finally:
        server.close()
        await server.wait_closed()
    assert (tmp_path / "browser" / "storage-state.json.unreadable").read_text() == state
    assert not list((tmp_path / "browser").glob("*.partial"))
