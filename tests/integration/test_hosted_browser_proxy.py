"""A hosted browser really reaches the web only through the forced egress proxy."""

import asyncio

import pytest

pytest.importorskip("playwright")


async def test_hosted_browser_uses_egress_proxy(monkeypatch):
    seen = []

    async def proxy(reader, writer):
        request = await reader.readuntil(b"\r\n\r\n")
        seen.append(request.split(b"\r\n", 1)[0].decode())
        body = b"<html><title>via-proxy</title></html>"
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s"
                     % (len(body), body))
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(proxy, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setenv("RUNE_CLOUD_WORKER", "1")
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{port}")
    for name in ("HTTPS_PROXY", "https_proxy", "http_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)

    from rune.capabilities.browser.core import _get_browser
    from rune.capabilities.browser.session import browser_session

    try:
        async with browser_session() as session:
            try:
                _browser, page = await _get_browser("managed")
            except Exception as exc:
                # Skip only when the browser binary is missing; any other launch error is a failure.
                if "Executable doesn't exist" not in str(exc) and "playwright install" not in str(exc):
                    raise
                pytest.skip("Chromium is not installed")
            await page.goto("http://hosted-egress-check.invalid/")
            assert await page.title() == "via-proxy"
            await session.release_resources()
    finally:
        server.close()
        await server.wait_closed()
    assert seen and seen[0].startswith("GET http://hosted-egress-check.invalid/")
