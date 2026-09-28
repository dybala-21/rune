"""Exercise the desktop tab through the same API and tools used by a live run."""

import asyncio
import os
import socket
import struct
import subprocess
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response

from rune.api.computer import Computers, computer_router
from rune.capabilities.browser.capabilities import (
    BrowserActParams,
    BrowserObserveParams,
    browser_act,
    browser_observe,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.skipif(
    os.environ.get("RUNE_NATIVE_BROWSER_TESTS") != "1", reason="Requires the desktop Electron runtime",
)]


async def test_embedded_tab_shares_agent_and_human_state(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    computers = Computers()
    app = FastAPI()
    app.include_router(computer_router(computers, lambda: None))

    @app.get("/fixture")
    async def fixture():
        return HTMLResponse("""<title>Shared browser test</title>
        <input aria-label="Destination"><button id="save" onclick="window.saves=(window.saves||0)+1;this.textContent='Saved '+window.saves">Save</button>
        <a href="/fixture?next=1">Next</a><div style="height:2000px">Scroll area</div>""")

    @app.get("/download")
    async def download():
        return Response(b"shared browser download", media_type="application/octet-stream",
                        headers={"Content-Disposition": 'attachment; filename="result.txt"'})

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(.02)
    root = Path(__file__).resolve().parents[2]
    profile = tmp_path / "profile"
    profile.mkdir()
    process = subprocess.Popen(
        [str(root / "desktop/node_modules/.bin/electron"), str(root / "tests/fixtures/browser_host.cjs")],
        env={**os.environ, "RUNE_TEST_PROFILE": str(profile), "RUNE_TEST_ORIGIN": origin},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(150):
            if computers.native:
                break
            assert process.poll() is None, "Electron exited before registering"
            await asyncio.sleep(.1)
        assert computers.native, "Electron did not register its browser host"
        async with httpx.AsyncClient(base_url=origin, timeout=30, trust_env=False) as client:
            opened = await client.post("/api/computer/navigate", json={"sessionId": "shared", "url": origin + "/fixture"})
            assert opened.status_code == 200, opened.text
            state = opened.json()
            assert state["native"] and state["state"] == "manual"
            entry = computers.entries["shared"]
            page = entry.browser.page
            assert await page.title() == "Shared browser test"
            viewport = await page.evaluate("({width: innerWidth, height: innerHeight})")
            shot = await page.screenshot(scale="css")
            assert struct.unpack(">II", shot[16:24]) == (viewport["width"], viewport["height"])
            full = await page.screenshot(full_page=True, scale="css")
            assert struct.unpack(">II", full[16:24]) == (viewport["width"], await page.evaluate("document.documentElement.scrollHeight"))
            cropped = await page.screenshot(full_page=True, clip={"x": 20, "y": 1600, "width": 300, "height": 100}, scale="css")
            assert struct.unpack(">II", cropped[16:24]) == (300, 100)
            assert await page.evaluate("({width: innerWidth, height: innerHeight})") == viewport
            await page.get_by_role("textbox").fill("풍무동 → 군산")
            await page.get_by_role("button").click()
            assert await page.evaluate("window.saves") == 1

            await computers.claim("shared", "agent")
            assert entry.human
            async with computers.bind(entry):
                assert not (await browser_act(BrowserActParams(action="click", selector="#save"))).success
                assert not entry.human
                observed = await browser_observe(BrowserObserveParams())
                assert observed.success and "풍무동" in observed.output
                assert (await browser_act(BrowserActParams(action="click", selector="#save"))).success
            assert await page.evaluate("window.saves") == 2

            paused = await client.post("/api/computer/control", json={"sessionId": "shared", "lease": entry.lease, "action": "pause"})
            assert paused.status_code == 200
            taken = await client.post("/api/computer/control", json={"sessionId": "shared", "lease": entry.lease, "action": "takeover"})
            assert taken.status_code == 200
            await page.get_by_role("textbox").fill("사용자 수정")
            resumed = await client.post("/api/computer/control", json={"sessionId": "shared", "lease": entry.lease, "action": "resume"})
            assert resumed.status_code == 200
            async with computers.bind(entry):
                assert not (await browser_act(BrowserActParams(action="click", selector="#save"))).success
                assert (await browser_observe(BrowserObserveParams())).success
            assert await page.get_by_role("textbox").input_value() == "사용자 수정"
            async with page.expect_popup() as popup_event:
                await page.evaluate("window.open('/fixture?popup=1', '_blank')")
            popup = await popup_event.value
            await popup.wait_for_load_state()
            assert await popup.evaluate("!!window.opener")
            assert (await computers.snapshot(entry, preview=False))["tabId"] != state["tabId"]
            async with computers.bind(entry):
                blocked = await browser_act(BrowserActParams(action="click", selector="#save"))
                assert not blocked.success and blocked.metadata["action_status"] == "not_executed"
                assert (await browser_observe(BrowserObserveParams())).success
                assert (await browser_act(BrowserActParams(action="click", selector="#save"))).success
            assert await popup.evaluate("window.saves") == 1
            await popup.evaluate("() => { const link = document.createElement('a'); link.href='/download'; link.id='download'; link.textContent='Download'; document.body.prepend(link); }")
            async with popup.expect_download() as downloaded:
                await popup.locator("#download").click()
            artifact = await downloaded.value
            assert Path(await artifact.path()).read_text() == "shared browser download"
            computers.finish(entry, "agent")

            await popup.close()
            resumed_entry = await computers.claim("shared", "follow-up")
            assert resumed_entry is entry
            async with computers.bind(entry):
                assert (await browser_observe(BrowserObserveParams())).success
                assert entry.browser.page is page
                assert await page.get_by_role("textbox").input_value() == "사용자 수정"
                assert await page.evaluate("window.saves") == 2
            computers.finish(entry, "follow-up")

            await client.post("/api/computer/control", json={"sessionId": "shared", "lease": entry.lease, "action": "takeover"})
            new_tab = await client.post("/api/computer/navigate", json={"sessionId": "shared", "action": "new_tab", "url": origin + "/fixture?tab=2"})
            assert new_tab.status_code == 200, new_tab.text
            assert len(new_tab.json()["tabs"]) == 2
            assert "tab=2" in new_tab.json()["url"]

            host = computers.native.registration
            rejected = await client.post(host.address, json={"sessionId": "shared", "operation": "status"})
            assert rejected.status_code == 403
            forged = await client.post("/api/computer/native/host", headers={"Origin": "https://example.com"}, json=host.model_dump())
            assert forged.status_code == 403

            other = await client.post("/api/computer/navigate", json={"sessionId": "other", "url": origin + "/fixture?other=1"})
            assert other.status_code == 200
            cdp = await entry.browser.browser.new_browser_cdp_session()
            targets = await cdp.send("Target.getTargets")
            assert len(targets["targetInfos"]) == 2
            with pytest.raises(Exception, match="Unknown|Unsupported"):
                await cdp.send("Target.attachToTarget", {"targetId": other.json()["tabId"], "flatten": True})

            previous_host = computers.native
            previous_page = entry.browser.page
            process.terminate()
            await asyncio.to_thread(process.wait, 10)
            for _ in range(100):
                if not entry.browser.browser.is_connected():
                    break
                await asyncio.sleep(.02)
            disconnected = await client.get("/api/computer/state", params={"sessionId": "shared"})
            assert disconnected.status_code == 200 and "frameId" not in disconnected.json()
            process = subprocess.Popen(
                [str(root / "desktop/node_modules/.bin/electron"), str(root / "tests/fixtures/browser_host.cjs")],
                env={**os.environ, "RUNE_TEST_PROFILE": str(profile), "RUNE_TEST_ORIGIN": origin},
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            for _ in range(100):
                if computers.native is not previous_host:
                    break
                await asyncio.sleep(.05)
            assert computers.native is not previous_host
            reopened = await client.post("/api/computer/navigate", json={"sessionId": "shared", "url": origin + "/fixture"})
            assert reopened.status_code == 200, reopened.text
            assert reopened.json()["native"] and entry.browser.page is not previous_page
            blank = await client.post("/api/computer/navigate", json={"sessionId": "shared", "action": "new_tab"})
            assert blank.status_code == 200 and blank.json()["url"] == "about:blank"
    finally:
        await computers.close()
        process.terminate()
        await asyncio.to_thread(process.wait, 10)
        server.should_exit = True
        await task
