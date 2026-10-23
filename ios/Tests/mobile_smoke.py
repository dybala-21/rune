"""Chromium mobile-layout smoke test for the iOS composer bridge with engine calls mocked; not a WKWebView test.

Needs Playwright and `npm --prefix web run preview -- --host 127.0.0.1 --port 4179 --strictPort`.
"""

import asyncio
import json
from urllib.parse import urlsplit

from playwright.async_api import async_playwright


async def main():
    writes = []
    errors = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        context = await browser.new_context(viewport={"width": 390, "height": 760}, is_mobile=True, has_touch=True)
        await context.add_init_script("""
            window.EventSource = class extends EventTarget {
                static CLOSED = 2;
                readyState = 1;
                constructor() {
                    super();
                    setTimeout(() => this.dispatchEvent(new MessageEvent('connected', {
                        data: JSON.stringify({clientId: 'mobile-test'})
                    })), 50);
                }
                close() { this.readyState = 2; }
            };
        """)

        async def route_request(route):
            request = route.request
            url = urlsplit(request.url)
            if url.hostname != "127.0.0.1":
                await route.abort()
                return
            if not url.path.startswith("/api/"):
                await route.continue_()
                return
            response = {}
            if url.path == "/api/v1/rpc":
                method = request.post_data_json["method"]
                data = {
                    "config.get": {"activeModel": {"provider": "test", "model": "fixture"}, "memoryTuning": {}, "safetyTuning": {}},
                    "models.list": {"test": ["fixture"]},
                    "workspace.get": {"path": "/test-workspace"},
                    "workspace.recents": {"paths": []},
                    "sessions.list": {"sessions": [], "total": 0},
                    "runs.active": {"runIds": []},
                }.get(method, {})
                response = {"success": True, "data": data}
            elif url.path == "/api/runs/snapshot":
                response = {"run": None}
            elif url.path == "/api/v1/proactive/feed":
                response = {"suggestions": []}
            elif url.path == "/api/v1/auth/bootstrap":
                response = {"ok": True}
            else:
                if request.method == "POST":
                    writes.append(url.path)
            await route.fulfill(content_type="application/json", body=json.dumps(response))

        await context.route("**/*", route_request)
        page = await context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto("http://127.0.0.1:4179/")
        await page.wait_for_function("typeof window.runeImportText === 'function'")
        composer = page.locator("textarea").first
        await composer.fill("Summarize the following conversation.")
        transcript = "민수: 내일 10시 회의\nAlex: confirmed 🐱\n</script>${alert(1)}"
        assert await page.evaluate("text => window.runeImportText(text)", transcript)
        assert await composer.input_value() == "Summarize the following conversation.\n\n" + transcript
        assert not await page.evaluate("window.runeImportText('x'.repeat(100001))")
        assert not writes, f"Import unexpectedly submitted work: {writes}"
        for width, height in [(390, 760), (844, 346), (768, 960)]:
            await page.set_viewport_size({"width": width, "height": height})
            await composer.scroll_into_view_if_needed()
            bounds = await composer.bounding_box()
            assert bounds and bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= width + 1
            assert bounds["y"] >= 0 and bounds["y"] + bounds["height"] <= height + 1
        await page.set_viewport_size({"width": 390, "height": 760})
        await page.screenshot(path="/tmp/rune-ios-mobile-smoke.png")
        assert not errors, errors
        await browser.close()
    print("Mobile composer smoke passed: Unicode, literal text, existing draft, size rejection, no submission, three viewports.")


if __name__ == "__main__":
    asyncio.run(main())
