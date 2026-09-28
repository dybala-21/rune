"""Connect the existing browser tools to conversation tabs in the desktop app."""

from __future__ import annotations

import asyncio
import time
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field, field_validator


class NativeHostRegistration(BaseModel):
    address: str
    token: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("address")
    @classmethod
    def local_address(cls, value: str) -> str:
        url = urlparse(value)
        if (url.scheme != "http" or url.hostname != "127.0.0.1" or not url.port
                or url.username or url.password or url.path or url.query or url.fragment):
            raise ValueError("The native browser must listen on a loopback port.")
        return value


class NativeBrowserHost:
    def __init__(self, registration: NativeHostRegistration) -> None:
        self.registration = registration
        self.updated = time.monotonic()
        self.client = httpx.AsyncClient(
            base_url=registration.address, headers={"Authorization": f"Bearer {registration.token}"},
            timeout=15, trust_env=False,
        )

    @property
    def available(self) -> bool:
        return time.monotonic() - self.updated < 15

    async def request(self, session_id: str, operation: str, **params) -> dict:
        response = await self.client.post("/", json={"sessionId": session_id, "operation": operation, **params})
        if response.is_error:
            raise RuntimeError(response.json().get("error", "The desktop browser could not complete the request."))
        return response.json()

    async def close(self) -> None:
        await self.client.aclose()


async def select_native_page(session) -> None:
    state = await session.native_host.request(session.owner_id, "status")
    selected = state["targetId"]
    session.tabs = state["tabs"]
    if selected == session.native_target and session.page and not session.page.is_closed():
        return
    context = session.browser.contexts[0]
    pages = asyncio.Queue()
    context.on("page", pages.put_nowait)
    try:
        for page in context.pages:
            pages.put_nowait(page)
        async with asyncio.timeout(3):
            while True:
                page = await pages.get()
                if page.is_closed():
                    continue
                cdp = await context.new_cdp_session(page)
                try:
                    info = await cdp.send("Target.getTargetInfo")
                finally:
                    await cdp.detach()
                if info["targetInfo"]["targetId"] == selected:
                    session.page = page
                    session.native_target = selected
                    session.needs_observation = True
                    return
    except TimeoutError as exc:
        raise RuntimeError("The selected tab is not ready. Refresh the browser panel.") from exc
    finally:
        context.remove_listener("page", pages.put_nowait)


async def connect_native(session) -> None:
    from playwright.async_api import async_playwright

    session.profile = "native"
    result = await session.native_host.request(session.owner_id, "open")
    session.playwright = await async_playwright().start()
    session.browser = await session.playwright.chromium.connect_over_cdp(
        result["endpoint"], timeout=15000,
        headers={"Authorization": f"Bearer {session.native_host.registration.token}"},
    )
    await select_native_page(session)


async def set_native_control(entry, manual: bool) -> None:
    if entry.browser.profile == "native" and entry.browser.browser and entry.browser.browser.is_connected():
        await entry.browser.native_host.request(entry.session_id, "control", manual=manual)
