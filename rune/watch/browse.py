"""A product page opened in RUNE's own browser, where a shop answers a browser and not a program.

Many shops refuse a program that asks in its own name, or write the price into the page only once
their scripts have run. Such a page is opened here the way its owner would open it: in a browser
window on this machine, one page at a time.

- The browser is RUNE's own and says so: its name ends in RUNE's, and nothing hides that a program
  drives it. A check that asks for a person is not answered; the page then stays unread.
- Nothing is clicked, typed or signed in to, no file is taken, no sound is played, and no cookie
  outlives a page.
- The window is kept minimized. On a machine with no screen the browser runs without one.
- An address is checked like any other the watch reads: https, a site's own name, a public host.
  The browser follows redirects and loads what a page names by itself, so it is given one way out,
  ``rune.watch.gate``: no connection is made to a host that is not public, whatever asked for it.

What comes back is the page's markup after its scripts ran, for the same code that reads any other
page: only numbers leave it. Whether a page may be opened at all (a site that names RUNE in its
robots.txt and refuses it is not) is decided in ``rune.watch.observe``.
"""

from __future__ import annotations

import asyncio
import os
import socket
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from rune import __version__
from rune.connectors.transport import public_address
from rune.utils.logger import get_logger
from rune.watch.denied import denied
from rune.watch.fetch import MAX_PAGE, MIN_GAP, Page, host_of, refusal
from rune.watch.gate import Gate
from rune.watch.item import Unread

log = get_logger(__name__)

# Added to the browser's own name, so a site sees who opened the page and where to read about it.
NAME = f"RUNE/{__version__} (+https://github.com/dybala-21/rune#fetcher)"
# Set to 0, no page is opened in a browser on this machine.
SWITCH = "RUNE_WATCH_BROWSER"
# One page, from opening the browser to having the page's markup.
DEADLINE = 45.0
# How long the page itself is waited for, and after that its scripts.
ARRIVE = 25.0
SETTLE = 8.0
# How long putting a page down may take before the next one is opened anyway.
LEAVE = 5.0
# No sound. And no packet that goes around the proxy: a page's calls (WebRTC) would otherwise send
# them straight to any address, this machine's own among them. The browser with a window and the
# one without each know that rule by a different name, and ignore the other's.
FLAGS = ("--mute-audio", "--webrtc-ip-handling-policy=disable_non_proxied_udp",
         "--force-webrtc-ip-handling-policy=disable_non_proxied_udp")
# A browser reaches this machine's own addresses directly unless told to send them through its
# proxy as well.
ALSO_PROXIED = "<-loopback>"


@dataclass(slots=True)
class Window:
    """A started browser with one empty page in it, and how to shut it."""

    context: Any
    page: Any
    close: Callable[[], Awaitable[None]]


def usable() -> bool:
    """Whether pages may be opened in a browser here. Not on a hosted machine: what a shop sees
    there is a data centre, not its owner's own machine."""
    from rune.cloud.boundary import hosted

    return os.environ.get(SWITCH, "").strip() != "0" and not hosted()


async def open_window(out: str, *, screen: bool = True) -> Window:
    """Start the browser, with a window unless ``screen`` is false. ``out`` is the proxy every
    connection it makes goes through. Raises ``Unread("no_browser")`` where there is none to start."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise Unread("no_browser") from None
    driver = await async_playwright().start()
    browser = None
    try:
        # With a window where this machine has a screen: shops turn away a browser that has none.
        for headless in ((False, True) if screen else (True,)):
            try:
                browser = await driver.chromium.launch(headless=headless, chromium_sandbox=True, args=list(FLAGS),
                                                       proxy={"server": out, "bypass": ALSO_PROXIED})
                break
            except Exception as failed:
                log.debug("watch_browser_not_started", headless=headless, error=type(failed).__name__)
        if browser is None:
            raise Unread("no_browser")
        about = await (await browser.new_browser_cdp_session()).send("Browser.getVersion")
        context = await browser.new_context(user_agent=f"{about['userAgent']} {NAME}", accept_downloads=False,
                                            service_workers="block")
        page = await context.new_page()
        if not headless:
            await _minimize(context, page)
    except BaseException:
        try:
            if browser is not None:
                await browser.close()
        finally:
            await driver.stop()
        raise

    async def close() -> None:
        try:
            await browser.close()
        finally:
            await driver.stop()

    return Window(context, page, close)


async def _minimize(context: Any, page: Any) -> None:
    try:
        session = await context.new_cdp_session(page)
        window = await session.send("Browser.getWindowForTarget")
        await session.send("Browser.setWindowBounds",
                           {"windowId": window["windowId"], "bounds": {"windowState": "minimized"}})
        await session.detach()
    except Exception as failed:
        # The page is read all the same; its window is just in view while it is.
        log.debug("watch_browser_not_minimized", error=type(failed).__name__)


class Browser:
    """Opens pages for one run in one window behind one gate, both started when the first page
    needs them."""

    def __init__(self, *, open: Callable[[str], Awaitable[Window]] = open_window,
                 resolve: Callable[[str], Awaitable[str]] = public_address,
                 pause: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 clock: Callable[[], float] = time.monotonic, deadline: float = DEADLINE) -> None:
        self._open, self._resolve, self._pause, self._clock, self._deadline = open, resolve, pause, clock, deadline
        self._window: Window | None = None
        self._gate: Gate | None = None
        # Why a host is not asked, by host; "" for one that is public. And the address each public
        # one was found at: the only one the browser is connected to for it.
        self._hosts: dict[str, str] = {}
        self._addresses: dict[str, str] = {}
        self._last: dict[str, float] = {}
        self._documents: list[int] = []

    async def __aenter__(self) -> Browser:
        return self

    async def __aexit__(self, kind: type[BaseException] | None, error: BaseException | None,
                        trace: TracebackType | None) -> None:
        window, self._window = self._window, None
        gate, self._gate = self._gate, None
        # The window first: its connections end with it, and the gate is left with none to end.
        for started in (window, gate):
            if started is not None:
                try:
                    await started.close()
                except Exception as failed:
                    log.debug("watch_browser_close_failed", error=type(failed).__name__)

    async def page(self, url: str, *, registering: bool) -> Page:
        """The page at ``url`` as the browser shows it. Raises ``Unread`` when it may not or cannot be read."""
        if not usable():
            raise Unread("no_browser")
        first = host_of(url)
        try:
            async with asyncio.timeout(self._deadline):
                if why := await self._kept_out(first):
                    raise Unread(why)
                window = await self._started()
                last = self._last.get(first)
                if last is not None and (wait := last + MIN_GAP - self._clock()) > 0:
                    await self._pause(wait)
                try:
                    return await self._read(window.page, url, first, registering)
                finally:
                    self._last[first] = self._clock()
                    await self._leave(window)
        except TimeoutError:
            raise Unread("timeout") from None

    async def _started(self) -> Window:
        if self._window is None:
            gate = Gate(self._reached_at)
            try:
                window = await self._open(await gate.open())
                try:
                    window.context.on("page", lambda extra: self._only(window, extra))
                    window.page.on("response", self._note)
                except BaseException:
                    await window.close()
                    raise
            except BaseException:
                await gate.close()
                raise
            self._gate, self._window = gate, window
        return self._window

    async def _read(self, page: Any, url: str, first: str, registering: bool) -> Page:
        self._documents.clear()
        try:
            answer = await page.goto(url, wait_until="domcontentloaded", timeout=ARRIVE * 1000)
        except Exception as failed:
            # Only its kind: the browser's own wording repeats the address.
            log.debug("watch_browser_no_page", host=first, error=type(failed).__name__)
            raise Unread("timeout" if type(failed).__name__ == "TimeoutError" else "unreachable") from None
        try:
            await page.wait_for_load_state("networkidle", timeout=SETTLE * 1000)
        except Exception as failed:
            # A page that never goes quiet is read as it stands.
            log.debug("watch_browser_not_settled", host=first, error=type(failed).__name__)
        at = page.url
        host = host_of(at)
        if not registering and host != first:
            raise Unread("moved")
        if why := await self._kept_out(host):
            raise Unread(why)
        # The last document the page became: a script may have sent it on after the first one.
        status = self._documents[-1] if self._documents else (answer.status if answer is not None else 0)
        if status != 200:
            raise Unread(refusal(status), status)
        markup = await page.content()
        if len(markup) > MAX_PAGE:
            raise Unread("too_large")
        return Page(at, status, markup)

    async def _leave(self, window: Window) -> None:
        """Put the page down: its scripts stop, and what the site set is not carried to the next one."""
        try:
            async with asyncio.timeout(LEAVE):
                await window.page.goto("about:blank")
                await window.context.clear_cookies()
        except Exception as failed:
            log.debug("watch_browser_not_cleared", error=type(failed).__name__)

    async def _kept_out(self, host: str) -> str:
        """Why ``host`` is not asked, or "" for a public host. The gate asks this of every host the
        browser wants, one written as numbers among them: whether it is public is looked up the
        same way."""
        if host not in self._hosts:
            if denied(host):
                self._hosts[host] = "denied"
            else:
                try:
                    self._addresses[host] = await self._resolve(host)
                    self._hosts[host] = ""
                except socket.gaierror:
                    self._hosts[host] = "unreachable"
                except ValueError:
                    self._hosts[host] = "private"
        return self._hosts[host]

    async def _reached_at(self, host: str) -> str:
        """The address the browser is connected to for ``host``; "" for a host it is kept out of."""
        return "" if await self._kept_out(host) else self._addresses[host]

    def _note(self, response: Any) -> None:
        """Keep the status of each document the page itself became, in order."""
        try:
            request = response.request
            if request.is_navigation_request() and request.frame.parent_frame is None:
                self._documents.append(response.status)
        except Exception as failed:
            # A request with no frame behind it (a worker's) is not the page.
            log.debug("watch_browser_response_skipped", error=type(failed).__name__)

    async def _only(self, window: Window, extra: Any) -> None:
        """A page a script opened beside the one being read is closed: nothing is read from it."""
        if extra is not window.page:
            try:
                await extra.close()
            except Exception as failed:
                log.debug("watch_browser_extra_page", error=type(failed).__name__)
