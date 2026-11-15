"""Signing in to a site on a hosted machine: the owner does it, Rune keeps the session sealed.

`browser_login` opens the site in the conversation's browser and hands the browser to the owner,
with a card in the chat that shows it live. The owner signs in there; the password goes only into
the page, never into a tool argument, the model's context or the journal. Then:
- **Keep me signed in** opens the owner's approval page, where one passkey keeps the site
  (rune-broker's taint: a kept site stays reachable after the web closes, so the machine can't
  choose one itself). Rune then seals the site's cookies and storage through the broker.
- A site already kept (its session lapsed) needs only **Done**, and is sealed again.

Either way the machine is then marked private: a signed-in page shows the owner's data.

A kept session is stored only sealed, in `RUNE_HOME/browser/logins/`, and opened through the broker
the first time a page of the site loads in a new browser. The broker marks the machine private
before it opens one: a signed-in page shows the owner's data, so the rest of the web closes.

A local install has none of this: the owner signs in with the Computer panel's take-control.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

from pydantic import BaseModel, Field

from rune.types import CapabilityResult
from rune.utils.logger import get_logger

log = get_logger(__name__)

# How often the card's wait checks whether the owner kept the site on the approval page.
KEEP_POLL_SECONDS = 2.0
# Hosts asked for with one site, at most (the broker's limit counts the site itself).
MAX_HOSTS = 23
# The browser's size while the owner signs in: a phone-sized page fits the card.
LOGIN_VIEWPORT = {"width": 430, "height": 860}


class BrowserLoginParams(BaseModel):
    url: str = Field(description="The site's sign-in page, or its home page")
    again: bool = Field(False, description="Sign in again even though Rune keeps a sign-in for this site, "
                                           "because the page still asks to sign in")


def hosted() -> bool:
    from rune.connectors.client import hosted_approval

    return hosted_approval()


def site_of(host: str) -> str:
    """The name a site is kept under: its registrable part, as the owner knows it.

    `www.shop.co.kr` is `shop.co.kr`; `login.shop.com` is `shop.com`. Two-letter country domains
    with a short second level (co.kr, or.jp, com.au) keep three labels.
    """
    labels = host.lower().rstrip(".").split(".")
    if len(labels) >= 3 and len(labels[-1]) == 2 and len(labels[-2]) <= 3:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _belongs(host: str, site: str) -> bool:
    return host == site or host.endswith("." + site)


def _applies(domain: str, hosts: set[str]) -> bool:
    """Whether a cookie for `domain` is sent to any of `hosts`."""
    domain = domain.lstrip(".").lower()
    return any(host == domain or host.endswith("." + domain) for host in hosts)


def logins_dir() -> Path:
    from rune.utils.paths import rune_home

    return rune_home() / "browser" / "logins"


def _file(site: str) -> Path:
    return logins_dir() / f"{site}.json"


def saved() -> dict[str, dict[str, Any]]:
    """Sites with a kept session: {site: {"hosts", "sealed"}}. Unreadable files are skipped."""
    found: dict[str, dict[str, Any]] = {}
    directory = logins_dir()
    if not directory.is_dir():
        return found
    for path in directory.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(record, dict) and record.get("site") == path.stem and isinstance(record.get("sealed"), dict):
                found[path.stem] = {"hosts": [str(h) for h in record.get("hosts") or []], "sealed": record["sealed"]}
        except (OSError, ValueError) as exc:
            log.warning("kept_login_unreadable", site=path.stem, error=type(exc).__name__)
    return found


def _write(site: str, hosts: list[str], sealed: dict[str, Any]) -> None:
    directory = logins_dir()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(dir=directory, prefix=f".{site}.", suffix=".partial")
    partial = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"site": site, "hosts": hosts, "sealed": sealed}, stream)
        os.chmod(partial, 0o600)
        partial.replace(_file(site))
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def forget(site: str) -> None:
    _file(site).unlink(missing_ok=True)


async def keep(context: Any, site: str, hosts: list[str]) -> bool:
    """Seal the site's cookies and storage through the broker and store them. Whether kept."""
    from rune.connectors.client import call_broker

    state = await context.storage_state()
    wanted = set(hosts) | {site}
    session = {"cookies": [c for c in state.get("cookies") or [] if _applies(str(c.get("domain", "")), wanted)],
               "origins": [o for o in state.get("origins") or []
                           if (urlsplit(str(o.get("origin", ""))).hostname or "") in wanted]}
    answer = await call_broker("keep", {"site": site, "state": json.dumps(session, separators=(",", ":"))})
    if not answer.get("kept") or not isinstance(answer.get("sealed"), dict):
        return False
    await asyncio.to_thread(_write, site, sorted(wanted), answer["sealed"])
    return True


def _storage_script(origins: list[dict[str, Any]]) -> str:
    """A script that puts the kept storage into its origin's page."""
    items = {str(o.get("origin")): [[i.get("name"), i.get("value")] for i in o.get("localStorage") or []]
             for o in origins if o.get("origin")}
    # Escaped so a stored value can't close the <script> it is put in.
    kept = json.dumps(items).replace("<", "\\u003c")
    return ("(() => { const kept = " + kept + "; const items = kept[location.origin] || [];"
            " try { for (const [name, value] of items)"
            " if (localStorage.getItem(name) === null) localStorage.setItem(name, value); } catch (e) {} })();")


def _with_script(body: bytes, script: str) -> bytes:
    """`body` with `script` run first: right after <head>, or before everything."""
    tag = b"<script>" + script.encode() + b"</script>"
    head = body[:8192].lower().find(b"<head")
    close = body.find(b">", head) if head >= 0 else -1
    return body[:close + 1] + tag + body[close + 1:] if close >= 0 else tag + body


async def _open_into(context: Any, site: str, record: dict[str, Any]) -> str:
    """Add the site's kept cookies; its storage comes back as a script for its pages to run."""
    from rune.connectors.client import call_broker

    try:
        answer = await call_broker("open", {"site": site, "sealed": record["sealed"]})
    except Exception as exc:
        # Without it the page loads signed out; the owner can sign in again from the card.
        log.warning("kept_login_unavailable", site=site, error=type(exc).__name__)
        return ""
    if not answer.get("kept"):
        # The owner signed Rune out of the site on the approval page.
        await asyncio.to_thread(forget, site)
        return ""
    try:
        state = json.loads(str(answer.get("state") or "{}"))
        await context.add_cookies(state.get("cookies") or [])
        return _storage_script(state["origins"]) if state.get("origins") else ""
    except Exception as exc:
        log.warning("kept_login_unusable", site=site, error=type(exc).__name__)
        return ""


async def attach(context: Any) -> None:
    """On a hosted machine, the first page of a kept site in this browser opens its session first.

    Only when one is about to be used: opening a session marks the machine private. The request
    that opened it is fetched again, with the cookies just added. The kept storage goes into the
    site's first HTML page, before its own scripts run, and into every page after it. Playwright
    doesn't route a redirect's next request, so while the storage waits, a redirect within the
    site is followed by a page of our own instead, whose next request is routed.
    """
    if not hosted():
        return
    records = await asyncio.to_thread(saved)
    if not records:
        return
    by_host = {host: site for site, record in records.items() for host in record["hosts"]}
    opened: set[str] = set()
    storage: dict[str, str] = {}
    lock = asyncio.Lock()

    async def handle(route: Any, request: Any) -> None:
        site = by_host.get(urlsplit(request.url).hostname or "")
        if site is None or not request.is_navigation_request() or site in opened and site not in storage:
            await route.fallback()
            return
        async with lock:
            if site not in opened:
                opened.add(site)
                if script := await _open_into(context, site, records[site]):
                    storage[site] = script
        response = await route.fetch(max_redirects=0)
        script = storage.get(site, "")
        if script and 300 <= response.status < 400 and response.headers.get("location"):
            target = urljoin(request.url, response.headers["location"])
            if (urlsplit(target).hostname or "") in by_host and urlsplit(target).scheme in ("http", "https"):
                await route.fulfill(status=200, content_type="text/html", headers={"cache-control": "no-store"},
                                    body=f"<script>location.replace({json.dumps(target)})</script>")
                return
        if not script or response.status != 200 or "text/html" not in response.headers.get("content-type", ""):
            await route.fulfill(response=response)
            return
        await route.fulfill(response=response, body=_with_script(await response.body(), script))
        storage.pop(site, None)
        # Not while a page waits on this route: Playwright holds it until every page has the script.
        asyncio.get_running_loop().create_task(_later_pages(context, site, script))

    await context.route(lambda url: (urlsplit(url).hostname or "") in by_host, handle)


async def _later_pages(context: Any, site: str, script: str) -> None:
    try:
        await context.add_init_script(script=script)
    except Exception as exc:
        log.debug("kept_login_storage_skipped", site=site, error=type(exc).__name__)


async def browser_login(params: BrowserLoginParams) -> CapabilityResult:
    from rune.capabilities.browser.core import _get_browser, _validate_browser_url
    from rune.capabilities.browser.session import current_session
    from rune.connectors.client import call_broker
    from rune.safety.approval_context import approval_channel, parking

    if blocked := _validate_browser_url(params.url):
        return CapabilityResult(success=False, error=blocked)
    host = (urlsplit(params.url).hostname or "").lower()
    if not hosted():
        return CapabilityResult(success=False, error="Ask the user to sign in themselves: they take control of the "
                                                     "browser in the Computer panel, then resume.")
    site = site_of(host)
    session = current_session()
    hand = session.hand_to_owner
    login = getattr(approval_channel(), "login", None)
    if hand is None or login is None or parking() is not None:
        return CapabilityResult(success=False, error="Signing in needs the owner in the Rune app. Ask them to open "
                                                     "this conversation there and try again.",
                                metadata={"action_status": "not_executed"})
    kept_before = site in await asyncio.to_thread(saved)
    async with session.operation():
        _, page = await _get_browser("managed")
        await page.goto(params.url, wait_until="domcontentloaded", timeout=30000)
    if kept_before and not params.again:
        return CapabilityResult(success=True, output=(
            f"Opened {page.url} with Rune's kept sign-in for {site}. Observe the page: if it still asks to sign in, "
            "call browser_login again with again=true."))

    context = page.context
    seen: set[str] = {site}
    stopped = asyncio.Event()
    asked_hosts: list[str] = []

    async def ask() -> dict[str, Any]:
        hosts = sorted(h for h in seen if _belongs(h, site))[:MAX_HOSTS]
        nonlocal asked_hosts
        asked_hosts = hosts
        return await call_broker("login", {"site": site, "hosts": hosts})

    def record(request: Any) -> None:
        seen.add((urlsplit(request.url).hostname or "").lower())

    async def kept() -> bool | None:
        """Until the owner keeps the site on the approval page; asked again as its pages load more hosts."""
        last: list[str] = []
        while True:
            try:
                await asyncio.wait_for(stopped.wait(), KEEP_POLL_SECONDS)
                return None
            except TimeoutError:
                pass
            try:
                current = sorted(h for h in seen if _belongs(h, site))[:MAX_HOSTS]
                if current != last:
                    last = current
                    if (await ask()).get("kept"):
                        return True
                elif site in ((await call_broker("status", {})).get("sites") or []):
                    return True
            except Exception:  # noqa: BLE001 - a broker that blinks while the owner signs in; ask again
                continue

    try:
        first = await ask()
    except Exception as exc:
        return CapabilityResult(success=False, error=f"Connector broker unavailable ({type(exc).__name__})",
                                metadata={"action_status": "not_executed"})
    # A site the owner already keeps (its session lapsed) is sealed again on Done, with no passkey.
    already = kept_before or bool(first.get("kept"))
    page_url = str(first.get("approval_url") or "")
    keep_url = f"{page_url.rstrip('/')}/keep/{site}" if page_url.startswith("https://") and not already else ""

    async def until_stopped() -> None:
        await stopped.wait()

    context.on("request", record)
    restore = page.viewport_size
    await page.set_viewport_size(LOGIN_VIEWPORT)
    await hand(True)
    try:
        outcome, _ = await login(f"Sign in to {site}", f"Sign in to {site} in the browser below.",
                                 {"site": site, "url": page.url, "sessionId": session.owner_id,
                                  "keepUrl": keep_url},
                                 until_stopped if not keep_url else kept, stopped.set)
    finally:
        context.remove_listener("request", record)
        await hand(False)
        if restore:
            await page.set_viewport_size(restore)
    if outcome in ("declined", "timeout"):
        error = ("The owner chose not to sign in now. Don't ask again in this task unless they do."
                 if outcome == "declined" else "The owner didn't sign in in time.")
        return CapabilityResult(success=False, error=error, metadata={"action_status": "not_executed"})
    # What a signed-in page shows is the owner's data: the rest of the web closes, as after a
    # connector call. Only a site the owner keeps stays reachable. Unrecorded, nothing goes on.
    try:
        await call_broker("private", {})
    except Exception as exc:
        log.warning("login_private_unrecorded", site=site, error=type(exc).__name__)
        async with session.operation():
            await session.release_resources()
        return CapabilityResult(success=False, error="The sign-in couldn't be recorded with the broker, so the "
                                                     "browser was closed. Try again in a moment.",
                                metadata={"action_status": "not_executed"})
    if outcome == "kept" or already:
        try:
            stays = (f" Rune stays signed in to {site} for later tasks." if await keep(context, site, asked_hosts)
                     else " The sign-in could not be kept for later tasks.")
        except Exception as exc:
            log.warning("login_keep_failed", site=site, error=type(exc).__name__)
            stays = " The sign-in could not be kept for later tasks."
    else:
        stays = f" {site} isn't kept, so it closes along with the rest of the web."
    return CapabilityResult(success=True, output=(
        f"The owner signed in to {site} and handed the browser back.{stays} The page may have changed: "
        "observe it before acting."))
