"""Read one public page for the watch: under RUNE's own name, and only where the site lets it.

- RUNE says who it is and obeys robots.txt. A refusal is this reader's answer: it asks nothing
  again under another name, protocol or address. Whether the page is then opened in RUNE's own
  browser (``rune.watch.browse``) is decided in ``rune.watch.observe``, not here.
- An address is checked before anything is sent to it, and again at every redirect: https, a host
  that resolves only to public addresses, and the connection goes to the address that was checked.
- An answer that is slow, endless or inflates is cut off. No cookie is kept or sent.

What comes back is the page's text, or ``Unread`` with one of a fixed set of words. Nothing a site
wrote (a redirect's target, an error's wording) is passed on.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import time
import zlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import TracebackType
from urllib.parse import urljoin, urlsplit

import httpx

from rune import __version__
from rune.connectors.transport import public_address
from rune.utils.logger import get_logger
from rune.watch import robots
from rune.watch.denied import denied
from rune.watch.item import Unread

log = get_logger(__name__)

# Who is asking, and where a site's operator can read what RUNE reads and how to refuse it.
USER_AGENT = f"Mozilla/5.0 (compatible; RUNE/{__version__}; +https://github.com/dybala-21/rune#fetcher)"
ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
MAX_URL = 2048
MAX_PAGE = 4 * 1024 * 1024
DEADLINE = 20.0
# Between two requests to one host: at least this, or the pause its robots.txt asks for, up to the cap.
MIN_GAP = 2.0
MAX_GAP = 10.0
REGISTER_HOPS = 5
DAILY_HOPS = 3
ROBOTS_HOPS = 5
_REDIRECT = (301, 302, 303, 307, 308)
_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]+)""", re.IGNORECASE)
# What each part of a host's name is made of, once the name is in its ASCII form.
_LABEL = re.compile(r"[a-z0-9_-]+")


@dataclass(frozen=True, slots=True)
class Page:
    url: str  # where it was read from, after any redirects
    status: int
    text: str


@dataclass(frozen=True, slots=True)
class _Answer:
    status: int
    headers: httpx.Headers
    body: bytes


def host_of(url: str) -> str:
    """The host of an address the watch may read, in its ASCII form. Raises ``Unread("address")``
    for anything else: not https, another port, a login in the address, a bare IP, a name not
    written plainly, or too long."""
    if len(url) > MAX_URL:
        raise Unread("address")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise Unread("address") from None
    host = parts.hostname or ""
    if parts.scheme != "https" or port not in (None, 443) or parts.username is not None or parts.password is not None:
        raise Unread("address")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise Unread("address")  # a bare IP address, not a site's name
    try:
        name = host.encode("idna").decode("ascii").lower()
    except UnicodeError:
        raise Unread("address") from None
    # One spelling for one site. A name that ends in a dot, or holds a "%" a browser would turn
    # into one, is the same site under another name: its robots.txt, its pause and the terms list
    # would be asked about a site they do not know.
    if "." not in name or not all(_LABEL.fullmatch(part) for part in name.split(".")):
        raise Unread("address")
    return name


class Reader:
    """Reads pages for one run: robots.txt once per host, and a pause between requests to a host."""

    def __init__(self, *, client: httpx.AsyncClient | None = None,
                 resolve: Callable[[str], Awaitable[str]] = public_address,
                 pause: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 clock: Callable[[], float] = time.monotonic, deadline: float = DEADLINE) -> None:
        self._client = client
        self._given = client is not None
        self._resolve, self._pause, self._clock, self._deadline = resolve, pause, clock, deadline
        self._rules: dict[str, robots.Rules | str] = {}
        self._last: dict[str, float] = {}

    async def __aenter__(self) -> Reader:
        if self._client is None:
            # No proxy from the environment: the connection has to go to the address that was checked.
            self._client = httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(10.0), follow_redirects=False)
        return self

    async def __aexit__(self, kind: type[BaseException] | None, error: BaseException | None,
                        trace: TracebackType | None) -> None:
        if self._client is not None and not self._given:
            await self._client.aclose()
            self._client = None

    async def page(self, url: str, *, registering: bool) -> Page:
        """The page at ``url``. Raises ``Unread`` when it may not or cannot be read.

        A redirect is followed only as far as it stays a page the watch may read: every hop is
        checked like the first. When a page is being registered it may lead to another site (a short
        link); a page already watched must stay where it was, or it has moved.
        """
        first = ""
        for _ in range((REGISTER_HOPS if registering else DAILY_HOPS) + 1):
            host = host_of(url)  # only https on port 443 gets past this, so the host alone names the site
            if denied(host):
                raise Unread("denied")
            first = first or host
            if not registering and host != first:
                raise Unread("moved")
            rules = await self._robots(host)
            if not robots.allows(rules, url):
                raise Unread("robots")
            answer = await self._get(host, url, MAX_PAGE, gap=max(MIN_GAP, min(rules.crawl_delay, MAX_GAP)))
            if answer.status in _REDIRECT:
                target = answer.headers.get("location")
                if not target:
                    raise Unread("error", answer.status)
                url = urljoin(url, target)
                continue
            if answer.status == 200:
                return Page(url, answer.status, _text(answer))
            raise Unread(refusal(answer.status), answer.status)
        raise Unread("error" if registering else "moved")  # more redirects than are followed

    async def refuses_by_name(self, url: str) -> bool:
        """Whether the site's robots.txt has a group for RUNE that keeps it from ``url``. A file that
        can't be had, or one that only speaks to every program at once, names nobody."""
        host = host_of(url)
        if host not in self._rules:
            try:
                self._rules[host] = await self._read_robots(host)
            except Unread:
                return False  # the host itself can't be reached: opening the page will say so
        rules = self._rules[host]
        return isinstance(rules, robots.Rules) and rules.named and not robots.allows(rules, url)

    async def _robots(self, host: str) -> robots.Rules:
        if host not in self._rules:
            self._rules[host] = await self._read_robots(host)
        rules = self._rules[host]
        if isinstance(rules, str):
            raise Unread(rules)
        return rules

    async def _read_robots(self, host: str) -> robots.Rules | str:
        url = f"https://{host}/robots.txt"
        for hop in range(ROBOTS_HOPS + 1):
            try:
                at = host_of(url)
                if denied(at):
                    return "robots"  # a redirect into a site that is asked nothing: its file is not had
                answer = await self._get(at, url, robots.MAX_BYTES, cut=True)
            except Unread as failed:
                if hop == 0 and failed.kind in ("unreachable", "private"):
                    raise  # the host itself can't be reached: its page can't be either
                return "robots"  # RFC 9309 2.3.1.4: robots.txt can't be had, so assume it disallows
            if answer.status not in _REDIRECT:
                return robots.read(answer.status, answer.body)
            target = answer.headers.get("location")
            if not target:
                return "robots"
            url = urljoin(url, target)
        return "robots"

    async def _get(self, host: str, url: str, limit: int, *, gap: float = MIN_GAP, cut: bool = False) -> _Answer:
        assert self._client is not None, "use the reader in an `async with`"
        last = self._last.get(host)
        if last is not None and (wait := last + gap - self._clock()) > 0:
            await self._pause(wait)
        parts = urlsplit(url)
        try:
            # httpx's own timeouts bound each wait, not the request as a whole.
            async with asyncio.timeout(self._deadline):
                try:
                    address = await self._resolve(host)
                except socket.gaierror:
                    raise Unread("unreachable") from None
                except ValueError:
                    raise Unread("private") from None  # it resolves to an address that isn't public
                target = (f"https://{f'[{address}]' if ':' in address else address}{parts.path or '/'}"
                          + (f"?{parts.query}" if parts.query else ""))
                headers = {"Host": host, "User-Agent": USER_AGENT, "Accept": ACCEPT, "Accept-Encoding": "gzip"}
                async with self._client.stream("GET", target, headers=headers,
                                               extensions={"sni_hostname": host}) as response:
                    return _Answer(response.status_code, response.headers, await _body(response, limit, cut))
        except TimeoutError:
            raise Unread("timeout") from None
        except httpx.TimeoutException:
            raise Unread("timeout") from None
        except httpx.TransportError as failed:
            # No answer at all: refused, reset, or closed before one came.
            log.debug("watch_no_answer", host=host, error=type(failed).__name__)
            raise Unread("unreachable") from None
        except httpx.HTTPError as failed:
            log.debug("watch_bad_answer", host=host, error=type(failed).__name__)
            raise Unread("error") from None
        finally:
            self._last[host] = self._clock()
            self._client.cookies.clear()  # nothing a site set is sent back to it


async def _body(response: httpx.Response, limit: int, cut: bool) -> bytes:
    """The answer's body, unpacked here so that it can never grow past ``limit``: beyond it the read
    stops (``cut``) or fails as too large."""
    encoding = response.headers.get("content-encoding", "").strip().lower()
    if encoding in ("", "identity"):
        inflate = None
    elif encoding in ("gzip", "x-gzip"):
        inflate = zlib.decompressobj(zlib.MAX_WBITS | 16)
    else:
        raise Unread("error")  # an encoding that was not asked for
    body = bytearray()
    async for chunk in response.aiter_raw():
        if inflate is None:
            body += chunk
        else:
            try:
                while chunk and len(body) <= limit:
                    body += inflate.decompress(chunk, limit + 1 - len(body))
                    chunk = inflate.unconsumed_tail
            except zlib.error:
                raise Unread("error") from None
        if len(body) > limit:
            if cut:
                return bytes(body[:limit])
            raise Unread("too_large")
    return bytes(body)


def refusal(status: int) -> str:
    """What an answer that is not the page means, as the kind of refusal it is."""
    if status in (401, 403, 429):
        return "blocked"
    if status in (404, 410):
        return "gone"
    return "server" if 500 <= status < 600 else "error"


def _text(answer: _Answer) -> str:
    """The body as text in the encoding the page states (its header, else its meta tag), else UTF-8."""
    declared = re.search(r"charset\s*=\s*[\"']?\s*([A-Za-z0-9_.:-]+)", answer.headers.get("content-type", ""), re.IGNORECASE)
    names = [declared.group(1)] if declared else []
    if (stated := _CHARSET.search(answer.body[:4096])) is not None:
        names.append(stated.group(1).decode("ascii"))
    for name in (*names, "utf-8"):
        try:
            return answer.body.decode(name, "replace")
        except LookupError:
            continue  # a name Python doesn't know: try the next, UTF-8 last
    return answer.body.decode("utf-8", "replace")
