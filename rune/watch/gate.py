"""The one way out of the watch's browser: every connection it makes is asked for here first.

A browser follows redirects and looks names up by itself, so which hosts a page reaches cannot be
settled by reading the addresses the page names. The browser is started with this as its proxy
instead. It asks here for each connection by the host's name; the name is looked up once, and the
connection is made to that very address or not at all. A host that is not public gets none,
whatever sent the browser there: the page's own address, a redirect, a script, a picture.

Only tunnels are given. What passes through one is encrypted by the browser for the far end and
is not read here; a request the browser would send in the clear is refused.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from rune.utils.logger import get_logger

log = get_logger(__name__)

# How long the browser has to say what it wants, and how long reaching the host may take.
ASK = 10.0
REACH = 15.0
# How long the last tunnels may take to end when the gate is shut.
SHUT = 5.0
# No request for a tunnel has more header lines than this.
LINES = 64
# How much is passed on at a time.
PIECE = 64 * 1024

_REFUSED = b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
_NOT_REACHED = b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
_OPENED = b"HTTP/1.1 200 Connection Established\r\n\r\n"


class Gate:
    """Listens on this machine for one browser. ``where`` gives the address a host is reached at,
    or "" for a host that is not to be reached."""

    def __init__(self, where: Callable[[str], Awaitable[str]]) -> None:
        self._where = where
        self._server: asyncio.Server | None = None
        self._serving: set[asyncio.Task[None]] = set()
        self._shut = False

    async def open(self) -> str:
        """Start listening. Returns the address the browser is to use as its proxy."""
        self._server = await asyncio.start_server(self._accept, "127.0.0.1", 0)
        return f"http://127.0.0.1:{self._server.sockets[0].getsockname()[1]}"

    async def close(self) -> None:
        """Stop listening and end every tunnel still open."""
        server, self._server = self._server, None
        if server is None:
            return
        self._shut = True
        server.close()
        serving = list(self._serving)
        for task in serving:
            task.cancel()
        await asyncio.gather(*serving, return_exceptions=True)
        try:
            async with asyncio.timeout(SHUT):
                await server.wait_closed()
        except TimeoutError:
            log.debug("watch_gate_slow_to_shut")

    def _accept(self, asked: asyncio.StreamReader, answer: asyncio.StreamWriter) -> None:
        if self._shut:
            # Taken in just as the gate was shut: ended like the rest.
            answer.close()
            return
        task = asyncio.ensure_future(self._serve(asked, answer))
        self._serving.add(task)
        task.add_done_callback(lambda done: self._ended(done, answer))

    def _ended(self, task: asyncio.Task[None], answer: asyncio.StreamWriter) -> None:
        # Done here and not where the connection is served: one ended before it was first served,
        # as when the gate shuts right after taking it in, never gets that far.
        self._serving.discard(task)
        answer.close()

    async def _serve(self, asked: asyncio.StreamReader, answer: asyncio.StreamWriter) -> None:
        onward: asyncio.StreamWriter | None = None
        try:
            async with asyncio.timeout(ASK):
                host, port = _wanted(await _request(asked))
                address = await self._where(host) if host else ""
            if not address:
                answer.write(_REFUSED)
                await answer.drain()
                return
            try:
                async with asyncio.timeout(REACH):
                    back, onward = await asyncio.open_connection(address, port)
            except (OSError, TimeoutError) as failed:
                log.debug("watch_gate_not_reached", error=type(failed).__name__)
                answer.write(_NOT_REACHED)
                await answer.drain()
                return
            answer.write(_OPENED)
            await answer.drain()
            await _carry(asked, onward, back, answer)
        except Exception as failed:
            # The browser hung up, what it sent was no request, or something here broke: in each
            # case the connection is not made. Only the error's kind is logged.
            log.debug("watch_gate_dropped", error=type(failed).__name__)
        finally:
            if onward is not None:
                onward.close()


async def _request(asked: asyncio.StreamReader) -> str:
    """The first line of the browser's request, once the whole of its header has come."""
    first = await asked.readline()
    for _ in range(LINES):
        line = await asked.readline()
        if line in (b"\r\n", b"\n"):
            return first.decode("latin-1")
    raise ValueError("not a request")


def _wanted(first: str) -> tuple[str, int]:
    """The host and port a request asks a tunnel to; ("", 0) for a request that asks for none."""
    method, _, rest = first.partition(" ")
    if method != "CONNECT":
        return "", 0
    try:
        to = urlsplit("//" + rest.partition(" ")[0])
        host, port = to.hostname or "", to.port or 0   # a host's name comes back in small letters
    except ValueError:
        return "", 0
    return (host, port) if host and port else ("", 0)


async def _carry(asked: asyncio.StreamReader, onward: asyncio.StreamWriter,
                 back: asyncio.StreamReader, answer: asyncio.StreamWriter) -> None:
    """Pass bytes both ways until both ends are done, or either breaks."""
    async with asyncio.TaskGroup() as both:
        both.create_task(_pass(asked, onward))
        both.create_task(_pass(back, answer))


async def _pass(source: asyncio.StreamReader, sink: asyncio.StreamWriter) -> None:
    while piece := await source.read(PIECE):
        sink.write(piece)
        await sink.drain()
    # One end is done sending; the other may still have an answer on its way.
    if sink.can_write_eof():
        sink.write_eof()
