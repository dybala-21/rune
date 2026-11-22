"""The one way out of the watch's browser: which connections it makes, and to what address."""

import asyncio
import gc
import warnings
from types import SimpleNamespace

import pytest

from rune.watch import gate
from rune.watch.gate import Gate


class Far:
    """A host on this machine standing in for one far away: how often it was reached and left. It
    sends back what it is sent, with ``far:`` before it."""

    def __init__(self) -> None:
        self.reached = self.left = 0
        self.port = 0
        self._server: asyncio.Server | None = None

    async def __aenter__(self) -> "Far":
        self._server = await asyncio.start_server(self._answer, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *why: object) -> None:
        self._server.close()

    async def _answer(self, sent: asyncio.StreamReader, back: asyncio.StreamWriter) -> None:
        self.reached += 1
        try:
            while piece := await sent.read(100):
                back.write(b"far:" + piece)
                await back.drain()
        finally:
            self.left += 1
            back.close()

    async def was_left(self) -> int:
        """How many of its connections were ended from the other side, once that had time to show."""
        for _ in range(100):
            if self.left == self.reached:
                break
            await asyncio.sleep(0.01)
        return self.left


class Names:
    """Where each host is found, and every host that was looked up."""

    def __init__(self, **found: str) -> None:
        self.found = {name.replace("_", "."): address for name, address in found.items()}
        self.asked: list[str] = []

    async def where(self, host: str) -> str:
        self.asked.append(host)
        return self.found.get(host, "")


async def soon(what):
    """``what``, or a failed test where it would wait without end."""
    return await asyncio.wait_for(what, 3)


async def ask(out: str, request: bytes) -> tuple[int, asyncio.StreamReader, asyncio.StreamWriter]:
    """Send ``request`` to the gate at ``out``. Returns its answer's status, 0 for a door shut without one."""
    sent, back = await asyncio.open_connection("127.0.0.1", int(out.rsplit(":", 1)[1]))
    back.write(request)
    await back.drain()
    first = await soon(sent.readline())
    while first and (await soon(sent.readline())).strip():
        pass
    return (int(first.split()[1]) if first else 0), sent, back


async def tunnel(out: str, to: str) -> tuple[int, asyncio.StreamReader, asyncio.StreamWriter]:
    return await ask(out, f"CONNECT {to} HTTP/1.1\r\nHost: {to}\r\n\r\n".encode())


@pytest.fixture
async def far():
    async with Far() as host:
        yield host


async def opened(names: Names) -> tuple[Gate, str]:
    door = Gate(names.where)
    return door, await door.open()


async def test_a_tunnel_is_made_to_the_address_its_host_was_found_at_and_carries_both_ways(far):
    names = Names(shop_example="127.0.0.1")
    door, out = await opened(names)
    status, sent, back = await tunnel(out, f"Shop.Example:{far.port}")
    assert (status, names.asked) == (200, ["shop.example"]), "looked up by its name, in small letters"
    back.write(b"hello")
    await back.drain()
    # shop.example is no name any lookup knows: the far end was reached at the address handed over.
    assert await soon(sent.readexactly(9)) == b"far:hello" and far.reached == 1
    back.close()
    await door.close()


@pytest.mark.parametrize("to", ["lan.example:{port}", "127.0.0.1:{port}", "[::1]:{port}"])
async def test_a_host_that_is_kept_out_gets_no_connection(far, to):
    names = Names()
    door, out = await opened(names)
    status, _, back = await tunnel(out, to.format(port=far.port))
    assert (status, far.reached) == (403, 0)
    assert names.asked == [to.split(":{")[0].strip("[]")], "an address written as numbers is asked about like a name"
    back.close()
    await door.close()


@pytest.mark.parametrize("request_", [
    "GET http://shop.example:{port}/reboot HTTP/1.1\r\nHost: shop.example\r\n\r\n",   # in the clear
    "POST http://shop.example:{port}/ HTTP/1.1\r\nContent-Length: 0\r\n\r\n",
    "CONNECT shop.example HTTP/1.1\r\n\r\n",                                           # no port
    "CONNECT shop.example:99999 HTTP/1.1\r\n\r\n",
    "CONNECT shop.example:https HTTP/1.1\r\n\r\n",
    "CONNECT :{port} HTTP/1.1\r\n\r\n",
    "CONNECT\r\n\r\n",
    "connect shop.example:{port} HTTP/1.1\r\n\r\n",
    "\x16\x03\x01 not a request at all\r\n\r\n",
])
async def test_a_request_that_asks_for_no_tunnel_is_refused_and_nothing_is_looked_up(far, request_):
    names = Names(shop_example="127.0.0.1")
    door, out = await opened(names)
    status, _, back = await ask(out, request_.format(port=far.port).encode("latin-1"))
    assert (status, names.asked, far.reached) == (403, [], 0)
    back.close()
    await door.close()


async def test_a_request_that_never_ends_or_is_too_long_is_dropped(far, monkeypatch):
    names = Names(shop_example="127.0.0.1")
    door, out = await opened(names)
    to = f"shop.example:{far.port}"
    # Too many lines, and one line too long to be a header.
    for request in (f"CONNECT {to} HTTP/1.1\r\n" + "X-More: 1\r\n" * (gate.LINES + 1) + "\r\n",
                    f"CONNECT {to} HTTP/1.1\r\nX-Long: " + "x" * 70_000 + "\r\n\r\n"):
        status, _, back = await ask(out, request.encode())
        assert (status, far.reached) == (0, 0)
        back.close()
    # The browser hung up before its request was whole.
    sent, back = await asyncio.open_connection("127.0.0.1", int(out.rsplit(":", 1)[1]))
    back.write(f"CONNECT {to} HTTP/1.1\r\n".encode())
    back.write_eof()
    assert await soon(sent.read()) == b"" and far.reached == 0
    back.close()
    # And one that says nothing is not waited for without end.
    monkeypatch.setattr(gate, "ASK", 0.05)
    sent, back = await asyncio.open_connection("127.0.0.1", int(out.rsplit(":", 1)[1]))
    assert await soon(sent.read()) == b""
    back.close()
    assert names.asked == []
    await door.close()


async def test_a_lookup_that_hangs_or_breaks_leaves_the_door_shut(far, monkeypatch):
    monkeypatch.setattr(gate, "ASK", 0.05)

    async def hangs(host: str) -> str:
        await asyncio.sleep(5)
        return "127.0.0.1"

    async def breaks(host: str) -> str:
        raise RuntimeError("the lookup broke")

    unhandled: list[dict] = []
    asyncio.get_running_loop().set_exception_handler(lambda loop, context: unhandled.append(context))
    for where in (hangs, breaks):
        door = Gate(where)
        status, _, back = await tunnel(await door.open(), f"shop.example:{far.port}")
        assert (status, far.reached) == (0, 0)
        back.close()
        await door.close()
    gc.collect()
    await asyncio.sleep(0)
    assert unhandled == [], "it is noted and the connection ended, not left as an error nobody took"


async def test_a_host_that_does_not_answer_is_said_to_be_unreached(monkeypatch):
    async with Far() as gone:
        port = gone.port
    names = Names(shop_example="127.0.0.1")
    door, out = await opened(names)
    status, _, back = await tunnel(out, f"shop.example:{port}")   # nothing listens there any more
    assert status == 502
    back.close()

    monkeypatch.setattr(gate, "REACH", 0.05)

    async def never(*to: object) -> None:
        await asyncio.sleep(5)

    monkeypatch.setattr(gate.asyncio, "open_connection", never)
    request = f"CONNECT shop.example:{port} HTTP/1.1\r\n\r\n".encode()
    sent, back = await _plain(out)
    back.write(request)
    assert (await soon(sent.readline())).split()[1:2] == [b"502"], "nor is one that takes too long waited for"
    back.close()
    await door.close()


async def _plain(out: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """A connection to the gate made without ``asyncio.open_connection``, which one test replaces."""
    loop = asyncio.get_running_loop()
    sent = asyncio.StreamReader()
    transport, protocol = await loop.create_connection(lambda: asyncio.StreamReaderProtocol(sent), "127.0.0.1",
                                                       int(out.rsplit(":", 1)[1]))
    return sent, asyncio.StreamWriter(transport, protocol, sent, loop)


async def test_a_tunnel_ends_when_either_end_does(far):
    names = Names(shop_example="127.0.0.1")
    door, out = await opened(names)
    status, sent, back = await tunnel(out, f"shop.example:{far.port}")
    assert status == 200
    back.write(b"last")
    back.write_eof()
    assert await soon(sent.read()) == b"far:last", "what was on its way still arrives, and then the tunnel is shut"
    back.close()
    await door.close()


async def test_shutting_the_gate_ends_its_tunnels_and_it_listens_no_more(far):
    names = Names(shop_example="127.0.0.1")
    door = Gate(names.where)
    await door.close()   # one never opened has nothing to shut
    out = await door.open()
    status, sent, back = await tunnel(out, f"shop.example:{far.port}")
    silent = await _plain(out)   # a browser that has connected and not yet said what it wants
    await asyncio.sleep(0.01)
    late = await _plain(out)     # and one that connects just as the gate is shut
    assert status == 200
    began = asyncio.get_running_loop().time()
    with warnings.catch_warnings(record=True) as left_open:
        warnings.simplefilter("always", ResourceWarning)
        await soon(door.close())
        assert asyncio.get_running_loop().time() - began < 1, "nothing was left to wait for"
        for ended in (sent, silent[0], late[0]):
            assert await soon(ended.read()) == b""
        assert await far.was_left() == 1, "the far end of a tunnel is let go as well"
        gc.collect()
    assert not left_open, "and both its ends were shut here, not left for Python to tidy away"
    with pytest.raises(OSError):
        await asyncio.open_connection("127.0.0.1", int(out.rsplit(":", 1)[1]))
    await door.close()
    for end in (back, silent[1], late[1]):
        end.close()


async def test_a_connection_handed_over_as_the_gate_shuts_is_ended_unserved():
    # The loop may hand a connection over after the gate has ended the ones it knew of.
    names = Names(shop_example="127.0.0.1")
    door = Gate(names.where)
    await door.open()
    await door.close()
    ended: list[bool] = []
    door._accept(asyncio.StreamReader(), SimpleNamespace(close=lambda: ended.append(True)))
    await asyncio.sleep(0)
    assert ended == [True] and names.asked == []
