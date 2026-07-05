"""Embedded terminal capability — gating, tokens, and a PTY echo roundtrip."""

from __future__ import annotations

import asyncio

import pytest

from rune.api import terminal as term


@pytest.fixture(autouse=True)
def _clear_tokens(monkeypatch):
    term._tokens.clear()
    monkeypatch.delenv("RUNE_TERMINAL_ENABLED", raising=False)
    yield
    term._tokens.clear()


def test_disabled_by_default(monkeypatch):
    monkeypatch.setattr("rune.config.get_config", lambda: object())
    assert term.is_enabled() is False


def test_enabled_via_env(monkeypatch):
    monkeypatch.setenv("RUNE_TERMINAL_ENABLED", "1")
    assert term.is_enabled() is True


def test_token_is_single_use(tmp_path):
    tok = term.mint_token(str(tmp_path))
    assert term.redeem_token(tok) == str(tmp_path)
    # Second redeem fails — one-shot.
    assert term.redeem_token(tok) is None
    assert term.redeem_token("garbage") is None


async def test_pty_echo_roundtrip(tmp_path):
    session = term.TerminalSession(str(tmp_path))
    session.start()
    try:
        session.write("echo rune_terminal_ok\n")
        collected = ""
        deadline = asyncio.get_event_loop().time() + 5
        while "rune_terminal_ok" not in collected:
            if asyncio.get_event_loop().time() > deadline:
                break
            chunk = await asyncio.wait_for(session.out_queue.get(), timeout=5)
            if chunk is None:
                break
            collected += chunk.decode("utf-8", "replace")
        assert "rune_terminal_ok" in collected
    finally:
        session.close()


async def test_close_signals_disconnect(tmp_path):
    session = term.TerminalSession(str(tmp_path))
    session.start()
    session.close()
    # A None sentinel must be queued so the WS pump can end.
    got_sentinel = False
    for _ in range(10):
        item = await asyncio.wait_for(session.out_queue.get(), timeout=2)
        if item is None:
            got_sentinel = True
            break
    assert got_sentinel


def test_mint_token_bounded_without_connecting(tmp_path):
    # Mint far past the cap without ever redeeming — map stays bounded.
    for _ in range(term._MAX_TOKENS * 3):
        term.mint_token(str(tmp_path))
    assert len(term._tokens) <= term._MAX_TOKENS


async def test_backpressure_pauses_and_resumes_reader(tmp_path, monkeypatch):
    # Small caps make the flood deterministic: a few 64KB reads overflow.
    monkeypatch.setattr(term.TerminalSession, "_MAX_QUEUE", 4)
    monkeypatch.setattr(term.TerminalSession, "_RESUME_AT", 1)
    session = term.TerminalSession(str(tmp_path))
    session.start()
    try:
        # ~1MB to stdout → many readable events, no draining → must pause.
        session.write("head -c 1000000 /dev/zero | tr '\\0' 'x'\n")
        for _ in range(100):
            await asyncio.sleep(0.02)
            if session._reader_paused:
                break
        assert session._reader_paused, "reader should pause under flood"
        assert session.out_queue.qsize() <= session._MAX_QUEUE  # bounded
        # Drain below low-water — reader re-arms.
        for _ in range(session._MAX_QUEUE + 1):
            if session.out_queue.qsize() == 0:
                break
            session.out_queue.get_nowait()
            session.notify_consumed()
        assert session._reader_paused is False
    finally:
        session.close()
