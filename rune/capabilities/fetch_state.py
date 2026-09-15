"""Briefly back off repeated GET failures within one agent run."""

from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from urllib.parse import urldefrag


@dataclass
class FetchState:
    failures: OrderedDict[str, tuple[int, float]] = field(default_factory=OrderedDict)

    def retry_after(self, url: str) -> float:
        key = urldefrag(url)[0]
        count, deadline = self.failures.get(key, (0, 0))
        if deadline <= time.monotonic():
            self.failures.pop(key, None)
            return 0
        return deadline - time.monotonic() if count >= 2 else 0

    def failed(self, url: str) -> None:
        self.retry_after(url)
        key = urldefrag(url)[0]
        count, _ = self.failures.pop(key, (0, 0))
        self.failures[key] = (count + 1, time.monotonic() + 30)
        while len(self.failures) > 128:
            self.failures.popitem(last=False)

    def succeeded(self, url: str) -> None:
        self.failures.pop(urldefrag(url)[0], None)


_current: ContextVar[FetchState | None] = ContextVar("fetch_state", default=None)


def current_fetch_state() -> FetchState | None:
    return _current.get()


@contextmanager
def fetch_scope():
    token = _current.set(FetchState())
    try:
        yield _current.get()
    finally:
        _current.reset(token)
