"""Classify request failures for logging and retry decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx


@dataclass(frozen=True)
class RequestFailure:
    kind: str
    status: int | None = None
    retryable: bool = False
    retry_after: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def request_failure(exc: BaseException) -> RequestFailure:
    chain: list[BaseException] = []
    current = exc
    while current is not None and all(current is not previous for previous in chain):
        chain.append(current)
        current = current.__cause__ or current.__context__
    for error in reversed(chain):
        for kind, types, retryable in (
            ("connect_timeout", (httpx.ConnectTimeout,), True),
            ("pool_timeout", (httpx.PoolTimeout,), True),
            ("read_timeout", (httpx.ReadTimeout,), False),
            ("write_timeout", (httpx.WriteTimeout,), False),
            ("connection_error", (httpx.ConnectError,), True),
            ("read_error", (httpx.ReadError,), False),
            ("write_error", (httpx.WriteError,), False),
            ("protocol_error", (httpx.RemoteProtocolError,), False),
        ):
            if isinstance(error, types):
                return RequestFailure(kind, retryable=retryable)
    # A timeout without a transport stage may have reached the provider.
    if any(isinstance(error, TimeoutError) or type(error).__name__ in {"Timeout", "APITimeoutError"}
           for error in chain):
        return RequestFailure("timeout")
    for error in chain:
        status = getattr(error, "status_code", None)
        response = getattr(error, "response", None)
        status = status or getattr(response, "status_code", None)
        if isinstance(status, int):
            headers = getattr(response, "headers", None) or getattr(error, "headers", None) or {}
            delay = None
            try:
                value = headers.get("retry-after") or headers.get("Retry-After")
                if value is not None:
                    try:
                        delay = max(0.0, float(value))
                    except ValueError:
                        delay = max(0.0, (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                delay = None
            return RequestFailure("rate_limit" if status == 429 else "http_error", status,
                                  status in {429, 502, 503, 504}, delay)
    return RequestFailure(type(exc).__name__)
