"""Per-run latency and token usage without storing prompts or tool arguments."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any

from rune.llm.usage import merge_usage, token_counts

_current: ContextVar[dict | None] = ContextVar("run_timing", default=None)
_phase: ContextVar[str] = ContextVar("timing_phase", default="setup")
_TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens",
                 "cache_write_tokens", "reasoning_tokens")


def _usage_totals() -> dict:
    return {"calls": 0, "reported_calls": 0, "cache_write_unreported_calls": 0,
            "cost_usd": 0.0, "unpriced_calls": 0,
            **dict.fromkeys(_TOKEN_FIELDS, 0)}


def current_usage() -> dict | None:
    run = _current.get()
    return copy.deepcopy(run["usage"]) if run is not None else None


def _record_usage(run: dict, row: dict, response: Any, *, streaming: bool, request: dict) -> None:
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    counts = token_counts(usage)
    if counts is None:
        return
    previous = row.get("usage")
    counts = merge_usage(previous, counts)
    from rune.llm.pricing import estimate_request_cost
    old_cost = row.get("costUsd")
    cost = estimate_request_cost(row["model"], counts, request)
    row["costUsd"] = cost
    row["usage"] = counts
    missing_before = previous is not None and not previous["cache_write_reported"]
    missing_after = not counts["cache_write_reported"]
    for totals in (run["usage"], run["usage"]["by_model"][row["model"]]):
        totals["reported_calls"] += int(previous is None)
        totals["cost_usd"] += (cost or 0) - (old_cost or 0)
        totals["unpriced_calls"] += int(cost is None) - int(previous is not None and old_cost is None)
        totals["cache_write_unreported_calls"] += int(missing_after) - int(missing_before)
        for key in _TOKEN_FIELDS:
            totals[key] += counts[key] - (previous[key] if previous else 0)
    # The streaming loop accounts for its own usage; auxiliary calls share its budget here.
    budget = getattr(run["owner"], "_token_budget", None)
    if not streaming and budget is not None:
        budget.used += counts["total_tokens"] - (previous["total_tokens"] if previous else 0)


@contextmanager
def timing_phase(name: str):
    token = _phase.set(name)
    try:
        with span("phase", name=name):
            yield
    finally:
        _phase.reset(token)


@contextmanager
def span(kind: str, **fields: Any):
    run = _current.get()
    started = time.monotonic()
    row = {"kind": kind, "phase": _phase.get(), **fields}
    if run is not None:
        row["startMs"] = round((started - run["started"]) * 1000, 1)
        if len(run["spans"]) < 1024:
            run["spans"].append(row)
        else:
            run["droppedSpans"] += 1
    try:
        yield row
    except BaseException as exc:
        row["status"] = "interrupted"
        if kind == "model":
            from rune.llm.failures import request_failure
            row["error"] = request_failure(exc).to_dict()
        raise
    else:
        row["status"] = "finished"
    finally:
        row["durationMs"] = round((time.monotonic() - started) * 1000, 1)


def timed(kind: str, *, name_arg: int | None = None):
    def decorate(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            name = str(args[name_arg]) if name_arg is not None and len(args) > name_arg else fn.__name__
            with span(kind, name=name):
                return await fn(*args, **kwargs)
        return wrapper
    return decorate


@contextmanager
def capture_timing(owner=None):
    run = {"started": time.monotonic(), "spans": [], "droppedSpans": 0,
           "owner": owner, "usage": {**_usage_totals(), "by_model": {}}}
    token = _current.set(run)
    try:
        yield run
    finally:
        _current.reset(token)


def timing_snapshot(run: dict) -> dict:
    return {"totalMs": round((time.monotonic() - run["started"]) * 1000, 1),
            "spans": copy.deepcopy(run["spans"]), "droppedSpans": run["droppedSpans"],
            "usage": copy.deepcopy(run["usage"])}


def timed_run(fn):
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        with capture_timing(args[0] if args else None) as run:
            result = await fn(*args, **kwargs)
            result.timings = timing_snapshot(run)
            return result
    return wrapper


async def timed_completion(completion, params):
    run = _current.get()
    if run is None:
        return await completion(**params)
    measurement = span("model", model=params.get("model"),
                       reasoningEffort=params.get("reasoning_effort") or
                       (params.get("extra_body") or {}).get("reasoning_effort") or
                       (params.get("extra_body") or {}).get("reasoning", {}).get("effort"))
    started = time.monotonic()
    row = measurement.__enter__()
    if params.get("tools"):
        # Store only a hash of the tool catalog.
        catalog = json.dumps(params["tools"], sort_keys=True, separators=(",", ":"), default=str)
        row["toolCatalogHash"] = hashlib.sha256(catalog.encode()).hexdigest()[:16]
        row["toolCount"] = len(params["tools"])
    run["usage"]["calls"] += 1
    run["usage"]["by_model"].setdefault(row["model"], _usage_totals())["calls"] += 1
    try:
        response = await completion(**params)
    except BaseException as exc:
        measurement.__exit__(type(exc), exc, exc.__traceback__)
        raise
    if not params.get("stream"):
        _record_usage(run, row, response, streaming=False, request=params)
        measurement.__exit__(None, None, None)
        return response

    async def stream():
        try:
            async for chunk in response:
                _record_usage(run, row, chunk, streaming=True, request=params)
                elapsed = round((time.monotonic() - started) * 1000, 1)
                row.setdefault("firstEventMs", elapsed)
                row["lastEventMs"] = elapsed
                row["eventCount"] = row.get("eventCount", 0) + 1
                for choice in getattr(chunk, "choices", None) or []:
                    if getattr(getattr(choice, "delta", None), "content", None):
                        row.setdefault("firstTextMs", round((time.monotonic() - started) * 1000, 1))
                yield chunk
        except BaseException as exc:
            measurement.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            measurement.__exit__(None, None, None)
    return stream()
