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

from rune.agent.request_budget import bound_budget, scopes
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


def streamed_tokens() -> int:
    run = _current.get()
    return run["streamed_tokens"] if run is not None else 0


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
    delta = counts["total_tokens"] - (previous["total_tokens"] if previous else 0)
    updated = set()
    for scope in scopes(run):
        for totals in (scope["usage"], scope["usage"]["by_model"][row["model"]]):
            totals["reported_calls"] += int(previous is None)
            totals["cost_usd"] += (cost or 0) - (old_cost or 0)
            totals["unpriced_calls"] += int(cost is None) - int(previous is not None and old_cost is None)
            totals["cache_write_unreported_calls"] += int(missing_after) - int(missing_before)
            for key in _TOKEN_FIELDS:
                totals[key] += counts[key] - (previous[key] if previous else 0)
        if streaming:
            scope["streamed_tokens"] += delta
        budget = bound_budget(scope)
        if budget is not None and id(budget) not in updated:
            budget.used += delta
            updated.add(id(budget))
        reservation = row.get("reservation")
        if reservation in scope["reservations"]:
            scope["reservations"][reservation] = max(0, scope["reservations"][reservation] - delta)


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
    parent = _current.get()
    if parent is not None and getattr(parent["owner"], "_token_budget", None) is None:
        parent = None
    run = {"started": time.monotonic(), "spans": [], "droppedSpans": 0,
           "owner": owner, "usage": {**_usage_totals(), "by_model": {}},
           "streamed_tokens": 0, "reservations": {}, "budget_blocked": "", "parent": parent}
    token = _current.set(run)
    try:
        yield run
    finally:
        run["closed"] = True
        _current.reset(token)


def timing_snapshot(run: dict) -> dict:
    return {"totalMs": round((time.monotonic() - run["started"]) * 1000, 1),
            "spans": copy.deepcopy(run["spans"]), "droppedSpans": run["droppedSpans"],
            "usage": copy.deepcopy(run["usage"])}


def timed_run(fn):
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        with capture_timing(args[0] if args else None) as run:
            try:
                result = await fn(*args, **kwargs)
            finally:
                snapshot = timing_snapshot(run)
                if args and hasattr(args[0], "__dict__"):
                    args[0]._last_run_timings = snapshot
            result.timings = snapshot
            budget = bound_budget(run)
            if budget is not None:
                result.total_tokens_used = budget.used
            if run["budget_blocked"]:
                result.reason = "request_budget_exhausted"
                result.completion_check = {"requirement": "Model budget", "detail": run["budget_blocked"]}
            return result
    return wrapper


async def timed_completion(completion, params):
    run = _current.get()
    if run is None:
        return await completion(**params)
    from rune.agent.request_budget import BudgetExceeded, admit, input_estimate, settle

    estimated = input_estimate(params) if any(getattr(bound_budget(s), "total", None) is not None for s in scopes(run)) else 0
    reservation = 0
    for scope in scopes(run):
        try:
            params, allowance = admit(scope, params, estimated)
            if allowance:
                reservation = allowance
        except BudgetExceeded as exc:
            scope["budget_blocked"] = run["budget_blocked"] = str(exc)
            raise
    measurement = span("model", model=params.get("model"),
                       reasoningEffort=params.get("reasoning_effort") or
                       (params.get("extra_body") or {}).get("reasoning_effort") or
                       (params.get("extra_body") or {}).get("reasoning", {}).get("effort"))
    started = time.monotonic()
    row = measurement.__enter__()
    if reservation:
        row["reservation"] = id(row)
        for scope in scopes(run):
            scope["reservations"][row["reservation"]] = reservation
    if params.get("tools"):
        # Fingerprint the catalog without recording its contents.
        catalog = json.dumps(params["tools"], sort_keys=True, separators=(",", ":"), default=str)
        row["toolCatalogHash"] = hashlib.sha256(catalog.encode()).hexdigest()[:16]
        row["toolCount"] = len(params["tools"])
    for scope in scopes(run):
        scope["usage"]["calls"] += 1
        scope["usage"]["by_model"].setdefault(row["model"], _usage_totals())["calls"] += 1
    try:
        response = await completion(**params)
    except BaseException as exc:
        settle(run, row, exc)
        measurement.__exit__(type(exc), exc, exc.__traceback__)
        raise
    if not params.get("stream"):
        _record_usage(run, row, response, streaming=False, request=params)
        settle(run, row)
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
            settle(run, row, exc)
            measurement.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            settle(run, row)
            measurement.__exit__(None, None, None)
    return stream()
