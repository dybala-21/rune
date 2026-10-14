"""Per-run latency and token usage without storing prompts or tool arguments."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from types import SimpleNamespace
from typing import Any

from rune.agent.request_budget import bound_budget, remaining_budget, scopes
from rune.llm.usage import merge_usage, token_counts
from rune.utils.logger import get_logger

log = get_logger(__name__)
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


def remaining_tokens() -> int | None:
    run = _current.get()
    budgets = [remaining_budget(scope) for scope in scopes(run)] if run is not None else []
    values = [budget["tokens"] for budget in budgets if budget is not None]
    return min(values) if values else None


def check_isolated_worker_budget() -> None:
    from rune.agent.request_budget import BudgetExceeded

    run = _current.get()
    if run is not None and any(getattr(bound_budget(s), "total", None) is not None for s in scopes(run)):
        # Subprocess workers cannot yet reserve or settle against their parent's budget.
        run["budget_blocked"] = "Isolated workers cannot enforce the shared run budget. No worker was started."
        raise BudgetExceeded(run["budget_blocked"])


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
    if row.get("toolCount"):
        from rune.llm.pricing import reserve_request_cost
        next_input = max(row.get("reservedInput", 0), math.ceil(counts["input_tokens"] * 1.05))
        next_output = max(128, counts["output_tokens"])
        row["followupBudget"] = {"tokens": next_input + next_output,
                                  "costUsd": reserve_request_cost(request, next_input, next_output)}
    missing_before = previous is not None and not previous["cache_write_reported"]
    missing_after = not counts["cache_write_reported"]
    delta = counts["total_tokens"] - (previous["total_tokens"] if previous else 0)
    updated = set()
    for scope in scopes(run):
        if row.get("followupBudget"):
            scope["followup_budget"] = row["followupBudget"]
        if row.get("inputEstimate", 0) > 0:
            key = row["estimateKey"]
            estimates = scope.setdefault("input_estimates", {})
            estimates[key] = max(estimates.get(key, 1.0), counts["input_tokens"] / row["inputEstimate"])
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
        if reservation in scope.get("cost_reservations", {}) and cost is not None:
            scope["cost_reservations"][reservation] = max(0, scope["cost_reservations"][reservation] - (cost - (old_cost or 0)))


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
    if (parent is not None and getattr(parent["owner"], "_token_budget", None) is None
            and not any(getattr(getattr(parent["owner"], "_config", None), field, None) is not None
                        for field in ("cost_budget_usd", "model_request_limit"))):
        parent = None
    run = {"started": time.monotonic(), "spans": [], "droppedSpans": 0,
           "owner": owner, "usage": {**_usage_totals(), "by_model": {}},
           "streamed_tokens": 0, "reservations": {}, "cost_reservations": {},
           "input_estimates": getattr(owner, "input_estimates", {}), "budget_blocked": "", "parent": parent}
    token = _current.set(run)
    from rune.llm.context import model_scope
    try:
        with model_scope():
            yield run
    finally:
        run["closed"] = True
        _current.reset(token)


class _ExecutionTokens:
    def __init__(self, loop):
        self._loop = loop
        self.used = 0

    @property
    def total(self):
        override = self._loop._config.token_budget_override
        return override if override is not None else self._loop._token_budget.total


@contextmanager
def capture_execution(loop):
    # Keep preflight and follow-up usage when the loop resets its own counters.
    owner = SimpleNamespace(_config=loop._config, _token_budget=_ExecutionTokens(loop))
    with capture_timing(owner) as run:
        yield run


def timing_snapshot(run: dict) -> dict:
    return {"totalMs": round((time.monotonic() - run["started"]) * 1000, 1),
            "spans": copy.deepcopy(run["spans"]), "droppedSpans": run["droppedSpans"],
            "remainingBudget": remaining_budget(run),
            "usage": copy.deepcopy(run["usage"]),
            "inputEstimates": dict(run["input_estimates"]),
            **({"followupBudget": dict(run["followup_budget"])} if run.get("followup_budget") else {})}


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


async def close_stream(response) -> None:
    if close := getattr(response, "aclose", None):
        try:
            async with asyncio.timeout(5):
                await close()
        except Exception as exc:
            log.debug("model_stream_close_failed", error_type=type(exc).__name__)


class _MeasuredStream:
    def __init__(self, response, record, finish):
        self._response = response
        self._iterator = response.__aiter__()
        self._record = record
        self._finish = finish
        self._closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration
        try:
            chunk = await anext(self._iterator)
            self._record(chunk)
            return chunk
        except StopAsyncIteration:
            await self._close(None)
            raise
        except BaseException as exc:
            await self._close(exc)
            raise

    async def aclose(self):
        await self._close(GeneratorExit())

    async def _close(self, error):
        if self._closed:
            return
        self._closed = True
        try:
            self._finish(error)
        finally:
            response = self._response if hasattr(self._response, "aclose") else self._iterator
            await close_stream(response)


async def timed_completion(completion, params):
    run = _current.get()
    wire_params = {key: value for key, value in params.items() if key != "_rune_source_model"}
    if run is None:
        return await completion(**wire_params)
    from rune.agent.request_budget import (
        BudgetExceeded,
        admit,
        calibrated_input,
        check_request_budget,
        estimate_key,
        input_estimate,
        settle,
    )

    parents = tuple(scopes(run))
    reservation = 0
    try:
        # Check the budget before tokenizing the prompt.
        for scope in parents:
            check_request_budget(scope)
        estimated = input_estimate(params) if any(getattr(bound_budget(s), "total", None) is not None
            or getattr(getattr(s["owner"], "_config", None), "cost_budget_usd", None) is not None for s in parents) else 0
        reserved_input = calibrated_input(run, params, estimated)
        for scope in parents:
            params, allowance = admit(scope, params, reserved_input)
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
    row.update(inputEstimate=estimated, reservedInput=reserved_input, estimateKey=estimate_key(params))
    if reservation:
        row["reservation"] = id(row)
        for scope in scopes(run):
            scope["reservations"][row["reservation"]] = reservation
            if getattr(getattr(scope["owner"], "_config", None), "cost_budget_usd", None) is not None:
                from rune.llm.pricing import reserve_request_cost
                scope["cost_reservations"][row["reservation"]] = reserve_request_cost(
                    params, reserved_input, params.get("max_completion_tokens", params.get("max_tokens", 4096)))
    if params.get("tools"):
        # Store a hash, not the tool definitions.
        catalog = json.dumps(params["tools"], sort_keys=True, separators=(",", ":"), default=str)
        row["toolCatalogHash"] = hashlib.sha256(catalog.encode()).hexdigest()[:16]
        row["toolCount"] = len(params["tools"])
    for scope in scopes(run):
        scope["usage"]["calls"] += 1
        scope["usage"]["by_model"].setdefault(row["model"], _usage_totals())["calls"] += 1
    try:
        response = await completion(**{key: value for key, value in params.items() if key != "_rune_source_model"})
    except BaseException as exc:
        settle(run, row, exc)
        measurement.__exit__(type(exc), exc, exc.__traceback__)
        raise
    if not params.get("stream"):
        _record_usage(run, row, response, streaming=False, request=params)
        settle(run, row)
        measurement.__exit__(None, None, None)
        return response

    def record(chunk):
        _record_usage(run, row, chunk, streaming=True, request=params)
        elapsed = round((time.monotonic() - started) * 1000, 1)
        row.setdefault("firstEventMs", elapsed)
        row["lastEventMs"] = elapsed
        row["eventCount"] = row.get("eventCount", 0) + 1
        for choice in getattr(chunk, "choices", None) or []:
            if getattr(getattr(choice, "delta", None), "content", None):
                row.setdefault("firstTextMs", elapsed)

    def finish(error):
        settle(run, row, error)
        measurement.__exit__(type(error) if error else None, error, getattr(error, "__traceback__", None))

    return _MeasuredStream(response, record, finish)
