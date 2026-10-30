"""Admit model requests against the run's remaining budget."""

from __future__ import annotations

import hashlib
import json
import math


class BudgetExceeded(RuntimeError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.reason = "request_budget_exhausted"


def scopes(run: dict):
    while run is not None:
        yield run
        run = run.get("parent")


def bound_budget(run: dict):
    if "bound_budget" not in run:
        run["bound_budget"] = getattr(run["owner"], "_token_budget", None)
    return run["bound_budget"]


def estimate_key(params: dict) -> str:
    target = {key: params.get(key) for key in ("model", "api_base", "base_url", "vertex_location")}
    target["tools"] = [(tool.get("function") or {}).get("name", tool.get("type")) for tool in params.get("tools") or []]
    target["response_format"] = params.get("response_format")
    target["native_format"] = (params.get("output_config") or {}).get("format")
    return hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest()[:24]


def calibrated_input(run: dict, params: dict, estimate: int) -> int:
    key = estimate_key(params)
    scale = max((scope.get("input_estimates", {}).get(key, 1.0) for scope in scopes(run)), default=1.0)
    return math.ceil(estimate * (scale * 1.05 if scale > 1 else 1))


def remaining_budget(run: dict) -> dict | None:
    budget = bound_budget(run)
    if budget is None or not hasattr(budget, "total"):
        return None
    config = getattr(run["owner"], "_config", None)
    usage = run["usage"]
    calls = getattr(config, "model_request_limit", None)
    cost = getattr(config, "cost_budget_usd", None)
    tokens = max(0, budget.total - budget.used - sum(run["reservations"].values()))
    if run["budget_blocked"] or (cost is not None and (run.get("unpriced_finished") or usage["unpriced_calls"])):
        tokens = 0
    return {"tokens": tokens, "modelRequests": None if calls is None else max(0, calls - usage["calls"]),
            "costUsd": None if cost is None else max(0, cost - usage["cost_usd"] - sum(run.get("cost_reservations", {}).values()))}


def input_estimate(params: dict) -> int:
    from rune.utils.tokenizer import count_tokens

    def compact(value):
        if isinstance(value, list):
            return [compact(item) for item in value]
        if isinstance(value, dict):
            kind = value.get("type")
            if isinstance(kind, str) and kind in {"image_url", "input_image", "image"}:
                return " image" * 4096
            return {key: compact(item) for key, item in value.items()}
        return value

    content = {key: compact(params[key]) for key in ("messages", "tools", "response_format", "state", "questions") if key in params}
    if native_format := (params.get("output_config") or {}).get("format"):
        content["native_format"] = compact(native_format)
    if str(params.get("model", "")).startswith(("gemini/", "vertex_ai/gemini")):
        messages = content.get("messages", [])
        calls = {call.get("id") for message in messages if isinstance(message, dict)
                 and message.get("role") == "assistant" for call in message.get("tool_calls") or []
                 if isinstance(call, dict) and isinstance(call.get("id"), str)}
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            call_id = message.get("tool_call_id")
            if isinstance(call_id, str) and call_id in calls:
                # LiteLLM sends the thought signature only with the call.
                message["tool_call_id"] = call_id.split("__thought__", 1)[0]
    return count_tokens(json.dumps(content, ensure_ascii=False, default=str)) + 16


def check_request_budget(run: dict) -> None:
    if run.get("closed"):
        raise BudgetExceeded("The parent run ended; it cannot authorize further model requests.")
    if run["budget_blocked"]:
        raise BudgetExceeded(run["budget_blocked"])
    owner = run["owner"]
    budget = bound_budget(run)
    config = getattr(owner, "_config", None)
    usage = run["usage"]
    limit = getattr(config, "model_request_limit", None)
    if limit is not None and usage["calls"] >= limit:
        raise BudgetExceeded("The model request limit was reached. No further request was sent.")
    cost_limit = getattr(config, "cost_budget_usd", None)
    if cost_limit is not None:
        if run.get("unpriced_finished") or usage["unpriced_calls"]:
            raise BudgetExceeded("Model cost could not be determined; the configured cost limit cannot be checked.")
        if usage["cost_usd"] + sum(run.get("cost_reservations", {}).values()) >= cost_limit:
            raise BudgetExceeded("The recorded model cost reached the configured limit. No further request was sent.")
    if budget is None or not hasattr(budget, "total"):
        return
    remaining = budget.total - budget.used - sum(run["reservations"].values())
    if remaining <= 0:
        raise BudgetExceeded("The token budget was reached. No further request was sent.")


def admit(run: dict, params: dict, estimated_input: int | None = None) -> tuple[dict, int]:
    check_request_budget(run)
    budget = bound_budget(run)
    cost_limit = getattr(getattr(run["owner"], "_config", None), "cost_budget_usd", None)
    limited_tokens = budget is not None and hasattr(budget, "total")
    if not limited_tokens and cost_limit is None:
        return params, 0
    remaining = budget.total - budget.used - sum(run["reservations"].values()) if limited_tokens else None
    expected_input = input_estimate(params) if estimated_input is None else estimated_input
    if remaining is not None and expected_input >= remaining:
        raise BudgetExceeded("The next prompt exceeds the estimated remaining token budget.")
    params = dict(params)
    output_key = "max_completion_tokens" if "max_completion_tokens" in params else "max_tokens"
    output = params.get(output_key, 4096)
    if remaining is not None:
        output = min(output, remaining - expected_input)
    if cost_limit is not None:
        from rune.llm.pricing import reserve_request_cost
        base = reserve_request_cost(params, expected_input, 0)
        single = reserve_request_cost(params, expected_input, 1)
        if base is None or single is None:
            raise BudgetExceeded("Pricing is unavailable for this model or endpoint. No request was sent; the configured cost limit cannot be checked.")
        available = cost_limit - run["usage"]["cost_usd"] - sum(run.get("cost_reservations", {}).values())
        if base > available:
            raise BudgetExceeded("The next prompt exceeds the remaining cost budget. No request was sent.")
        if single > base:
            output = min(output, max(0, math.floor((available - base) / (single - base))))
    if output <= 0:
        raise BudgetExceeded("The output token allowance is zero. No request was sent.")
    # System One APIs do not accept output-token limits.
    if "messages" in params:
        params[output_key] = output
        params["num_retries"] = 0
        params["max_retries"] = 0
    return params, expected_input + output


def settle(run: dict, row: dict, error: BaseException | None = None) -> None:
    reservation = row.pop("reservation", None)
    if reservation is None:
        return
    rejected = rejected_before_generation(error)
    for scope in scopes(run):
        if (row.get("usage") is not None and error is None) or rejected:
            scope["reservations"].pop(reservation, None)
            scope.get("cost_reservations", {}).pop(reservation, None)
        else:
            scope["unpriced_finished"] = True
            if row.get("usage") is not None:
                row["usageIncomplete"] = True
                scope["usage"]["incomplete_calls"] = scope["usage"].get("incomplete_calls", 0) + 1


def rejected_before_generation(error: BaseException | None) -> bool:
    from rune.llm.failures import request_failure

    failure = request_failure(error) if error is not None else None
    return failure is not None and (failure.kind in {"connect_timeout", "pool_timeout", "connection_error"}
                                    or failure.status in {400, 401, 403, 404, 422, 429})
