"""Admit model requests against the run's remaining budget."""

from __future__ import annotations

import json


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

    content = {key: compact(params[key]) for key in ("messages", "tools", "state", "questions") if key in params}
    return count_tokens(json.dumps(content, ensure_ascii=False, default=str)) + 16


def admit(run: dict, params: dict, estimated_input: int | None = None) -> tuple[dict, int]:
    if run.get("closed"):
        raise BudgetExceeded("The parent run ended; it cannot authorize further model requests.")
    if run["budget_blocked"]:
        raise BudgetExceeded(run["budget_blocked"])
    owner = run["owner"]
    budget = bound_budget(run)
    if budget is None or not hasattr(budget, "total"):
        return params, 0
    config = getattr(owner, "_config", None)
    usage = run["usage"]
    limit = getattr(config, "model_request_limit", None)
    if limit is not None and usage["calls"] >= limit:
        raise BudgetExceeded("The model request limit was reached. No further request was sent.")
    cost_limit = getattr(config, "cost_budget_usd", None)
    if cost_limit is not None:
        if run.get("unpriced_finished") or usage["unpriced_calls"]:
            raise BudgetExceeded("Model cost could not be determined; the configured cost limit cannot be checked.")
        if usage["cost_usd"] >= cost_limit:
            raise BudgetExceeded("The recorded model cost reached the configured limit. No further request was sent.")
    remaining = budget.total - budget.used - sum(run["reservations"].values())
    if remaining <= 0:
        raise BudgetExceeded("The token budget was reached. No further request was sent.")
    expected_input = input_estimate(params) if estimated_input is None else estimated_input
    if expected_input >= remaining:
        raise BudgetExceeded("The next prompt exceeds the estimated remaining token budget.")
    params = dict(params)
    output_key = "max_completion_tokens" if "max_completion_tokens" in params else "max_tokens"
    output = min(params.get(output_key, 4096), remaining - expected_input)
    if output <= 0:
        raise BudgetExceeded("The output token allowance is zero. No request was sent.")
    # Jev's choice API has no output-token parameter.
    if "messages" in params:
        params[output_key] = output
        params["num_retries"] = 0
        params["max_retries"] = 0
    return params, expected_input + output


def settle(run: dict, row: dict, error: BaseException | None = None) -> None:
    reservation = row.pop("reservation", None)
    if reservation is None:
        return
    from rune.llm.failures import request_failure

    failure = request_failure(error) if error is not None else None
    # Release reservations for requests rejected before generation.
    rejected = failure is not None and (failure.kind in {"connect_timeout", "pool_timeout", "connection_error"}
                                        or failure.status in {400, 401, 403, 404, 422, 429})
    for scope in scopes(run):
        if (row.get("usage") is not None and error is None) or rejected:
            scope["reservations"].pop(reservation, None)
        else:
            scope["unpriced_finished"] = True
            if row.get("usage") is not None:
                row["usageIncomplete"] = True
                scope["usage"]["incomplete_calls"] = scope["usage"].get("incomplete_calls", 0) + 1
