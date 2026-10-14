"""Carry a routine's recorded usage across executions and restarts."""

from __future__ import annotations

import math
from types import SimpleNamespace

from rune.agent.request_budget import BudgetExceeded


def check_followup_budget(owner, previous: dict, *, verify_artifacts: bool = False) -> None:
    """Check that verification and a final reply fit before consuming approval."""
    saved = (previous.get("timings") or {}).get("followupBudget")
    if saved is None:
        return
    tokens, cost = saved.get("tokens"), saved.get("costUsd")
    limit = owner._config.cost_budget_usd
    requests = 2 if verify_artifacts else 1
    calls = owner._config.model_request_limit
    if (isinstance(tokens, int) and tokens * requests > owner._token_budget.total
            or calls is not None and calls < requests
            or limit is not None and (cost is None or cost * requests > limit)):
        raise BudgetExceeded("Too little budget remains for a post-action check. The action was not executed; its approval is still pending.")


def restore_input_estimates(previous: dict | None) -> dict:
    saved = ((previous or {}).get("timings") or {}).get("inputEstimates") or {}
    if not isinstance(saved, dict):
        return {}
    return {key: value for key, value in list(saved.items())[-64:]
            if isinstance(key, str) and len(key) == 24 and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 1}


def recorded_usage(result: dict) -> dict:
    usage = (result.get("timings") or {}).get("usage") or {}
    empty = bool(result.get("reused") or result.get("not_started"))

    def count(key):
        value = usage.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    calls, reported, tokens = count("calls"), count("reported_calls"), count("total_tokens")
    complete = calls is not None and reported == calls and not usage.get("incomplete_calls")
    cost = usage.get("cost_usd")
    cost_valid = isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0
    return {
        "modelRequests": 0 if empty else calls or 0,
        "tokens": 0 if empty else tokens or 0,
        "costUsd": 0.0 if empty or not cost_valid else cost,
        "requestsKnown": empty or calls is not None,
        "tokensKnown": empty or (complete and tokens is not None),
        "costKnown": empty or (complete and cost_valid and usage.get("unpriced_calls") == 0),
    }


def budget_state(policy, usage: dict) -> dict:
    remaining = {}
    reason = ""
    for limit, used, known in (("total_tokens", "tokens", "tokensKnown"),
                               ("total_model_requests", "modelRequests", "requestsKnown"),
                               ("total_cost_usd", "costUsd", "costKnown")):
        cap = getattr(policy, limit)
        remaining[used] = None if cap is None else max(0, cap - usage[used])
        label = {"tokens": "total token", "modelRequests": "total model request", "costUsd": "total estimated cost"}[used]
        if cap is not None and not reason:
            if not usage[known]:
                reason = f"Earlier usage is incomplete; the task's {label} limit cannot be checked."
            elif remaining[used] <= 0:
                reason = f"The task's {label} limit was reached."
    return {"used": usage, "remaining": remaining, "blocked": bool(reason), "reason": reason}


def budget_owner(policy, usage: dict):
    from rune.config import get_config

    state = budget_state(policy, usage)
    if state["blocked"]:
        raise BudgetExceeded(state["reason"])
    remaining = state["remaining"]
    config = get_config().llm

    def smaller(*values):
        return min(value for value in values if value is not None) if any(v is not None for v in values) else None

    return SimpleNamespace(
        _token_budget=SimpleNamespace(total=smaller(policy.token_budget, remaining["tokens"]), used=0),
        _config=SimpleNamespace(
            model_request_limit=smaller(config.model_request_limit, remaining["modelRequests"]),
            cost_budget_usd=smaller(config.cost_budget_usd, remaining["costUsd"]),
        ),
    )
