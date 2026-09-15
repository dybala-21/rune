"""Retry explicit parameter incompatibilities without changing the model."""

from collections.abc import Awaitable, Callable
from typing import Any

from rune.agent.model_traits import (
    is_max_tokens_rename_error,
    is_reasoning_effort_error,
    is_responses_only_error,
    is_temperature_error,
    needs_responses_api,
    note_completion_tokens_required,
    note_reasoning_effort_rejected,
    note_responses_only,
    note_temperature_rejected,
    reasoning_effort_rejected,
    traits,
)
from rune.agent.timing import timed_completion
from rune.llm.reasoning import apply_reasoning_control
from rune.utils.logger import get_logger

log = get_logger(__name__)


def _route_responses(params: dict[str, Any]) -> bool:
    model = params["model"]
    provider, separator, name = model.partition("/")
    if not separator:
        params["model"] = f"openai/responses/{model}"
    elif provider in {"openai", "azure"}:
        if not name.startswith("responses/"):
            params["model"] = f"{provider}/responses/{name}"
    else:
        return False
    params.setdefault("store", False)
    # LiteLLM's chat parameter filter drops newer reasoning levels.
    # Pass the native fields through its Responses escape hatch.
    body = dict(params.get("extra_body") or {})
    body["store"] = params["store"]
    effort = params.pop("reasoning_effort", None)
    if effort is not None:
        body["reasoning"] = {**(body.get("reasoning") or {}), "effort": effort}
    params["extra_body"] = body
    return True


async def compatible_completion(
    completion: Callable[..., Awaitable[Any]], bad_request: type[Exception],
    kwargs: dict[str, Any],
) -> Any:
    params = dict(kwargs)
    apply_reasoning_control(params)
    model = params["model"]
    capabilities = traits(model)
    if reasoning_effort_rejected(model):
        params.pop("reasoning_effort", None)
    if not capabilities.temperature:
        params.pop("temperature", None)
    if capabilities.max_completion_tokens and "max_tokens" in params:
        params["max_completion_tokens"] = params.pop("max_tokens")
    routed = False
    if capabilities.responses_api or needs_responses_api(model) or "/responses/" in model:
        routed = _route_responses(params)
    # Each correction can happen once: endpoint, token cap, temperature, effort.
    retries_left = 4
    while True:
        try:
            return await timed_completion(completion, params)
        except bad_request as exc:
            if retries_left == 0:
                raise
            retries_left -= 1
            if not routed and is_responses_only_error(exc) and _route_responses(params):
                routed = True
                note_responses_only(model)
                log.warning("responses_api_required", model=model)
            elif "temperature" in params and is_temperature_error(exc):
                note_temperature_rejected(model)
                params.pop("temperature")
                log.warning("temperature_unsupported_retry", model=model)
            elif "max_tokens" in params and is_max_tokens_rename_error(exc):
                note_completion_tokens_required(model)
                params["max_completion_tokens"] = params.pop("max_tokens")
                log.warning("completion_token_parameter_retry", model=model)
            elif "reasoning_effort" in params and is_reasoning_effort_error(exc):
                note_reasoning_effort_rejected(model)
                params.pop("reasoning_effort")
                log.warning("reasoning_effort_unsupported_retry", model=model)
            else:
                raise
