"""Retry explicit parameter incompatibilities without changing the model."""

from collections.abc import Awaitable, Callable
from typing import Any

from rune.agent.model_traits import (
    is_temperature_error,
    note_completion_tokens_required,
    note_temperature_rejected,
    traits,
)
from rune.llm.reasoning import apply_reasoning_control
from rune.utils.logger import get_logger

log = get_logger(__name__)


async def compatible_completion(
    completion: Callable[..., Awaitable[Any]], bad_request: type[Exception],
    kwargs: dict[str, Any],
) -> Any:
    params = dict(kwargs)
    apply_reasoning_control(params)
    model = params["model"]
    capabilities = traits(model)
    if not capabilities.temperature:
        params.pop("temperature", None)
    if capabilities.max_completion_tokens and "max_tokens" in params:
        params["max_completion_tokens"] = params.pop("max_tokens")
    if capabilities.responses_api:
        provider, separator, name = model.partition("/")
        if not separator:
            params["model"] = f"openai/responses/{model}"
        elif provider in {"openai", "azure"} and not name.startswith("responses/"):
            params["model"] = f"{provider}/responses/{name}"
        if "responses/" in params["model"]:
            params.setdefault("store", False)
            # LiteLLM's chat parameter filter drops newer reasoning levels.
            # Pass the native Responses fields through its documented escape hatch.
            body = dict(params.get("extra_body") or {})
            body["store"] = params["store"]
            effort = params.pop("reasoning_effort", None)
            if effort is not None:
                body["reasoning"] = {
                    **(body.get("reasoning") or {}),
                    **(effort if isinstance(effort, dict) else {"effort": effort}),
                }
            params["extra_body"] = body
    retries_left = 2
    while True:
        try:
            return await completion(**params)
        except bad_request as exc:
            if retries_left == 0:
                raise
            retries_left -= 1
            message = str(exc).lower()
            if "temperature" in params and is_temperature_error(exc):
                note_temperature_rejected(model)
                params.pop("temperature")
                log.warning("temperature_unsupported_retry", model=model)
            elif ("max_tokens" in params and "max_completion_tokens" in message
                  and "max_tokens" in message and "unsupported" in message):
                note_completion_tokens_required(model)
                params["max_completion_tokens"] = params.pop("max_tokens")
                log.warning("completion_token_parameter_retry", model=model)
            else:
                raise
