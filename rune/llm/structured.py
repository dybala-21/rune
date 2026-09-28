"""Select the response constraint supported by the configured model."""

from copy import deepcopy
from functools import lru_cache

from rune.utils.logger import get_logger

log = get_logger(__name__)
_ANTHROPIC_NATIVE = {"anthropic/claude-opus-5", "claude-opus-5"}


@lru_cache(maxsize=128)
def _format_type(model: str) -> str | None:
    from rune.llm.xai import model_name as xai_model_name

    if xai_model_name(model) is not None:
        return "json_schema"
    if model in _ANTHROPIC_NATIVE:
        return "json_schema"
    from rune.agent.litellm_adapter import litellm

    try:
        if litellm.supports_response_schema(model=model):
            return "json_schema"
        if "response_format" in (litellm.get_supported_openai_params(model=model) or []):
            return "json_object"
    except Exception as exc:
        log.debug("response_schema_support_unknown", model=model, error=type(exc).__name__)
    log.warning("response_schema_unavailable", model=model)
    return None


def supported_format(model: str, requested: dict) -> dict | None:
    kind = _format_type(model)
    return requested if kind == "json_schema" else {"type": kind} if kind else None


def native_output_config(model: str, requested: dict) -> dict | None:
    """Use Claude's JSON output API when the adapter's model catalog is stale."""
    if model not in _ANTHROPIC_NATIVE or requested.get("type") != "json_schema":
        return None
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    from rune.llm.reasoning import _register_anthropic_control, reasoning_control

    _register_anthropic_control(model, reasoning_control(model))
    # The adapter filters unsupported wire constraints; Rune validates the result.
    # https://platform.claude.com/docs/en/build-with-claude/structured-outputs
    schema = AnthropicConfig().filter_anthropic_output_schema(deepcopy(requested["json_schema"]["schema"]))
    return {"format": {"type": "json_schema", "schema": schema}}
