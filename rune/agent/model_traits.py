"""Supplement LiteLLM's model catalog with documented settings and API rejections."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, replace
from functools import lru_cache

from rune.llm.reasoning import reasoning_control


@dataclass(frozen=True)
class ModelTraits:
    # Apply Anthropic cache boundaries and remove trailing assistant messages.
    anthropic_wire: bool = False
    # Send speed="fast" only to models that accept it.
    speed_param: bool = False
    # Omit temperature for families whose support is misreported by LiteLLM.
    temperature: bool = True
    max_completion_tokens: bool = False
    responses_api: bool = False
    vision: bool | None = None


_DEFAULT = ModelTraits()

# First match wins; put specific models before families.
_STATIC: tuple[tuple[tuple[str, ...], ModelTraits], ...] = (
    (("claude-opus-5",), ModelTraits(anthropic_wire=True, speed_param=True, temperature=False, vision=True)),
    (("claude", "opus"), ModelTraits(anthropic_wire=True, speed_param=True)),
    (("anthropic", "opus"), ModelTraits(anthropic_wire=True, speed_param=True)),
    (("claude",), ModelTraits(anthropic_wire=True)),
    (("anthropic",), ModelTraits(anthropic_wire=True)),
    (("gpt-6-astra",), ModelTraits(
        temperature=False, max_completion_tokens=True, responses_api=True, vision=True,
    )),
    (("gpt-6.1-sol",), ModelTraits(
        temperature=False, max_completion_tokens=True, responses_api=True, vision=True,
    )),
    (("gpt-6-sol",), ModelTraits(
        temperature=False, max_completion_tokens=True, responses_api=True, vision=True,
    )),
    (("gpt-6-luna",), ModelTraits(
        temperature=False, max_completion_tokens=True, responses_api=True, vision=True,
    )),
    (("gpt-5.6",), ModelTraits(temperature=False, max_completion_tokens=True, responses_api=True)),
    (("gpt-5",), ModelTraits(temperature=False)),
)

# Keep rejections within the endpoint and request features that produced them.
_TEMPERATURE_REJECTED: set[Hashable] = set()
_COMPLETION_TOKENS_REQUIRED: set[Hashable] = set()
_RESPONSES_ONLY: set[Hashable] = set()


def traits(model: str, *, cache_key: Hashable | None = None) -> ModelTraits:
    """Combine model settings with rejections observed for this request context."""
    from rune.llm.endpoints import capability_model
    m = capability_model(model or "").lower()
    found = _DEFAULT
    for needles, entry in _STATIC:
        if all(n in m for n in needles):
            found = entry
            break
    from rune.llm.xai import model_name as xai_model_name
    if xai_model_name(model) is not None:
        found = replace(found, vision=True)
    key = model if cache_key is None else cache_key
    if found.temperature and key in _TEMPERATURE_REJECTED:
        found = replace(found, temperature=False)
    if key in _COMPLETION_TOKENS_REQUIRED:
        found = replace(found, max_completion_tokens=True)
    return found


def supports_reasoning_effort(model: str) -> bool:
    """Reasoning capability alone does not imply an adjustable effort."""
    return bool(reasoning_efforts(model))


def reasoning_efforts(model: str) -> tuple[str, ...]:
    return reasoning_control(model).efforts


def effective_reasoning_effort(model: str, configured: str | None) -> str | None:
    """A setting from another model must not become an invalid API parameter."""
    return configured if configured in reasoning_efforts(model) else None


def supports_vision(model: str) -> bool:
    """Use confirmed model traits when the SDK's catalog has not caught up."""
    declared = traits(model).vision
    if declared is not None:
        return declared
    from rune.llm.endpoints import accepts_images, capability_model
    # A server's catalog says nothing about the model it serves; the user declares it.
    if (accepts := accepts_images(model)) is not None:
        return accepts
    return _sdk_supports_vision(capability_model(model))


@lru_cache(maxsize=256)
def _sdk_supports_vision(model: str) -> bool:
    try:
        import litellm
        return bool(litellm.supports_vision(model=model))
    except Exception:
        return False


def note_temperature_rejected(model: Hashable) -> None:
    """Record that *model* rejected temperature; traits() reflects it."""
    _TEMPERATURE_REJECTED.add(model)


def note_completion_tokens_required(model: Hashable) -> None:
    _COMPLETION_TOKENS_REQUIRED.add(model)


def note_responses_only(model: Hashable) -> None:
    """Record that *model* needs /v1/responses for a request carrying tools."""
    _RESPONSES_ONLY.add(model)


def needs_responses_api(model: Hashable) -> bool:
    """Whether *model* has refused tools on chat/completions in this process."""
    return model in _RESPONSES_ONLY


def is_responses_only_error(exc: Exception) -> bool:
    """Check whether the provider requires the Responses endpoint."""
    m = str(exc).lower()
    if "v1/responses" in m or "/responses" in m:
        return True
    return "chat/completions" in m and (
        "not supported" in m or "not a chat model" in m
    )


def is_max_tokens_rename_error(exc: Exception) -> bool:
    """Check whether the provider requires max_completion_tokens."""
    m = str(exc).lower()
    return "max_tokens" in m and "max_completion_tokens" in m


def is_reasoning_effort_error(exc: Exception) -> bool:
    """Whether a BadRequest is about reasoning_effort being unacceptable."""
    m = str(exc).lower()
    return "reasoning_effort" in m and (
        "support" in m or "deprecat" in m or "invalid" in m
    )


def is_temperature_error(exc: Exception) -> bool:
    """Whether a BadRequest is about temperature (unsupported/invalid/deprecated)."""
    m = str(exc).lower()
    return "temperature" in m and (
        "support" in m or "deprecat" in m or "invalid" in m
    )
