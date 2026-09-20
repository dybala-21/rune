"""Model request settings that supplement LiteLLM's catalog.

Static overrides cover model families; provider errors add corrections
for individual models during the process lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from rune.llm.reasoning import reasoning_control


@dataclass(frozen=True)
class ModelTraits:
    # Anthropic request shaping applies: cache_control breakpoints and
    # the no-assistant-tail rule.
    anthropic_wire: bool = False
    # Send speed="fast" only to models that accept it.
    speed_param: bool = False
    # Omit temperature for families whose support is misreported by LiteLLM.
    temperature: bool = True
    max_completion_tokens: bool = False
    responses_api: bool = False
    vision: bool | None = None


_DEFAULT = ModelTraits()

# (required substrings, traits): the first row whose substrings all
# appear in the lowercased model id wins, so specific families must
# stay above general ones.
_STATIC: tuple[tuple[tuple[str, ...], ModelTraits], ...] = (
    (("claude-opus-5",), ModelTraits(anthropic_wire=True, speed_param=True, temperature=False, vision=True)),
    (("claude", "opus"), ModelTraits(anthropic_wire=True, speed_param=True)),
    (("anthropic", "opus"), ModelTraits(anthropic_wire=True, speed_param=True)),
    (("claude",), ModelTraits(anthropic_wire=True)),
    (("anthropic",), ModelTraits(anthropic_wire=True)),
    (("gpt-6-astra",), ModelTraits(
        temperature=False, max_completion_tokens=True, responses_api=True, vision=True,
    )),
    (("gpt-5.6",), ModelTraits(temperature=False, max_completion_tokens=True, responses_api=True)),
    (("gpt-5",), ModelTraits(temperature=False)),
)

# Cache provider rejections by resolved model ID for this process.
_TEMPERATURE_REJECTED: set[str] = set()
_COMPLETION_TOKENS_REQUIRED: set[str] = set()
# Reasoning support alone does not guarantee support alongside tools.
_REASONING_EFFORT_REJECTED: set[str] = set()
# Route models to Responses after the provider rejects Chat Completions.
_RESPONSES_ONLY: set[str] = set()


def traits(model: str) -> ModelTraits:
    """Traits for a resolved model id: static table + learned overlay."""
    m = (model or "").lower()
    found = _DEFAULT
    for needles, entry in _STATIC:
        if all(n in m for n in needles):
            found = entry
            break
    from rune.llm.xai import model_name as xai_model_name
    if xai_model_name(model) is not None:
        found = replace(found, vision=True)
    if found.temperature and model in _TEMPERATURE_REJECTED:
        found = replace(found, temperature=False)
    if model in _COMPLETION_TOKENS_REQUIRED:
        found = replace(found, max_completion_tokens=True)
    return found


@lru_cache(maxsize=256)
def supports_reasoning_effort(model: str) -> bool:
    """Reasoning capability alone does not imply an adjustable effort."""
    return bool(reasoning_efforts(model))


@lru_cache(maxsize=256)
def reasoning_efforts(model: str) -> tuple[str, ...]:
    return reasoning_control(model).efforts


def effective_reasoning_effort(model: str, configured: str | None) -> str | None:
    """A setting from another model must not become an invalid API parameter."""
    return configured if configured in reasoning_efforts(model) else None


@lru_cache(maxsize=256)
def supports_vision(model: str) -> bool:
    """Use confirmed model traits when the SDK's catalog has not caught up."""
    declared = traits(model).vision
    if declared is not None:
        return declared
    try:
        import litellm
        return bool(litellm.supports_vision(model=model))
    except Exception:
        return False


def note_temperature_rejected(model: str) -> None:
    """Record that *model* rejected temperature; traits() reflects it."""
    _TEMPERATURE_REJECTED.add(model)


def note_completion_tokens_required(model: str) -> None:
    _COMPLETION_TOKENS_REQUIRED.add(model)


def note_reasoning_effort_rejected(model: str) -> None:
    """Record that *model* refused reasoning_effort alongside tools."""
    _REASONING_EFFORT_REJECTED.add(model)


def reasoning_effort_rejected(model: str) -> bool:
    """Whether *model* has refused reasoning_effort in this process.

    Deliberately not cached, and deliberately not folded into
    :func:`supports_reasoning_effort`: that one is ``lru_cache``d, so a True
    answered before the first rejection would mask everything learned after it.
    """
    return model in _REASONING_EFFORT_REJECTED


def note_responses_only(model: str) -> None:
    """Record that *model* needs /v1/responses for a request carrying tools."""
    _RESPONSES_ONLY.add(model)


def needs_responses_api(model: str) -> bool:
    """Whether *model* has refused tools on chat/completions in this process."""
    return model in _RESPONSES_ONLY


def is_responses_only_error(exc: Exception) -> bool:
    """Whether a BadRequest is the endpoint pointing at /v1/responses.

    Two distinct refusals mean the same thing: "function tools ... are not
    supported ... use /v1/responses", and "this model is not supported in the
    v1/chat/completions endpoint".
    """
    m = str(exc).lower()
    if "v1/responses" in m or "/responses" in m:
        return True
    return "chat/completions" in m and (
        "not supported" in m or "not a chat model" in m
    )


def is_max_tokens_rename_error(exc: Exception) -> bool:
    """Whether the provider is asking for max_completion_tokens instead.

    Newer OpenAI models refuse the older spelling outright, and that refusal
    arrives before any complaint about tools — so without handling it the run
    never gets far enough to learn the model needs /v1/responses.
    """
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
