"""What each model family accepts on the wire, declared in one table.

Scattered per-model checks are how the fast-mode 400 happened: `speed`
went to a model that rejects the whole request over it, every round
failed instantly, and the shorter wall-clock read as a speed-up. With
the rules in one place, a new model generation is a row edit, not a
hunt for call sites.

The table only holds what litellm's own model database gets wrong or
doesn't know; facts it carries reliably (output caps, context windows)
are read from it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache


@dataclass(frozen=True)
class ModelTraits:
    # Anthropic request shaping applies: cache_control breakpoints and
    # the no-assistant-tail rule.
    anthropic_wire: bool = False
    # Accepts speed="fast". Not advisory — an unsupported model rejects
    # the whole request (measured on haiku: every round failed, and the
    # run "finished" in a third of the time having done nothing).
    speed_param: bool = False
    # Accepts a temperature parameter. litellm's drop_params strips it
    # for the models its DB knows about; a False here covers a family
    # the DB has wrong (gpt-5.5 rejects it while listed as supported).
    temperature: bool = True


_DEFAULT = ModelTraits()

# (required substrings, traits): the first row whose substrings all
# appear in the lowercased model id wins, so specific families must
# stay above general ones.
_STATIC: tuple[tuple[tuple[str, ...], ModelTraits], ...] = (
    (("claude", "opus"), ModelTraits(anthropic_wire=True, speed_param=True)),
    (("anthropic", "opus"), ModelTraits(anthropic_wire=True, speed_param=True)),
    (("claude",), ModelTraits(anthropic_wire=True)),
    (("anthropic",), ModelTraits(anthropic_wire=True)),
    (("gpt-5",), ModelTraits(temperature=False)),
)

# Models whose temperature rejection we only learn from the API's own
# error (exact resolved id, kept for the process). The static table
# can't enumerate these ahead of time — claude-opus-4-8 rejects
# temperature while claude-opus-4-6 accepts it.
_TEMPERATURE_REJECTED: set[str] = set()
# Models that reason and take tools, but refuse both in one request. litellm's
# capability DB answers "does it reason", which is true and not the question, so
# this is learned from the provider's own 400 rather than declared up front.
_REASONING_EFFORT_REJECTED: set[str] = set()


def traits(model: str) -> ModelTraits:
    """Traits for a resolved model id: static table + learned overlay."""
    m = (model or "").lower()
    found = _DEFAULT
    for needles, entry in _STATIC:
        if all(n in m for n in needles):
            found = entry
            break
    if found.temperature and model in _TEMPERATURE_REJECTED:
        found = replace(found, temperature=False)
    return found


@lru_cache(maxsize=256)
def supports_reasoning_effort(model: str) -> bool:
    """Whether *model* accepts a ``reasoning_effort``.

    Trusts litellm's model-capability DB rather than a hand-kept list: it is
    correct per model where a static list drifts — o1 takes one but o1-mini
    does not, claude-opus-4-5 reasons while claude-opus-4 does not, gemini-2.5
    reasons, deepseek-reasoner reasons but takes no effort param. Unknown or
    lookup failure → False (the selector simply won't show; the model still
    runs). drop_params is on, so a stray effort on a model that reasons but
    ignores it is dropped, not an error.
    """
    try:
        import litellm
        return bool(litellm.supports_reasoning(model=model))
    except Exception:
        return False


@lru_cache(maxsize=256)
def supports_vision(model: str) -> bool:
    """Whether *model* accepts image content in a user message.

    Same reasoning as supports_reasoning_effort: litellm's capability DB beats
    a hand-kept list. Unknown or lookup failure → False, so an image is
    described in text rather than sent as content the model would reject.
    """
    try:
        import litellm
        return bool(litellm.supports_vision(model=model))
    except Exception:
        return False


def note_temperature_rejected(model: str) -> None:
    """Record that *model* rejected temperature; traits() reflects it."""
    _TEMPERATURE_REJECTED.add(model)


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
