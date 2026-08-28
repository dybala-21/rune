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
    # Accepts a `reasoning_effort` (low/medium/high). litellm passes it
    # through to OpenAI and maps it to adaptive thinking for Claude 4.6+/5.
    # Only set for models that support it — others reject or ignore it.
    reasoning_effort: bool = False


_DEFAULT = ModelTraits()

# (required substrings, traits): the first row whose substrings all
# appear in the lowercased model id wins, so specific families must
# stay above general ones.
_STATIC: tuple[tuple[tuple[str, ...], ModelTraits], ...] = (
    # Reasoning-capable Claude (adaptive thinking via litellm): the 4.6
    # generation and Claude 5. Matched by exact version substrings so
    # opus-4-5 (no thinking) doesn't slip in. Above the general rows.
    (("claude-opus-5",), ModelTraits(anthropic_wire=True, speed_param=True, reasoning_effort=True)),
    (("claude-sonnet-5",), ModelTraits(anthropic_wire=True, reasoning_effort=True)),
    (("claude-opus-4-6",), ModelTraits(anthropic_wire=True, speed_param=True, reasoning_effort=True)),
    (("claude-sonnet-4-6",), ModelTraits(anthropic_wire=True, reasoning_effort=True)),
    (("claude", "opus"), ModelTraits(anthropic_wire=True, speed_param=True)),
    (("anthropic", "opus"), ModelTraits(anthropic_wire=True, speed_param=True)),
    (("claude",), ModelTraits(anthropic_wire=True)),
    (("anthropic",), ModelTraits(anthropic_wire=True)),
    # OpenAI reasoning models: the GPT-5 family and the o-series.
    (("gpt-5",), ModelTraits(temperature=False, reasoning_effort=True)),
    (("o1",), ModelTraits(temperature=False, reasoning_effort=True)),
    (("o3",), ModelTraits(temperature=False, reasoning_effort=True)),
    (("o4",), ModelTraits(temperature=False, reasoning_effort=True)),
)

# Models whose temperature rejection we only learn from the API's own
# error (exact resolved id, kept for the process). The static table
# can't enumerate these ahead of time — claude-opus-4-8 rejects
# temperature while claude-opus-4-6 accepts it.
_TEMPERATURE_REJECTED: set[str] = set()


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


def note_temperature_rejected(model: str) -> None:
    """Record that *model* rejected temperature; traits() reflects it."""
    _TEMPERATURE_REJECTED.add(model)


def is_temperature_error(exc: Exception) -> bool:
    """Whether a BadRequest is about temperature (unsupported/invalid/deprecated)."""
    m = str(exc).lower()
    return "temperature" in m and (
        "support" in m or "deprecat" in m or "invalid" in m
    )
