"""Model-specific reasoning controls and the parameters they accept."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from rune.utils.logger import get_logger

log = get_logger(__name__)
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_BASIC = ("low", "medium", "high")
_EXTENDED = (*_BASIC, "xhigh", "max")


@dataclass(frozen=True)
class ReasoningControl:
    efforts: tuple[str, ...] = ()
    wire: Literal["litellm", "responses", "anthropic", "gemini", "xai"] = "litellm"
    adaptive: bool = False
    budgets: tuple[tuple[str, int], ...] = ()


# Exact model IDs; a newer sibling does not inherit another model's API contract.
# https://developers.openai.com/api/docs/models/gpt-6-astra
# https://developers.openai.com/api/docs/models/gpt-5.6-sol
_OPENAI = {
    "gpt-6-astra": ReasoningControl(_EXTENDED, "responses"),
    "gpt-5.6-sol": ReasoningControl(("none", *_EXTENDED), "responses"),
    "gpt-5.6": ReasoningControl(("none", *_EXTENDED), "responses"),
}

# https://platform.claude.com/docs/en/build-with-claude/effort
_ANTHROPIC = {
    "claude-opus-4-5": ReasoningControl(_BASIC, "anthropic"),
    "claude-opus-4-6": ReasoningControl((*_BASIC, "max"), "anthropic", adaptive=True),
    "claude-sonnet-4-6": ReasoningControl((*_BASIC, "max"), "anthropic", adaptive=True),
    **{name: ReasoningControl(_EXTENDED, "anthropic", adaptive=True) for name in (
        "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    )},
}

# GenerateContent uses levels for Gemini 3 and token budgets for Gemini 2.5.
# The budgets are Rune presets, not provider-defined effort levels.
# https://ai.google.dev/gemini-api/docs/generate-content/thinking
_BUDGETS = (("low", 1024), ("medium", 2048), ("high", 4096))
_GEMINI = {
    "gemini-3-pro-preview": ReasoningControl(("low", "high"), "gemini"),
    "gemini-3.1-pro-preview": ReasoningControl(_BASIC, "gemini"),
    "gemini-3-flash-preview": ReasoningControl(("minimal", *_BASIC), "gemini"),
    "gemini-2.5-pro": ReasoningControl(_BASIC, "gemini", budgets=_BUDGETS),
    **{name: ReasoningControl(
        ("none", *_BASIC), "gemini", budgets=(("none", 0), *_BUDGETS),
    ) for name in ("gemini-2.5-flash", "gemini-2.5-flash-lite")},
}


def reasoning_model_key(model: str) -> str:
    """Stable preference key across LiteLLM's transport prefixes."""
    provider, sep, name = model.strip().partition("/")
    if not sep:
        name = provider
        if name.startswith("claude-"):
            provider = "anthropic"
        elif name.startswith("grok-"):
            provider = "xai"
        else:
            provider = "openai"
    if provider in {"openai", "azure"}:
        name = name.removeprefix("responses/")
    # Rune's Gemini provider selects Vertex when service-account credentials exist.
    if provider == "vertex_ai":
        provider = "gemini"
    return f"{provider}/{name}"


@lru_cache(maxsize=256)
def reasoning_control(model: str) -> ReasoningControl:
    provider, name = reasoning_model_key(model).split("/", 1)
    if provider == "xai":
        from rune.llm.xai import REASONING, model_name
        return ReasoningControl(REASONING.get(model_name(model), ()), "xai")
    catalog = {"openai": _OPENAI, "anthropic": _ANTHROPIC, "gemini": _GEMINI}.get(provider, {})
    known = catalog.get(name)
    # Dated snapshots of a documented model keep that model's control surface.
    if known is None:
        known = catalog.get(re.sub(r"-(?:\d{8}|\d{4}-\d{2}-\d{2})$", "", name))
    if known is not None:
        return known
    try:
        import litellm

        params = litellm.get_supported_openai_params(model=model) or []
        info = litellm.get_model_info(model) or {}
        if "reasoning_effort" in params:
            return ReasoningControl(tuple(
                level for level in _LEVELS
                if info.get(f"supports_{level}_reasoning_effort") is True
            ))
    except Exception as exc:
        log.debug("reasoning_metadata_unavailable", model=model, error=type(exc).__name__)
    return ReasoningControl()


def configured_reasoning_effort(model: str) -> str | None:
    from rune.config import get_config

    value = get_config().llm.reasoning_efforts.get(reasoning_model_key(model))
    if value is None:
        return None
    return value if value in reasoning_control(model).efforts else None


@lru_cache(maxsize=128)
def _register_anthropic_control(model: str, control: ReasoningControl) -> None:
    import litellm

    name = reasoning_model_key(model).split("/", 1)[1]
    # LiteLLM validates output_config and thinking against its own model map.
    # Overlay only documented capabilities, preserving prices and token limits.
    litellm.register_model({name: {
        **litellm.model_cost.get(name, {}),
        "litellm_provider": "anthropic",
        "mode": "chat",
        "supports_reasoning": True,
        "supports_output_config": True,
        "supports_adaptive_thinking": control.adaptive,
        **{f"supports_{level}_reasoning_effort": level in control.efforts for level in _LEVELS},
    }})


def apply_reasoning_control(params: dict) -> None:
    """Validate an explicit effort before LiteLLM can drop or reinterpret it."""
    effort = params.get("reasoning_effort")
    if effort is None:
        params.pop("reasoning_effort", None)
        return
    control = reasoning_control(params["model"])
    if not isinstance(effort, str) or effort not in control.efforts:
        raise ValueError(f"Unsupported reasoning effort {effort!r} for {params['model']}")
    if control.wire == "anthropic":
        _register_anthropic_control(params["model"], control)
        params.pop("reasoning_effort")
        params["output_config"] = {**(params.get("output_config") or {}), "effort": effort}
        if control.adaptive:
            params["thinking"] = {"type": "adaptive"}
            params.pop("temperature", None)
    elif control.wire == "gemini":
        params.pop("reasoning_effort")
        budget = dict(control.budgets).get(effort)
        params["thinkingConfig"] = (
            {"thinkingBudget": budget} if budget is not None else {"thinkingLevel": effort}
        )
    elif control.wire == "xai":
        # Older LiteLLM catalogs otherwise drop documented Grok effort levels.
        params.pop("reasoning_effort")
        params["extra_body"] = {**(params.get("extra_body") or {}), "reasoning_effort": effort}
