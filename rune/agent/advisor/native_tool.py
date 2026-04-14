"""Claude native ``advisor_20260301`` tool integration.

Fail-open on mismatch; function=None (server-side dispatch); schema
passthrough (no OpenAI envelope); observability via usage.iterations.

Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rune.agent.advisor.tiers import (
    extract_provider_and_model,
    is_claude_native_eligible,
)
from rune.agent.tool_adapter import ToolWrapper
from rune.utils.logger import get_logger

log = get_logger(__name__)

_BETA_HEADER_VALUE = "advisor-tool-2026-03-01"
_NATIVE_TOOL_TYPE = "advisor_20260301"
_NATIVE_TOOL_NAME = "advisor"


@dataclass(slots=True, frozen=True)
class NativeAdvisorConfig:
    """Runtime configuration for the native advisor path.

    ``enabled=False`` means the caller should fall back to RUNE's
    policy-driven advisor; this config is inert.
    """

    enabled: bool
    advisor_model: str
    max_uses: int = 3

    @property
    def beta_headers(self) -> dict[str, str]:
        """Return the Anthropic beta header dict when enabled, else {}.

        Used by ``litellm_adapter.StreamResult`` to pass
        ``extra_headers`` to ``litellm.acompletion``.
        """
        if not self.enabled:
            return {}
        return {"anthropic-beta": _BETA_HEADER_VALUE}


def _native_opt_in_enabled() -> bool:
    """Opt-in via RUNE_ADVISOR_NATIVE=1. Default OFF."""
    from rune.agent.advisor.runtime_toggle import parse_env_bool
    return parse_env_bool("RUNE_ADVISOR_NATIVE", default=False)


def resolve_native_config(
    executor_model: str,
    advisor_model_full: str | None,
    max_uses: int = 3,
) -> NativeAdvisorConfig:
    """Build a ``NativeAdvisorConfig`` by tier-matching the pair.

    ``executor_model`` is the full ``provider/model`` string RUNE uses
    internally (e.g. ``'anthropic/claude-haiku-4-5-20251001'``).
    ``advisor_model_full`` is the env-provided advisor id, or None if
    the advisor service is disabled.

    Returns an inert config (``enabled=False``) when **any** of the
    following holds:

    - ``RUNE_ADVISOR_NATIVE`` is not set / not truthy (default OFF)
    - ``advisor_model_full`` is missing
    - the pair does not match Anthropic's official compatibility matrix

    Never raises — the caller treats this as a boolean gate.
    """
    if not _native_opt_in_enabled():
        return NativeAdvisorConfig(enabled=False, advisor_model="")
    if not advisor_model_full:
        return NativeAdvisorConfig(enabled=False, advisor_model="")
    exec_provider, exec_model = extract_provider_and_model(executor_model)
    adv_provider, adv_model = extract_provider_and_model(advisor_model_full)
    if not is_claude_native_eligible(
        exec_provider, exec_model, adv_provider, adv_model,
    ):
        return NativeAdvisorConfig(enabled=False, advisor_model="")
    return NativeAdvisorConfig(
        enabled=True,
        advisor_model=adv_model,
        max_uses=max_uses,
    )


def build_native_tool_wrapper(
    config: NativeAdvisorConfig,
) -> ToolWrapper | None:
    """Build a ``ToolWrapper`` representing the native advisor tool.

    The wrapper has ``function=None`` so the client dispatch layer
    ignores it; it only reaches Anthropic's server as part of the
    ``tools`` list in the Messages API request.

    Returns ``None`` when the config is disabled so callers can use
    the result as a simple existence check.
    """
    if not config.enabled:
        return None
    return ToolWrapper(
        name=_NATIVE_TOOL_NAME,
        description=(
            "Consult a stronger reviewer model for strategic guidance. "
            "Takes no parameters — the advisor sees the full conversation "
            "automatically. Returns a plan, correction, or stop signal."
        ),
        json_schema={
            "type": _NATIVE_TOOL_TYPE,
            "name": _NATIVE_TOOL_NAME,
            "model": config.advisor_model,
            "max_uses": config.max_uses,
        },
        function=None,
    )


def is_native_schema(tool_schema: Any) -> bool:
    """Return True if the given schema is a passthrough native advisor
    tool — used by ``tools_to_openai_schema`` to skip the OpenAI
    ``function`` envelope when serializing."""
    if not isinstance(tool_schema, dict):
        return False
    return str(tool_schema.get("type", "")).startswith("advisor_")


def extract_synthetic_events_from_usage(
    usage: Any,
    *,
    fallback_advisor_model: str = "",
) -> list[dict[str, Any]]:
    """Parse ``usage.iterations[]`` for ``advisor_message`` entries.

    Anthropic's native path runs advisor sub-inferences server-side,
    so there is no client-side ``consult()`` call we can intercept.
    Instead, the billed usage block breaks down into per-iteration
    entries; each ``type=='advisor_message'`` entry is one advisor
    call. We rebuild the same dict shape that
    ``AdvisorBudget.call_history`` uses so the loop's persistence
    tail can UPSERT them into ``advisor_events`` uniformly.

    Never raises; parse failures become an empty list + a WARNING log
    (CLAUDE.md forbids silent error swallowing).
    """
    events: list[dict[str, Any]] = []
    try:
        if usage is None:
            return events
        iterations = getattr(usage, "iterations", None)
        if iterations is None and isinstance(usage, dict):
            iterations = usage.get("iterations", [])
        if not iterations:
            return events
        for it in iterations:
            if isinstance(it, dict):
                entry_type = it.get("type")
                model = it.get("model", "") or fallback_advisor_model
                output_tokens = int(it.get("output_tokens") or 0)
            else:
                # LiteLLM ModelResponse iteration objects are
                # attribute-based — dispatch defensively.
                entry_type = getattr(it, "type", None)
                model = getattr(it, "model", "") or fallback_advisor_model
                output_tokens = int(getattr(it, "output_tokens", 0) or 0)
            if entry_type != "advisor_message":
                continue
            events.append(
                {
                    "trigger": "native",
                    "action": "continue",
                    "provider": "anthropic",
                    "model": model,
                    "output_tokens": output_tokens,
                    "latency_ms": 0,
                    "plan_injected": True,
                    "stuck_reason": "",
                    "error_code": "",
                }
            )
    except Exception as exc:
        log.warning(
            "native_advisor_usage_parse_failed",
            error=str(exc)[:200],
        )
    return events
