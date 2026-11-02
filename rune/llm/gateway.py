"""Route every LiteLLM request through an operator's LLM gateway instead of holding provider keys.

Set ``RUNE_LLM_GATEWAY_URL`` (hosted deployments) and each provider's requests go to
``<gateway>/<provider>`` in that provider's native wire format. The machine running
RUNE then holds no provider key: ``RUNE_LLM_GATEWAY_KEY`` is an optional scoped,
revocable gateway credential, and without it a placeholder is sent so the gateway (or
the host proxy in front of it) can attach the caller's identity instead.

Unset, nothing here changes behaviour.
"""

from __future__ import annotations

from typing import Any

from rune.llm.context import model_environment
from rune.llm.target import ModelCompatibilityError

# Providers whose public APIs a gateway can front transparently. Local or user-supplied
# endpoints (ollama, openai_compatible) and cloud-tenant ones (azure, vertex) are not
# routed: they would either bypass the gateway or need the user's own cloud account.
ROUTED_PROVIDERS = frozenset({"openai", "anthropic", "xai", "gemini", "mistral", "deepseek", "cohere"})
_PLACEHOLDER_KEY = "rune-gateway"


class GatewayConfigError(ModelCompatibilityError):
    """Raised before any request is sent; the failover policy aborts on it instead of switching."""


def gateway_url() -> str | None:
    """The validated gateway origin, or None when gateway mode is off."""
    value = (model_environment("RUNE_LLM_GATEWAY_URL") or "").strip()
    if not value:
        return None
    from rune.utils.service_url import service_url

    try:
        return service_url(value, allow_path=True)
    except ValueError as exc:
        raise GatewayConfigError(f"RUNE_LLM_GATEWAY_URL is not usable: {exc}") from exc


def gateway_active() -> bool:
    """True when requests must go through the gateway, including when it is misconfigured."""
    try:
        return gateway_url() is not None
    except GatewayConfigError:
        # A broken gateway setting must not silently fall back to direct provider keys.
        return True


def gateway_options(provider: str, extra: dict[str, Any]) -> dict[str, Any]:
    """Return request options that send ``provider`` through the gateway, or raise if it cannot be."""
    base = gateway_url()
    if base is None:
        return extra
    if provider not in ROUTED_PROVIDERS:
        raise GatewayConfigError(
            f"{provider} models are not available through the LLM gateway. "
            f"Use one of: {', '.join(sorted(ROUTED_PROVIDERS))}. No request was sent."
        )
    routed = {k: v for k, v in extra.items() if k not in {"api_base", "base_url", "api_key"}}
    routed["api_base"] = f"{base}/{provider}"
    routed["api_key"] = (model_environment("RUNE_LLM_GATEWAY_KEY") or "").strip() or _PLACEHOLDER_KEY
    return routed


def is_gateway_base(api_base: object) -> bool:
    """True when a request's api_base is this deployment's gateway (so public list prices apply)."""
    try:
        base = gateway_url()
    except GatewayConfigError:
        return False
    return bool(base and isinstance(api_base, str) and (api_base == base or api_base.startswith(base + "/")))
