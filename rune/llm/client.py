"""LLM client for RUNE.

Ported from src/llm/client.ts - multi-provider support via LiteLLM,
health checks with caching, and provider availability detection.
"""

from __future__ import annotations

import asyncio
import socket
import time
from dataclasses import dataclass

import httpx

from rune.config import get_config
from rune.types import LLMAvailabilityStatus, ModelTier, Provider
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Health check cache

@dataclass(slots=True)
class _HealthEntry:
    healthy: bool
    reason: str
    checked_at: float


_health_cache: dict[str, _HealthEntry] = {}
_DEFAULT_CACHE_TTL = 15.0  # seconds
_DEFAULT_TIMEOUT = 1.5  # seconds


async def _check_dns(hostname: str, timeout: float = _DEFAULT_TIMEOUT) -> bool:
    """Non-blocking DNS lookup."""
    loop = asyncio.get_running_loop()
    try:
        await asyncio.wait_for(
            loop.getaddrinfo(hostname, 443, family=socket.AF_INET),
            timeout=timeout,
        )
        return True
    except (TimeoutError, socket.gaierror, OSError):
        return False


# What the local server actually has installed, newest first, refreshed as a
# by-product of every health check. Configured model names go stale — the
# shipped defaults sat unusable for months because the names predated tool
# calling — so the name in config is a recommendation and a last resort,
# and the installed list is the truth when it is known.
_ollama_installed: list[str] | None = None


async def _check_ollama(timeout: float = _DEFAULT_TIMEOUT) -> bool:
    """Check Ollama availability via HTTP GET /api/tags."""
    global _ollama_installed
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                try:
                    models = resp.json().get("models", [])
                    _ollama_installed = [
                        m["name"] for m in sorted(
                            models, key=lambda m: m.get("modified_at", ""),
                            reverse=True)
                        if "name" in m
                    ]
                except (ValueError, KeyError, TypeError):
                    log.debug("ollama_tags_unparseable")
            return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def refresh_ollama_installed_sync(timeout: float = 0.3) -> None:
    """Fill the installed list right now, bounded, for sync callers.

    The agent loop builds its model profile from config directly and never
    passes through resolve_model, so the async probe cannot help it — the
    profile is built exactly once per run, before any task would get a turn.
    One bounded localhost round-trip at that moment costs single-digit
    milliseconds when the server is up and at most *timeout* when it is not,
    in which case the configured name was the only answer anyway.
    """
    global _ollama_installed
    if _ollama_installed is not None:
        return
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=timeout)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            _ollama_installed = [
                m["name"] for m in sorted(
                    models, key=lambda m: m.get("modified_at", ""),
                    reverse=True)
                if "name" in m
            ]
    except (httpx.HTTPError, OSError, ValueError, KeyError, TypeError):
        # Cache the miss too. Leaving it None means every caller re-probes and
        # re-waits out the timeout on a machine with no Ollama.
        _ollama_installed = []
        log.debug("ollama_sync_probe_failed")


def installed_ollama_models() -> list[str]:
    """Ollama models pulled on this machine, newest first, for model pickers.

    Returns only what a previous probe already cached — it never probes itself,
    because callers include an asyncio request handler where a synchronous
    round-trip would stall the whole server. Use ``prime_ollama_installed``
    off the event loop to fill it. Embedding models are left out for the same
    reason pick_ollama_model skips them: they cannot drive the agent loop.
    """
    return [n for n in (_ollama_installed or []) if "embed" not in n.lower()]


async def prime_ollama_installed(timeout: float = 0.3) -> None:
    """Fill the installed-model cache without blocking the event loop.

    ``refresh_ollama_installed_sync`` only skips its probe when the list is
    already set, so a machine without Ollama would re-block on every call.
    Running it in a thread keeps that cost off the loop either way.
    """
    if _ollama_installed is not None:
        return
    try:
        await asyncio.to_thread(refresh_ollama_installed_sync, timeout)
    except Exception:
        log.debug("ollama_prime_failed")


def pick_ollama_model(configured: str) -> str:
    """The configured model if it is installed, else one that is.

    Embedding models never drive the agent loop, so they are passed over.
    With no health check yet, or an empty server, the configured name stands
    — it doubles as the recommendation of what to pull.
    """
    installed = _ollama_installed
    if not installed:
        return configured
    if configured in installed:
        return configured
    for name in installed:
        if "embed" not in name.lower():
            return name
    return configured


def loop_model_string(provider: str, model: str) -> str:
    """Agent-loop model string: bare id for openai, provider/model otherwise.

    Single owner of the formatting rule — the loop, fast lane, and failover
    restore must all produce the same format or LiteLLM routing breaks for
    one of them.
    """
    return model if provider == "openai" else f"{provider}/{model}"


# LLM Client

class LLMClient:
    """Multi-provider LLM client using LiteLLM under the hood."""

    def __init__(self) -> None:
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the client and verify provider availability."""
        if self._initialized:
            return
        self._initialized = True
        log.info("llm_client_initialized")

    async def check_provider_health(
        self, provider: Provider, *, cache_ttl: float = _DEFAULT_CACHE_TTL,
    ) -> tuple[bool, str]:
        """Check if a provider is reachable. Returns (healthy, reason)."""
        now = time.monotonic()
        cached = _health_cache.get(provider)
        if cached and (now - cached.checked_at) < cache_ttl:
            return cached.healthy, cached.reason

        healthy: bool
        reason: str

        if provider == Provider.OPENAI:
            healthy = await _check_dns("api.openai.com")
            reason = "OK" if healthy else "DNS lookup failed for api.openai.com"
        elif provider == Provider.ANTHROPIC:
            healthy = await _check_dns("api.anthropic.com")
            reason = "OK" if healthy else "DNS lookup failed for api.anthropic.com"
        elif provider == Provider.OLLAMA:
            healthy = await _check_ollama()
            reason = "OK" if healthy else "Ollama not reachable at localhost:11434"
        else:
            healthy = False
            reason = f"Unknown provider: {provider}"

        _health_cache[provider] = _HealthEntry(healthy=healthy, reason=reason, checked_at=now)
        return healthy, reason

    async def get_availability(self) -> LLMAvailabilityStatus:
        """Check all providers and return availability status."""
        providers = [Provider.OPENAI, Provider.ANTHROPIC, Provider.OLLAMA]
        results = await asyncio.gather(
            *(self.check_provider_health(p) for p in providers)
        )

        available = []
        details: dict[str, dict] = {}

        for provider, (healthy, reason) in zip(providers, results, strict=False):
            details[provider] = {"healthy": healthy, "reason": reason}
            if healthy:
                available.append(provider)

        return LLMAvailabilityStatus(
            ready=len(available) > 0,
            available_providers=available,
            blocked_reason="none" if available else "no_provider",
            details=details,
        )

    def _effective_provider(self, provider: Provider | None) -> Provider:
        """Provider to call when none is passed explicitly.

        Uses the session choice (``active_provider``, set by ``-p`` / ``/model``)
        before the static ``default_provider``. Without this, subsystems that
        call with no explicit provider (classifier, gates, learning) route to
        ``default_provider`` even when the user selected another provider.
        """
        if provider is not None:
            return provider
        config = get_config()
        active = getattr(config.llm, "active_provider", None)
        if active:
            try:
                return Provider(active)
            except ValueError:
                pass
        return Provider(config.llm.default_provider)

    def resolve_model(self, tier: ModelTier, provider: Provider | None = None) -> str:
        """Resolve a model ID from tier and provider."""
        config = get_config()
        provider = self._effective_provider(provider)

        # Local providers (ollama) usually have a single model installed, so the
        # per-tier defaults (e.g. fast=llama3.2) are typically NOT present. When
        # the user selected a model for the session, use it for every tier;
        # otherwise aux calls (consolidation, classifier, gates) hit an
        # uninstalled tier model and fail. Cloud providers keep per-tier models
        # (all reachable via API), so the cheaper fast tier still applies there.
        if provider == Provider.OLLAMA:
            active = (getattr(config.llm, "active_model", None) or "").strip()
            if active:
                return active

        models_config = config.llm.models
        tier_models = getattr(models_config, provider.value, models_config.ollama)
        resolved = getattr(tier_models, tier, tier_models.best)

        # The configured name is only a guess about what the local server
        # holds; when the server has told us, what is installed wins. The
        # refresh is a no-op once the list is known, so only the first
        # resolution in a process pays the bounded localhost round-trip —
        # a scheduled-task version of this left the first few resolutions
        # answering from the guess, and they went to a model that was not
        # there.
        if provider == Provider.OLLAMA:
            refresh_ollama_installed_sync()
            return pick_ollama_model(resolved)
        return resolved

    async def completion(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        tier: ModelTier = ModelTier.BEST,
        provider: Provider | None = None,
        temperature: float = 0.0,
        max_tokens: int = 16_384,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        cache_system: bool = False,
        timeout: float = 600.0,
    ) -> dict:
        """Send a completion request via LiteLLM.

        Returns the raw LiteLLM response dict.
        """
        # Through the adapter's lazy accessor so drop_params and the debug
        # suppression are set before any request goes out — a plain
        # `import litellm` configures nothing.
        from rune.agent.litellm_adapter import litellm

        resolved_model = model or self.resolve_model(tier, provider)

        # Prepend the LiteLLM provider prefix. Use the same effective provider as
        # resolve_model so the prefix matches the chosen provider.
        effective_provider = self._effective_provider(provider)
        _PREFIX_MAP = {
            Provider.ANTHROPIC: "anthropic/",
            Provider.GEMINI: "gemini/",
            Provider.AZURE: "azure/",
            Provider.OLLAMA: "ollama/",
        }
        prefix = _PREFIX_MAP.get(effective_provider, "")
        provider_extra: dict = {}

        if effective_provider == Provider.GEMINI:
            import os as _os
            creds = _os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
            if creds and _os.path.isfile(creds):
                prefix = "vertex_ai/"
                proj = _os.environ.get("VERTEX_PROJECT") or _os.environ.get(
                    "VERTEXAI_PROJECT"
                )
                loc = (
                    _os.environ.get("VERTEX_LOCATION")
                    or _os.environ.get("VERTEXAI_LOCATION")
                    or "us-central1"
                )
                if proj:
                    provider_extra["vertex_project"] = proj
                provider_extra["vertex_location"] = loc

        if prefix and not resolved_model.startswith(prefix):
            resolved_model = f"{prefix}{resolved_model}"

        # Clamp to model's hard output cap; the traits table + the retry
        # below handle models litellm's DB gets wrong (see model_traits).
        from rune.agent.litellm_adapter import _clamp_max_tokens
        from rune.agent.model_traits import (
            traits,
        )
        effective_max_tokens = _clamp_max_tokens(resolved_model, max_tokens)

        kwargs: dict = {
            "model": resolved_model,
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "timeout": timeout,
            **provider_extra,
        }
        if traits(resolved_model).temperature:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = tools
        if response_format is not None:
            from rune.llm.structured import supported_format
            selected_format = supported_format(resolved_model, response_format)
            if selected_format is not None:
                kwargs["response_format"] = selected_format
            if response_format.get("type") == "json_schema" and (selected_format or {}).get("type") != "json_schema":
                import json

                contract = "Return JSON matching this schema:\n" + json.dumps(
                    response_format["json_schema"]["schema"], separators=(",", ":"))
                kwargs["messages"] = [{"role": "system", "content": contract}, *messages]

        if cache_system:
            from rune.agent.litellm_adapter import _apply_anthropic_cache_control

            kwargs["messages"] = _apply_anthropic_cache_control(resolved_model, kwargs["messages"])

        from rune.llm.request_params import compatible_completion

        response = await compatible_completion(litellm.acompletion, litellm.BadRequestError, kwargs)
        return response  # type: ignore[return-value]


# Module-level singleton

_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
