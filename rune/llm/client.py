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


async def _check_ollama(timeout: float = _DEFAULT_TIMEOUT) -> bool:
    """Check Ollama availability via HTTP GET /api/tags."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


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

    def resolve_model(self, tier: ModelTier, provider: Provider | None = None) -> str:
        """Resolve a model ID from tier and provider."""
        config = get_config()
        provider = provider or Provider(config.llm.default_provider)

        models_config = config.llm.models
        if provider == Provider.OPENAI:
            tier_models = models_config.openai
        elif provider == Provider.ANTHROPIC:
            tier_models = models_config.anthropic
        else:
            tier_models = models_config.ollama

        return getattr(tier_models, tier, tier_models.best)

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
        timeout: float = 600.0,
    ) -> dict:
        """Send a completion request via LiteLLM.

        Returns the raw LiteLLM response dict.
        """
        import litellm

        resolved_model = model or self.resolve_model(tier, provider)

        # Prepend provider prefix for LiteLLM if needed
        config = get_config()
        effective_provider = provider or Provider(config.llm.default_provider)
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

        # Clamp max_tokens to the model's hard output cap. Gemini 2.0
        # flash-lite caps at 8192; sending 16384 returns a 400.
        from rune.agent.litellm_adapter import _clamp_max_tokens
        effective_max_tokens = _clamp_max_tokens(resolved_model, max_tokens)

        kwargs: dict = {
            "model": resolved_model,
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "timeout": timeout,
            **provider_extra,
        }
        # gpt-5 models only support temperature=1; skip temperature param for them
        model_lower = resolved_model.lower()
        if "gpt-5" in model_lower and "gpt-5." not in model_lower:
            pass  # omit temperature entirely
        else:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = tools

        response = await litellm.acompletion(**kwargs)
        return response  # type: ignore[return-value]


# Module-level singleton

_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
