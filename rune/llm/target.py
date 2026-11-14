"""Resolve logical models and their transport without losing provider identity."""

from dataclasses import dataclass, field
from pathlib import Path

from rune.llm.context import model_environment

# https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-pro
_VERTEX_LOCATIONS = {
    "gemini-3.1-pro-preview": frozenset({"global"}),
    "gemini-3.1-pro-preview-customtools": frozenset({"global"}),
}
_PROVIDERS = {"openai", "anthropic", "xai", "gemini", "vertex_ai", "azure", "ollama",
              "openai_compatible", "mistral", "deepseek", "cohere"}


class ModelCompatibilityError(ValueError):
    pass


@dataclass(frozen=True)
class ModelTarget:
    provider: str
    model: str
    wire_model: str
    options: tuple[tuple[str, str], ...] = field(default=(), repr=False)

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model}"

    def kwargs(self) -> dict:
        return {"model": self.wire_model, **dict(self.options)}


def vertex_active() -> bool:
    credentials = model_environment("GOOGLE_APPLICATION_CREDENTIALS")
    return bool(credentials and Path(credentials).is_file())


def vertex_options(model: str, options: dict | None = None) -> dict:
    extra = dict(options or {})
    location = (extra.get("vertex_location") or model_environment("VERTEX_LOCATION")
                or model_environment("VERTEXAI_LOCATION") or "auto")
    supported = _VERTEX_LOCATIONS.get(model)
    if location == "auto":
        location = "global"
    if supported is not None and location not in supported:
        raise ModelCompatibilityError(
            f"Vertex model {model} is not available in {location}. "
            f"Select a supported location: {', '.join(sorted(supported))}. The configured location was not changed."
        )
    project = model_environment("VERTEX_PROJECT") or model_environment("VERTEXAI_PROJECT")
    if project:
        extra.setdefault("vertex_project", project)
    extra["vertex_location"] = location
    return extra


def resolve_target(model: str, provider: str | None = None, *, options: dict | None = None) -> ModelTarget:
    extra = dict(options or {})
    name = model.strip()
    prefix, separator, suffix = name.partition("/")
    declared = prefix if separator else name.split(":", 1)[0] if ":" in name else None
    if (provider and declared in _PROVIDERS and declared != provider
            and {provider, declared} != {"gemini", "vertex_ai"}):
        raise ModelCompatibilityError(f"Model {model!r} belongs to {declared}, not {provider}.")
    from rune.llm.gateway import gateway_options, gateway_url

    gateway = gateway_url()
    if separator and prefix not in _PROVIDERS and provider is None:
        if gateway:
            gateway_options(prefix, extra)  # raises: unrouted providers must not bypass the gateway
        return ModelTarget(prefix, suffix, name, tuple(extra.items()))
    if separator and prefix in _PROVIDERS:
        provider, name = prefix, suffix
    elif ":" in name and name.split(":", 1)[0] in _PROVIDERS:
        provider, name = name.split(":", 1)
    provider = provider or next((p for stem, p in (
        ("claude-", "anthropic"), ("grok-", "xai"), ("gemini-", "gemini")) if name.startswith(stem)), "openai")
    if gateway:
        gateway_options(provider, extra)  # refuse unrouted providers before building local options
    if provider == "ollama":
        from rune.llm.endpoints import ollama_v1
        extra.setdefault("api_base", ollama_v1())
        extra.setdefault("api_key", "ollama")
        wire = f"openai/{name}"
    elif provider == "openai_compatible":
        from rune.llm.endpoints import connection_options, split_model
        endpoint, server_model = split_model(name)
        extra = connection_options(endpoint, extra)
        wire = f"openai/{server_model}"
    elif provider == "vertex_ai" or provider == "gemini" and not gateway and vertex_active():
        extra = vertex_options(name, extra)
        wire = f"vertex_ai/{name}"
        provider = "gemini"
    elif provider == "azure":
        from rune.llm.azure import connection_options
        extra = connection_options(extra)
        wire = f"azure/{name}"
    else:
        wire = name if provider == "openai" and not separator else f"{provider}/{name}"
    if provider not in {"ollama", "openai_compatible"} and not wire.startswith("vertex_ai/"):
        from rune.llm.models import PROVIDER_ENV_KEYS

        # Resolve aliases inside the run's environment snapshot as well. Leaving
        # them to the SDK would let a live os.environ edit change in-flight auth.
        for key_name in PROVIDER_ENV_KEYS.get(provider, ()):
            if key := model_environment(key_name):
                extra.setdefault("api_key", key)
                break
    if gateway:
        extra = gateway_options(provider, extra)
    return ModelTarget(provider, name.removeprefix("responses/"), wire, tuple(extra.items()))


def preflight_model(model: str, provider: str | None = None, *, cost_limit: float | None = None) -> ModelTarget:
    """Check endpoint compatibility and budget support without making a request."""
    target = resolve_target(model, provider)
    if cost_limit is not None:
        from rune.llm.pricing import request_price_known

        if not request_price_known({**target.kwargs(), "_rune_source_model": target.key}):
            raise ModelCompatibilityError(
                "Pricing is unavailable for this model or endpoint. No request was sent; "
                "the configured cost limit cannot be checked."
            )
    return target
