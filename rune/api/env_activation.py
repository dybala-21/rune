"""Conservative activation guidance for environment edits, without client resets."""

from typing import Literal

from pydantic import BaseModel

from rune.llm.azure import ENDPOINT_NAMES, KEY_NAMES
from rune.llm.endpoints import OLLAMA_NAMES
from rune.llm.models import PROVIDER_ENV_KEYS
from rune.types import Provider

_SELECTABLE_PROVIDERS = frozenset(provider.value for provider in Provider)


class EnvActivationInfo(BaseModel):
    state: Literal["next_run", "restart_required", "unknown"]
    reason: str


# These are read by model_environment/resolve_target on new requests.
# Client caches include explicit keys and endpoints; in-flight model_scope keeps
# the previous environment, including known provider key aliases.
_NEXT_RUN_KEYS = frozenset({
    *(key for keys in PROVIDER_ENV_KEYS.values() for key in keys),
    *KEY_NAMES, *ENDPOINT_NAMES, *OLLAMA_NAMES, "AZURE_API_VERSION",
    "VERTEX_PROJECT", "VERTEXAI_PROJECT", "VERTEX_LOCATION", "VERTEXAI_LOCATION",
    "RUNE_LLM_GATEWAY_URL", "RUNE_LLM_GATEWAY_KEY",
    "TYPESAFE_API_KEY", "AZURE_DECISION_API_KEY",
})
_VERTEX_KEYS = frozenset({"GOOGLE_APPLICATION_CREDENTIALS", "VERTEX_PROJECT", "VERTEXAI_PROJECT",
                          "VERTEX_LOCATION", "VERTEXAI_LOCATION"})
_RESTART_KEYS = frozenset({
    # Vertex's ADC cache is keyed by project and explicit credentials, not the
    # GOOGLE_APPLICATION_CREDENTIALS path. Do not clear a global SDK cache here.
    "GOOGLE_APPLICATION_CREDENTIALS", "BRAVE_API_KEY",
    "RUNE_HOME", "RUNE_LOG_LEVEL", "RUNE_LOG_JSON", "RUNE_LOG_FILE",
    "TELEGRAM_BOT_TOKEN", "RUNE_TELEGRAM_TOKEN", "DISCORD_BOT_TOKEN", "RUNE_DISCORD_TOKEN",
    "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "RUNE_SLACK_BOT_TOKEN", "RUNE_SLACK_APP_TOKEN",
})


def provider_for_env_key(key: str) -> str | None:
    for provider, names in PROVIDER_ENV_KEYS.items():
        if key in names:
            # Only offer a model handoff when model.set accepts this provider.
            return provider if provider in _SELECTABLE_PROVIDERS else None
    if key in (*KEY_NAMES, *ENDPOINT_NAMES, "AZURE_API_VERSION"):
        return "azure"
    if key in OLLAMA_NAMES:
        return "ollama"
    if key in _VERTEX_KEYS:
        return "gemini"
    if key.startswith("RUNE_ENDPOINT_") and key.endswith("_API_KEY"):
        return "openai_compatible"
    return None


def env_activation(key: str, *, scope: str | None = None) -> EnvActivationInfo:
    """Describe when the known consumer reads a changed value; never claim a connection."""
    from rune.cloud.boundary import hosted

    if hosted() and scope == "project":
        return EnvActivationInfo(
            state="unknown",
            reason="Hosted engines ignore project environment files. The running environment was updated; save in user scope for persistence after restart.",
        )
    if key in _RESTART_KEYS:
        reason = ("Restart the engine to reload cached Google credentials."
                  if key == "GOOGLE_APPLICATION_CREDENTIALS"
                  else "Restart the engine to reload settings retained by this service.")
        return EnvActivationInfo(state="restart_required", reason=reason)
    if key in _NEXT_RUN_KEYS or key.startswith("RUNE_ENDPOINT_") and key.endswith("_API_KEY"):
        return EnvActivationInfo(
            state="next_run",
            reason="Model settings are read for new runs; ongoing runs keep their existing settings.",
        )
    return EnvActivationInfo(
        state="unknown",
        reason="The environment was updated, but when this setting takes effect is not known.",
    )
