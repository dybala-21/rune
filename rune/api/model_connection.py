"""Describe selected-model setup without probing a provider or constructing its SDK."""

from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel

from rune.llm.context import llm_settings, model_environment


class ModelSetupAction(BaseModel):
    target: Literal["environment", "azure", "localModels", "models", "operator"]
    label: str
    key: str | None = None


class ModelConnectionInfo(BaseModel):
    # Configured means local prerequisites exist, never authenticated or reachable.
    state: Literal["needs_setup", "configured", "unknown"]
    provider: str
    model: str
    reason: str
    action: ModelSetupAction | None = None


def model_connection(provider: str, model: str) -> ModelConnectionInfo:
    """Report only evidence available locally for this exact selection.

    Do not use provider health, discovery, preflight or resolve_target here: some
    resolve DNS, load provider clients or validate hosted destinations over the network.
    """
    from rune.cloud.boundary import hosted
    from rune.llm.gateway import ROUTED_PROVIDERS, GatewayConfigError, gateway_url
    from rune.llm.models import PROVIDER_ENV_KEYS
    from rune.llm.target import ModelCompatibilityError, vertex_active, vertex_options

    def result(state, reason, target=None, label=None, key=None):
        action = ModelSetupAction(target=target, label=label, key=key) if target else None
        return ModelConnectionInfo(state=state, provider=provider, model=model, reason=reason,
                                   action=action)

    if not model.strip() or model == "unknown":
        return result("needs_setup", "Select a model before starting a run.", "models", "Choose model")

    # A Gemini selection may explicitly request Vertex instead of the API-key route.
    name = model
    declared, separator, suffix = model.partition("/")
    if not separator:
        declared, separator, suffix = model.partition(":")
    known = {*PROVIDER_ENV_KEYS, "vertex_ai", "ollama", "openai_compatible"}
    explicit_vertex = bool(separator and declared == "vertex_ai")
    if separator and declared in known:
        if declared != provider and {declared, provider} != {"gemini", "vertex_ai"}:
            return result("needs_setup", "The selected model belongs to a different provider.",
                          "models", "Choose model")
        name = suffix.removeprefix("responses/")

    try:
        gateway = gateway_url()
    except GatewayConfigError:
        return result("needs_setup", "The engine's model gateway URL is not usable.", "environment",
                      "Review gateway settings", "RUNE_LLM_GATEWAY_URL")
    if gateway:
        if provider not in ROUTED_PROVIDERS or explicit_vertex:
            return result("needs_setup", "This model cannot use the engine's configured gateway.",
                          "models", "Choose a supported model")
        return result("configured", "Gateway routing is configured; provider access has not been verified.")

    if provider == "ollama":
        return result("unknown", "Ollama availability and the selected model's installation have not been checked.",
                      "localModels", "Review local model settings")

    if provider == "openai_compatible":
        from rune.llm.endpoints import configured, private_host

        endpoint_name, separator, server_model = name.partition("/")
        entry = configured().get(endpoint_name)
        if not separator or not server_model or entry is None:
            return result("needs_setup", "Select a model from a saved compatible endpoint.",
                          "localModels", "Set up an endpoint")
        url = urlsplit(entry.base_url)
        if hosted() and (url.scheme != "https" or private_host(url.hostname)):
            return result("needs_setup", "Hosted engines require a public HTTPS model endpoint.",
                          "localModels", "Review endpoint settings")
        return result("unknown", "The endpoint is saved; model availability and authentication have not been checked.",
                      "localModels", "Review endpoint settings")

    if provider in {"gemini", "vertex_ai"} and (explicit_vertex or provider == "vertex_ai" or vertex_active()):
        try:
            vertex_options(name)
        except ModelCompatibilityError:
            return result("needs_setup", "The selected Vertex model does not support the configured location.",
                          "environment", "Review Vertex location", "VERTEX_LOCATION")
        return result("unknown", "Vertex uses cloud credentials; identity, project access and model access have not been verified.",
                      "environment", "Review Vertex credentials", "GOOGLE_APPLICATION_CREDENTIALS")

    if provider == "azure":
        from rune.llm import azure

        try:
            azure.resource_endpoint(azure.endpoint())
        except (ModelCompatibilityError, ValueError):
            return result("needs_setup", "An HTTPS Azure resource endpoint is required for this deployment.",
                          "operator" if hosted() else "azure",
                          "View setup details" if hosted() else "Set up Azure")
        if not any(model_environment(key).strip() for key in azure.KEY_NAMES):
            if model_environment("AZURE_AD_TOKEN").strip():
                return result("unknown", "Azure token authentication and deployment access have not been verified.")
            return result("needs_setup", "An Azure resource credential is required for this deployment.",
                          "operator" if hosted() else "azure",
                          "View setup details" if hosted() else "Set up Azure")
        if name not in llm_settings().azure_deployments:
            return result("unknown", "Azure credentials are present; the selected deployment is not registered in settings.",
                          "operator" if hosted() else "azure",
                          "View setup details" if hosted() else "Review Azure deployment")
        return result("configured", "Azure endpoint, credential and deployment are configured; access has not been verified.")

    keys = PROVIDER_ENV_KEYS.get(provider)
    if not keys:
        return result("unknown", "Setup requirements for this provider are not known.", "models", "Choose model")
    if any(model_environment(key).strip() for key in keys):
        return result("configured", "An API key is present; provider access and model availability have not been verified.")
    # An invalid credentials-file path does not activate Vertex in the runtime.
    if provider == "gemini" and model_environment("GOOGLE_APPLICATION_CREDENTIALS").strip():
        return result("needs_setup", "The configured Google credentials file is unavailable and no Gemini API key is present.",
                      "environment", "Review Gemini credentials", "GOOGLE_APPLICATION_CREDENTIALS")
    return result("needs_setup", "An API key is required for the selected model provider.",
                  "environment", "Add API key", keys[0])
