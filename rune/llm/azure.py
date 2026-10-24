"""Azure endpoints, deployments, and model capabilities."""

from urllib.parse import urlsplit

from rune.llm.context import llm_settings, model_environment

GPT6_MODELS = ("gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna", "gpt-6-astra")
KEY_NAMES = ("AZURE_API_KEY", "AZURE_OPENAI_API_KEY")
ENDPOINT_NAMES = ("AZURE_API_BASE", "AZURE_OPENAI_ENDPOINT")


def environment_name(names: tuple[str, ...]) -> str:
    return next((name for name in names if model_environment(name).strip()), names[0])


def endpoint() -> str:
    return model_environment(environment_name(ENDPOINT_NAMES)).strip()


def connection_options(options: dict) -> dict:
    from rune.llm.target import ModelCompatibilityError

    result = dict(options)
    base = str(result.get("api_base") or endpoint()).strip().rstrip("/")
    parsed = urlsplit(base)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in {"", "/openai/v1"}):
        raise ModelCompatibilityError("Set a valid HTTPS Azure endpoint in Settings before selecting an Azure model.")
    result["api_base"] = base.removesuffix("/openai/v1")
    result.setdefault("api_version", model_environment("AZURE_API_VERSION") or "v1")
    key = model_environment(environment_name(KEY_NAMES)).strip()
    if key:
        result.setdefault("api_key", key)
    return result


def capability_model(model: str) -> str:
    provider, separator, name = model.partition("/")
    if provider != "azure" or not separator:
        return model
    name = name.removeprefix("responses/")
    return f"openai/{llm_settings().azure_deployments.get(name, name)}"


def deployment_names() -> list[str]:
    settings = llm_settings()
    names = list(settings.azure_deployments)
    # Keep older configurations selectable without a deployment mapping.
    if settings.active_provider == "azure" and settings.active_model and settings.active_model not in names:
        names.append(settings.active_model)
    return names


def tier_deployment(model: str) -> str:
    settings = llm_settings()
    deployments = settings.azure_deployments
    if model in deployments:
        return model
    if name := next((name for name, base in deployments.items() if base == model), None):
        return name
    if settings.active_provider == "azure" and settings.active_model:
        return settings.active_model
    return next(iter(deployments), model)


def register_streaming(model: str) -> None:
    if not model.startswith("azure/") or capability_model(model).removeprefix("openai/") not in GPT6_MODELS:
        return
    import litellm

    deployment = model.removeprefix("azure/").removeprefix("responses/")
    key = f"azure/{deployment}"
    existing = litellm.model_cost.get(key, {})
    # LiteLLM buffers responses when it cannot identify the deployment's streaming support.
    if existing.get("supports_native_streaming") is not True:
        litellm.register_model({key: {**existing, "litellm_provider": "azure", "mode": existing.get("mode", "chat"),
                                    "supports_native_streaming": True}})
