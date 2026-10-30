"""Ollama and OpenAI-compatible inference servers chosen by the user.

vLLM, SGLang, LM Studio, llama.cpp server and hosted open-weight services all
serve the OpenAI chat API, so one provider reaches them through LiteLLM's
``openai/`` route with the server's ``/v1`` base.
https://docs.litellm.ai/docs/providers/openai_compatible
"""

from __future__ import annotations

import ipaddress
import re
import socket
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx

from rune.llm.context import llm_settings, model_environment
from rune.utils.logger import get_logger

if TYPE_CHECKING:
    from rune.config.schema import CompatibleEndpoint

log = get_logger(__name__)

PROVIDER = "openai_compatible"
OLLAMA_PORT = 11434
OLLAMA_NAMES = ("OLLAMA_API_BASE", "OLLAMA_HOST")
_DISCOVERY_TTL = 300.0
# A server that was down is retried sooner than a healthy list is refreshed.
_FAILED_DISCOVERY_TTL = 30.0
_discovered: dict[str, tuple[float, list[str]]] = {}


# Ollama

def ollama_root() -> str:
    """Ollama server root without ``/v1``.

    ``OLLAMA_HOST`` follows Ollama's own form, where the scheme and port may be
    omitted and a bind-all address names the local server.
    """
    for name in OLLAMA_NAMES:
        value = model_environment(name).strip()
        if not value:
            continue
        url = urlsplit(value if "://" in value else f"http://{value}")
        if url.scheme not in {"http", "https"} or not url.hostname:
            log.warning("ollama_base_ignored", variable=name)
            continue
        host = "localhost" if url.hostname in {"0.0.0.0", "::"} else url.hostname
        host = f"[{host}]" if ":" in host else host
        path = url.path.rstrip("/").removesuffix("/v1")
        return f"{url.scheme}://{host}:{url.port or OLLAMA_PORT}{path}"
    return f"http://localhost:{OLLAMA_PORT}"


def ollama_v1() -> str:
    return f"{ollama_root()}/v1"


def is_ollama_request(model: str, provider_extra: object) -> bool:
    """Whether a request goes to Ollama, judged by the source model or its base."""
    extra = provider_extra if isinstance(provider_extra, dict) else {}
    source = str(extra.get("_rune_source_model") or model)
    if source.startswith("ollama/"):
        return True
    base = str(extra.get("api_base") or "").rstrip("/")
    return bool(base) and base == ollama_v1()


# OpenAI-compatible endpoints

def configured() -> dict[str, CompatibleEndpoint]:
    return dict(getattr(llm_settings(), "endpoints", None) or {})


def key_name(endpoint: str) -> str:
    return f"RUNE_ENDPOINT_{re.sub(r'[^A-Z0-9]', '_', endpoint.upper())}_API_KEY"


def split_model(model: str) -> tuple[str, str]:
    """``<endpoint>/<server model id>``; the server id may itself contain slashes."""
    from rune.llm.target import ModelCompatibilityError

    if not model:
        raise ModelCompatibilityError(
            "No model is selected for the OpenAI-compatible endpoints. "
            "Pick one in the model picker, or list models for the endpoint in Settings.")
    endpoint, separator, name = model.partition("/")
    if not separator or not name:
        raise ModelCompatibilityError(
            f"OpenAI-compatible models are named <endpoint>/<model>, not {model!r}.")
    if endpoint not in configured():
        raise ModelCompatibilityError(
            f"No OpenAI-compatible endpoint named {endpoint!r}. Add it in Settings first.")
    return endpoint, name


def connection_options(endpoint: str, options: dict[str, Any]) -> dict[str, Any]:
    result = dict(options)
    result.setdefault("api_base", configured()[endpoint].base_url)
    check_destination(result["api_base"])
    # The OpenAI client refuses to send without a key even when the server ignores it.
    result.setdefault("api_key", model_environment(key_name(endpoint)).strip() or "none")
    return result


def _address(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    # Resolvers also accept legacy IPv4 forms such as 0x7f.1 or 2130706433.
    try:
        return ipaddress.ip_address(socket.inet_ntoa(socket.inet_aton(host)))
    except OSError:
        return None


def private_host(host: str | None) -> bool:
    """Whether a host is not on the public internet.

    IP literals use the IANA special-purpose registry (``is_global``), which
    also covers IPv4-mapped IPv6, CGNAT/Tailscale and reserved ranges. Names
    count as private when they are single-label or use a local-only suffix.
    """
    host = (host or "").lower().rstrip(".")
    address = _address(host)
    if address is not None:
        return not address.is_global
    return host == "localhost" or "." not in host or host.endswith(
        (".localhost", ".local", ".lan", ".home.arpa", ".internal"))


def check_destination(base_url: str) -> None:
    """In hosted mode, refuse endpoints that resolve to non-public addresses.

    Validation already rejects private names and literals; resolving here also
    catches public names pointed at private addresses. A name that changes its
    answer between this check and the request (DNS rebinding) is not covered.
    """
    from rune.cloud.boundary import hosted

    if not hosted():
        return
    from rune.llm.target import ModelCompatibilityError

    url = urlsplit(base_url)
    host = url.hostname or ""
    if url.scheme != "https" or private_host(host):
        raise ModelCompatibilityError("Hosted Rune can only connect to public HTTPS endpoints.")
    try:
        answers = socket.getaddrinfo(host, url.port or 443, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise ModelCompatibilityError(f"Could not resolve {host}.") from exc
    if any(private_host(str(answer[4][0])) for answer in answers):
        raise ModelCompatibilityError("Hosted Rune can only connect to public HTTPS endpoints.")


def is_free(endpoint: str) -> bool:
    """Self-hosted servers bill nothing per token; hosted services are unpriced."""
    entry = configured().get(endpoint)
    if entry is None:
        return False
    if entry.free is not None:
        return entry.free
    return private_host(urlsplit(entry.base_url).hostname)


def self_hosted(provider: str, model: str) -> bool:
    """Ollama or a free OpenAI-compatible endpoint: requests stay on the user's servers."""
    if provider == "ollama":
        return True
    endpoint = endpoint_of(f"{provider}/{model}")
    return endpoint is not None and is_free(endpoint)


def endpoint_of(source_model: str) -> str | None:
    provider, _, rest = source_model.partition("/")
    if provider != PROVIDER:
        return None
    endpoint = rest.partition("/")[0]
    return endpoint if endpoint in configured() else None


def guided(source_model: str) -> bool:
    endpoint = endpoint_of(source_model)
    return endpoint is not None and configured()[endpoint].tool_mode == "guided"


def model_ids() -> list[str]:
    """Selectable ``<endpoint>/<model>`` ids: pinned models, then discovered ones."""
    result: list[str] = []
    for endpoint, entry in configured().items():
        names = list(entry.models) or _discovered.get(endpoint, (0.0, []))[1]
        result.extend(f"{endpoint}/{name}" for name in names)
    settings = llm_settings()
    if settings.active_provider == PROVIDER and settings.active_model and settings.active_model not in result:
        result.append(settings.active_model)
    return result


def tier_model() -> str:
    """The model every tier runs, since other server models may not be loaded.

    The selected model, else the configured default, else the first one
    available. Empty when there is none, so the request fails with a clear
    message instead of a model name from another provider.
    """
    settings = llm_settings()
    for provider, model in ((settings.active_provider, settings.active_model),
                            (settings.default_provider, settings.default_model)):
        if provider == PROVIDER and model and endpoint_of(f"{PROVIDER}/{model}"):
            return str(model)
    ids = model_ids()
    return ids[0] if ids else ""


def capability_model(model: str) -> str:
    """The id used to look up model capabilities: the request's own ``openai/`` route."""
    provider, separator, rest = model.partition("/")
    if provider == PROVIDER and separator:
        return f"openai/{rest.partition('/')[2] or rest}"
    from rune.llm.azure import capability_model as azure_capability

    return azure_capability(model)


def accepts_images(model: str) -> bool | None:
    """Whether an endpoint model takes images, as the user declared; None for other providers."""
    endpoint = endpoint_of(model)
    if endpoint is None:
        return None
    return model.partition("/")[2].partition("/")[2] in configured()[endpoint].vision_models


def _headers(endpoint: str) -> dict[str, str]:
    key = model_environment(key_name(endpoint)).strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


async def discover(endpoint: str, *, timeout: float = 3.0) -> list[str]:
    """Models the server reports at ``GET /models``; cached for the pickers."""
    entry = configured()[endpoint]
    check_destination(entry.base_url)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False, follow_redirects=False) as client:
        response = await client.get(f"{entry.base_url}/models", headers=_headers(endpoint))
    response.raise_for_status()
    names = sorted({str(item["id"]) for item in response.json().get("data", []) if item.get("id")})
    _discovered[endpoint] = (time.monotonic(), names)
    return names


async def refresh_discovered() -> None:
    """Refresh stale discovery results without failing the caller."""
    now = time.monotonic()
    for endpoint, entry in configured().items():
        checked, names = _discovered.get(endpoint, (None, []))
        ttl = _DISCOVERY_TTL if names else _FAILED_DISCOVERY_TTL
        if entry.models or (checked is not None and now - checked < ttl):
            continue
        try:
            await discover(endpoint, timeout=1.5)
        except (httpx.HTTPError, OSError, ValueError, KeyError, TypeError) as exc:
            _discovered[endpoint] = (now, [])
            log.debug("endpoint_discovery_failed", endpoint=endpoint, error=type(exc).__name__)


def forget_discovered() -> None:
    _discovered.clear()


_TOOL_HINT = ("If the model supports tools, vLLM needs --enable-auto-tool-choice and --tool-call-parser "
              "and llama.cpp needs --jinja. Otherwise set Tool mode to Guided.")


def _error_message(response: httpx.Response) -> str:
    """The server's own error text: OpenAI-style ``error.message``, else the raw body."""
    try:
        error = response.json().get("error")
        message = error.get("message") if isinstance(error, dict) else error
    except (ValueError, AttributeError):
        message = None
    return str(message or response.text)[:300]


async def diagnose(endpoint: str, model: str | None = None, *, timeout: float = 120.0) -> dict[str, Any]:
    """List models and send one small tool request to show whether the agent can run.

    The first request to a local server may include loading the model, hence the
    long timeout. Reasoning models spend completion tokens before any tool call
    (vLLM counts reasoning against ``max_tokens``), so the budget leaves room for
    that, and a response cut off by the limit is reported as inconclusive.
    https://docs.vllm.ai/en/latest/features/reasoning_outputs.html
    """
    from rune.llm.target import ModelCompatibilityError

    entry = configured()[endpoint]
    try:
        check_destination(entry.base_url)
    except ModelCompatibilityError as exc:
        return {"reachable": False, "models": [], "tools": "unknown", "detail": str(exc)}
    names: list[str] = []
    listing = ""
    try:
        names = await discover(endpoint, timeout=10.0)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in {401, 403}:
            return {"reachable": False, "models": [], "tools": "unknown", "detail": "The API key was rejected."}
        listing = f"GET /models returned {status}. "
    except (httpx.HTTPError, OSError, ValueError, KeyError, TypeError) as exc:
        return {"reachable": False, "models": [], "tools": "unknown",
                "detail": f"Could not reach the server ({type(exc).__name__}). Check the URL, usually ending in /v1."}
    model = model or (entry.models[0] if entry.models else names[0] if names else None)
    if model is None:
        return {"reachable": True, "models": names, "tools": "unknown",
                "detail": listing + "No model to test. List the model names in Settings."}
    result: dict[str, Any] = {"reachable": True, "models": names, "model": model}
    request = {
        "model": model, "max_tokens": 1024, "tool_choice": "auto",
        "messages": [{"role": "user", "content": "Call the ping tool now."}],
        "tools": [{"type": "function", "function": {
            "name": "ping", "description": "Reply to a ping.",
            "parameters": {"type": "object", "properties": {}}}}],
    }
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False, follow_redirects=False) as client:
            response = await client.post(f"{entry.base_url}/chat/completions",
                                         json=request, headers=_headers(endpoint))
    except (httpx.HTTPError, OSError) as exc:
        return {**result, "tools": "unknown", "detail": listing + f"The test request failed ({type(exc).__name__})."}
    if response.status_code == 200:
        try:
            choice = response.json()["choices"][0]
            calls = choice["message"].get("tool_calls")
            finish = choice.get("finish_reason")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            return {**result, "tools": "unknown", "detail": listing + "The server returned an unexpected response."}
        if calls:
            return {**result, "tools": "native", "detail": listing + "The model returned a native tool call."}
        if finish == "length":
            return {**result, "tools": "unknown", "detail": listing + (
                "The model used its whole test budget, likely on reasoning, before answering. "
                "Run a short task to check tool use.")}
        return {**result, "tools": "text", "detail": listing + (
            "The server accepts tools but the model answered in text. Guided mode may work better.")}
    message = _error_message(response)
    rejected = response.status_code == 400 and "tool" in message.lower()
    detail = f"Tool calls were rejected: {message}. {_TOOL_HINT}" if rejected else (
        f"The test request returned {response.status_code}: {message}")
    return {**result, "tools": "unsupported" if rejected else "unknown", "detail": listing + detail}
