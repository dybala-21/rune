"""Ollama and OpenAI-compatible inference servers chosen by the user.

vLLM, SGLang, LM Studio, llama.cpp server and hosted open-weight services all
serve the OpenAI chat API, so one provider reaches them through LiteLLM's
``openai/`` route with the server's ``/v1`` base.
https://docs.litellm.ai/docs/providers/openai_compatible
"""

from __future__ import annotations

import ipaddress
import re
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
    # The OpenAI client refuses to send without a key even when the server ignores it.
    result.setdefault("api_key", model_environment(key_name(endpoint)).strip() or "none")
    return result


def private_host(host: str | None) -> bool:
    """Loopback, private-network, link-local or single-label hosts."""
    host = (host or "").lower()
    if host == "localhost" or host.endswith((".localhost", ".local", ".lan", ".internal")) or "." not in host:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local


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


def tier_model(model: str) -> str:
    """Every tier runs the selected server model; others may not be loaded."""
    settings = llm_settings()
    if settings.active_provider == PROVIDER and settings.active_model:
        return str(settings.active_model)
    ids = model_ids()
    return model if model in ids or not ids else ids[0]


def _headers(endpoint: str) -> dict[str, str]:
    key = model_environment(key_name(endpoint)).strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


async def discover(endpoint: str, *, timeout: float = 3.0) -> list[str]:
    """Models the server reports at ``GET /models``; cached for the pickers."""
    entry = configured()[endpoint]
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
        if entry.models or now - _discovered.get(endpoint, (-_DISCOVERY_TTL, []))[0] < _DISCOVERY_TTL:
            continue
        try:
            await discover(endpoint, timeout=1.5)
        except (httpx.HTTPError, OSError, ValueError, KeyError, TypeError) as exc:
            _discovered[endpoint] = (now, [])
            log.debug("endpoint_discovery_failed", endpoint=endpoint, error=type(exc).__name__)


def forget_discovered() -> None:
    _discovered.clear()


_TOOL_HINT = ("The server rejected tool calls. vLLM needs --enable-auto-tool-choice and "
              "--tool-call-parser; llama.cpp needs --jinja. Otherwise set Tool mode to Guided.")


async def diagnose(endpoint: str, model: str | None = None, *, timeout: float = 30.0) -> dict[str, Any]:
    """List models and send one small tool request to show whether the agent can run."""
    try:
        names = await discover(endpoint, timeout=5.0)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        detail = "The API key was rejected." if status in {401, 403} else f"GET /models returned {status}."
        return {"reachable": False, "models": [], "tools": "unknown", "detail": detail}
    except (httpx.HTTPError, OSError, ValueError, KeyError, TypeError) as exc:
        return {"reachable": False, "models": [], "tools": "unknown",
                "detail": f"Could not reach the server ({type(exc).__name__}). Check the URL ends in /v1."}
    entry = configured()[endpoint]
    model = model or (entry.models[0] if entry.models else names[0] if names else None)
    if model is None:
        return {"reachable": True, "models": names, "tools": "unknown", "detail": "The server lists no models."}
    request = {
        "model": model, "max_tokens": 32, "tool_choice": "auto",
        "messages": [{"role": "user", "content": "Call the ping tool."}],
        "tools": [{"type": "function", "function": {
            "name": "ping", "description": "Reply to a ping.",
            "parameters": {"type": "object", "properties": {}}}}],
    }
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False, follow_redirects=False) as client:
            response = await client.post(f"{entry.base_url}/chat/completions",
                                         json=request, headers=_headers(endpoint))
    except (httpx.HTTPError, OSError) as exc:
        return {"reachable": True, "models": names, "model": model, "tools": "unknown",
                "detail": f"The test request failed ({type(exc).__name__})."}
    if response.status_code == 200:
        try:
            calls = response.json()["choices"][0]["message"].get("tool_calls")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            calls = None
        if calls:
            return {"reachable": True, "models": names, "model": model, "tools": "native",
                    "detail": "The model returned a native tool call."}
        return {"reachable": True, "models": names, "model": model, "tools": "text",
                "detail": "The server accepts tools but the model answered in text. Guided mode may work better."}
    body = response.text[:300]
    rejected = response.status_code == 400 and "tool" in body.lower()
    return {"reachable": True, "models": names, "model": model, "tools": "unsupported" if rejected else "unknown",
            "detail": _TOOL_HINT if rejected else f"The test request returned {response.status_code}: {body}"}
