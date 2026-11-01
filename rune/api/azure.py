"""Explicit, non-persisting checks for Azure resource credentials and deployments."""

import asyncio
import re
from time import monotonic
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from rune.llm import azure
from rune.llm.context import llm_settings, model_environment
from rune.llm.target import ModelCompatibilityError
from rune.utils.logger import get_logger

log = get_logger(__name__)
_NAME = r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$"
_AZURE_HOSTS = (".openai.azure.com", ".services.ai.azure.com", ".cognitiveservices.azure.com",
                ".openai.azure.us", ".cognitiveservices.azure.us", ".openai.azure.cn",
                ".cognitiveservices.azure.cn")


class AzureConnection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    endpoint: str = Field(min_length=1, max_length=2048)
    apiKey: SecretStr | None = Field(None, exclude=True, repr=False)


class AzureTest(AzureConnection):
    deployment: str | None = Field(None, pattern=_NAME)
    model: str | None = Field(None, pattern=_NAME)


def _credentials(req: AzureConnection) -> tuple[str, str]:
    from rune.cloud.boundary import hosted

    if hosted():
        raise HTTPException(403, "Azure settings are only available in local Rune.")
    try:
        base = azure.resource_endpoint(req.endpoint)
        parsed = urlsplit(base)
        if not (parsed.hostname or "").endswith(_AZURE_HOSTS) or parsed.port not in {None, 443}:
            raise ValueError("not an Azure resource")
    except (ModelCompatibilityError, ValueError) as exc:
        raise HTTPException(400, "Enter an HTTPS Azure resource endpoint, optionally ending in /openai/v1.") from exc
    key = req.apiKey.get_secret_value().strip() if req.apiKey else ""
    if not key:
        try:
            saved = azure.resource_endpoint(azure.endpoint())
        except ModelCompatibilityError:
            saved = ""
        if base != saved:
            raise HTTPException(400, "Enter an API key for this endpoint. The saved key belongs to a different resource.")
        key = model_environment(azure.environment_name(azure.KEY_NAMES)).strip()
    if not key or not key.isascii() or any(ord(char) < 32 or ord(char) == 127 for char in key):
        raise HTTPException(400, "Enter a valid Azure API key.")
    return base, key


def _http_error(status: int, *, deployment: bool) -> str:
    if status == 401:
        return "Azure rejected the API key. Check the key and resource endpoint."
    if status == 403:
        return "Azure denied access. Check resource permissions, key authentication, and network restrictions."
    if status == 404:
        return ("Deployment not found. Use its exact name from Foundry, not a catalog model ID."
                if deployment else "The model catalog is unavailable at this endpoint. Check the Azure resource URL.")
    if status == 429:
        return "Azure is rate limiting this resource or its quota is exhausted. Try again later."
    if 300 <= status < 400:
        return "Azure returned a redirect. Use the final resource endpoint; credentials were not forwarded."
    if status == 400:
        return "Azure rejected the test request. Check the deployment, underlying model ID, and supported API."
    return f"Azure returned HTTP {status}. Try again or check the resource in Foundry."


async def _request(req: AzureConnection, *, deployment: str | None = None, model: str | None = None) -> dict:
    base, key = _credentials(req)
    path = "/models"
    body = None
    responses = False
    if deployment:
        from rune.agent.model_traits import traits

        settings = llm_settings()
        # A diagnostic is one bounded request, not an agent run or a retry loop.
        if settings.model_request_limit == 0 or settings.cost_budget_usd is not None:
            return {"ok": False, "detail": "Deployment tests are disabled by the configured request or cost limit. "
                    "Use Test connection to check credentials without generating tokens."}
        model = model or settings.azure_deployments.get(deployment, deployment)
        responses = traits(f"openai/{model}").responses_api
        prompt = "Reply with exactly OK."
        if responses:
            path = "/responses"
            body = {"model": deployment, "input": prompt, "max_output_tokens": 256, "store": False}
        else:
            path = "/chat/completions"
            cap = "max_completion_tokens" if traits(f"openai/{model}").max_completion_tokens else "max_tokens"
            body = {"model": deployment, "messages": [{"role": "user", "content": prompt}], cap: 256}
    started = monotonic()
    try:
        async with asyncio.timeout(60 if deployment else 15):
            async with httpx.AsyncClient(timeout=60 if deployment else 15, trust_env=False, follow_redirects=False) as client:
                response = await client.request("POST" if body else "GET", f"{base}/openai/v1{path}",
                                                headers={"api-key": key}, **({"json": body} if body else {}))
    except (TimeoutError, httpx.TimeoutException):
        log.debug("azure_check_timeout", deployment=bool(deployment))
        return {"ok": False, "detail": "Azure did not respond before the test timed out. Try again later."}
    except (httpx.HTTPError, OSError) as exc:
        log.debug("azure_check_failed", error=type(exc).__name__)
        return {"ok": False, "detail": "Could not reach Azure. Check the endpoint, network, and TLS configuration."}
    result = {"statusCode": response.status_code, "latencyMs": round((monotonic() - started) * 1000)}
    if response.status_code != 200:
        return {**result, "ok": False, "detail": _http_error(response.status_code, deployment=bool(deployment))}
    try:
        data = response.json()
        if not deployment:
            entries = data["data"]
            if not isinstance(entries, list) or not all(isinstance(item, dict) and isinstance(item.get("id"), str) for item in entries):
                raise ValueError("invalid catalog")
            names = sorted({item["id"] for item in entries if re.fullmatch(_NAME, item["id"])})
            skipped = sum(not re.fullmatch(_NAME, item["id"]) for item in entries)
            return {**result, "ok": True, "models": names,
                    "detail": "Endpoint and API key verified. This is the model catalog, not a list of your deployments."
                    + (f" {skipped} model IDs cannot be registered in Rune and were omitted." if skipped else "")}
        if responses:
            finished = data.get("status") == "completed"
            content = "".join(part.get("text", "") for item in data.get("output", [])
                              if item.get("type") == "message" for part in item.get("content", [])
                              if part.get("type") == "output_text")
        else:
            choice = data["choices"][0]
            finished = choice.get("finish_reason") == "stop"
            content = choice["message"].get("content")
        if not finished or not isinstance(content, str) or not content.strip():
            return {**result, "ok": False, "detail": "Azure accepted the request but did not return a complete text answer "
                    "within the short test. The deployment may need a larger reasoning budget or a different API."}
        return {**result, "ok": True, "detail": "Deployment connected: Azure returned a complete text response. "
                "Streaming and tool use were not tested."}
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        log.debug("azure_check_invalid_response", deployment=bool(deployment))
        return {**result, "ok": False, "detail": "Azure returned an unexpected response. Check the resource endpoint and model API."}


def azure_router(auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1/azure", dependencies=[Depends(auth)])

    @router.post("/models")
    async def models(req: AzureConnection) -> dict:
        return await _request(req)

    @router.post("/test")
    async def test(req: AzureTest) -> dict:
        result = await _request(req, deployment=req.deployment, model=req.model)
        result.pop("models", None)
        return result

    return router
