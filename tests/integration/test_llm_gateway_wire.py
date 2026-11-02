"""LiteLLM actually sends gateway-mode requests to <gateway>/<provider> with no provider key."""

import json

import litellm
import pytest
from fastapi import FastAPI, Request, Response

from rune.llm.request_params import compatible_completion
from rune.llm.target import resolve_target
from tests.integration.test_cloud_transport import serving

SECRET = "sk-local-provider-key-must-stay-home"


def fake_gateway(seen):
    app = FastAPI()

    @app.post("/{path:path}")
    async def handle(path: str, request: Request):
        seen.append({"path": "/" + path, "headers": dict(request.headers), "body": (await request.body()).decode()})
        if path.startswith("anthropic"):
            return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5",
                    "content": [{"type": "text", "text": "pong"}], "stop_reason": "end_turn",
                    "usage": {"input_tokens": 3, "output_tokens": 1}}
        if path.endswith("responses"):
            return {"id": "resp_1", "object": "response", "created_at": 0, "status": "completed", "model": "gpt-5.6",
                    "output": [{"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
                                "content": [{"type": "output_text", "text": "pong", "annotations": []}]}],
                    "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}}
        return {"id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "gpt-5.6",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "pong"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}

    return app


@pytest.mark.parametrize(("model", "path"), [
    ("anthropic/claude-sonnet-5", "/anthropic/v1/messages"),
    ("gpt-5.6", "/openai/responses"),
    ("openai/gpt-4o", "/openai/chat/completions"),
])
async def test_completion_goes_to_gateway_without_provider_key(model, path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.delenv("RUNE_LLM_GATEWAY_KEY", raising=False)
    seen = []
    async with serving(fake_gateway(seen)) as origin:
        monkeypatch.setenv("RUNE_LLM_GATEWAY_URL", origin)
        target = resolve_target(model)
        response = await compatible_completion(
            litellm.acompletion, litellm.BadRequestError,
            {**target.kwargs(), "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5,
             "_rune_source_model": target.key, "num_retries": 0})
    assert response.choices[0].message.content == "pong"
    assert len(seen) == 1 and seen[0]["path"] == path
    assert SECRET not in json.dumps(seen)
    sent = seen[0]["headers"].get("x-api-key") or seen[0]["headers"].get("authorization", "")
    assert "rune-gateway" in sent


@pytest.mark.parametrize(("model", "prefix"), [
    ("gemini/gemini-3.1-pro-preview", "/gemini/"),
    ("xai/grok-4.7", "/xai/"),
    ("mistral/mistral-large-latest", "/mistral/"),
    ("deepseek/deepseek-chat", "/deepseek/"),
    ("cohere/command-r-plus", "/cohere"),
])
async def test_every_routed_provider_reaches_its_gateway_prefix_without_a_key(model, prefix, monkeypatch):
    """Pins the per-provider path the gateway must serve; the response format is not under test."""
    for name in ("GEMINI_API_KEY", "XAI_API_KEY", "MISTRAL_API_KEY", "DEEPSEEK_API_KEY", "COHERE_API_KEY"):
        monkeypatch.setenv(name, SECRET)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    seen = []
    app = FastAPI()

    @app.post("/{path:path}")
    async def record(path: str, request: Request):
        seen.append({"path": "/" + path, "headers": dict(request.headers), "body": (await request.body()).decode()})
        return Response(status_code=503)

    async with serving(app) as origin:
        monkeypatch.setenv("RUNE_LLM_GATEWAY_URL", origin)
        target = resolve_target(model)
        with pytest.raises((litellm.exceptions.ServiceUnavailableError, litellm.exceptions.APIError,
                            litellm.exceptions.InternalServerError, litellm.exceptions.APIConnectionError)):
            await compatible_completion(
                litellm.acompletion, litellm.BadRequestError,
                {**target.kwargs(), "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5,
                 "_rune_source_model": target.key, "num_retries": 0})
    assert seen and all(entry["path"].startswith(prefix) for entry in seen)
    assert SECRET not in json.dumps(seen)
