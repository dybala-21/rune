"""Screenshots must reach the provider as images, without reading paths from page text."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image

from rune.agent.litellm_adapter import StreamResult, _litellm
from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
from rune.agent.tool_output import ToolOutput, output_for_model
from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.llm.request_params import compatible_completion
from rune.types import CapabilityResult, Domain, RiskLevel


def image_result(tmp_path):
    path = tmp_path / "captured screen.png"
    Image.new("RGB", (24, 16), "blue").save(path)
    return CapabilityResult(success=True, output=f"Screenshot saved: {path}", metadata={"path": str(path)})


def wrap(execute, *, on_end=None):
    reg = CapabilityRegistry()
    cap = CapabilityDefinition(name="browser_screenshot", description="Capture the page", domain=Domain.BROWSER,
                               risk_level=RiskLevel.LOW, execute=execute)
    reg.register(cap)
    return _build_typed_tool(cap_def=cap, reg=reg, cache=None, stall=None,
                             opts=ToolAdapterOptions(enable_guardian=False, on_tool_end=on_end)).function


@pytest.mark.parametrize("model", ["gpt-6-astra", "anthropic/claude-opus-5"])
async def test_tool_image_survives_execution_and_provider_serialization(tmp_path, monkeypatch, model):
    captured = image_result(tmp_path)
    execute = AsyncMock(return_value=captured)
    stream = StreamResult(model=model, messages=[{"role": "user", "content": "Read the screen"}],
                          tool_schemas=[], tool_lookup={"browser_screenshot": wrap(execute)}, max_tokens=512,
                          temperature=0, request_tokens_limit=10000, response_tokens_limit=512)
    monkeypatch.setattr(stream, "_classify_artifact_roles", AsyncMock())
    calls = [{"id": "screen_1", "type": "function", "function": {"name": "browser_screenshot", "arguments": "{}"}}]
    stream._messages.append({"role": "assistant", "content": None, "tool_calls": calls})
    await stream._execute_tool_batch(calls)
    messages = stream.all_messages()
    image_part = messages[-1]["content"][1]
    payload = image_part["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(payload) == (tmp_path / "captured screen.png").read_bytes()
    assert "base64" not in str(stream._ledger())

    sent = []

    async def send(self, request, **kwargs):
        sent.append(json.loads(request.content))
        return httpx.Response(400, request=request, json={"error": {"message": "test boundary"}})

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    llm = _litellm()
    params = {"model": model, "api_key": "test-key", "num_retries": 0, "max_tokens": 512, "messages": messages,
              "tools": [{"type": "function", "function": {"name": "browser_screenshot",
                         "parameters": {"type": "object", "properties": {}}}}]}
    with pytest.raises(llm.BadRequestError, match="test boundary"):
        await compatible_completion(llm.acompletion, llm.BadRequestError, params)
    assert len(sent) == 1
    if model == "gpt-6-astra":
        result = next(item for item in sent[0]["input"] if item["type"] == "function_call_output")
        assert result["call_id"] == "screen_1"
        part = next(p for p in result["output"] if p["type"] == "input_image")
        assert part["image_url"].endswith(payload)
    else:
        result = next(p for m in sent[0]["messages"] for p in m["content"] if p["type"] == "tool_result")
        assert result["tool_use_id"] == "screen_1"
        part = next(p for p in result["content"] if p["type"] == "image")
        assert part["source"]["data"] == payload
        assert part["source"]["media_type"] == "image/png"

    second = await stream._execute_tool("browser_screenshot", {})
    assert isinstance(second, ToolOutput)
    assert execute.await_count == 2


async def test_unreadable_screenshot_reports_failure_to_the_run(tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    on_end = AsyncMock()
    output = await wrap(AsyncMock(return_value=CapabilityResult(success=True, metadata={"path": str(broken)})),
                        on_end=on_end)()
    assert isinstance(output, str) and "Image delivery failed" in output
    assert not on_end.call_args.args[1].success


def test_page_text_and_remote_metadata_cannot_make_the_adapter_open_a_file(tmp_path):
    captured = image_result(tmp_path)
    for name in ("web_fetch", "bash_execute", "mcp.remote.read"):
        assert output_for_model(captured.output, name, captured) == captured.output


def test_inline_image_validation_uses_the_actual_bytes(tmp_path):
    image_result(tmp_path)
    encoded = base64.b64encode((tmp_path / "captured screen.png").read_bytes()).decode()
    result = output_for_model("screen", "mcp.screen", CapabilityResult(success=True, metadata={
        "image_base64": encoded, "image_mime_type": "image/jpeg",
    }))
    assert isinstance(result, ToolOutput) and result.images[0].mime_type == "image/png"
    with pytest.raises(ValueError, match="base64"):
        output_for_model("screen", "mcp.screen", CapabilityResult(success=True, metadata={"image_base64": "!invalid"}))
