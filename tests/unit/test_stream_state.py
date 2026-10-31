"""Replay model continuation metadata through the real streaming loop."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from litellm import ModelResponseStream

from rune.agent.litellm_adapter import StreamResult
from rune.llm.stream_state import StreamState


def chunk(delta=None, finish=None):
    return ModelResponseStream(choices=[{"index": 0, "delta": delta or {}, "finish_reason": finish}])


@pytest.mark.parametrize("model,state", [
    ("gpt-6-astra", {"reasoning_items": [{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"}]}),
    ("xai/grok-4.7", {"reasoning_items": [{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"}]}),
    ("vertex_ai/gemini-3.1-pro-preview", {"provider_specific_fields": {"thought_signatures": [{"text": "ready", "thought_signature": "opaque"}]}}),
    ("anthropic/claude-opus-5", {"thinking_blocks": [{"type": "thinking", "thinking": "context", "signature": "opaque"}]}),
])
async def test_tool_round_keeps_state_without_extra_calls(model, state, monkeypatch):
    monkeypatch.setenv("RUNE_ARTIFACT_PROVENANCE", "0")
    requests = []
    tc = {"index": 0, "id": "call_1", "type": "function", "function": {
        "name": "file_read", "arguments": '{"path":"input.txt"}',
        "provider_specific_fields": {"thought_signature": "function-signature"}},
        "provider_specific_fields": {"thought_signature": "call-signature"}}
    original = deepcopy(state)
    async def complete(**params):
        requests.append(deepcopy(params))
        async def generate():
            if len(requests) == 1:
                yield chunk({"tool_calls": [tc]})
                yield chunk(state, "tool_calls")
            else:
                yield chunk({"content": "Done."}, "stop")
        return generate()
    async def read(**kwargs):
        return "file contents"
    sr = StreamResult(model=model, messages=[{"role": "user", "content": "Read input.txt"}],
        tool_schemas=[{"type": "function", "function": {"name": "file_read", "parameters": {"type": "object"}}}],
        tool_lookup={"file_read": read}, max_tokens=1000, temperature=0,
        request_tokens_limit=100_000, response_tokens_limit=1000)
    with patch("litellm.acompletion", new=complete):
        output = "".join([part async for part in sr.stream_text()])
    assert output == "Done." and len(requests) == 2
    assistant = next(m for m in requests[1]["messages"] if m["role"] == "assistant")
    for key, value in state.items():
        assert assistant[key] == value
    call = assistant["tool_calls"][0]
    assert call["provider_specific_fields"] == tc["provider_specific_fields"]
    assert state == original
    if "responses/" in requests[0]["model"]:
        bodies = [p["extra_body"] for p in requests]
        assert bodies[0]["prompt_cache_key"] == bodies[1]["prompt_cache_key"]
        assert "reasoning.encrypted_content" in bodies[0]["include"]
        assert bodies[0]["store"] is False


def test_thinking_chunks_and_redacted_blocks_keep_order():
    state = StreamState()
    for block in [
        {"type": "thinking", "thinking": "first "},
        {"type": "thinking", "thinking": "block"},
        {"type": "thinking", "signature": "sig-"},
        {"type": "thinking", "signature": "one"},
        {"type": "redacted_thinking", "data": "redacted"},
        {"type": "thinking", "signature": "signature-only"},
    ]:
        state.observe(SimpleNamespace(thinking_blocks=[block]))
    assert state.message()["thinking_blocks"] == [
        {"type": "thinking", "thinking": "first block", "signature": "sig-one"},
        {"type": "redacted_thinking", "data": "redacted"},
        {"type": "thinking", "thinking": "", "signature": "signature-only"},
    ]
    assert StreamState().message("plain") == {"role": "assistant", "content": "plain"}


async def test_responses_routes_preserve_settings_and_optional_fields():
    from rune.llm.pricing import rates_for
    from rune.llm.request_params import compatible_completion
    captured = []
    async def complete(**params):
        captured.append(params)
    for model in ["gpt-6-astra", "xai/grok-4.7"]:
        await compatible_completion(complete, ValueError, {"model": model, "messages": [],
            "reasoning_effort": "medium", "extra_body": {"include": ["message.output_text.logprobs"]},
            "prompt_cache_key": "task-a"})
        params = captured[-1]
        assert "/responses/" in params["model"]
        assert params["extra_body"]["reasoning"]["effort"] == "medium"
        assert "reasoning_effort" not in params["extra_body"]
        assert params["extra_body"]["include"] == ["message.output_text.logprobs", "reasoning.encrypted_content"]
        assert rates_for(params["model"]) == rates_for(model)


def test_responses_state_survives_sdk_request_conversion():
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )
    item = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"}
    state = StreamState()
    state.observe({"reasoning_items": [item]})
    message = state.message()
    message["tool_calls"] = [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]
    handler = LiteLLMResponsesTransformationHandler()
    items, _ = handler.convert_chat_completion_messages_to_responses_api([message,
        {"role": "tool", "tool_call_id": "c1", "content": "data"}])
    assert [i["type"] for i in items] == ["reasoning", "function_call", "function_call_output"]
    assert items[0]["encrypted_content"] == "opaque"


def test_signature_budget_counts_the_wire_value_once():
    from rune.agent.request_budget import input_estimate
    signature = "abcdefgh" * 1000
    call = {"id": f"c1__thought__{signature}", "type": "function",
            "function": {"name": "read", "arguments": "{}"},
            "provider_specific_fields": {"thought_signature": signature}}
    params = {"model": "vertex_ai/gemini-3.1-pro-preview", "messages": [
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": "data"}]}
    native = deepcopy(params)
    native["messages"][0]["tool_calls"][0]["id"] = "c1"
    native["messages"][1]["tool_call_id"] = "c1"
    assert input_estimate(params) == input_estimate(native)
    assert params["messages"][0]["tool_calls"][0]["id"] == f"c1__thought__{signature}"


def test_duplicate_sdk_thinking_fields_are_not_replayed_twice():
    blocks = [{"type": "thinking", "thinking": "text", "signature": "opaque"}]
    state = StreamState()
    state.observe({"thinking_blocks": blocks, "provider_specific_fields": {"thinking_blocks": blocks}})
    assert state.message() == {"role": "assistant", "thinking_blocks": blocks}


async def test_xai_system_instructions_survive_the_sdk_wire_conversion():
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )
    from litellm.llms.xai.responses.transformation import XAIResponsesAPIConfig

    from rune.llm.request_params import compatible_completion
    requests = []
    async def complete(**params):
        requests.append(params)
    await compatible_completion(complete, ValueError, {
        "model": "xai/grok-4.7", "messages": [
            {"role": "system", "content": "Preserve the input contract."},
            {"role": "user", "content": "Fix the calculation."}],
        "reasoning_effort": "medium", "prompt_cache_key": "task-a", "max_tokens": 1000})
    params = requests[0]
    wire = LiteLLMResponsesTransformationHandler().transform_request(
        model="grok-4.7", messages=params["messages"],
        optional_params={k: v for k, v in params.items() if k not in {"model", "messages"}},
        litellm_params={}, headers={}, litellm_logging_obj=SimpleNamespace())
    native = XAIResponsesAPIConfig().map_openai_params(wire, "grok-4.7", drop_params=True)
    assert native["input"][0] == {"type": "message", "role": "system", "content": [
        {"type": "input_text", "text": "Preserve the input contract."}]}
    assert native["reasoning"]["effort"] == "medium"
    assert native["prompt_cache_key"] == "task-a"
    assert native["include"] == ["reasoning.encrypted_content"]


def test_model_switch_keeps_visible_history_and_tool_pairs_without_foreign_signatures():
    from rune.llm.stream_state import continuation_messages
    messages = [{"role": "assistant", "content": "Read the source.", "_rune_state_model": "model-a",
        "thinking_blocks": [{"type": "thinking", "thinking": "internal", "signature": "opaque"}],
        "tool_calls": [{"id": "c1__thought__opaque", "type": "function", "function": {
            "name": "read", "arguments": "{}", "provider_specific_fields": {"thought_signature": "opaque"}}}]},
        {"role": "tool", "tool_call_id": "c1__thought__opaque", "content": "original source"}]
    original = deepcopy(messages)
    same = continuation_messages(messages, "model-a")
    assert same[0]["thinking_blocks"] == messages[0]["thinking_blocks"]
    assert "_rune_state_model" not in same[0]
    switched = continuation_messages(messages, "model-b")
    assert "thinking_blocks" not in switched[0]
    assert switched[0]["content"] == "Read the source."
    assert switched[1]["content"] == "original source"
    assert switched[0]["tool_calls"][0]["id"] == switched[1]["tool_call_id"] == "c1"
    assert switched[0]["tool_calls"][0]["function"] == {"name": "read", "arguments": "{}"}
    assert messages == original


def test_unsigned_partial_thinking_is_not_sent_as_a_signed_block():
    state = StreamState()
    state.observe({"thinking_blocks": [{"type": "thinking", "thinking": "unfinished"}]})
    assert state.message("partial answer") == {"role": "assistant", "content": "partial answer"}
