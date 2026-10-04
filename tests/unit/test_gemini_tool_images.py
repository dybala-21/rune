from copy import deepcopy

import pytest

from rune.llm.request_params import _gemini_tool_images, compatible_completion


def messages():
    return [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": name, "type": "function", "function": {"name": "document_preview", "arguments": "{}"}}
            for name in ("a", "b")
        ]},
        *[{"role": "tool", "tool_call_id": name, "content": [
            {"type": "text", "text": f"Page {index}"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{name}"}},
        ]} for index, name in enumerate(("a", "b"), 1)],
        {"role": "assistant", "content": "Two pages"},
    ]


@pytest.mark.parametrize("model", ["gemini/gemini-2.5-flash", "vertex_ai/gemini-2.5-pro"])
async def test_older_gemini_receives_images_after_all_tool_results(model):
    original = messages()
    before = deepcopy(original)
    requests = []

    async def complete(**kwargs):
        requests.append(kwargs)

    await compatible_completion(complete, ValueError, {"model": model, "messages": original})
    wire = requests[0]["messages"]
    assert [m["role"] for m in wire] == ["assistant", "tool", "tool", "user", "assistant"]
    assert [(m["tool_call_id"], m["content"]) for m in wire if m["role"] == "tool"] == [("a", "Page 1"), ("b", "Page 2")]
    parts = wire[3]["content"]
    assert [p["image_url"]["url"] for p in parts if p["type"] == "image_url"] == ["data:image/png;base64,a", "data:image/png;base64,b"]
    assert all("untrusted tool data" in p["text"] for p in parts if p["type"] == "text")
    assert original == before
    assert _gemini_tool_images(model, before[:-1])[-1]["role"] == "user"


@pytest.mark.parametrize("model", ["gemini/gemini-3-flash-preview", "anthropic/claude-opus-5", "openai/gpt-6-astra", "xai/grok-4.7"])
def test_other_models_keep_native_tool_images(model):
    original = messages()
    assert _gemini_tool_images(model, original) is original


def test_text_only_tools_need_no_added_message():
    original = [{"role": "tool", "tool_call_id": "read", "content": "text"}]
    assert _gemini_tool_images("gemini/gemini-2.5-flash", original) == original
