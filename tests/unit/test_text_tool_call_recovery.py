"""Recover text-emitted tool calls from local models.

Small ollama models (qwen2.5-coder, gemma) sometimes emit a tool call as JSON in
the message content instead of as native tool_calls, so the agent loop would run
nothing. These tests cover the parser; the name must match a known tool so real
text answers are not misread as calls.
"""

from __future__ import annotations

from rune.agent.litellm_adapter import _extract_text_tool_calls, _iter_json_objects

KNOWN = {"file_write", "file_read", "bash_execute"}


def test_recovers_plain_json_tool_call():
    text = '{"name": "file_write", "arguments": {"path": "a.py", "content": "x=1"}}'
    calls = _extract_text_tool_calls(text, KNOWN)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "file_write"
    assert '"path": "a.py"' in calls[0]["function"]["arguments"]


def test_recovers_fenced_and_prose_wrapped():
    text = (
        "Sure, I'll write it.\n```json\n"
        '{"name": "file_write", "arguments": {"path": "b.py", "content": "y=2"}}\n'
        "```\nDone."
    )
    calls = _extract_text_tool_calls(text, KNOWN)
    assert len(calls) == 1 and calls[0]["function"]["name"] == "file_write"


def test_handles_nested_braces_in_content():
    # content itself contains braces — balanced scan must not break.
    text = '{"name": "file_write", "arguments": {"path": "c.py", "content": "d = {1: 2}"}}'
    calls = _extract_text_tool_calls(text, KNOWN)
    assert len(calls) == 1
    assert "d = {1: 2}" in calls[0]["function"]["arguments"]


def test_alt_shapes_tool_and_parameters():
    text = '{"tool": "bash_execute", "parameters": {"command": "ls"}}'
    calls = _extract_text_tool_calls(text, KNOWN)
    assert len(calls) == 1 and calls[0]["function"]["name"] == "bash_execute"


def test_nested_function_shape():
    text = '{"function": {"name": "file_read", "arguments": {"path": "z.py"}}}'
    calls = _extract_text_tool_calls(text, KNOWN)
    assert len(calls) == 1 and calls[0]["function"]["name"] == "file_read"


def test_unknown_name_is_not_a_tool_call():
    # A real text answer that happens to be JSON must NOT be treated as a call.
    text = '{"name": "Alice", "arguments": {"age": 30}}'
    assert _extract_text_tool_calls(text, KNOWN) == []


def test_plain_prose_yields_nothing():
    assert _extract_text_tool_calls("Here is the answer: 42.", KNOWN) == []


def test_multiple_calls_recovered_in_order():
    text = (
        '{"name": "file_read", "arguments": {"path": "a"}} then '
        '{"name": "file_write", "arguments": {"path": "b", "content": "c"}}'
    )
    calls = _extract_text_tool_calls(text, KNOWN)
    assert [c["function"]["name"] for c in calls] == ["file_read", "file_write"]


def test_iter_json_objects_skips_braces_in_strings():
    objs = _iter_json_objects('{"k": "a } b { c"}')
    assert objs == ['{"k": "a } b { c"}']


def test_empty_known_set_recovers_nothing():
    text = '{"name": "file_write", "arguments": {}}'
    assert _extract_text_tool_calls(text, set()) == []
