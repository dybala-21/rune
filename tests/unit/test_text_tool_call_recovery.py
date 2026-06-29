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


# Guided-decoding schema (constrained tool-call output for local models)

def test_guided_flag_default_on(monkeypatch):
    from rune.agent.litellm_adapter import _guided_tools_enabled
    # Default ON (the per-model gate restricts it to local ollama); opt out with =0/off.
    monkeypatch.delenv("RUNE_GUIDED_TOOLS", raising=False)
    assert _guided_tools_enabled() is True
    monkeypatch.setenv("RUNE_GUIDED_TOOLS", "1")
    assert _guided_tools_enabled() is True
    monkeypatch.setenv("RUNE_GUIDED_TOOLS", "off")
    assert _guided_tools_enabled() is False


def test_build_action_schema_enum_names_plus_final():
    # Generic shape: tool name constrained to the real set via enum, arguments a
    # free object. A per-tool anyOf over dozens of rich param schemas is rejected
    # by ollama's grammar engine ("invalid JSON schema"), so it stays small to
    # scale to the full tool set.
    from rune.agent.litellm_adapter import _build_action_schema
    schemas = [
        {"function": {"name": "file_write", "parameters": {"type": "object"}}},
        {"function": {"name": "bash_execute", "parameters": {"type": "object"}}},
    ]
    schema = _build_action_schema(schemas)
    branches = schema["anyOf"]
    assert len(branches) == 2  # one tool-call branch + one final branch
    call = next(b for b in branches if "tool" in b["properties"])
    assert call["properties"]["tool"]["enum"] == ["file_write", "bash_execute"]
    assert call["properties"]["arguments"] == {"type": "object"}
    assert any("final" in b["properties"] for b in branches)


def test_build_action_schema_no_final_when_disallowed():
    # allow_final=False drops the {final} branch so a weak model can't bail to a
    # prose answer before producing a real artifact — only a tool call is valid.
    from rune.agent.litellm_adapter import _build_action_schema
    schemas = [{"function": {"name": "file_write", "parameters": {"type": "object"}}}]
    schema = _build_action_schema(schemas, allow_final=False)
    assert "anyOf" not in schema  # single tool-call branch, no final escape
    assert schema["required"] == ["tool", "arguments"]
    assert schema["properties"]["tool"]["enum"] == ["file_write"]


def test_arg_key_aliases_normalized():
    # Weak models emit filename/cmd/code; RUNE tools use path/command/content.
    from rune.agent.litellm_adapter import _extract_text_tool_calls
    text = '{"tool": "file_write", "arguments": {"filename": "a.py", "content": "x=1"}}'
    calls = _extract_text_tool_calls(text, {"file_write"})
    import json
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["path"] == "a.py" and "filename" not in args


def test_arg_alias_not_applied_when_canonical_present():
    from rune.agent.litellm_adapter import _normalize_arg_keys
    out = _normalize_arg_keys({"path": "real.py", "filename": "ignore.py"})
    assert out["path"] == "real.py"  # canonical wins; alias dropped


def test_edit_with_content_redirects_to_write():
    # Weak models pass the whole new file as `content` to file_edit, which fails
    # schema validation. Treat that as a full-file write.
    from rune.agent.litellm_adapter import _redirect_edit_to_write
    args = {"path": "a.py", "content": "x = 1\n", "replace": "", "all": False}
    fn = _redirect_edit_to_write("file_edit", args)
    assert fn == "file_write"
    assert "replace" not in args and "all" not in args  # edit-only keys dropped
    assert args["content"] == "x = 1\n"


def test_real_edit_with_search_is_not_redirected():
    # A genuine search/replace edit must stay file_edit.
    from rune.agent.litellm_adapter import _redirect_edit_to_write
    args = {"path": "a.py", "search": "x = 1", "replace": "x = 2"}
    assert _redirect_edit_to_write("file_edit", args) == "file_edit"


def test_non_edit_tool_untouched():
    from rune.agent.litellm_adapter import _redirect_edit_to_write
    args = {"path": "a.py", "content": "x = 1"}
    assert _redirect_edit_to_write("file_write", args) == "file_write"
