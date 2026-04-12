"""Unit tests for the provider-agnostic advisor response normalizer.

Covers the common provider output shapes:
- OpenAI dict choices
- LiteLLM ModelResponse-like objects
- Anthropic list-of-content-blocks
- DeepSeek-R1 <think> tags
- Qwen QwQ <reasoning> tags
- Local models with fenced tool_call blocks
- System prompt echo (weak local models)
"""

from __future__ import annotations

from types import SimpleNamespace

from rune.agent.advisor.normalizer import (
    extract_text,
    normalize,
    strip_system_echo,
    strip_thinking_blocks,
    strip_tool_call_attempts,
)


class TestExtractText:
    def test_plain_string(self):
        assert extract_text("hello") == "hello"

    def test_openai_dict_shape(self):
        raw = {
            "choices": [
                {"message": {"content": "NEXT: continue\n1. step"}}
            ],
        }
        assert "NEXT: continue" in extract_text(raw)

    def test_litellm_object_shape(self):
        msg = SimpleNamespace(content="NEXT: abort")
        choice = SimpleNamespace(message=msg)
        raw = SimpleNamespace(choices=[choice])
        assert extract_text(raw) == "NEXT: abort"

    def test_anthropic_content_blocks(self):
        raw = [
            {"type": "text", "text": "NEXT: continue"},
            {"type": "text", "text": "1. step one"},
        ]
        result = extract_text(raw)
        assert "NEXT: continue" in result
        assert "step one" in result

    def test_dict_with_content_key(self):
        assert extract_text({"content": "hi"}) == "hi"

    def test_none_returns_empty(self):
        assert extract_text(None) == ""

    def test_nested_content_list(self):
        raw = {
            "choices": [
                {"message": {"content": [{"type": "text", "text": "hi"}]}}
            ],
        }
        assert extract_text(raw) == "hi"


class TestStripThinkingBlocks:
    def test_deepseek_think_tag(self):
        text = "<think>long reasoning</think>\nNEXT: continue"
        result = strip_thinking_blocks(text)
        assert "<think>" not in result
        assert "NEXT: continue" in result

    def test_multiline_think(self):
        text = "<think>\nline 1\nline 2\n</think>\nNEXT: abort"
        result = strip_thinking_blocks(text)
        assert "line 1" not in result
        assert "NEXT: abort" in result

    def test_qwq_reasoning_tag(self):
        text = "<reasoning>some reasoning</reasoning>\nNEXT: continue\n1. do x"
        result = strip_thinking_blocks(text)
        assert "<reasoning>" not in result
        assert "some reasoning" not in result
        assert "NEXT: continue" in result

    def test_case_insensitive(self):
        assert "<THINK>" not in strip_thinking_blocks("<THINK>x</THINK>ok")

    def test_no_tags_unchanged(self):
        text = "NEXT: continue\n1. step"
        assert strip_thinking_blocks(text) == text


class TestStripToolCallAttempts:
    def test_fenced_tool_call(self):
        text = "NEXT: continue\n```tool_call\n{\"name\":\"x\"}\n```\n1. step"
        result = strip_tool_call_attempts(text)
        assert "tool_call" not in result.lower() or "```" not in result
        assert "NEXT: continue" in result

    def test_fenced_json(self):
        text = "NEXT: abort\n```json\n{\"action\":\"x\"}\n```"
        result = strip_tool_call_attempts(text)
        assert "```" not in result

    def test_xml_tool_use(self):
        text = "NEXT: continue\n<tool_use>do something</tool_use>\n1. step"
        result = strip_tool_call_attempts(text)
        assert "<tool_use>" not in result
        assert "NEXT: continue" in result

    def test_function_call_tag(self):
        text = "NEXT: abort\n<function_call>call</function_call>"
        result = strip_tool_call_attempts(text)
        assert "<function_call>" not in result


class TestStripSystemEcho:
    def test_drops_leading_echo(self):
        text = "You are an ADVISOR. You will not...\nNEXT: continue\n1. step"
        result = strip_system_echo(text)
        assert "NEXT: continue" in result
        assert not result.startswith("You are an ADVISOR")

    def test_keeps_content_after_first_real_line(self):
        text = "NEXT: continue\nYou are an ADVISOR (in middle)\n1. step"
        result = strip_system_echo(text)
        # Second-line echo is preserved; only leading echoes are dropped
        assert "NEXT: continue" in result

    def test_no_echo_unchanged(self):
        text = "NEXT: continue\n1. step"
        assert strip_system_echo(text) == text


class TestNormalizePipeline:
    def test_openai_plain(self):
        raw = {
            "choices": [{"message": {"content": "NEXT: continue\n1. step"}}],
        }
        out = normalize(raw)
        assert out.startswith("NEXT: continue")

    def test_deepseek_r1_full_pipeline(self):
        raw = (
            "<think>I should tell them to retry</think>\n"
            "NEXT: retry_tool:file_read\n"
            "1. read main.py again"
        )
        out = normalize(raw)
        assert "<think>" not in out
        assert "NEXT: retry_tool:file_read" in out

    def test_contaminated_local_model(self):
        raw = (
            "You are an ADVISOR. You will not take actions.\n"
            "<think>reasoning</think>\n"
            "NEXT: abort\n"
            "```tool_call\nbogus\n```\n"
            "1. stop execution"
        )
        out = normalize(raw)
        assert "NEXT: abort" in out
        assert "tool_call" not in out.lower() or "```" not in out
        assert "<think>" not in out
