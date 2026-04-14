"""Provider-agnostic response normalization.

Different providers return the advisor's text in different shapes:

- OpenAI / LiteLLM: ``choices[0].message.content`` (str)
- Anthropic via litellm: may surface ``[{"type":"text","text":...}]``
- DeepSeek-R1 / QwQ: inline ``<think>...</think>`` reasoning blocks
- Local llama.cpp: may echo the system prompt, emit fenced ``tool_call``
  blocks, or wrap output in ``answer:`` prefixes.

The normalizer runs these structural strips BEFORE the verb parser.
Regex here is only applied to STRUCTURED markers (XML-like tags, fenced
blocks), never to natural language — in keeping with the project rule
against language-specific pattern matching.
"""

from __future__ import annotations

import re
from typing import Any

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_REASONING_TAG_RE = re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_FENCE_RE = re.compile(
    r"```(?:tool_call|tool_code|json)\s*\n.*?\n```",
    re.DOTALL | re.IGNORECASE,
)
_XML_TOOL_USE_RE = re.compile(
    r"<tool_use>.*?</tool_use>|<function_call>.*?</function_call>",
    re.DOTALL | re.IGNORECASE,
)
_SYSTEM_ECHO_PREFIXES = (
    "You are an ADVISOR",
    "You are the advisor",
    "System:",
)


def extract_text(raw: Any) -> str:
    """Pull the textual payload out of a provider response.

    Accepts:
    - Plain strings (already extracted)
    - LiteLLM ``ModelResponse`` objects
    - Anthropic-style ``[{"type":"text","text":...}]`` lists
    - Dicts shaped like ``{"content": ..., "message": ...}``
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if isinstance(raw, dict):
        if "choices" in raw:
            try:
                choice = raw["choices"][0]
                message = choice.get("message") if isinstance(choice, dict) else None
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        return extract_text(content)
            except (IndexError, KeyError, TypeError):
                pass
        for key in ("content", "text", "output_text"):
            val = raw.get(key)
            if isinstance(val, str):
                return val
            if isinstance(val, list):
                return extract_text(val)
        return ""
    # LiteLLM ModelResponse object
    try:
        choices = getattr(raw, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return extract_text(content)
    except (AttributeError, IndexError, TypeError):
        pass
    return ""


def strip_thinking_blocks(text: str) -> str:
    """Remove structured reasoning tags (``<think>``, ``<reasoning>``).

    Only matches literal XML-like tags emitted by DeepSeek-R1, QwQ, and
    similar reasoning models. Does not touch free-form text.
    """
    text = _THINK_TAG_RE.sub("", text)
    text = _REASONING_TAG_RE.sub("", text)
    return text


def strip_tool_call_attempts(text: str) -> str:
    """Remove fenced tool-call blocks and XML tool-use tags.

    The advisor is not supposed to call tools. If a provider emits a
    tool-call block anyway (common for OpenAI executors chained to an
    advisor call), we strip it so the verb parser isn't confused.
    """
    text = _TOOL_CALL_FENCE_RE.sub("", text)
    text = _XML_TOOL_USE_RE.sub("", text)
    return text


def strip_system_echo(text: str) -> str:
    """Drop leading lines that echo the system prompt (common with small
    local models that have weak instruction following)."""
    lines = text.splitlines()
    kept: list[str] = []
    skipping = True
    for line in lines:
        stripped = line.strip()
        if skipping and any(stripped.startswith(p) for p in _SYSTEM_ECHO_PREFIXES):
            continue
        if stripped:
            skipping = False
        kept.append(line)
    return "\n".join(kept)


def normalize(raw: Any) -> str:
    """Run the full normalization pipeline."""
    text = extract_text(raw)
    text = strip_thinking_blocks(text)
    text = strip_tool_call_attempts(text)
    text = strip_system_echo(text)
    return text.strip()
