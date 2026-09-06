"""Run a request on /v1/responses and hand the loop chat-completions chunks.

gpt-6-astra, the gpt-5.6 family and gpt-5.3-codex reject tools on
/v1/chat/completions — measured live, and not fixable by dropping
reasoning_effort, since they carry a default one and "none" is refused. An
agent step always carries tools, so those models cannot take a step without
this.

The streaming loop reads four fields off a chunk: delta.content,
delta.tool_calls, finish_reason, usage. The responses event stream is
reshaped into those rather than the loop being rewritten for a second format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)


# Request translation


def to_responses_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Flatten chat tool schemas into the responses shape.

    Responses puts the definition at the top level and requires
    ``additionalProperties: false``. ``strict`` stays off: these schemas come
    from capability signatures, and strict mode rejects constructs it cannot
    model.
    """
    out: list[dict[str, Any]] = []
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict):
            continue
        params = dict(fn.get("parameters") or {"type": "object", "properties": {}})
        if params.get("type") == "object":
            params.setdefault("additionalProperties", False)
        out.append({
            "type": "function",
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": params,
            "strict": False,
        })
    return out


def to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert chat messages into responses input items.

    Tool calls and their results are top-level items here, not fields on an
    assistant message, so one assistant turn can expand into several items.
    Dropping either would hide from the model what it just ran.
    """
    items: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")

        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": _as_text(msg.get("content")),
            })
            continue

        if role == "assistant":
            content = msg.get("content")
            if content:
                items.append({"role": "assistant", "content": _as_text(content)})
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                items.append({
                    "type": "function_call",
                    "call_id": call.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "") or "{}",
                })
            continue

        items.append({"role": role or "user", "content": _as_text(msg.get("content"))})
    return items


def _as_text(content: Any) -> str:
    """Multimodal content arrives as parts; keep the text and note the rest."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") in ("text", "input_text")
        ]
        return "\n".join(x for x in parts if x)
    return str(content)


# Stream translation — the four fields the loop reads


@dataclass(slots=True)
class _Fn:
    name: str = ""
    arguments: str = ""


@dataclass(slots=True)
class _ToolCallDelta:
    index: int = 0
    id: str = ""
    function: _Fn = field(default_factory=_Fn)


@dataclass(slots=True)
class _Delta:
    content: str | None = None
    tool_calls: list[_ToolCallDelta] | None = None


@dataclass(slots=True)
class _Choice:
    delta: _Delta = field(default_factory=_Delta)
    finish_reason: str | None = None


@dataclass(slots=True)
class _Chunk:
    """A chat-completions chunk as far as the streaming loop is concerned."""

    choices: list[_Choice] = field(default_factory=list)
    usage: Any = None


def _event_type(event: Any) -> str:
    raw = getattr(event, "type", None)
    if raw is None and isinstance(event, dict):
        raw = event.get("type")
    return str(raw or "").rsplit(".", 1)[-1].lower()


def _get(event: Any, key: str) -> Any:
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


class ResponsesToChatStream:
    """Iterate a responses event stream as chat-completions chunks.

    output_index numbers the function calls, and is passed through as the
    tool-call index the loop accumulates on.
    """

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def __aiter__(self) -> ResponsesToChatStream:
        self._it = self._stream.__aiter__()
        return self

    async def __anext__(self) -> _Chunk:
        while True:
            event = await self._it.__anext__()
            chunk = self._translate(event)
            if chunk is not None:
                return chunk

    def _translate(self, event: Any) -> _Chunk | None:
        kind = _event_type(event)

        if kind == "output_text_delta":
            return _Chunk(choices=[_Choice(delta=_Delta(content=_get(event, "delta") or ""))])

        if kind == "output_item_added":
            item = _get(event, "item") or {}
            if (_get(item, "type") or "") != "function_call":
                return None
            return _Chunk(choices=[_Choice(delta=_Delta(tool_calls=[
                _ToolCallDelta(
                    index=int(_get(event, "output_index") or 0),
                    id=str(_get(item, "call_id") or ""),
                    function=_Fn(name=str(_get(item, "name") or "")),
                )
            ]))])

        if kind == "function_call_arguments_delta":
            return _Chunk(choices=[_Choice(delta=_Delta(tool_calls=[
                _ToolCallDelta(
                    index=int(_get(event, "output_index") or 0),
                    function=_Fn(arguments=str(_get(event, "delta") or "")),
                )
            ]))])

        if kind == "response_completed":
            response = _get(event, "response") or {}
            has_call = any(
                (_get(o, "type") or "") == "function_call"
                for o in (_get(response, "output") or [])
            )
            return _Chunk(
                choices=[_Choice(finish_reason="tool_calls" if has_call else "stop")],
                usage=_get(response, "usage"),
            )

        if kind in ("response_failed", "response_incomplete"):
            response = _get(event, "response") or {}
            log.warning(
                "responses_stream_unfinished",
                kind=kind,
                detail=str(_get(response, "incomplete_details") or "")[:120],
            )
            return _Chunk(choices=[_Choice(finish_reason="length")])

        # created / in_progress / part boundaries / *_done carry nothing new:
        # the deltas above already delivered the content.
        return None


def build_responses_kwargs(chat_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Rewrite an acompletion kwargs dict for ``litellm.aresponses``."""
    out: dict[str, Any] = {
        "model": chat_kwargs["model"],
        "input": to_responses_input(chat_kwargs.get("messages") or []),
        "stream": chat_kwargs.get("stream", True),
    }
    tools = to_responses_tools(chat_kwargs.get("tools"))
    if tools:
        out["tools"] = tools
    # max_tokens is the chat spelling.
    limit = chat_kwargs.get("max_completion_tokens") or chat_kwargs.get("max_tokens")
    if limit:
        out["max_output_tokens"] = limit
    for passthrough in ("reasoning_effort", "extra_headers", "api_key", "api_base"):
        if chat_kwargs.get(passthrough) is not None:
            out[passthrough] = chat_kwargs[passthrough]
    return out

