"""Live-verify Anthropic prompt caching (doc step-1 acceptance gate).

Runs the ACTUAL `_apply_anthropic_cache_control` from the adapter, sends the
same large system prompt twice, and prints the cache token counters.

PASS criterion (token-optimization-analysis.md §2):
    call #2 reports cache_read_input_tokens > 0

Usage:
    ANTHROPIC_API_KEY=sk-ant-... uv run python scripts/verify_cache_control.py

Cost: two small Sonnet calls (~a cent). No writes, no side effects.
"""

import asyncio
import os
import sys

import litellm

from rune.agent.litellm_adapter import _apply_anthropic_cache_control

# Sonnet 4.5 min cache length = 1,024 tokens. Pad the system prompt well past it.
MODEL = "claude-sonnet-4-5-20250929"
BIG_SYSTEM = (
    "You are RUNE, a local-first assistant. "
    + ("Follow every instruction precisely and verify results. " * 250)
)


def _cache_read(usage) -> int:
    for attr in ("cache_read_input_tokens", "cache_read_tokens"):
        v = getattr(usage, attr, None)
        if v:
            return int(v)
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        v = getattr(details, "cached_tokens", None)
        if v:
            return int(v)
    return 0


def _cache_write(usage) -> int:
    for attr in ("cache_creation_input_tokens", "cache_creation_tokens"):
        v = getattr(usage, attr, None)
        if v:
            return int(v)
    return 0


async def _one_call(label: str):
    messages = [
        {"role": "system", "content": BIG_SYSTEM},
        {"role": "user", "content": "Reply with the single word: ok"},
    ]
    messages = _apply_anthropic_cache_control(MODEL, messages)
    # Confirm the breakpoint was actually attached before spending money.
    sys_content = messages[0]["content"]
    assert isinstance(sys_content, list) and sys_content[-1].get("cache_control"), (
        "cache_control was not applied — adapter regression"
    )
    resp = await litellm.acompletion(model=MODEL, messages=messages, max_tokens=8)
    usage = resp.usage
    read, write = _cache_read(usage), _cache_write(usage)
    print(
        f"  [{label}] input={usage.prompt_tokens} "
        f"cache_write={write} cache_read={read}"
    )
    return read, write


async def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 2
    print(f"Model: {MODEL}  (system ~{len(BIG_SYSTEM)//4} tok)")
    print("Call #1 (expect cache WRITE, read=0):")
    _, write1 = await _one_call("1")
    print("Call #2 (expect cache READ > 0):")
    read2, _ = await _one_call("2")

    print("\n--- VERDICT ---")
    if read2 > 0:
        print(f"PASS ✅  call #2 cache_read_input_tokens = {read2} (> 0)")
        return 0
    print(f"FAIL ❌  call #2 cache_read = {read2}. write#1 was {write1}.")
    print("If write#1 was 0 too, LiteLLM did not pass cache_control through — "
          "check litellm version / block-list format.")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
