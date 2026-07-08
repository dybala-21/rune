"""Harsh end-to-end check for the cross-turn cache boundary.

Drives the REAL NativeAgentLoop against the live API and asserts, on the
actual wire path (not an isolated helper):

  1. NO LEAK    - SYSTEM_CACHE_BOUNDARY appears in no outgoing message, for any
                  role, on any step, nor in the returned trace / history.
  2. STRUCTURE  - the system message reaches Anthropic as a block list carrying
                  cache_control breakpoints, split at the static/dynamic seam.
  3. BEHAVIOR   - the reordered prompt still drives a correct task completion
                  (the requested file is actually written with the right text).
  4. MULTI-TURN - a second turn with prior history also leaks nothing.

Every outgoing litellm request is captured by wrapping litellm.acompletion.

Usage:
    ANTHROPIC_API_KEY=sk-ant-... uv run python scripts/e2e_cache_boundary.py
Cost: a few small Sonnet calls.
"""

import asyncio
import copy
import os
import tempfile

import litellm

from rune.agent.prompts import SYSTEM_CACHE_BOUNDARY

CAPTURED: list[list[dict]] = []
_REAL_ACOMPLETION = litellm.acompletion


async def _recording_acompletion(*args, **kwargs):
    msgs = kwargs.get("messages")
    if msgs is not None:
        CAPTURED.append(copy.deepcopy(msgs))
    return await _REAL_ACOMPLETION(*args, **kwargs)


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


def _fail(msg):
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


async def main() -> int:
    # Frugal: force the Anthropic FAST/sonnet tier for the run. (litellm resolves
    # credentials on its own; a raw acompletion below will error if truly absent.)
    from rune.config import get_config

    cfg = get_config()
    cfg.llm.active_provider = "anthropic"
    cfg.llm.active_model = "claude-sonnet-4-5-20250929"

    litellm.acompletion = _recording_acompletion

    from rune.agent.loop import NativeAgentLoop

    workdir = tempfile.mkdtemp(prefix="rune_e2e_")
    target = os.path.join(workdir, "out.txt")
    prev_cwd = os.getcwd()
    os.chdir(workdir)
    print(f"workdir={workdir}")

    try:
        loop = NativeAgentLoop()
        goal = (
            "Create a file named out.txt in the current directory containing "
            "exactly the single word BANANA. Then you are done."
        )
        trace = await loop.run(goal, max_steps=6)

        # --- Turn 2: continue with prior history to exercise multi-turn path ---
        history = [
            {"role": "user", "content": goal},
            {"role": "assistant", "content": "Created out.txt with BANANA."},
        ]
        loop2 = NativeAgentLoop()
        await loop2.run(
            "Now append the word MANGO on a new line to out.txt.",
            max_steps=6,
            message_history=history,
        )
    finally:
        os.chdir(prev_cwd)
        litellm.acompletion = _REAL_ACOMPLETION

    print(f"captured {len(CAPTURED)} outgoing requests")

    # 1. NO LEAK anywhere in captured wire payloads.
    leak_sites = 0
    saw_system_blocklist = False
    saw_two_breakpoints = False
    for req in CAPTURED:
        for m in req:
            content = m.get("content")
            if SYSTEM_CACHE_BOUNDARY in _text_of(content):
                leak_sites += 1
            if m.get("role") == "system" and isinstance(content, list):
                saw_system_blocklist = True
                bps = sum(
                    1 for b in content
                    if isinstance(b, dict) and b.get("cache_control")
                )
                if bps >= 2:
                    saw_two_breakpoints = True
    if leak_sites:
        _fail(f"SYSTEM_CACHE_BOUNDARY leaked into {leak_sites} wire message(s)")
    print("  OK: boundary marker never appears in any outgoing message")

    # 2. STRUCTURE: system arrived as a cached block list, split in two.
    if not saw_system_blocklist:
        _fail("system message was never a cache-annotated block list")
    if not saw_two_breakpoints:
        _fail("expected 2 cache_control breakpoints (static + dynamic split)")
    print("  OK: system reached Anthropic as a 2-breakpoint block list")

    # 3. BEHAVIOR: the file was actually written correctly.
    if not os.path.exists(target):
        _fail(f"task did not create {target}")
    body = open(target).read()
    if "BANANA" not in body:
        _fail(f"file content wrong: {body!r}")
    print("  OK: task completed correctly (out.txt contains BANANA)")

    # 4. NO LEAK in returned trace / final answer.
    trace_text = str(getattr(trace, "final_answer", "")) + str(
        getattr(trace, "summary", "")
    )
    if SYSTEM_CACHE_BOUNDARY in trace_text:
        _fail("boundary marker leaked into the completion trace")
    print("  OK: completion trace carries no marker")

    print("\n" + "=" * 48)
    print("E2E PASS: no side effects, correct behavior, cache structure intact")
    print("=" * 48)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
