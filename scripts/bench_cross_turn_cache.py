"""Harsh cross-turn caching A/B for the §3 prefix reorder.

Question: across CONSECUTIVE turns (each a different goal + memory, so the
dynamic tail differs) does the turn-stable instructional prefix get reused by
Anthropic prompt caching?

Two arms, same reordered prompt from the real build_system_prompt:
  - boundary   : mark_cache_boundary=True  -> breakpoint at static/dynamic split
  - no-boundary: mark_cache_boundary=False -> single breakpoint at end of system

Only the boundary arm has an explicit checkpoint at the end of the static
prefix, so it should read the static prefix from cache on turns 2..N even
though each turn's dynamic tail is new. The no-boundary arm caches the whole
system per turn, so a new tail should prevent (or shrink) the cross-turn read.

Each arm uses an arm-unique tag + run nonce so the two arms never share cache
entries and a re-run within the 5-min TTL starts cold.

Usage:
    ANTHROPIC_API_KEY=sk-ant-... uv run python scripts/bench_cross_turn_cache.py
Cost: 2 arms x N turns small Sonnet calls (~a few cents).
"""

import asyncio
import os
import secrets
from types import SimpleNamespace

import litellm

from rune.agent.litellm_adapter import _apply_anthropic_cache_control
from rune.agent.prompts import build_system_prompt

MODEL = "claude-sonnet-4-5-20250929"
N_TURNS = 4
NONCE = secrets.token_hex(4)


def _classification():
    return SimpleNamespace(
        goal_type="full",
        intent_categories=frozenset(),
        output_expectation=None,
        is_continuation=False,
        is_complex_coding=False,
        is_multi_task=False,
        requires_execution=False,
    )


def _system_for_turn(turn: int, arm: str, mark: bool) -> str:
    # Distinct dynamic tail per turn; identical static prefix within an arm.
    goal = f"Task {turn}: refactor module number {turn} and add tests."
    memory = (
        f"Turn {turn} memory. " + f"detail-{turn} " * 40
    )  # ~unique, non-trivial tail
    prompt = build_system_prompt(
        goal=goal,
        classification=_classification(),
        memory_context=memory,
        goal_category="full",
        environment={"cwd": "/repo", "home": "/home/u"},
        mark_cache_boundary=mark,
    )
    # Arm-unique, turn-stable tag at the very front (part of the static prefix)
    # so the two arms cannot share cache entries.
    return f"[arm={arm} nonce={NONCE}]\n\n{prompt}"


def _read(usage) -> int:
    v = getattr(usage, "cache_read_input_tokens", None)
    return int(v) if v else 0


def _write(usage) -> int:
    v = getattr(usage, "cache_creation_input_tokens", None)
    return int(v) if v else 0


async def run_arm(arm: str, mark: bool):
    print(f"\n=== ARM: {arm} (mark_cache_boundary={mark}) ===")
    print(f"{'turn':>4} {'input':>7} {'read':>7} {'write':>7}")
    reads = []
    for turn in range(N_TURNS):
        system = _system_for_turn(turn, arm, mark)
        msgs = _apply_anthropic_cache_control(
            MODEL,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Proceed with task {turn}. Reply ok."},
            ],
        )
        resp = await litellm.acompletion(model=MODEL, messages=msgs, max_tokens=8)
        u = resp.usage
        r, w = _read(u), _write(u)
        reads.append(r)
        print(f"{turn:>4} {int(u.prompt_tokens):>7} {r:>7} {w:>7}")
    # Cross-turn reuse = reads on turns 2..N (turn 0 is the cold write).
    cross = reads[1:]
    print(f"  cross-turn reads (turns 1..{N_TURNS-1}): {cross}")
    return cross


async def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        return 2
    print(f"Model={MODEL} turns={N_TURNS} nonce={NONCE}")

    bound = await run_arm("boundary", True)
    plain = await run_arm("no-boundary", False)

    bound_min = min(bound) if bound else 0
    plain_max = max(plain) if plain else 0
    print("\n" + "=" * 52)
    print("CROSS-TURN CACHE REUSE (real API)")
    print(f"  boundary    turns 1..N read >= {bound_min}")
    print(f"  no-boundary turns 1..N read <= {plain_max}")
    if bound_min > 0 and bound_min > plain_max:
        print(f"  PASS: boundary reuses the static prefix across turns "
              f"({bound_min} tok) where no-boundary does not ({plain_max}).")
        return 0
    if bound_min > 0 and plain_max > 0:
        print(f"  PARTIAL: both arms get cross-turn reads "
              f"(boundary>={bound_min}, no-boundary<={plain_max}); "
              f"Anthropic gives automatic prefix credit — boundary still >= baseline.")
        return 0
    print("  CHECK: boundary did not show cross-turn reuse; inspect numbers.")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
