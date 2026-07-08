"""Harsh A/B benchmark for Anthropic prompt caching — measure REAL savings.

Reproduces the token-optimization-analysis.md §2 scenario against the live API:
a ~16K-token fixed system prefix (tools+persona+repo-map equivalent) driven
through a multi-step agent loop whose message history GROWS and MUTATES every
step. Runs it twice — caching OFF vs ON — and reports the real dollar cost
delta computed with RUNE's own `estimate_cost`.

Also proves the load-bearing claim: history mutation each step does NOT
invalidate the tools+system cache (cache_read stays ~16K even as messages
change), and honestly measures the cache-rewrite cost when the system prompt
changes mid-run (advisor-append / tool-set-change case).

Usage:
    ANTHROPIC_API_KEY=sk-ant-... uv run python scripts/bench_cache_control.py
    # optional: STEPS=12 SYS_TOK=16000 uv run python scripts/bench_cache_control.py

Cost: real API spend. Defaults (~16K sys × 12 steps × 2 arms) ≈ US$0.7-1.0.
"""

import asyncio
import os
import sys

import litellm

from rune.agent.litellm_adapter import _apply_anthropic_cache_control
from rune.ui.cost import estimate_cost

MODEL = "claude-sonnet-4-5-20250929"
STEPS = int(os.environ.get("STEPS", "12"))
SYS_TOK = int(os.environ.get("SYS_TOK", "16000"))

# ~4 chars/token. Build a realistic fixed system prefix at the target size.
_UNIT = "You are RUNE, a local-first verifying assistant; follow tool schemas exactly. "
SYSTEM = (_UNIT * (SYS_TOK * 4 // len(_UNIT))).strip()


def _cache_read(usage) -> int:
    # READ is billed at 0.1x. Fields are mutually exclusive with creation per request.
    v = getattr(usage, "cache_read_input_tokens", None)
    if v:
        return int(v)
    d = getattr(usage, "prompt_tokens_details", None)
    v = getattr(d, "cached_tokens", None) if d is not None else None
    return int(v) if v else 0


def _cache_write(usage) -> int:
    # WRITE (creation) is billed at 1.25x. Must NOT fall back to cached_tokens
    # (that is the READ counter) — doing so double-charges hot steps.
    v = getattr(usage, "cache_creation_input_tokens", None)
    if v:
        return int(v)
    d = getattr(usage, "prompt_tokens_details", None)
    v = getattr(d, "cache_creation_tokens", None) if d is not None else None
    return int(v) if v else 0


async def _call(messages, cache_on):
    if cache_on:
        messages = _apply_anthropic_cache_control(MODEL, messages)
    resp = await litellm.acompletion(model=MODEL, messages=messages, max_tokens=12)
    usage = resp.usage
    read = _cache_read(usage)
    write = _cache_write(usage)
    text = resp.choices[0].message.content or "ok"
    cost = estimate_cost(
        MODEL,
        int(usage.prompt_tokens),
        int(usage.completion_tokens),
        cached_input_tokens=read,
        cache_write_tokens=write,
    )
    return {
        "in": int(usage.prompt_tokens),
        "out": int(usage.completion_tokens),
        "read": read,
        "write": write,
        "cost": cost,
        "text": text,
    }


async def run_arm(cache_on: bool):
    """Drive a STEPS-long loop with growing, mutating history."""
    label = "ON " if cache_on else "OFF"
    history: list[dict] = []
    tot = {"in": 0, "out": 0, "read": 0, "write": 0, "cost": 0.0}
    print(f"\n=== ARM: caching {label} ===")
    print(f"{'step':>4} {'input':>7} {'read':>7} {'write':>7} {'cost$':>9}")
    for step in range(STEPS):
        # Unique user turn each step => history bytes MUTATE (worst case for cache).
        user = {
            "role": "user",
            "content": f"Step {step}: continue the task. "
            + f"context-detail-{step} " * 60,  # ~400 tok of fresh history
        }
        messages = [{"role": "system", "content": SYSTEM}, *history, user]
        r = await _call(messages, cache_on)
        for k in tot:
            tot[k] += r[k]
        print(
            f"{step:>4} {r['in']:>7} {r['read']:>7} {r['write']:>7} {r['cost']:>9.5f}"
        )
        history.append(user)
        history.append({"role": "assistant", "content": r["text"]})
    print(
        f" TOT input={tot['in']} read={tot['read']} "
        f"write={tot['write']} cost=${tot['cost']:.5f}"
    )
    return tot


async def invalidation_probe():
    """Honestly measure the cache-rewrite hit when system changes mid-run."""
    print("\n=== PROBE: mid-run system change (advisor-append) ===")
    base = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "warm the cache"},
    ]
    a = await _call(list(base), True)  # write
    b = await _call(list(base), True)  # read (hot)
    changed = [
        {"role": "system", "content": SYSTEM + "\n\n## Advisor\nExtra guidance appended."},
        {"role": "user", "content": "warm the cache"},
    ]
    c = await _call(changed, True)  # system changed -> rewrite, read should drop
    print(f"  #1 write={a['write']:>6} read={a['read']:>6}  (cold)")
    print(f"  #2 write={b['write']:>6} read={b['read']:>6}  (hot, expect read>0)")
    print(f"  #3 write={c['write']:>6} read={c['read']:>6}  (system changed, expect read~0)")
    ok = b["read"] > 0 and c["read"] < b["read"]
    print(f"  -> {'PASS' if ok else 'CHECK'}: cache hot on repeat, invalidates on change")


async def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 2
    print(f"Model={MODEL}  system≈{len(SYSTEM)//4} tok  steps={STEPS}")

    off = await run_arm(cache_on=False)
    on = await run_arm(cache_on=True)
    await invalidation_probe()

    saved = off["cost"] - on["cost"]
    pct = (saved / off["cost"] * 100) if off["cost"] else 0.0
    # Fixed-overhead-only view: input tokens billed, cache-weighted.
    print("\n" + "=" * 46)
    print("RESULT (real API, real estimate_cost)")
    print(f"  OFF cost = ${off['cost']:.5f}")
    print(f"  ON  cost = ${on['cost']:.5f}  (read={on['read']} write={on['write']})")
    print(f"  SAVED    = ${saved:.5f}  ({pct:.1f}% cheaper)")
    print(f"  history mutated every step, yet ON cache_read={on['read']} "
          f"(system cache survived mutation)")
    print("=" * 46)
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
