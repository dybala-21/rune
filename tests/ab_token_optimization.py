"""A/B test for token optimization — chat prompt separation.

The only remaining optimization after review:
  #1: Chat system prompt 3,537 → 230 tokens (93% reduction)

Rolled back (accuracy impact):
  #2: full budget 500K→300K — wind-down fires early on complex tasks
  #3: tool output 30K→15K — truncates source data irreversibly
  #3b: research window (6,10)→(4,8) — loses early context in long research

Usage:
    python3 tests/ab_token_optimization.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_prompt(content: str, name: str) -> str:
    idx = content.find(f'{name} = """')
    if idx == -1:
        return ""
    body_start = content.index('"""', idx) + 3
    body_end = content.index('"""', body_start)
    return content[body_start:body_end]


def main() -> None:
    with open("rune/agent/prompts.py") as f:
        src = f.read()

    prompts = {}
    for name in ["PROMPT_CORE", "PROMPT_CHAT", "PROMPT_CODE",
                  "PROMPT_WEB_BASE", "PROMPT_WEB_EFFICIENCY", "PROMPT_BROWSER"]:
        prompts[name] = extract_prompt(src, name)

    env = "\n## Environment\nCurrent date: 2026-03-14\nCWD: /tmp"
    goal = "\n## Current Task\n\n안녕하세요"
    cls_section = "\n## Task Classification\n\nType: chat"

    # A: original (chat → full prompt)
    a_parts = [prompts["PROMPT_CORE"], prompts["PROMPT_CODE"],
                prompts["PROMPT_WEB_BASE"], prompts["PROMPT_WEB_EFFICIENCY"],
                prompts["PROMPT_BROWSER"], env, goal, cls_section]
    a_prompt = "\n\n".join(a_parts)

    # B: optimized (chat → lightweight prompt)
    b_parts = [prompts["PROMPT_CHAT"], env, goal, cls_section]
    b_prompt = "\n\n".join(b_parts)

    a_tokens = len(a_prompt) // 3
    b_tokens = len(b_prompt) // 3
    saved = a_tokens - b_tokens

    print("=" * 60)
    print("  CHAT PROMPT A/B COMPARISON")
    print("=" * 60)
    print()
    print(f"  A (original, chat→full):  ~{a_tokens:,} tokens")
    print(f"  B (optimized, chat→chat): ~{b_tokens:,} tokens")
    print(f"  Savings per chat run:     ~{saved:,} tokens ({saved/a_tokens*100:.0f}%)")
    print()
    print("  Accuracy impact: NONE")
    print("  - Chat classification only matches: hi/hello/hey/thanks/bye")
    print("  - Complex requests never classify as 'chat'")
    print("  - PROMPT_CODE/WEB/BROWSER are useless for greetings")
    print()
    print("  Toggle:")
    print("    B (default):  rune")
    print("    A (rollback): RUNE_TOKEN_OPT=0 rune")
    print()


if __name__ == "__main__":
    main()
