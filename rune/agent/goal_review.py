"""Review helpers for the ``/goal`` loop.

Two LLM-backed factories that plug into :class:`~rune.agent.goal_loop.GoalLoop`:

* ``make_adversarial_review_fn``: an allow/block gate on a candidate that
  already passed the inner gate and deterministic validation. Returns block
  on error.
* ``make_ssc_critique_fn``: every N iterations the model critiques whether its
  own output exploited a spec loophole and proposes a tighter criterion. This
  is advisory, recorded to progress.md for a human to adopt.

Same conventions as ``goal_spec.py``: lazy shared LLM client, robust JSON
parse, conservative fallback. The client is injectable so this can be
unit-tested offline.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from rune.agent.goal_loop import GoalSpec, ReviewContext
from rune.agent.goal_spec import LLMLike, _extract_text
from rune.utils.logger import get_logger

log = get_logger(__name__)


def _spec_block(spec: GoalSpec) -> str:
    ac = "\n".join(f"- {c}" for c in spec.acceptance_criteria) or "- (none)"
    con = "\n".join(f"- {c}" for c in spec.constraints) or "- (none)"
    return f"GOAL: {spec.goal}\nACCEPTANCE CRITERIA:\n{ac}\nCONSTRAINTS:\n{con}"


_REVIEW_SYSTEM = """\
You are an adversarial reviewer judging whether a candidate genuinely meets the \
spec. You are given the SPEC, DETERMINISTIC EVIDENCE (real exit codes and \
output of the spec's own validation commands, executed by the loop; this is \
ground truth you can trust), and the agent's CLAIM (prose; do NOT trust it, \
use it only for context).

Decide on the EVIDENCE, not the claim:
- If the deterministic evidence shows the spec's validation commands all \
exited 0 and the acceptance criteria are objectively satisfied by that \
evidence, respond allow=true. Absence of prose is NOT a reason to block when \
the evidence already proves the commands passed.
- Block (allow=false) only if the evidence shows failures, the criteria are \
not actually covered, or the work is gamed.

You are ALSO given the CHANGED SOURCE. A green exit code is necessary but NOT \
sufficient; inspect the source and BLOCK even at exit 0 when the tests do not \
genuinely prove the acceptance criteria: empty/no-assertion tests, t.Skip / \
skipped or always-true assertions, expected==expected or hard-coded to the \
test, swallowed/ignored errors, masked failures, or criteria met only \
technically-but-not-meaningfully. If the changed source shows the criteria are \
genuinely exercised and asserted AND the evidence shows commands passed, \
allow=true.

The CHANGED SOURCE may begin with a SOURCE MANIFEST listing the project files. \
Each is tagged [changed] (modified this iteration) or [baseline] (already \
present, unchanged this iteration) and [shown] or [omitted: cap]. Judge the \
actual source you can see: do NOT block solely because a file is [baseline] \
or unchanged this iteration - stable, already-correct code that genuinely \
meets the criteria is acceptable. Only lean allow=false when a file clearly \
relevant to the acceptance criteria is "omitted: cap" or truncated so its \
correctness cannot be verified from what is shown.

Respond with ONLY a JSON object:
{"allow": true|false, "reason": "<one sentence grounded in evidence/source>"}"""

_SSC_SYSTEM = """\
You perform Specification Self-Correction. Inspect whether the work below \
exploited a loophole or ambiguity in the spec rather than fulfilling its \
intent. Respond with ONLY a JSON object:
{"gamed": true|false, "critique": "<short>", "spec_patch": "<a tighter \
acceptance criterion to close the loophole, or empty>"}"""


async def _ask(llm: LLMLike, system: str, user: str, tier: str) -> dict[str, Any]:
    from rune.utils.fast_serde import json_decode

    resp = await llm.completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tier=tier,  # type: ignore[call-arg]
        # Some models return empty visible text at small caps; too small a cap
        # would make the review always empty and so always block.
        max_tokens=2048,
        timeout=30.0,
    )
    text = _extract_text(resp).strip()
    if not text:
        raise ValueError("empty LLM response")
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    data = json_decode(text)
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    return data


def make_adversarial_review_fn(
    *, llm: LLMLike | None = None, tier: str = "best"
) -> Callable[[ReviewContext], Awaitable[tuple[bool, str]]]:
    async def _review(rc: ReviewContext) -> tuple[bool, str]:
        try:
            client = llm
            if client is None:
                from rune.llm.client import get_llm_client

                client = get_llm_client()
            artifact = rc.artifact.strip() or "(no changed source captured)"
            data = await _ask(
                client,
                _REVIEW_SYSTEM,
                f"{_spec_block(rc.spec)}\n\n"
                f"DETERMINISTIC EVIDENCE (ground truth):\n"
                f"{rc.validation_output[:6000]}\n\n"
                f"CHANGED SOURCE - may start with a MANIFEST (judge whether "
                f"tests genuinely assert the criteria):\n{artifact[:20000]}\n\n"
                f"AGENT CLAIM (untrusted context only):\n{rc.claim[:2000]}",
                tier,
            )
            allow = bool(data.get("allow", False))
            reason = str(data.get("reason") or "").strip()
            return allow, reason or ("approved" if allow else "blocked by reviewer")
        except Exception as exc:  # on error return block (GoalLoop blocks)
            log.debug("adversarial_review_fallback", error=str(exc)[:200])
            return False, f"reviewer unavailable ({type(exc).__name__})"

    return _review


def make_ssc_critique_fn(
    *, llm: LLMLike | None = None, tier: str = "fast"
) -> Callable[[GoalSpec, str, int], Awaitable[str]]:
    async def _critique(spec: GoalSpec, answer: str, iteration: int) -> str:
        try:
            client = llm
            if client is None:
                from rune.llm.client import get_llm_client

                client = get_llm_client()
            data = await _ask(
                client,
                _SSC_SYSTEM,
                f"{_spec_block(spec)}\n\nWORK SO FAR:\n{answer[:6000]}",
                tier,
            )
            if not bool(data.get("gamed", False)):
                return ""
            crit = str(data.get("critique") or "").strip()
            patch = str(data.get("spec_patch") or "").strip()
            return (crit + (f" | proposed: {patch}" if patch else "")).strip()
        except Exception as exc:  # advisory only; ignore failures
            log.debug("ssc_critique_skipped", error=str(exc)[:200])
            return ""

    return _critique
