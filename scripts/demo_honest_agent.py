"""The honest-agent demo: the same hard task, run two ways.

A: an UNVERIFIED agent (one local-model pass, trust its self-report) — the
   common design. On a task beyond the model it ships broken code and reports
   success anyway ("fabricated success", the documented #1 agent failure mode).
B: RUNE's VERIFIED path — run the project's real tests, refuse to finish on a
   failing check, and (when an escalation profile is configured) escalate one
   attempt to a stronger model. It ships only verified work, or says it could
   not, instead of claiming a success it cannot back up.

Both modes use the same local model, so the difference shown is verification,
not capability. Run:

    RUNE_GUIDED_TOOLS=1 \
    RUNE_DEMO_ESC_PROVIDER=anthropic RUNE_DEMO_ESC_MODEL=claude-sonnet-4-5-20250929 \
    uv run python scripts/demo_honest_agent.py

Set RUNE_DEMO_ESC_* (and that provider's key) to show the escalation payoff;
omit them to show RUNE failing honestly with no stronger model on hand. Mode B
sends one attempt to the cloud only when an escalation profile is set.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL = os.environ.get("RUNE_DEMO_MODEL", "qwen2.5-coder:7b")
ESC_PROVIDER = os.environ.get("RUNE_DEMO_ESC_PROVIDER", "")
ESC_MODEL = os.environ.get("RUNE_DEMO_ESC_MODEL", "")

TASK = (
    "Implement an RPN (reverse polish notation) calculator in rpn.py: "
    "eval_rpn(tokens: list[str]) -> int evaluates a postfix expression with "
    "operators + - * / over integers, using integer division truncating toward "
    "zero, and raises ValueError on malformed input. Make test_rpn.py pass."
)
TESTS = """\
import pytest
from rpn import eval_rpn
def test_add(): assert eval_rpn(['2','3','+'])==5
def test_chain(): assert eval_rpn(['4','13','5','/','+'])==6
def test_mul_sub(): assert eval_rpn(['5','1','2','+','4','*','+','3','-'])==14
def test_trunc_toward_zero(): assert eval_rpn(['-7','2','/'])==-3
def test_malformed_raises():
    with pytest.raises(ValueError): eval_rpn(['1','+'])
"""


def _run_tests(cwd: str) -> tuple[int, int]:
    """Return (passed, total) by running the real test suite."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_rpn.py"],
        cwd=cwd, capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    import re
    p = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
    f = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
    e = 1 if "error" in out.lower() and p + f == 0 else 0
    return p, p + f + e


def _seed(cwd: str) -> None:
    Path(cwd, "test_rpn.py").write_text(TESTS)


async def mode_a_unverified(cwd: str) -> tuple[str, int, int]:
    """One local pass, trust the self-report. Returns (claim, passed, total)."""
    from rune.agent.loop import NativeAgentLoop
    loop = NativeAgentLoop()
    collected: list[str] = []
    loop.on("text_delta", lambda d: collected.append(d))
    await loop.run(TASK, context={"workspace_root": cwd})
    claim = "".join(collected).strip()[-300:] or "(no final message)"
    passed, total = _run_tests(cwd)
    return claim, passed, total


async def mode_b_verified(cwd: str) -> tuple[str, int, int]:
    """RUNE's verified loop (+ escalation if configured). Returns (verdict, p, t)."""
    from rune.agent.goal_loop import GoalLoop, GoalLoopConfig, GoalSpec
    from rune.agent.goal_runtime import GoalRuntime
    from rune.agent.goal_validate import make_validate_fn
    from rune.agent.loop import NativeAgentLoop
    from rune.config import get_config

    cfg = get_config()
    has_esc = bool(ESC_PROVIDER and ESC_MODEL)
    if has_esc:
        cfg.llm.escalation_provider = ESC_PROVIDER
        cfg.llm.escalation_model = ESC_MODEL
        cfg.goal_loop.escalate_on_stuck = True

    runtime = GoalRuntime(loop=NativeAgentLoop(), channel="cli", conversation_id="demo")
    gl = GoalLoop(
        GoalLoopConfig(max_iterations=2, stagnation_window=2,
                       adversarial_review=False, iteration_timeout_s=300),
        run_fn=runtime.run_fn,
        validate_fn=make_validate_fn(cwd=cwd, timeout_s=120.0),
        persist_fn=runtime.persist_fn,
        answer_of=runtime.answer_of,
        escalate_fn=runtime.escalate_run_fn if has_esc else None,
        workspace=Path(cwd) / ".rune" / "goal",
    )
    res = await gl.run(GoalSpec(goal=TASK, validation_commands=["python -m pytest -q test_rpn.py"]))
    passed, total = _run_tests(cwd)
    verdict = "verified" if res.success else f"not verified ({res.stop_cause})"
    return verdict, passed, total


async def main() -> None:
    # Pin the local executor model for every tier (otherwise failover resolves a
    # default tier model that may not be installed).
    from rune.config import get_config
    _llm = get_config().llm
    _llm.active_provider = "ollama"
    _llm.active_model = MODEL

    base = tempfile.mkdtemp(prefix="honest_demo_", dir="/tmp")
    a_dir, b_dir = Path(base, "a"), Path(base, "b")
    a_dir.mkdir(parents=True)
    b_dir.mkdir(parents=True)
    _seed(str(a_dir))
    _seed(str(b_dir))

    print("=" * 64)
    print(f"HONEST-AGENT DEMO  (local model: {MODEL})")
    print("task: RPN calculator (beyond a small local model in one pass)")
    print("=" * 64)

    os.chdir(a_dir)
    claim, ap, at = await mode_a_unverified(str(a_dir))
    print("\n[A] UNVERIFIED agent — trusts its own self-report, ships it")
    print(f"    agent says: \"{claim.splitlines()[-1] if claim.splitlines() else claim}\"")
    print(f"    actual tests: {ap}/{at} passed"
          + ("   <-- claimed done, but it is WRONG (fabricated success)"
             if ap < at else ""))

    os.chdir(b_dir)
    verdict, bp, bt = await mode_b_verified(str(b_dir))
    esc = "with escalation" if (ESC_PROVIDER and ESC_MODEL) else "local only, no escalation profile"
    print(f"\n[B] RUNE — verifies against the real tests ({esc})")
    print(f"    RUNE says: \"{verdict}\"")
    print(f"    actual tests: {bp}/{bt} passed"
          + ("   <-- verified correct" if verdict == "verified" and bp == bt
             else "   <-- did NOT claim success it cannot back up (honest fail)"))

    print("\n" + "=" * 64)
    print("CONTRAST: the unverified agent reported success on broken code;")
    print("RUNE shipped only what its tests confirmed, or said it could not.")
    print("Trust is bounded by the tests: verification is only as strong as them.")
    print("=" * 64)


if __name__ == "__main__":
    asyncio.run(main())
