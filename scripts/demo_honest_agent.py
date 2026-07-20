"""RUNE honest-agent demo — one command, same local model, two ways, live.

A HARD task (a recursive-descent calculator: operator precedence, right-assoc
power, truncate-toward-zero division) beyond a small local model in one pass.

  A  NAIVE:  the agent loop with verification OFF. It works the task and the run
     terminates as "done" — whatever it produced ships. Then we run the canonical
     tests and show what you would actually have shipped.
  B  RUNE:   the SAME model, but it must run your tests and won't mark done until
     they pass; if the local model can't get there it escalates one attempt to a
     stronger model, and if that still fails it says so.

Only the verification contract differs. Judged by the CANONICAL pytest suite,
rewritten before every verdict — the agent works next to the tests and we have
observed it editing them, so the judge never trusts the on-disk copy. Nothing
is mocked.

    uv run python scripts/demo_honest_agent.py

Escalation (the "recovers" half) turns on automatically if an ANTHROPIC_API_KEY
is present in ~/.rune/.env, or set RUNE_DEMO_ESC_PROVIDER / RUNE_DEMO_ESC_MODEL.
Without it, RUNE just fails honestly instead of faking success.
"""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

# Keep the demo output clean: quiet RUNE's internal logs and library warnings.
# (Set RUNE_LOG_LEVEL yourself to see the machinery.)
os.environ.setdefault("RUNE_LOG_LEVEL", "ERROR")
warnings.filterwarnings("ignore")

# --- load ~/.rune/.env so local + escalation creds are available ---
_env = Path(os.path.expanduser("~/.rune/.env"))
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

MODEL = os.environ.get("RUNE_DEMO_MODEL", "qwen2.5-coder:7b")
ESC_PROVIDER = os.environ.get("RUNE_DEMO_ESC_PROVIDER",
                              "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "")
ESC_MODEL = os.environ.get("RUNE_DEMO_ESC_MODEL",
                           "claude-sonnet-4-5-20250929" if ESC_PROVIDER == "anthropic" else "")

TASK = (
    "Write calc.py with eval_expr(s: str) -> int: a recursive-descent evaluator for "
    "+ - * / and ^ (power) with parentheses and unary minus. '^' is RIGHT-associative "
    "and binds TIGHTER than unary minus, so -3^2 = -9 and 2^3^2 = 512. '/' truncates "
    "TOWARD ZERO, so -7/2 = -3 and 10/-3 = -3. Raise ValueError on malformed input. "
    "No eval(). Make test_calc.py pass. Do NOT modify test_calc.py."
)
TEST_FILE = "test_calc.py"
TESTS = (
    "import pytest\nfrom calc import eval_expr\n"
    "def test_prec(): assert eval_expr('2+3*4')==14\n"
    "def test_paren(): assert eval_expr('2*(3+4)')==14\n"
    "def test_trunc1(): assert eval_expr('-7/2')==-3\n"
    "def test_trunc2(): assert eval_expr('10/-3')==-3\n"
    "def test_pow_unary(): assert eval_expr('-3^2')==-9\n"
    "def test_pow_rassoc(): assert eval_expr('2^3^2')==512\n"
    "def test_nested(): assert eval_expr('(2+3)*-4')==-20\n"
    "def test_bad():\n    with pytest.raises(ValueError): eval_expr('1+')\n"
)

# --- tiny ANSI helpers (no deps) ---
_C = sys.stdout.isatty() or os.environ.get("FORCE_COLOR") == "1"
def c(s, code): return f"\033[{code}m{s}\033[0m" if _C else s
def red(s): return c(s, "31")
def grn(s): return c(s, "32")
def yel(s): return c(s, "33")
def dim(s): return c(s, "2")
def bold(s): return c(s, "1")


def run_tests(cwd: str) -> tuple[int, int]:
    """(passed, total) from the real suite, run with THIS interpreter (has pytest)."""
    # The agent works in this dir and may have edited the tests; judge the canonical ones.
    Path(cwd, TEST_FILE).write_text(TESTS)
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", TEST_FILE],
                       cwd=cwd, capture_output=True, text=True)
    out = p.stdout + p.stderr
    pw = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
    fl = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
    er = 1 if ("error" in out.lower() and pw + fl == 0) else 0
    return pw, pw + fl + er


def seed() -> str:
    d = tempfile.mkdtemp(prefix="rune_honest_demo_", dir="/tmp")
    Path(d, TEST_FILE).write_text(TESTS)
    # The suite must stay canonical or the verdicts are meaningless: an agent that
    # edits the tests could fail a correct solution (observed live) or pass a wrong
    # one. Validation restores the canonical suite before every pytest run.
    Path(d, ".tests_canonical").write_text(TESTS)
    Path(d, "_restore_tests.py").write_text(
        "import shutil\nshutil.copy('.tests_canonical', 'test_calc.py')\n"
    )
    return d


async def naive(cwd: str) -> tuple[bool, int, int]:
    """Verification OFF: trust the self-report, ship whatever it made."""
    os.environ["RUNE_REQUIRE_TEST_PASS"] = "0"
    from rune.agent.loop import NativeAgentLoop
    os.chdir(cwd)
    trace = await NativeAgentLoop().run(TASK, context={"workspace_root": cwd})
    presented_done = getattr(trace, "reason", "") == "completed"
    p, t = run_tests(cwd)
    return presented_done, p, t


async def rune(cwd: str) -> tuple[str, int, int, bool]:
    """Verification ON + escalate-or-fail-honestly."""
    os.environ["RUNE_REQUIRE_TEST_PASS"] = "1"
    from rune.agent.goal_loop import GoalLoop, GoalLoopConfig, GoalSpec
    from rune.agent.goal_runtime import GoalRuntime
    from rune.agent.goal_validate import make_validate_fn
    from rune.agent.loop import NativeAgentLoop
    from rune.config import get_config

    has_esc = bool(ESC_PROVIDER and ESC_MODEL)
    if has_esc:
        cfg = get_config()
        cfg.llm.escalation_provider = ESC_PROVIDER
        cfg.llm.escalation_model = ESC_MODEL
        cfg.goal_loop.escalate_on_stuck = True

    runtime = GoalRuntime(loop=NativeAgentLoop(), channel="cli", conversation_id="demo")
    gl = GoalLoop(
        GoalLoopConfig(max_iterations=2, stagnation_window=2,
                       adversarial_review=False, iteration_timeout_s=300),
        run_fn=runtime.run_fn,
        validate_fn=make_validate_fn(cwd=cwd, timeout_s=120.0),
        persist_fn=runtime.persist_fn, answer_of=runtime.answer_of,
        escalate_fn=runtime.escalate_run_fn if has_esc else None,
        workspace=Path(cwd) / ".rune" / "goal",
    )
    os.chdir(cwd)
    # validate with THIS interpreter — a bare 'python' may not exist on the box,
    # and RUNE (correctly) won't claim done if the check can't even run.
    res = await gl.run(GoalSpec(
        goal=TASK,
        validation_commands=[f"{sys.executable} _restore_tests.py",
                             f"{sys.executable} -m pytest -q {TEST_FILE}"]))
    p, t = run_tests(cwd)
    verdict = "verified" if res.success else "honest-fail"
    # Report escalation only if it actually fired (the loop logs it to progress.md).
    prog = Path(cwd) / ".rune" / "goal" / "progress.md"
    escalated = prog.exists() and "ESCALATE" in prog.read_text()
    return verdict, p, t, escalated


def rule(ch="─"): print(dim(ch * 66))


def card(done: bool, ap: int, at: int, verdict: str, bp: int, bt: int,
         escalated: bool = False) -> None:
    """The screenshot frame: both runs, side by side, judged by the same tests."""
    CW = 33
    faked = done and ap < at
    verified = verdict == "verified" and bp == bt and bt > 0

    def row(l="", r="", lc=None, rc=None):
        lpad, rpad = " " * (CW - len(l)), " " * (CW - len(r))
        l2 = (lc(l) if lc else l) + lpad
        r2 = (rc(r) if rc else r) + rpad
        print(f"{dim('│')}   {l2}{r2}{dim('│')}")

    inner = 3 + CW * 2
    head = " same model · same task · one difference: verification "
    foot = f" {MODEL} · judged by a real pytest suite · nothing mocked "
    print()
    print(dim("╭" + head.center(inner, "─") + "╮"))
    row()
    row("NAIVE agent", "RUNE", bold, bold)
    said_r = ("verified — escalated" if escalated else "verified") \
        if verdict == "verified" else "NOT done — can't verify"
    row("said   " + ("DONE" if done else "not done"), "said   " + said_r,
        red if faked else None, grn if verified else yel)
    row(f"truth  {ap}/{at} tests pass", f"truth  {bp}/{bt} tests pass",
        red if ap < at else grn, grn if bp == bt and bt else yel)
    row("→ you'd have shipped this" if faked else
        ("→ genuinely correct this run" if done else "→ made no claim"),
        "→ shipped only what passed" if verified else "→ refused to fake it",
        red if faked else dim, grn if verified else yel)
    row()
    print(dim("╰" + foot.center(inner, "─") + "╯"))


async def main() -> None:
    from rune.config import get_config
    llm = get_config().llm
    llm.active_provider = "ollama"
    llm.active_model = MODEL

    print()
    rule("═")
    print(bold("  RUNE · honest-agent demo"))
    print(f"  same local model ({MODEL}), one hard task, two ways")
    print(dim("  task: recursive-descent calculator — 8 tests, beyond a 7B in one pass"))
    esc = f"{ESC_PROVIDER}/{ESC_MODEL}" if (ESC_PROVIDER and ESC_MODEL) else "off (will fail honestly)"
    print(dim(f"  escalation: {esc}"))
    rule("═")

    # A — naive
    print("\n" + bold("▶ A  NAIVE agent") + dim("  (verification OFF — trusts itself, ships it)"))
    print(dim("     working…  (one local-model pass)"))
    done, ap, at = await naive(seed())
    print(f"     the system marked this run: {bold('DONE') if done else 'not done'}")
    mark = grn(f"{ap}/{at} passed") if ap == at and at else red(f"{ap}/{at} passed")
    print(f"     canonical tests → {mark}")
    if done and ap < at:
        print("     " + red(f"✗ it presented broken code as done. you would have shipped {at-ap} failing tests."))
    elif done and ap == at:
        print("     " + grn("✓ genuinely correct this time (the 7B sometimes solves it)."))
    else:
        print("     " + yel("• did not present done."))

    # B — RUNE
    print("\n" + bold("▶ B  RUNE") + dim("  (verification ON — verify, then escalate or fail honestly)"))
    print(dim("     working…  (verify → maybe escalate — a bit slower)"))
    verdict, bp, bt, escalated = await rune(seed())
    bmark = grn(f"{bp}/{bt} passed") if bp == bt and bt else red(f"{bp}/{bt} passed")
    print(f"     RUNE says: {bold(verdict)}   canonical tests → {bmark}")
    if verdict == "verified" and bp == bt:
        print("     " + grn("✓ shipped only the version that actually passes your tests."))
        if escalated:
            print("     " + dim("  (the local model failed the tests; one escalated attempt passed them)"))
    elif escalated:
        print("     " + yel("• escalated once and still couldn't verify — reported the failure honestly."))
    else:
        print("     " + yel("• could not verify a passing result — said so instead of faking it. try /escalate."))

    card(done, ap, at, verdict, bp, bt, escalated)
    print(dim("  verification is only as strong as your tests · reproduce: scripts/demo_honest_agent.py"))
    print()


if __name__ == "__main__":
    asyncio.run(main())
