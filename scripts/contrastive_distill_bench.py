"""Contrastive winner-vs-loser distillation bench.

Failure-driven rule extraction keeps learning the shallowest cause of a
failure ("check the files exist"), never the underlying convention. This
bench asks two things: does contrasting the test-verified WINNER against
the failing attempts recover the convention, and does the extracted rule
transfer to a held-out task with a different surface?

Phases (resumable; state lands in CD_OUT as JSON):
  collect   K single-shot runs per family on the TRAIN task (convention lives
            ONLY in the provided tests) -> winners + losers with evidence.
  extract   arm A = product failure-only rule; arm B = contrastive rule.
  transfer  held-out task, convention omitted from the prompt, grading tests
            written AFTER the run; R reps per arm in {none, A, B}.

Config via env: CD_MODEL (default qwen2.5-coder:7b), CD_PROVIDER (ollama),
CD_K (default 6), CD_REPS (default 4), CD_OUT (default /tmp/cd_bench),
CD_FAMILIES (comma list), CD_PHASE (collect|extract|transfer|all).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL = os.environ.get("CD_MODEL", "qwen2.5-coder:7b")
PROVIDER = os.environ.get("CD_PROVIDER", "ollama")
K = int(os.environ.get("CD_K", "6"))
REPS = int(os.environ.get("CD_REPS", "4"))
OUT = Path(os.environ.get("CD_OUT", "/tmp/cd_bench"))

# Conventions are chosen so the Python-NATURAL implementation is WRONG: the
# generator can only satisfy the tests by noticing the convention in them,
# and can only pass the held-out task if the injected rule carries it.
FAMILIES: dict[str, dict] = {
    "trunc_div": {
        "train_task": (
            "Implement eval_rpn(tokens: list[str]) -> int in solution.py, "
            "evaluating reverse-polish notation with + - * /. "
            "The provided tests in test_solution.py define the exact expected "
            "behaviour; make them pass. Write only solution.py."
        ),
        "train_tests": (
            "from solution import eval_rpn\n"
            "def test_add(): assert eval_rpn(['2','3','+'])==5\n"
            "def test_mul(): assert eval_rpn(['4','5','*'])==20\n"
            "def test_div(): assert eval_rpn(['9','3','/'])==3\n"
            "def test_div_neg(): assert eval_rpn(['-7','2','/'])==-3\n"
            "def test_div_neg2(): assert eval_rpn(['10','-3','/'])==-3\n"
        ),
        "heldout_task": (
            "Implement split_budget(total: int, parts: int) -> int in "
            "solution.py: the integer amount each part receives when an "
            "integer budget is divided into equal integer parts. Budgets and "
            "parts may be negative. Write only solution.py."
        ),
        "heldout_tests": (
            "from solution import split_budget\n"
            "def test_even(): assert split_budget(10,2)==5\n"
            "def test_neg(): assert split_budget(-7,2)==-3\n"
            "def test_neg2(): assert split_budget(7,-2)==-3\n"
        ),
    },
    # Keep the surface trivial: on complex surfaces (parsers) attempts crash
    # structurally long before the convention matters, so no convention
    # evidence is ever produced. round() is banker's rounding in Python, which
    # makes away-from-zero anti-natural even on a one-line function.
    "round_half": {
        "train_task": (
            "Implement round_units(x: float) -> int in solution.py. "
            "The provided tests in test_solution.py define the exact expected "
            "behaviour; make them pass. Write only solution.py."
        ),
        "train_tests": (
            "from solution import round_units\n"
            "def test_down(): assert round_units(2.4)==2\n"
            "def test_up(): assert round_units(2.6)==3\n"
            "def test_half(): assert round_units(2.5)==3\n"
            "def test_half2(): assert round_units(3.5)==4\n"
            "def test_half_neg(): assert round_units(-2.5)==-3\n"
        ),
        "heldout_task": (
            "Implement star_rating(avg: float) -> int in solution.py: convert "
            "a product's average review score (may be negative for adjusted "
            "scores) to the nearest whole number of stars. Write only "
            "solution.py."
        ),
        "heldout_tests": (
            "from solution import star_rating\n"
            "def test_low(): assert star_rating(3.2)==3\n"
            "def test_half(): assert star_rating(3.5)==4\n"
            "def test_half_even(): assert star_rating(4.5)==5\n"
            "def test_half_neg(): assert star_rating(-0.5)==-1\n"
        ),
    },
    "median_lower": {
        "train_task": (
            "Implement median_score(scores: list[int]) -> int in solution.py. "
            "The provided tests in test_solution.py define the exact expected "
            "behaviour; make them pass. Write only solution.py."
        ),
        "train_tests": (
            "from solution import median_score\n"
            "def test_odd(): assert median_score([5,1,3])==3\n"
            "def test_single(): assert median_score([7])==7\n"
            "def test_even(): assert median_score([1,2,3,4])==2\n"
            "def test_even2(): assert median_score([10,20])==10\n"
        ),
        "heldout_task": (
            "Implement p50_latency(samples: list[int]) -> int in solution.py: "
            "the median latency of a list of request timings in ms. Write "
            "only solution.py."
        ),
        "heldout_tests": (
            "from solution import p50_latency\n"
            "def test_odd(): assert p50_latency([30,10,20])==20\n"
            "def test_even(): assert p50_latency([100,200,300,400])==200\n"
            "def test_even2(): assert p50_latency([50,150])==50\n"
        ),
    },
    "ceil_div": {
        "train_task": (
            "Implement minutes_billed(seconds: int) -> int in solution.py. "
            "The provided tests in test_solution.py define the exact expected "
            "behaviour; make them pass. Write only solution.py."
        ),
        "train_tests": (
            "from solution import minutes_billed\n"
            "def test_zero(): assert minutes_billed(0)==0\n"
            "def test_exact(): assert minutes_billed(120)==2\n"
            "def test_partial(): assert minutes_billed(61)==2\n"
            "def test_short(): assert minutes_billed(1)==1\n"
        ),
        "heldout_task": (
            "Implement boxes_needed(items: int, per_box: int) -> int in "
            "solution.py: how many boxes are required to pack all the items. "
            "Write only solution.py."
        ),
        "heldout_tests": (
            "from solution import boxes_needed\n"
            "def test_exact(): assert boxes_needed(9,3)==3\n"
            "def test_partial(): assert boxes_needed(10,3)==4\n"
            "def test_small(): assert boxes_needed(1,5)==1\n"
            "def test_zero(): assert boxes_needed(0,5)==0\n"
        ),
    },
    # The right-assoc version of this family (2^3^2==512, -3^2==-9) is a
    # dead end: it coincides with Python's ** semantics, so capable models
    # pass with no rule and weak ones only fail structurally. Flipping the
    # convention to left-assoc keeps it anti-natural.
    "pow_lassoc": {
        "train_task": (
            "Implement eval_expr(expr: str) -> int in solution.py for "
            "arithmetic expressions with + - * / ^ parentheses and unary "
            "minus. The provided tests in test_solution.py define the exact "
            "expected behaviour; make them pass. Write only solution.py."
        ),
        "train_tests": (
            "from solution import eval_expr\n"
            "def test_prec(): assert eval_expr('2+3*4')==14\n"
            "def test_paren(): assert eval_expr('2*(3+4)')==14\n"
            "def test_pow(): assert eval_expr('2^3')==8\n"
            "def test_pow_lassoc(): assert eval_expr('2^3^2')==64\n"
            "def test_pow_unary(): assert eval_expr('-3^2')==9\n"
        ),
        "heldout_task": (
            "Implement eval_cell(formula: str, cells: dict[str, int]) -> int "
            "in solution.py: a spreadsheet cell formula evaluator. Formulas "
            "use cell names (A1, B2), integer literals, + - * ^ and unary "
            "minus, e.g. eval_cell('A1+B2*2', {'A1':1,'B2':3}) == 7. "
            "Write only solution.py."
        ),
        "heldout_tests": (
            "from solution import eval_cell\n"
            "def test_basic(): assert eval_cell('A1+B2*2',{'A1':1,'B2':3})==7\n"
            "def test_lassoc(): assert eval_cell('C1^C2^C3',{'C1':2,'C2':3,'C3':2})==64\n"
            "def test_unary(): assert eval_cell('-A1^2',{'A1':3})==9\n"
        ),
    },
}


def _run_tests(cwd: str) -> tuple[int, int, str]:
    """(passed, total, concise failure evidence) for test_solution.py."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=line", "test_solution.py"],
        cwd=cwd, capture_output=True, text=True, timeout=60,
    )
    out = proc.stdout + proc.stderr
    p = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
    f = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
    if p + f == 0:  # collection error / no solution — structural failure
        return 0, 1, "structural: " + out.strip().splitlines()[-1][:200] if out.strip() else "no output"
    ev_lines = [ln.strip() for ln in out.splitlines()
                if ln.strip().startswith("/") and "assert" in ln][:4]
    return p, p + f, "; ".join(e[-160:] for e in ev_lines)


async def _agent_run(task: str, cwd: str, memory: str | None) -> None:
    """One single-shot agent run — no verify gate, no retries (organic)."""
    os.environ["RUNE_REQUIRE_TEST_PASS"] = "0"
    from rune.agent.loop import NativeAgentLoop

    ctx: dict = {"workspace_root": cwd}
    if memory:
        ctx["memory_context"] = memory  # string shape — reaches the model (prompts.py:711)
    prev = os.getcwd()
    os.chdir(cwd)
    try:
        await NativeAgentLoop().run(task, context=ctx)
    finally:
        os.chdir(prev)


# ---------------------------------------------------------------- collect ---

async def _one_attempt(fam: str, k: int, winner_model: str | None = None) -> dict:
    spec = FAMILIES[fam]
    d = tempfile.mkdtemp(prefix=f"cd_{fam}_{k}_", dir="/tmp")
    Path(d, "test_solution.py").write_text(spec["train_tests"])
    await _agent_run(spec["train_task"], d, memory=None)
    sol = Path(d, "solution.py")
    code = sol.read_text()[:4000] if sol.exists() else ""
    p, t, ev = _run_tests(d)
    rec = {"attempt": k, "passed": p == t and t > 0,
           "score": f"{p}/{t}", "evidence": ev, "code": code}
    if winner_model:
        rec["winner_model"] = winner_model
    print(f"  {fam} attempt {k + 1}: {p}/{t} "
          f"{'WIN' if rec['passed'] else ''}{f' [{winner_model}]' if winner_model else ''}")
    return rec


async def collect(fam: str) -> dict:
    pool = [await _one_attempt(fam, k) for k in range(K)]
    return {"family": fam, "pool": pool}


async def supplement_winner(fam: str, entry: dict) -> dict:
    """Escalation-shaped winner: when the weak generator never passes, let the
    configured stronger local model produce the verified winner (RUNE's real
    ladder: weak fails honestly -> escalate). Marked in the pool for honest
    reporting — the contrast is then cross-model."""
    winner_model = os.environ.get("CD_WINNER_MODEL", "")
    if not winner_model or any(a["passed"] for a in entry["pool"]):
        return entry
    from rune.config import get_config

    llm = get_config().llm
    prev_m, prev_p = llm.active_model, llm.active_provider
    llm.active_model = winner_model
    llm.active_provider = os.environ.get("CD_WINNER_PROVIDER", prev_p)
    try:
        for _ in range(int(os.environ.get("CD_WINNER_K", "3"))):
            rec = await _one_attempt(fam, len(entry["pool"]), winner_model=winner_model)
            entry["pool"].append(rec)
            if rec["passed"]:
                break
    finally:
        llm.active_model, llm.active_provider = prev_m, prev_p
    return entry


# ---------------------------------------------------------------- extract ---

_CONTRAST_PROMPT = (
    "Several attempts at the same coding task were graded by the task's own "
    "tests. One PASSED all tests; the others FAILED.\n\n"
    "PASSING solution:\n```python\n{winner}\n```\n\n"
    "{losers}\n"
    "Compare the passing solution with the failing ones. Identify the SPECIFIC "
    "behavioral convention (an exact semantic choice, e.g. a rounding "
    "direction, an operator associativity, an ordering) that the passing "
    "solution implements and at least one failing attempt got wrong.\n"
    "Write ONE rule (under 40 words) that:\n"
    "1. states the convention GENERALLY — do NOT mention this task, its "
    "function names, or its input format (it must apply to any future task "
    "touching the same semantics);\n"
    "2. names the natural-but-WRONG implementation choice the failing "
    "attempts made;\n"
    "3. names the correct implementation;\n"
    "4. includes ONE minimal concrete example: an input, the wrong result "
    "the natural choice gives, and the correct result.\n"
    "If the failures are only structural (missing file, syntax, unsupported "
    "input) and no semantic convention separates them, reply NONE.\n"
    "Format: key_name: rule description\nRule:"
)


async def _llm(prompt: str) -> str | None:
    from rune.llm.client import get_llm_client
    from rune.types import ModelTier

    resp = await get_llm_client().completion(
        messages=[{"role": "user", "content": prompt}],
        tier=ModelTier.FAST, max_tokens=600,
    )
    text = ""
    if isinstance(resp, dict):
        ch = resp.get("choices", [])
        text = ch[0].get("message", {}).get("content", "") if ch else ""
    else:
        try:
            text = resp.choices[0].message.content
        except (AttributeError, IndexError):
            pass
    text = (text or "").strip().lstrip("- •").strip()
    return None if not text or text.upper().startswith("NONE") else text


async def extract(fam: str, pool: list[dict]) -> dict:
    winners = [a for a in pool if a["passed"]]
    losers = [a for a in pool if not a["passed"] and a["evidence"]]
    if not winners or not losers:
        return {"family": fam, "contrast": False,
                "note": f"no contrast (winners={len(winners)} losers={len(losers)})"}

    # Arm A — the product's existing failure-only learning path.
    from rune.memory.rule_learner import generate_rule_from_failure

    seen: set[str] = set()
    rules_a: list[str] = []
    for lo in losers:
        if lo["evidence"] in seen:
            continue
        seen.add(lo["evidence"])
        r = await generate_rule_from_failure("collect_attempt", lo["evidence"], "code_modify")
        if r:
            rules_a.append(r)
        if len(rules_a) >= 3:  # mirrors best_of._MAX_FAILURE_RULES
            break

    # Arm B — contrastive winner-vs-loser (same extractor tier as A).
    loser_blocks = "".join(
        f"FAILING attempt {i + 1} (failures: {lo['evidence'][:300]}):\n"
        f"```python\n{lo['code'][:2500]}\n```\n\n"
        for i, lo in enumerate(losers[:2])
    )
    rule_b = await _llm(_CONTRAST_PROMPT.format(winner=winners[0]["code"][:2500],
                                                losers=loser_blocks))
    return {"family": fam, "contrast": True,
            "winner_model": winners[0].get("winner_model", MODEL),
            "rules_a": rules_a, "rule_b": rule_b}


# --------------------------------------------------------------- transfer ---

async def transfer(fam: str, rules: dict) -> dict:
    # Transfer target is its own axis (RUNE supports local AND cloud models);
    # headroom is model-dependent, so the none arm doubles as the validity
    # check — none≈pass means this family has no headroom for this model.
    t_model = os.environ.get("CD_TRANSFER_MODEL", MODEL)
    t_provider = os.environ.get("CD_TRANSFER_PROVIDER", PROVIDER)
    from rune.config import get_config

    llm = get_config().llm
    prev_m, prev_p = llm.active_model, llm.active_provider
    llm.active_model, llm.active_provider = t_model, t_provider

    spec = FAMILIES[fam]
    arms: dict[str, str | None] = {"none": None}
    if rules.get("contrast"):
        if rules.get("rules_a"):
            arms["A"] = "## Learned Rules\n" + "\n".join(f"- {r}" for r in rules["rules_a"])
        if rules.get("rule_b"):
            arms["B"] = f"## Learned Rules\n- {rules['rule_b']}"
    tally: dict[str, list[bool]] = {}
    for arm, memory in arms.items():
        tally[arm] = []
        for r in range(REPS):
            d = tempfile.mkdtemp(prefix=f"cd_t_{fam}_{arm}_{r}_", dir="/tmp")
            # Held-out: agent NEVER sees the grading tests (written after).
            await _agent_run(spec["heldout_task"], d, memory=memory)
            Path(d, "test_solution.py").write_text(spec["heldout_tests"])
            p, t, _ = _run_tests(d)
            ok = p == t and t > 0
            tally[arm].append(ok)
            print(f"  {fam} transfer[{t_model}] {arm} #{r + 1}/{REPS}: "
                  f"{'pass' if ok else 'FAIL'} ({p}/{t})")
    llm.active_model, llm.active_provider = prev_m, prev_p
    return {"family": fam, "transfer_model": t_model,
            "pass_rate": {arm: f"{sum(v)}/{len(v)}" for arm, v in tally.items()}}


# ------------------------------------------------------------------- main ---

def _load(name: str) -> dict:
    f = OUT / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _save(name: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=2))


async def main() -> int:
    from rune.config import get_config

    llm = get_config().llm
    llm.active_provider = PROVIDER
    llm.active_model = MODEL

    fams = [f for f in os.environ.get("CD_FAMILIES", "").split(",") if f] or list(FAMILIES)
    phase = os.environ.get("CD_PHASE", "all")
    print(f"Contrastive distill bench — model={MODEL} K={K} reps={REPS} "
          f"families={fams} phase={phase} out={OUT}")

    if phase in ("collect", "all"):
        pools = _load("pools")
        for fam in fams:
            if fam not in pools:
                pools[fam] = await collect(fam)
                _save("pools", pools)
            else:
                print(f"  {fam}: pool cached ({len(pools[fam]['pool'])} attempts)")
            pools[fam] = await supplement_winner(fam, pools[fam])
            _save("pools", pools)

    if phase in ("extract", "all"):
        # The extractor model is its own experimental axis; hold it constant
        # across arms A and B within a run.
        extractor = os.environ.get("CD_EXTRACTOR_MODEL", MODEL)
        llm.active_model = extractor
        pools, rules = _load("pools"), _load("rules")
        for fam in fams:
            if fam in rules:
                continue
            rules[fam] = await extract(fam, pools[fam]["pool"])
            rules[fam]["extractor"] = extractor
            _save("rules", rules)
        llm.active_model = MODEL
        for fam in fams:
            r = rules[fam]
            print(f"\n  {fam}: contrast={r.get('contrast')}")
            for a in r.get("rules_a", []):
                print(f"    A(failure-only): {a}")
            print(f"    B(contrastive):  {r.get('rule_b') or r.get('note')}")

    if phase in ("transfer", "all"):
        rules, results = _load("rules"), _load("results")
        for fam in fams:
            if fam in results:
                continue
            results[fam] = await transfer(fam, rules[fam])
            _save("results", results)
        print("\n" + "=" * 56)
        print(f"TRANSFER (held-out, non-duplicate; n={REPS}/arm)")
        for fam in fams:
            print(f"  {fam}: {results[fam]['pass_rate']}")
        print("  expectation: A tracks none; hypothesis: B>none where contrast existed.")
        print("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
