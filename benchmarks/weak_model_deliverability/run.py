#!/usr/bin/env python3
"""Weak-model deliverability benchmark.

Runs each task (benchmarks/weak_model_deliverability/tasks.py) through an agent
in an isolated workspace, then scores the produced files with HIDDEN ground-truth
verifiers (the agent's own tests are ignored). It measures the thing that
separates a harness from a bare model wrapper on weak local models: does the
agent produce a VERIFIABLE artifact at all, and how correct is it.

The same runner+verifier scores ANY agent, so head-to-head is reproducible:

  # RUNE (default)
  python benchmarks/weak_model_deliverability/run.py --model qwen2.5-coder:7b

  # Any OpenAI-compatible agent via a command template ({task} and {workspace}
  # are substituted; the command runs with cwd = the isolated workspace):
  python benchmarks/weak_model_deliverability/run.py \
      --agent-cmd 'python /path/to/hermes/run_agent.py --query {task} \
                   --model qwen2.5-coder:7b --base_url http://localhost:11434/v1 \
                   --api_key ollama --max_turns 25'

Honest by construction: scoring is independent of the agent, so a weak agent
that writes lenient tests still fails the hidden checks.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from tasks import TASKS  # noqa: E402

_DEFAULT_RUNE_CMD = "{rune} --message {task} --provider ollama --model {model}"


def _agent_command(args: argparse.Namespace, instruction: str) -> list[str]:
    if args.agent_cmd:
        tmpl = args.agent_cmd
    else:
        # Default: the RUNE CLI from this repo's venv (or PATH).
        rune_bin = args.rune_bin or "rune"
        tmpl = _DEFAULT_RUNE_CMD.format(rune=rune_bin, task="{task}", model=args.model)
    # shlex first, then substitute the task as a single argv element so spaces/
    # quotes in the instruction never break the command.
    parts = shlex.split(tmpl)
    return [instruction if p == "{task}" else p.replace("{model}", args.model) for p in parts]


# Workspaces/homes go under /tmp, NOT the default tempdir. On macOS the default
# is /var/folders/...; agent file_write (which resolves paths) writes there fine
# but the workspace then reads back EMPTY in practice, so a real artifact scores
# as 0. /tmp (-> /private/tmp) is stable for both write and read. Resolved to the
# canonical path so glob matches where file_write lands.
_TMP_BASE = str(Path("/tmp").resolve())


def _make_home() -> str:
    """A fresh isolated, empty RUNE_HOME under /tmp (cold: no accumulated memory)."""
    return tempfile.mkdtemp(prefix="wmd_home_", dir=_TMP_BASE)


def run_task(task: dict, args: argparse.Namespace) -> dict:
    ws = tempfile.mkdtemp(prefix=f"wmd_{task['id']}_", dir=_TMP_BASE)
    home = _make_home()
    try:
        workspace = Path(ws).resolve()
        env = {**os.environ, "RUNE_HOME": home}
        cmd = _agent_command(args, task["instruction"])
        t0 = time.time()
        out = ""
        try:
            proc = subprocess.run(
                cmd,
                cwd=workspace,
                env=env,
                timeout=args.timeout,
                capture_output=True,
                text=True,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        elapsed = time.time() - t0

        produced = sorted(p.name for p in workspace.glob("*.py"))
        checks = task["verify"](workspace)
        score = sum(1 for _, ok, _ in checks if ok)
        ol = out.lower()
        return {
            "task": task["id"],
            "produced_artifact": task["module"] in produced,
            "files": produced,
            "score": score,
            "max": task["max"],
            "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in checks],
            "seconds": round(elapsed, 1),
            # Real tool execution = an "-> tool" action line actually ran. (The
            # guided-decoding system prompt mentions tool names, so a substring
            # match on "file_write" would false-positive — require the action arrow.)
            "rule_injected": "applying" in ol and "learned rule" in ol,
            "tool_called": "-> " in out,
        }
    finally:
        shutil.rmtree(ws, ignore_errors=True)
        shutil.rmtree(home, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Weak-model deliverability benchmark")
    ap.add_argument("--model", default="qwen2.5-coder:7b", help="Model id (weak local by default)")
    ap.add_argument("--rune-bin", default="", help="Path to the rune CLI (default: 'rune' on PATH)")
    ap.add_argument(
        "--agent-cmd", default="", help="Command template for a non-RUNE agent; use {task}"
    )
    ap.add_argument(
        "--trials", type=int, default=1, help="Repeat each task N times (weak models are noisy)"
    )
    ap.add_argument("--task", default="", help="Run only this task id (default: all)")
    ap.add_argument("--timeout", type=int, default=900, help="Per-task timeout (s)")
    ap.add_argument("--json", default="", help="Write full results JSON to this path")
    args = ap.parse_args()

    print(
        f"# weak-model deliverability — model={args.model} trials={args.trials}\n"
    )
    tasks = [t for t in TASKS if not args.task or t["id"] == args.task]
    results = []
    for task in tasks:
        delivered = 0
        scores = []
        for t in range(args.trials):
            r = run_task(task, args)
            r["trial"] = t + 1
            results.append(r)
            delivered += int(r["produced_artifact"])
            scores.append(r["score"])
            art = "✓ artifact" if r["produced_artifact"] else "✗"
            sig = f"rules={'Y' if r['rule_injected'] else 'n'} tool={'Y' if r['tool_called'] else 'n'}"
            print(
                f"## {r['task']} {t + 1}/{args.trials}: {r['score']}/{r['max']} [{art}] {sig} ({r['seconds']}s) {r['files']}"
            )
        inj = sum(1 for x in results if x["task"] == task["id"] and x["rule_injected"])
        print(
            f"   => {task['id']}: artifact {delivered}/{args.trials} | rules-injected {inj}/{args.trials} | scores {scores}\n"
        )

    n = len(results)
    delivered_total = sum(1 for r in results if r["produced_artifact"])
    inj_total = sum(1 for r in results if r["rule_injected"])
    tool_total = sum(1 for r in results if r["tool_called"])
    print(
        f"# OVERALL (n={n}): artifact {delivered_total}/{n} | "
        f"rules-injected {inj_total}/{n} | tool-called {tool_total}/{n}"
    )

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"# wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
