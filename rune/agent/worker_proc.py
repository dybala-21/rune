"""Run one subtask in its own process inside an isolated worktree.

A separate process is needed because cwd is process-global, so coroutine workers
can't each have their own. The parent sets RUNE_ISOLATION_ROOT and RUNE_WORKER.

  python -m rune.agent.worker_proc --spec <json> --result <json>
  spec:   {goal, root, provider?, model?, max_iterations?}
  result: {ok, answer, iterations, actions, trace_reason, error}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys


async def _run(spec: dict) -> dict:
    root = spec["root"]
    os.environ["RUNE_ISOLATION_ROOT"] = root
    os.environ["RUNE_WORKER"] = "1"
    os.chdir(root)

    from rune.agent.loop import NativeAgentLoop
    from rune.types import AgentConfig

    cfg = AgentConfig(_overridden=True)
    if spec.get("provider"):
        cfg.provider = spec["provider"]
    if spec.get("model"):
        cfg.model = spec["model"]
    if spec.get("max_iterations"):
        cfg.max_iterations = int(spec["max_iterations"])

    loop = NativeAgentLoop(config=cfg)
    loop._auto_skill = False

    async def _approve(_cmd: str, _reason: str) -> bool:
        return True  # non-interactive; Guardian + isolation still apply
    try:
        loop.set_approval_callback(_approve)
    except Exception:
        pass

    actions = {"n": 0}

    async def _count(_info: dict) -> None:
        actions["n"] += 1
    loop.on("tool_call", _count)

    trace = await loop.run(spec["goal"], context=spec.get("context"))
    answer = (getattr(loop, "_last_answer_text", "") or "").strip()
    return {
        "ok": True,
        "answer": answer,
        "iterations": int(getattr(trace, "final_step", 0) or 0),
        "actions": actions["n"],
        "trace_reason": getattr(trace, "reason", ""),
        "error": "",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="path to spec JSON")
    ap.add_argument("--result", required=True, help="path to write result JSON")
    args = ap.parse_args(argv)

    try:
        with open(args.spec, encoding="utf-8") as fh:
            spec = json.load(fh)
    except Exception as exc:
        _write(args.result, {"ok": False, "error": f"bad spec: {exc}",
                             "answer": "", "iterations": 0, "actions": 0,
                             "trace_reason": "spec_error"})
        return 1

    try:
        result = asyncio.run(_run(spec))
    except Exception as exc:
        result = {"ok": False, "error": str(exc)[:500], "answer": "",
                  "iterations": 0, "actions": 0, "trace_reason": "worker_error"}
    _write(args.result, result)
    return 0 if result.get("ok") else 1


def _write(path: str, data: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
