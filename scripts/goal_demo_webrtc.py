"""Live /goal demo: drive the real GoalLoop core (same machinery as
app._do_goal_loop) to build a Go pion/webrtc loopback project from scratch,
gated by `go build` / `go test`. Bounded run."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

PROJECT = Path("/Users/gmldns46/test-workspace/go-webrtc-goal")
PROJECT.mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT)
LOG = PROJECT / "DRIVER_LOG.md"


def w(s: str) -> None:
    print(s, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(s + "\n")


REQUEST = (
    "Build, from scratch in the current working directory, a Go module that "
    "uses github.com/pion/webrtc/v4 to establish a loopback WebRTC "
    "PeerConnection between two in-process peers: perform SDP offer/answer and "
    "exchange ICE candidates locally (no external signaling server), open a "
    "DataChannel, and send the exact text 'hello-webrtc' from peer A to peer "
    "B. Include a Go test that asserts peer B receives exactly that message "
    "within 10 seconds. Initialize go.mod (module example.com/gowebrtc, go "
    "1.26), run `go mod tidy` to fetch pion/webrtc. The project MUST satisfy: "
    "`go build ./...` exits 0 AND `go test ./...` exits 0."
)


async def main() -> None:
    from rune.agent.goal_loop import GoalIteration, GoalLoop, GoalLoopConfig
    from rune.agent.goal_review import make_adversarial_review_fn
    from rune.agent.goal_runtime import GoalRuntime
    from rune.agent.goal_spec import crystallize_goal
    from rune.agent.goal_validate import make_validate_fn
    from rune.agent.loop import NativeAgentLoop

    LOG.write_text("# /goal live run - Go pion/webrtc\n\n", encoding="utf-8")
    w(f"started {time.strftime('%Y-%m-%d %H:%M:%S')}  cwd={PROJECT}")

    # This env's rune config has active_provider=gemini with an invalid Vertex
    # key (the agent's priority-0 failover profile). OpenAI/Anthropic keys are
    # valid. Pin the agent to Anthropic IN-PROCESS only (does not write the
    # user's config file). Pure environment fix; /goal code is unchanged.
    from rune.config import get_config

    # Probed in this env: anthropic id 'claude-sonnet-4.5' -> NotFound;
    # openai gpt-5.4 works (same key that made crystallize succeed).
    _llm = get_config().llm
    _llm.active_provider = "openai"
    _llm.active_model = "gpt-5.4"
    w(f"pinned agent provider -> {_llm.active_provider}:{_llm.active_model}")

    cr = await crystallize_goal(REQUEST)
    w(f"crystallize: ambiguous={cr.ambiguous} | notice={cr.notice}")
    w(f"  acceptance_criteria={cr.spec.acceptance_criteria}")
    w(f"  validation(from LLM)={cr.spec.validation_commands}")
    if cr.ambiguous:
        w("ABORTED (ambiguous): " + " | ".join(cr.clarifications))
        return

    # Pin the objective gate regardless of crystallizer phrasing.
    cr.spec.validation_commands = ["go build ./...", "go test ./..."]

    loop = NativeAgentLoop()
    rt = GoalRuntime(loop, channel="cli")

    def on_it(it: GoalIteration) -> None:
        w(
            f"[iter {it.n}] verdict={it.verdict} reason={it.reason or '-'} "
            f"evidence={it.evidence:.2f} tokens={it.tokens} "
            f"validation_passed={it.validation_passed} "
            f"review_passed={it.review_passed}"
        )

    gl = GoalLoop(
        GoalLoopConfig(
            max_iterations=6,
            max_total_tokens=1_500_000,
            stagnation_window=3,
            evidence_threshold=0.8,
            adversarial_review=True,
            ssc_interval=0,
        ),
        run_fn=rt.run_fn,
        validate_fn=make_validate_fn(cwd=str(PROJECT), timeout_s=240.0),
        persist_fn=rt.persist_fn,
        answer_of=rt.answer_of,
        on_iteration=on_it,
        review_fn=make_adversarial_review_fn(),
        workspace=PROJECT / ".rune" / "goal",
    )

    t0 = time.time()
    try:
        res = await asyncio.wait_for(gl.run(cr.spec), timeout=1500)
        w(
            f"\n## RESULT success={res.success} stop_cause={res.stop_cause} "
            f"iterations={len(res.iterations)} secs={time.time() - t0:.0f}"
        )
        w("final_answer (head):\n" + (res.final_answer or "")[:1500])
    except TimeoutError:
        w(f"\n## RESULT wall-clock guard hit after {time.time() - t0:.0f}s")
    except Exception as exc:  # surface honestly
        w(f"\n## RESULT driver error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
