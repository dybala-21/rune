"""Live /goal: drive the real GoalLoop core (same machinery as
app._do_goal_loop, Phase 5/5.1/5.2 active) to build a Rust webrtc-rs loopback
project from scratch, gated by `cargo build` / `cargo test`. Model = Claude
Sonnet. Bounded run; reports token usage."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

PROJECT = Path("/Users/gmldns46/test-workspace/rust-webrtc-goal")
PROJECT.mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT)
LOG = PROJECT / "DRIVER_LOG.md"

MODEL = "claude-sonnet-4-5-20250929"  # real Sonnet + in rune cost table


def w(s: str) -> None:
    print(s, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(s + "\n")


REQUEST = (
    "Build, from scratch in the current working directory, a Rust Cargo "
    "project that uses the `webrtc` crate (webrtc-rs) to establish a loopback "
    "RTCPeerConnection between two in-process peers: perform SDP offer/answer "
    "locally (no external signaling), open a DataChannel, and send the exact "
    "text 'hello-webrtc' from peer A to peer B. Include a Rust test (tokio "
    "async, #[tokio::test]) that asserts peer B receives exactly that message "
    "within 10 seconds. Create Cargo.toml (package rust_webrtc, edition 2021) "
    "with dependencies on `webrtc` and `tokio` (full features). The project "
    "MUST satisfy: `cargo build` exits 0 AND `cargo test` exits 0."
)


async def main() -> None:
    from rune.agent.goal_loop import GoalIteration, GoalLoop, GoalLoopConfig
    from rune.agent.goal_review import make_adversarial_review_fn
    from rune.agent.goal_runtime import GoalRuntime
    from rune.agent.goal_spec import crystallize_goal
    from rune.agent.goal_validate import make_validate_fn
    from rune.agent.loop import NativeAgentLoop

    LOG.write_text("# /goal live run - Rust webrtc-rs\n\n", encoding="utf-8")
    w(f"started {time.strftime('%Y-%m-%d %H:%M:%S')}  cwd={PROJECT}")

    from rune.config import get_config

    _llm = get_config().llm
    _llm.active_provider = "anthropic"
    _llm.active_model = MODEL
    w(f"pinned agent provider -> anthropic:{MODEL}")

    cr = await crystallize_goal(REQUEST)
    w(f"crystallize: ambiguous={cr.ambiguous} | notice={cr.notice}")
    w(f"  acceptance_criteria={cr.spec.acceptance_criteria}")
    if cr.ambiguous:
        w("ABORTED (ambiguous): " + " | ".join(cr.clarifications))
        return

    cr.spec.validation_commands = ["cargo build", "cargo test"]

    loop = NativeAgentLoop()
    rt = GoalRuntime(loop, channel="cli")
    tokens_by_iter: list[int] = []

    def on_it(it: GoalIteration) -> None:
        tokens_by_iter.append(it.tokens)
        w(
            f"[iter {it.n}] verdict={it.verdict} reason={it.reason or '-'} "
            f"evidence={it.evidence:.2f} tokens={it.tokens} "
            f"validation_passed={it.validation_passed} "
            f"review_passed={it.review_passed}"
        )

    gl = GoalLoop(
        GoalLoopConfig(
            max_iterations=8,
            max_total_tokens=3_000_000,
            stagnation_window=3,
            evidence_threshold=0.8,
            adversarial_review=True,
            ssc_interval=0,
        ),
        run_fn=rt.run_fn,
        validate_fn=make_validate_fn(cwd=str(PROJECT), timeout_s=900.0),
        persist_fn=rt.persist_fn,
        answer_of=rt.answer_of,
        on_iteration=on_it,
        review_fn=make_adversarial_review_fn(),
        artifact_fn=rt.make_artifact_fn(),
        workspace=PROJECT / ".rune" / "goal",
    )

    t0 = time.time()
    try:
        res = await asyncio.wait_for(gl.run(cr.spec), timeout=2400)
        total = sum(tokens_by_iter)
        # rune pricing for the pinned model (input,output per 1K USD)
        try:
            from rune.ui.cost import estimate_cost

            c70 = estimate_cost(MODEL, int(total * 0.7), total - int(total * 0.7))
            c50 = estimate_cost(MODEL, int(total * 0.5), total - int(total * 0.5))
            cost = f"${c70:.2f}-${c50:.2f} (70:30 to 50:50 in:out split assumed)"
        except Exception as exc:  # pragma: no cover
            cost = f"n/a ({exc})"
        w(
            f"\n## RESULT success={res.success} stop_cause={res.stop_cause} "
            f"iterations={len(res.iterations)} secs={time.time() - t0:.0f}"
        )
        w(f"## TOKENS total={total} per_iter={tokens_by_iter}")
        w(f"## COST ({MODEL}) {cost}")
        w("final_answer (head):\n" + (res.final_answer or "")[:1500])
    except TimeoutError:
        w(f"\n## RESULT wall-clock guard hit after {time.time() - t0:.0f}s")
    except Exception as exc:  # surface honestly
        w(f"\n## RESULT driver error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
