"""Live smoke for the quality_gate LLM judge.

Five scenarios hit a real LiteLLM client to verify the judge actually
rejects hollow-success answers (multilingual), accepts clean ones, and
survives concurrent + oversized inputs.

Run:
  set -a && source ~/.rune/.env && set +a
  uv run python scripts/_pr1_live_smoke.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class Outcome:
    name: str
    ok: bool
    detail: str
    duration_ms: float


async def _run_one(
    name: str,
    task,
    result,
    expect_failure: bool,
    *,
    llm_client: Any,
) -> Outcome:
    from rune.agent.quality_gate import check_task_quality

    t0 = time.monotonic()
    qc = await check_task_quality(task, result, llm_client=llm_client)
    dt = (time.monotonic() - t0) * 1000

    if expect_failure and qc.passed:
        return Outcome(name, False, f"expected fail; got pass score={qc.score:.2f}", dt)
    if not expect_failure and not qc.passed:
        return Outcome(
            name,
            False,
            f"expected pass; got fail score={qc.score:.2f} issues={qc.issues}",
            dt,
        )
    return Outcome(
        name,
        True,
        f"score={qc.score:.2f} issues={len(qc.issues)} suggestion={'Y' if qc.suggestion else 'N'}",
        dt,
    )


async def main() -> int:
    from rune.agent.quality_gate import AgentResult, TaskInfo, check_task_quality
    from rune.llm.client import get_llm_client

    client = get_llm_client()

    # S1: Korean hollow-success — agent claims complete but text says failure.
    s1_task = TaskInfo(id="s1", role="researcher", goal="search file")
    s1_result = AgentResult(
        success=True,
        answer=(
            "작업을 완료했습니다. 하지만 파일을 찾지 못했고, API 접근에 실패했으며, "
            "결과적으로 요청한 작업을 제대로 수행할 수 없었습니다. "
            "추가 조치가 필요합니다."
        ),
        iterations=4,
        duration_ms=8000.0,
    )

    # S2: Mixed Korean+English — error masking detection
    s2_task = TaskInfo(id="s2", role="researcher", goal="research")
    s2_result = AgentResult(
        success=True,
        answer=(
            "Task complete. 다만 unable to access the database 했고, "
            "the search returned no results. 결국 the operation could not be "
            "completed as expected. Further investigation is required."
        ),
        iterations=5,
        duration_ms=6000.0,
    )

    # S3: Clean genuine success — judge must NOT false-positive
    s3_task = TaskInfo(id="s3", role="researcher", goal="research")
    s3_result = AgentResult(
        success=True,
        answer=(
            "Research complete. Reviewed three primary sources at "
            "https://example.com/a , https://example.com/b , https://example.com/c "
            "and synthesized a structured summary. The implementation uses pytest "
            "with asyncio_mode=auto, FAISS HNSW for vector retrieval, and APSW "
            "for SQLite WAL mode. All findings are documented with concrete "
            "file paths and line numbers."
        ),
        iterations=6,
        duration_ms=9000.0,
    )

    # S4: concurrent — 4 simultaneous gate calls
    s4_pairs = [
        (
            TaskInfo(id=f"s4-{i}", role="researcher", goal="x"),
            AgentResult(
                success=True,
                answer=(
                    f"Iteration {i}. Reviewed sources at "
                    f"https://example.com/{i}-a, https://example.com/{i}-b, "
                    f"https://example.com/{i}-c. All findings consistent."
                )
                * 2,
                iterations=5,
                duration_ms=5000.0,
            ),
        )
        for i in range(4)
    ]

    # S5: oversized answer (5000 chars) — truncation path with real LLM
    big_answer = "Analysis: " + ("filler text without any failure indication. " * 200)
    s5_task = TaskInfo(id="s5", role="researcher", goal="big")
    s5_result = AgentResult(
        success=True,
        answer=big_answer,
        iterations=6,
        duration_ms=7000.0,
    )

    outcomes: list[Outcome] = []

    outcomes.append(
        await _run_one("S1 ko-hollow-success", s1_task, s1_result, True, llm_client=client)
    )
    outcomes.append(
        await _run_one("S2 mixed-error-masking", s2_task, s2_result, True, llm_client=client)
    )
    outcomes.append(
        await _run_one("S3 clean-success-no-fp", s3_task, s3_result, False, llm_client=client)
    )

    # S4 concurrent
    t0 = time.monotonic()
    s4_results = await asyncio.gather(
        *[check_task_quality(t, r, llm_client=client) for t, r in s4_pairs]
    )
    s4_dt = (time.monotonic() - t0) * 1000
    s4_ok = all(qc.passed for qc in s4_results)
    outcomes.append(
        Outcome(
            "S4 concurrent-x4",
            s4_ok,
            f"all_pass={s4_ok} per_call_avg_ms={s4_dt / len(s4_pairs):.0f}",
            s4_dt,
        )
    )

    outcomes.append(
        await _run_one("S5 oversized-5000ch", s5_task, s5_result, False, llm_client=client)
    )

    print("\n--- PR-1 deep live smoke ---")
    width = max(len(o.name) for o in outcomes) + 2
    for o in outcomes:
        flag = "✓" if o.ok else "✗"
        print(f"{flag}  {o.name:<{width}}  {o.duration_ms:>6.0f}ms  {o.detail}")

    failed = [o for o in outcomes if not o.ok]
    print()
    if failed:
        print(f"FAILED {len(failed)}/{len(outcomes)}")
        return 1
    print(f"PASSED {len(outcomes)}/{len(outcomes)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
