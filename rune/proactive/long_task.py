"""Long tasks: work that runs as a series of steps, parked in between.

A long task is a scheduled job with no clock schedule (``@task``). Each step is one bounded
background run (its policy's ``timeout_seconds``, at most 30 minutes). A step ends by parking
(``task_park``: the progress so far, and when to go on) or by finishing the task. While it is
parked nothing runs, so a hosted machine may sleep. The next step starts at ``wake_at``, fired
like a scheduled minute: by the local dispatcher, or by the host, which reads ``wakeAt`` from
``cron.list`` and passes it to ``cron.fire``. Each step picks up from the previous one's progress
note and output (``rune.proactive.routine_context``).

The job's ``max_runs`` bounds the steps. Its policy's ``total_tokens`` and ``total_cost_usd`` bound
the whole task, as for any routine. A machine may cap both (``RUNE_TASK_STEP_LIMIT``,
``RUNE_TASK_TOKEN_LIMIT``), for example per plan.

The owner hears about it when it matters: done, failed, out of steps or budget, waiting for an
approval, or a step the agent parked with ``notify`` (a finding, half way). Quiet steps are seen
in the app.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

SCHEDULE = "@task"
# The longest a step may park before the next one starts.
MAX_PARK_MINUTES = 7 * 24 * 60


def _limit(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def step_limit() -> int:
    """The most steps a task may take on this machine."""
    return _limit("RUNE_TASK_STEP_LIMIT", 50)


def token_limit() -> int:
    """The most tokens a task may use, over all its steps, on this machine."""
    return _limit("RUNE_TASK_TOKEN_LIMIT", 2_000_000)


def is_task(schedule: str) -> bool:
    return schedule.strip() == SCHEDULE


def most_steps(max_runs: int | None) -> int:
    """The steps a task may take: its own limit, within this machine's, which is checked at every
    step so raising ``max_runs`` later can't pass it."""
    return min(max_runs, step_limit()) if max_runs is not None else step_limit()


# A step that ended without parking or finishing goes on after this long.
DEFAULT_RESUME_MINUTES = 5


def resumed(task: dict | None, run_count: int, max_runs: int | None, last_state: str | None,
            last_minute: int | None = None, last_id: str | None = None) -> dict:
    """The state of a task the owner turns on again: its next step due now (after the minute of its
    last one, which already has a record). That last step counts as settled, so it isn't settled
    again. Raises why it can't go on."""
    if last_state == "recovery_required":
        raise ValueError("Its last step stopped part way. Check what it did and reconcile it before resuming.")
    if last_state == "needs_approval":
        raise ValueError("Its step is waiting for your approval; it goes on once you decide.")
    if last_state in {"running", "accepted"}:
        raise ValueError("A step of it is running now; it goes on by itself.")
    if run_count >= most_steps(max_runs):
        raise ValueError(f"It has taken {run_count} steps, the most it may; allow more steps to go on.")
    kept = {k: v for k, v in (task or {}).items() if k in {"step", "notice"}}
    if last_id:
        kept["step"] = last_id
    if task and task.get("wake_at") is not None and not task.get("ended"):
        return {**kept, "wake_at": task["wake_at"]}
    now = int(time.time() // 60)
    return {**kept, "wake_at": max(now, last_minute + 1) if last_minute is not None else now}



def can_run(job, run_count: int, max_runs: int | None) -> bool:
    """Whether a task's next step could start, so it is worth waking the machine for."""
    return (job.status == "active" and not job.policy.expired and run_count < most_steps(max_runs))


# The step being run, when it is a long task's: where `task_park` leaves its request, and how it
# stops the run.
_slot: ContextVar[dict[str, Any] | None] = ContextVar("long_task_park", default=None)


def park_slot(control: Any) -> tuple[dict[str, Any], Any]:
    """For the step run under `control`: only that run may park it, not a subagent it starts."""
    slot: dict[str, Any] = {"control": control}
    return slot, _slot.set(slot)


def release_slot(token: Any) -> None:
    _slot.reset(token)


def current_slot() -> dict[str, Any] | None:
    return _slot.get()


def in_step() -> bool:
    return _slot.get() is not None


@contextmanager
def outside_step() -> Iterator[None]:
    """Run work (a subagent) that must not park the step it is part of."""
    token = _slot.set(None)
    try:
        yield
    finally:
        _slot.reset(token)


def park(progress: str, resume_in_minutes: int, notify: bool, done: bool = False) -> int:
    """Record the running step's park request and stop the step. Returns the minute it resumes.

    `done`: the whole task is done, and `progress` is its final answer."""
    from rune.agent.run_control import current_control

    slot = _slot.get()
    if slot is None:
        raise ValueError("task_park works only inside a long task's step")
    if current_control() is not slot["control"]:
        raise ValueError("Only the step itself parks the task; finish this part and report back instead")
    resume_at = int(time.time() // 60) + max(1, min(resume_in_minutes, MAX_PARK_MINUTES))
    slot.update(progress=progress.strip()[:4000], resume_at=resume_at, notify=bool(notify), done=bool(done))
    slot["control"].stop()
    return resume_at


def parked_result(result: dict[str, Any], slot: dict[str, Any] | None) -> None:
    """A step that asked to park, and stopped cleanly, ends as parked with its progress note."""
    if not slot or "resume_at" not in slot or result.get("execution_unknown") or result.get("pending_approval"):
        return
    if slot.get("done"):
        # The step said the whole task is done: its note is the answer.
        result.update(status="completed", success=True, verified=False, error=None, task_done=True,
                      output=slot["progress"] or result.get("output") or "")
        if "outcome" in result:
            result["outcome"].update(completionStatus="completed", reason="task_done")
        return
    result.update(status="parked", success=True, verified=False, error=None,
                  output=slot["progress"] or result.get("output") or "",
                  park={"resume_at": slot["resume_at"], "notify": slot["notify"]})
    if "outcome" in result:
        result["outcome"].update(completionStatus="incomplete", reason="parked")


def step_brief(step: int, max_runs: int | None, minutes: int) -> str:
    """What every step is told about the task it belongs to, the one continued after an approval too."""
    of = f" of at most {max_runs}" if max_runs else ""
    return (f"This is step {step}{of} of a long task that runs in the background. This step has at most "
            f"{minutes} minutes; stop well before that, since a step cut off by its time limit needs the "
            "owner to check it before the task goes on. Do one bounded chunk of the work toward the goal, "
            "then call task_park with what is done and what the next step should do. Only when the whole "
            "goal is done, call task_park with done=true and the final answer; the owner is told. A step "
            "that ends any other way is taken as unfinished, and the task goes on.")


def after_step(job, result: dict[str, Any], run_count: int, max_runs: int | None, *,
               enabled: bool = True) -> tuple[dict[str, Any], str | None]:
    """What the task becomes after a step, and what to tell the owner (None: nothing).

    Returns the job's new ``task`` state: parked with ``wake_at``, ended with ``ended`` (and the job
    then paused, so nothing fires it again), or ``paused`` when the owner turned it off mid-step.
    """
    status = str(result.get("status") or "")
    park = result.get("park") or {}
    output = str(result.get("output") or "").strip()
    error = str(result.get("error") or "").strip()
    if not enabled and status in {"interrupted", "cancelled"}:
        # The owner turned it off during the step: on hold, not over. Turning it on goes on.
        return {"wake_at": None, "paused": True}, None
    # A park the observed-files check marked unverified is still a park.
    if (park.get("resume_at") and status in {"parked", "unverified"} and not result.get("execution_unknown")
            and not result.get("pending_approval")):
        if run_count >= most_steps(max_runs):
            return ({"wake_at": None, "ended": "step_limit"},
                    f"Stopped after {run_count} steps, the most it may take. Where it got to: {output}")
        return {"wake_at": int(park["resume_at"])}, (output if park.get("notify") else None)
    if status == "needs_approval":
        # The approval continues the step; until then nothing else fires it.
        return {"wake_at": None}, "Waiting for your approval to go on."
    if result.get("task_done") and status in {"completed", "verified", "unverified"}:
        return {"wake_at": None, "ended": "done"}, output or "Done."
    if status in {"completed", "verified", "unverified"} and not result.get("execution_unknown"):
        # It ended without saying the task is done: unfinished, so the next step goes on from it.
        if run_count >= most_steps(max_runs):
            return ({"wake_at": None, "ended": "step_limit"},
                    f"Stopped after {run_count} steps, the most it may take. Where it got to: {output}")
        return {"wake_at": int(time.time() // 60) + DEFAULT_RESUME_MINUTES}, None
    if status == "budget_exhausted":
        return {"wake_at": None, "ended": status}, f"Stopped: it used up its budget. {error}".strip()
    if status == "interrupted":
        return ({"wake_at": None, "ended": status},
                "Stopped part way through a step (it ran out of time, changed while it ran, or the machine "
                "stopped). Check what it did in the app and mark it checked, then turn it on again to go on "
                "from there.")
    return {"wake_at": None, "ended": status or "failed"}, f"Stopped ({status or 'failed'}): {error or output}".strip()
