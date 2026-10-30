"""Read scheduled results without importing approval state."""

from pathlib import Path


def scheduled_tasks(owner: str, workspace: str, *, task_id: str = "", status: str = "",
                    limit: int = 20, offset: int = 0) -> list[dict]:
    from rune.capabilities.cron import _get_store, _row_to_cronjob
    from rune.proactive.execution_store import ExecutionStore
    from rune.utils.paths import rune_data, user_workspace

    jobs = [_row_to_cronjob(row) for row in _get_store().list_cron_jobs()]
    jobs = [job for job in jobs if job.owner == owner
            and str(Path(job.policy.workspace or str(user_workspace())).resolve()) == workspace
            and (not task_id or job.id == task_id)]
    jobs.sort(key=lambda job: (job.last_run_at, job.id), reverse=True)
    path = rune_data() / "routine-executions.db"
    store = ExecutionStore(path) if path.exists() else None
    try:
        tasks = []
        for job in jobs:
            previous = store.latest_result(f"cron:{job.id}:") if store else None
            current = bool(previous and previous.get("routine_goal") == job.goal
                           and previous.get("routine_command") == job.command
                           and previous.get("workspace") == job.policy.workspace)
            last_status = (previous or {}).get("status", "not_run")
            task_status = job.status
            if status and status not in {task_status, last_status}:
                continue
            tasks.append({"id": job.id, "goal": job.goal or job.command, "status": task_status,
                          "category": "scheduled", "routine_id": job.id,
                          "checkpoint": {"requests": [job.goal or job.command],
                                         "last_turn": {"status": last_status, "matches_current_goal": current,
                                                       "output": str((previous or {}).get("output", ""))[:2000],
                                                       "error": str((previous or {}).get("error") or "")[:800]},
                                         "files": job.policy.output_paths}})
            if len(tasks) >= offset + limit:
                break
        return tasks[offset:offset + limit]
    finally:
        if store is not None:
            store.close()
