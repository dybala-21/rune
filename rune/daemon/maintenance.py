"""Check free disk space, recover spending records, and prune expired data."""

from __future__ import annotations

import asyncio
import shutil
import time

from rune.config.loader import get_config
from rune.utils.logger import get_logger
from rune.utils.paths import rune_data

log = get_logger(__name__)


def check_storage() -> None:
    minimum = get_config().runtime.min_free_disk_mb * 1024 * 1024
    if minimum and shutil.disk_usage(rune_data()).free < minimum:
        raise RuntimeError("Rune has too little free disk space to safely start new work. Free some space and retry.")


async def recover_spend() -> None:
    from rune.agent.spend_budget import SpendLedger
    try:
        await asyncio.to_thread(SpendLedger().recover)
    except Exception:
        # Paid requests still need a successful ledger reservation.
        log.exception("budget_recovery_failed")


class Maintenance:
    def __init__(self, memory_store) -> None:
        self._store = memory_store
        self._last_prune = 0.0
        self.healthy = True

    async def tick(self) -> None:
        await recover_spend()
        try:
            check_storage()
            self.healthy = True
        except (OSError, RuntimeError) as exc:
            self.healthy = False
            log.warning("storage_unavailable", error=str(exc))
        now = time.monotonic()
        if self.healthy and self._last_prune and now - self._last_prune < 6 * 3600:
            return
        try:
            from rune.agent.spend_budget import SpendLedger
            from rune.tools.file import cleanup_trash
            await asyncio.to_thread(cleanup_trash)
            await asyncio.to_thread(SpendLedger().prune)
            if self._store is not None:
                self._store.prune_tool_calls(max_age_days=30)
            self._last_prune = now
            check_storage()
            self.healthy = True
        except Exception:
            self.healthy = False
            log.exception("maintenance_failed")
