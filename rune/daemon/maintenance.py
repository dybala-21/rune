"""Check free disk space, recover spending records, prune expired data, and clear away what an
earlier version left behind."""

from __future__ import annotations

import asyncio
import json
import shutil
import time

from rune.config.loader import get_config
from rune.utils.logger import get_logger
from rune.utils.paths import rune_data, rune_home

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


def remove_retired_extension() -> bool:
    """Delete the Browser Bridge extension an earlier version copied into RUNE's own folder, and
    say whether one was deleted.

    That version had its owner load the folder in Chrome. A loaded copy keeps asking this
    machine's ports for a relay and attaches Chrome's debugger for whichever program answers.
    RUNE no longer runs one. With the folder gone Chrome does not start the extension again; a
    copy that is running goes on until Chrome is next started, which only its owner can cut short.
    """
    folder = rune_home() / "extension" / "rune-browser-bridge"
    try:
        # Only the folder RUNE wrote: whatever else is kept under that name is left alone.
        if json.loads((folder / "manifest.json").read_text(encoding="utf-8")).get("name") != "RUNE Browser Bridge":
            return False
        if folder.is_symlink():
            folder.unlink()
        else:
            shutil.rmtree(folder)
    except FileNotFoundError:
        return False
    except (OSError, ValueError, AttributeError) as failed:
        log.warning("retired_extension_not_removed", folder=str(folder), error=str(failed))
        return False
    try:
        folder.parent.rmdir()
    except OSError:
        log.debug("retired_extension_folder_kept", folder=str(folder.parent))
    log.warning("retired_extension_removed", folder=str(folder),
                next='a copy loaded in Chrome stops when Chrome is next started; to stop it now, remove '
                     '"RUNE Browser Bridge" at chrome://extensions')
    return True


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
