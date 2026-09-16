"""Inspect the installed helper and report its permission setup status."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

from rune.utils.logger import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=8)
def _signing(bundle: str, executable_stamp: tuple[int, int], plist_stamp: tuple[int, int]) -> str:
    try:
        result = subprocess.run(["codesign", "--display", "--verbose=2", bundle],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            if "Signature=adhoc" in result.stderr:
                return "ad_hoc"
            if any(line.startswith("Authority=") for line in result.stderr.splitlines()):
                return "certificate"
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.debug("native_signing_inspection_failed", error=str(exc))
    return "unknown"


def installation_info(executable: Path) -> dict:
    bundle = executable.parents[2]
    info = {"appPath": str(bundle), "signing": "unknown"}
    try:
        binary = executable.stat()
        plist = (bundle / "Contents/Info.plist").stat()
    except OSError:
        return info
    info["signing"] = _signing(str(bundle), (binary.st_mtime_ns, binary.st_size), (plist.st_mtime_ns, plist.st_size))
    return info
