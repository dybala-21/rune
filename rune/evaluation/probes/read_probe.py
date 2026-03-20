"""File reading probe for RUNE evaluation system.

Tests the file.read capability by creating a temporary file,
reading it back, and verifying the content matches.
"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from typing import Any

from rune.evaluation.probes.types import Probe
from rune.evaluation.types import ProbeResult
from rune.utils.logger import get_logger

log = get_logger(__name__)

_TEST_CONTENT = "RUNE_READ_PROBE_VERIFICATION_STRING_42"


class ReadProbe(Probe):
    """Tests the file.read capability.

    Creates a temporary file with known content, reads it back
    via the agent's file reading mechanism, and verifies the result.
    """

    @property
    def name(self) -> str:
        return "file.read"

    async def run(self, context: dict[str, Any]) -> ProbeResult:
        """Execute the read probe.

        Parameters:
            context: May contain a "read_fn" async callable for custom
                file reading. Falls back to direct filesystem read.

        Returns:
            ProbeResult indicating whether file reading works correctly.
        """
        tmp_path: Path | None = None
        try:
            # Create a temporary file with known content
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                prefix="rune_probe_",
                delete=False,
            ) as tmp:
                tmp.write(_TEST_CONTENT)
                tmp_path = Path(tmp.name)

            # Read back using provided function or direct read
            read_fn = context.get("read_fn")
            if read_fn is not None:
                content = await read_fn(str(tmp_path))
            else:
                content = tmp_path.read_text(encoding="utf-8")

            # Verify
            success = content.strip() == _TEST_CONTENT
            score = 1.0 if success else 0.0

            return ProbeResult(
                probe_name=self.name,
                success=success,
                output=content.strip(),
                expected=_TEST_CONTENT,
                score=score,
            )

        except Exception as exc:
            log.error("read_probe_error", error=str(exc))
            return ProbeResult(
                probe_name=self.name,
                success=False,
                output=f"Error: {exc}",
                expected=_TEST_CONTENT,
                score=0.0,
            )

        finally:
            if tmp_path is not None and tmp_path.exists():
                with contextlib.suppress(OSError):
                    tmp_path.unlink()
