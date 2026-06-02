"""Output prefix constants for RUNE capability results.

Ported from src/capabilities/output-prefixes.ts - machine-parseable
prefixes injected by the tool adapter and parsed by the loop summary.

These prefixes are language-neutral (English) and serve as a protocol
between ``format_tool_result`` and ``make_tool_summary``.
"""

from __future__ import annotations

import re

# -- file.read --
FILE_READ_PATH_PREFIX = "Path: "

# -- code.analyze --
CODE_ANALYZE_FILE_PREFIX = "File: "
CODE_ANALYZE_LINES_PREFIX = "Lines: "
CODE_ANALYZE_EXPORTED_MARKER = "(exported)"

# -- code.findRefs --
CODE_REFS_HEADER_SUFFIX = " refs:"
CODE_REFS_DEF_HEADER = "Definitions "
CODE_REFS_IMPORT_HEADER = "Import refs "

# -- code.impact --
CODE_IMPACT_SYMBOL_PREFIX = "Symbol: "
CODE_IMPACT_TOTAL_PREFIX = "Total impact: "
CODE_IMPACT_DIRECT_PREFIX = "Direct impact"

# -- bash --
BASH_CMD_PREFIX = "[cmd: "
BASH_EXIT_PREFIX = "] [exit: "

# -- failure markers (protocol between format_tool_output and the adapter's
#    failure detector / batch-stop logic). Changing these strings requires
#    updating both producers and consumers, so they live here as the single
#    source of truth. --
ERROR_PREFIX = "[ERROR]"
BLOCKED_PREFIX = "[BLOCKED]"
DENIED_PREFIX = "[DENIED]"
# Emitted when a write/edit executed but produced no change (phantom action).
NO_CHANGES_MARKER = "NO CHANGES DETECTED"
# Browser act/find miss marker.
ELEMENT_NOT_FOUND_MARKER = "Element not found"


def looks_like_failure_output(formatted: str) -> bool:
    """Return True when a formatted tool result encodes a failure.

    Operates on the machine-parseable prefixes emitted by
    ``_format_tool_output`` rather than re-deriving success from free text,
    so it stays language-neutral. ``head`` is scanned for prefixes that may
    appear after a leading ``[cmd: ...]`` segment; whole-string markers are
    checked everywhere.
    """
    head = formatted[:300]
    return (
        ERROR_PREFIX in head
        or ELEMENT_NOT_FOUND_MARKER in head
        or NO_CHANGES_MARKER in formatted
        or formatted.startswith("Error")
        or formatted.startswith(BLOCKED_PREFIX)
        or formatted.startswith(DENIED_PREFIX)
    )


def escape_for_regex(s: str) -> str:
    """Escape a string so it can be used as a literal in a regex pattern."""
    return re.escape(s)
