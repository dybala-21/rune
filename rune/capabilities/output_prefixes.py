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


def escape_for_regex(s: str) -> str:
    """Escape a string so it can be used as a literal in a regex pattern."""
    return re.escape(s)
