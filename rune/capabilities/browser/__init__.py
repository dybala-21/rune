"""Browser capabilities package for RUNE.

Public API — preserves all existing import paths so that
``from rune.capabilities.browser import X`` continues to work.
"""

from rune.capabilities.browser.capabilities import (  # noqa: F401
    register_browser_capabilities,
)
from rune.capabilities.browser.core import (  # noqa: F401
    _close_browser,
)
