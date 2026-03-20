"""RUNE - AI Development Environment.

A Python port of the RUNE TypeScript AI agent system.
"""

__version__ = "0.1.0"
__app_name__ = "rune"
__codename__ = "dybala"

# Phase 9: Lazy imports - heavy third-party libs and submodules are loaded
# on first access rather than at import time.
from rune._lazy import install_lazy_imports as _install_lazy_imports

_install_lazy_imports()
del _install_lazy_imports
