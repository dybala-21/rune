"""Lazy-import machinery for RUNE.

Phase 9 optimization - heavy third-party modules are only imported when
first accessed, shaving hundreds of milliseconds off CLI startup.

Usage (inside ``rune/__init__.py``)::

    from rune._lazy import install_lazy_imports
    install_lazy_imports()

After that, ``rune.pydantic_ai``, ``rune.playwright``, etc. resolve on
first attribute access rather than at import time.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

# Modules that are expensive to import and should be deferred.
# Keys   = attribute name exposed on the ``rune`` package
# Values = fully-qualified module to import

LAZY_MODULES: dict[str, str] = {
    "pydantic_ai": "pydantic_ai",
    "playwright": "playwright",
    "faiss": "faiss",
    "tree_sitter": "tree_sitter",
    "tiktoken": "tiktoken",
    "litellm": "litellm",
}

# Submodules of the ``rune`` package that should also be lazy.
LAZY_SUBMODULES: dict[str, str] = {
    "agent": "rune.agent",
    "api": "rune.api",
    "browser": "rune.browser",
    "capabilities": "rune.capabilities",
    "channels": "rune.channels",
    "config": "rune.config",
    "conversation": "rune.conversation",
    "daemon": "rune.daemon",
    "evaluation": "rune.evaluation",
    "identity": "rune.identity",
    "integration": "rune.integration",
    "intelligence": "rune.intelligence",
    "llm": "rune.llm",
    "mcp": "rune.mcp",
    "memory": "rune.memory",
    "proactive": "rune.proactive",
    "safety": "rune.safety",
    "services": "rune.services",
    "skills": "rune.skills",
    "ui": "rune.ui",
    "voice": "rune.voice",
}


def _make_module_getattr(
    package_name: str,
    lazy_externals: dict[str, str],
    lazy_submodules: dict[str, str],
) -> Any:
    """Return a ``__getattr__`` function suitable for a package's ``__init__``.

    The returned function intercepts attribute lookups on the package module
    and performs just-in-time imports for the entries in *lazy_externals* and
    *lazy_submodules*.
    """

    def __getattr__(name: str) -> Any:
        # 1. Lazy external library (e.g., ``rune.tiktoken``)
        if name in lazy_externals:
            module = importlib.import_module(lazy_externals[name])
            # Cache on the package so subsequent access is instant.
            setattr(sys.modules[package_name], name, module)
            return module

        # 2. Lazy submodule (e.g., ``rune.agent``)
        if name in lazy_submodules:
            module = importlib.import_module(lazy_submodules[name])
            setattr(sys.modules[package_name], name, module)
            return module

        raise AttributeError(f"module {package_name!r} has no attribute {name!r}")

    return __getattr__


def install_lazy_imports() -> None:
    """Patch ``rune.__getattr__`` to enable lazy imports.

    Safe to call multiple times - subsequent calls are no-ops.
    """
    rune_pkg = sys.modules.get("rune")
    if rune_pkg is None:
        return  # Package not loaded yet; nothing to patch.

    # Guard against double-patching.
    if getattr(rune_pkg, "_lazy_installed", False):
        return

    rune_pkg.__getattr__ = _make_module_getattr(  # type: ignore[attr-defined]
        "rune",
        LAZY_MODULES,
        LAZY_SUBMODULES,
    )
    rune_pkg._lazy_installed = True  # type: ignore[attr-defined]
