"""Tool Registry - manages all available Tool instances.

Ported from src/tools/registry.ts.  Singleton pattern via
:func:`get_tool_registry`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rune.tools.base import Tool, ToolInfo, get_tool_info
from rune.utils.logger import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


class ToolRegistry:
    """Central registry for :class:`Tool` instances.

    On construction the three built-in tools (File, Process, Browser) are
    registered automatically.
    """

    def __init__(self, *, auto_register_defaults: bool = True) -> None:
        self._tools: dict[str, Tool] = {}
        if auto_register_defaults:
            self._register_default_tools()

    # -- default registration ------------------------------------------------

    def _register_default_tools(self) -> None:
        """Register the built-in tools (lazy imports to avoid circular deps)."""
        from rune.tools.browser import BrowserTool
        from rune.tools.file import FileTool
        from rune.tools.process import ProcessTool

        self.register(FileTool())
        self.register(ProcessTool())
        self.register(BrowserTool())

    # -- public API ----------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Register (or overwrite) a tool."""
        if tool.name in self._tools:
            log.warning("tool_already_registered", name=tool.name)
        self._tools[tool.name] = tool
        log.debug("tool_registered", name=tool.name)

    def unregister(self, name: str) -> bool:
        """Remove a tool by name.  Returns True if it was found."""
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Tool | None:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def list_info(self) -> list[ToolInfo]:
        """Return lightweight :class:`ToolInfo` for every tool."""
        return [get_tool_info(t) for t in self._tools.values()]

    def get_by_domain(self, domain: str) -> list[Tool]:
        """Return tools whose domain matches *domain*."""
        return [t for t in self._tools.values() if t.domain == domain]

    def find_by_action(self, action: str) -> Tool | None:
        """Find the first tool that supports *action*."""
        for tool in self._tools.values():
            if action in tool.actions:
                return tool
        return None

    async def health_check(self) -> dict[str, bool]:
        """Run health_check on every tool."""
        results: dict[str, bool] = {}
        for name, tool in self._tools.items():
            try:
                results[name] = await tool.health_check()
            except Exception:
                results[name] = False
        return results


# Singleton

_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Return the singleton :class:`ToolRegistry`, creating it on first call."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
