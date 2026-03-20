"""Cognitive cache for the RUNE agent. Avoids redundant tool calls.

Ported from src/agent/cognitive-cache.ts (696 lines). LRU cache with
file generation tracking, bash mutation detection, and knowledge inventory.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

# Constants

MAX_CACHE_ENTRIES = 50
MAX_CACHEABLE_OUTPUT_CHARS = 100_000
PREVIEW_HEAD_LINES = 3
PREVIEW_TAIL_LINES = 3
SMART_EXPAND_THRESHOLD_LINES = 500

# Bash mutating patterns - detect commands that change the filesystem

BASH_MUTATING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+", re.I),
    re.compile(r"\bmv\s+", re.I),
    re.compile(r"\bcp\s+", re.I),
    re.compile(r"\bmkdir\s+", re.I),
    re.compile(r"\btouch\s+", re.I),
    re.compile(r"\bchmod\s+", re.I),
    re.compile(r"\bchown\s+", re.I),
    re.compile(r"\bln\s+", re.I),
    re.compile(r"\bsed\s+-i", re.I),
    re.compile(r"\bawk\s+.*-i\s+inplace", re.I),
    re.compile(r"\btee\s+", re.I),
    re.compile(r"\b(>\s*|>>)\s*\S+"),
    re.compile(r"\bgit\s+(add|commit|push|reset|checkout|merge|rebase|stash|cherry-pick|revert)\b", re.I),
    re.compile(r"\bnpm\s+(install|uninstall|update|ci|init)\b", re.I),
    re.compile(r"\bpip\s+(install|uninstall)\b", re.I),
    re.compile(r"\byarn\s+(add|remove|install)\b", re.I),
    re.compile(r"\bpnpm\s+(add|remove|install)\b", re.I),
    re.compile(r"\bcargo\s+(add|remove|install|build)\b", re.I),
    re.compile(r"\bdocker\s+(build|run|rm|rmi|create|compose)\b", re.I),
    re.compile(r"\bpatch\b", re.I),
    re.compile(r"\btruncate\b", re.I),
    re.compile(r"\bdd\b", re.I),
    re.compile(r"\bunzip\b", re.I),
    re.compile(r"\btar\s+.*x", re.I),
    re.compile(r"\bcurl\s+.*-[oO]", re.I),
    re.compile(r"\bwget\b", re.I),
    re.compile(r"\bpython[3]?\s+.*\.(py|pyw)\b", re.I),
    re.compile(r"\bnode\s+\S+\.js\b", re.I),
    re.compile(r"\bmake\b", re.I),
]


# Data classes

@dataclass(slots=True)
class CacheEntry:
    """A single cached tool result."""
    key: str
    capability_name: str
    params: dict[str, Any]
    preview: str
    char_count: int
    estimated_tokens: int
    cached_at_step: int
    cached_at: float  # monotonic timestamp
    hit_count: int = 0
    summary: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CacheHitResult:
    """Returned when a cache hit is found."""
    output: str
    entry: CacheEntry


@dataclass(slots=True)
class KnowledgeInventory:
    """Summary of what the agent has learned during the session."""
    files_read: list[str] = field(default_factory=list)
    searches_performed: list[str] = field(default_factory=list)
    code_analyses: list[str] = field(default_factory=list)
    tokens_saved: int = 0
    hit_count: int = 0


# SessionToolCache

class SessionToolCache:
    """LRU-based cognitive cache for tool results within an agent session.

    Tracks file generations to detect stale cache entries after mutations.
    Supports bash mutation detection for automatic invalidation.
    """

    def __init__(self, max_entries: int = MAX_CACHE_ENTRIES) -> None:
        self._max_entries = max_entries
        self.entries: dict[str, CacheEntry] = {}
        self.file_generations: dict[str, int] = {}
        self.access_order: list[str] = []
        self._hit_count = 0
        self._miss_count = 0
        self._tokens_saved = 0

    # -- Properties ----------------------------------------------------------

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def miss_count(self) -> int:
        return self._miss_count

    @property
    def tokens_saved(self) -> int:
        return self._tokens_saved

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    def stats(self) -> dict[str, int]:
        """Return summary statistics for knowledge inventory (#5)."""
        return {
            "entries": len(self.entries),
            "hits": self._hit_count,
            "misses": self._miss_count,
            "tokens_saved": self._tokens_saved,
        }

    # -- Key generation ------------------------------------------------------

    def generate_key(self, cap_name: str, params: dict[str, Any]) -> str | None:
        """Generate a cache key based on capability name and parameters.

        Returns ``None`` if the capability is not cacheable.
        """
        match cap_name:
            case "file_read":
                file_path = params.get("file_path") or params.get("path", "")
                normalized = self._normalize_path(file_path)
                gen = self._get_path_generation(normalized)
                offset = params.get("offset", 0)
                limit = params.get("limit", 0)
                return f"file_read:{normalized}:{offset}:{limit}:g{gen}"

            case "file_list":
                dir_path = self._normalize_path(params.get("path", "."))
                gen = self._get_path_generation(dir_path)
                depth = params.get("depth", 3)
                return f"file_list:{dir_path}:{depth}:g{gen}"

            case "file_search":
                pattern = params.get("pattern", "")
                path = self._normalize_path(params.get("path", "."))
                return f"file_search:{path}:{pattern}"

            case "code_analyze":
                file_path = self._normalize_path(params.get("file_path", ""))
                gen = self._get_path_generation(file_path)
                return f"code_analyze:{file_path}:g{gen}"

            case "code_find_def":
                symbol = params.get("symbol", "")
                path = self._normalize_path(params.get("path", "."))
                return f"code_find_def:{path}:{symbol}"

            case "code_find_refs":
                symbol = params.get("symbol", "")
                path = self._normalize_path(params.get("path", "."))
                return f"code_find_refs:{path}:{symbol}"

            case "project_map":
                path = self._normalize_path(params.get("path", "."))
                gen = self._get_path_generation(path)
                return f"project_map:{path}:g{gen}"

            case "web_search":
                query = params.get("query", "")
                h = hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()[:12]
                return f"web_search:{h}"

            case "web_fetch":
                url = params.get("url", "")
                h = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:12]
                return f"web_fetch:{h}"

            case "think":
                # Think is never cached; always unique reasoning
                return None

            case "bash_execute":
                # Bash commands are not cached due to side effects
                return None

            case _:
                # Browser and other capabilities are not cached
                return None

    # -- Cache operations ----------------------------------------------------

    def get(
        self, key: str, cap_name: str, params: dict[str, Any],
    ) -> CacheHitResult | None:
        """Look up a cached result. Returns ``None`` on miss."""
        entry = self.entries.get(key)
        if entry is None:
            self._miss_count += 1
            return None

        # Validate freshness for file-based entries
        if cap_name in ("file_read", "file_list", "code_analyze", "project_map"):
            file_path = self._normalize_path(
                params.get("file_path") or params.get("path", "")
            )
            current_gen = self._get_path_generation(file_path)
            expected_gen = entry.meta.get("generation", -1)
            if current_gen != expected_gen:
                # Stale entry, remove and miss
                self._remove_entry(key)
                self._miss_count += 1
                return None

        # Cache hit
        entry.hit_count += 1
        self._hit_count += 1
        self._tokens_saved += entry.estimated_tokens
        self._touch_access_order(key)

        output = self._format_cache_hit_output(entry)
        return CacheHitResult(output=output, entry=entry)

    def set(
        self,
        key: str,
        cap_name: str,
        params: dict[str, Any],
        result: Any,
        step_number: int,
    ) -> None:
        """Store a tool result in the cache."""
        output_text = str(getattr(result, "output", result) or "")
        if len(output_text) > MAX_CACHEABLE_OUTPUT_CHARS:
            log.debug("cache_skip_too_large", key=key, chars=len(output_text))
            return

        char_count = len(output_text)
        estimated_tokens = max(1, char_count // 4)
        preview = self._extract_preview(output_text)
        summary = self._generate_summary(cap_name, params, output_text)
        meta = self._extract_meta(cap_name, params, output_text)

        entry = CacheEntry(
            key=key,
            capability_name=cap_name,
            params=dict(params),
            preview=preview,
            char_count=char_count,
            estimated_tokens=estimated_tokens,
            cached_at_step=step_number,
            cached_at=time.monotonic(),
            summary=summary,
            meta=meta,
        )

        # Track file generation
        if cap_name in ("file_read", "file_list", "code_analyze", "project_map"):
            file_path = self._normalize_path(
                params.get("file_path") or params.get("path", "")
            )
            gen = self._get_path_generation(file_path)
            entry.meta["generation"] = gen

        # Evict if at capacity
        if len(self.entries) >= self._max_entries and key not in self.entries:
            self._evict_oldest()

        self.entries[key] = entry
        self._touch_access_order(key)

    def get_file_read_from_full_cache(self, file_path: str) -> CacheHitResult | None:
        """Look up a full file read (offset=0, limit=0) in the cache."""
        normalized = self._normalize_path(file_path)
        gen = self._get_path_generation(normalized)
        key = f"file_read:{normalized}:0:0:g{gen}"
        return self.get(key, "file_read", {"file_path": file_path})

    def get_file_info(self, file_path: str) -> dict[str, Any] | None:
        """Get cached metadata about a file (char_count, lines, etc.)."""
        normalized = self._normalize_path(file_path)
        gen = self._get_path_generation(normalized)
        key = f"file_read:{normalized}:0:0:g{gen}"
        entry = self.entries.get(key)
        if entry is None:
            return None
        return {
            "char_count": entry.char_count,
            "estimated_tokens": entry.estimated_tokens,
            "cached_at_step": entry.cached_at_step,
            "hit_count": entry.hit_count,
            **entry.meta,
        }

    # -- Invalidation --------------------------------------------------------

    def invalidate_file(self, file_path: str) -> None:
        """Invalidate all cache entries related to a file path."""
        normalized = self._normalize_path(file_path)
        # Bump generation for this path
        self.file_generations[normalized] = self._get_path_generation(normalized) + 1
        # Remove affected entries
        self._invalidate_all_file_entries(normalized)

    def invalidate_from_bash(self, command: str, success: bool) -> None:
        """Invalidate cache entries affected by a bash command.

        Detects mutating commands and invalidates relevant file entries.
        """
        if not success:
            return

        is_mutating = any(pat.search(command) for pat in BASH_MUTATING_PATTERNS)
        if not is_mutating:
            return

        # Extract potential file paths from the command
        # and invalidate all file-related entries conservatively
        keys_to_remove: list[str] = []
        for key, entry in self.entries.items():
            if self._is_entry_affected_by_mutation(entry, command):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            self._remove_entry(key)

        if keys_to_remove:
            log.debug(
                "cache_invalidated_from_bash",
                command=command[:80],
                removed=len(keys_to_remove),
            )

    # -- Knowledge inventory -------------------------------------------------

    def build_knowledge_inventory(self) -> KnowledgeInventory:
        """Build a summary of knowledge gathered during this session."""
        files_read: list[str] = []
        searches: list[str] = []
        analyses: list[str] = []

        for entry in self.entries.values():
            match entry.capability_name:
                case "file_read":
                    path = entry.params.get("file_path") or entry.params.get("path", "")
                    if path and path not in files_read:
                        files_read.append(path)
                case "web_search" | "file_search":
                    query = entry.params.get("query") or entry.params.get("pattern", "")
                    if query and query not in searches:
                        searches.append(query)
                case "code_analyze" | "code_find_def" | "code_find_refs":
                    target = (
                        entry.params.get("file_path")
                        or entry.params.get("symbol", "")
                    )
                    if target and target not in analyses:
                        analyses.append(target)

        return KnowledgeInventory(
            files_read=files_read,
            searches_performed=searches,
            code_analyses=analyses,
            tokens_saved=self._tokens_saved,
            hit_count=self._hit_count,
        )

    # -- Partial clear / eviction --------------------------------------------

    def partial_clear(self, max_entries: int | None = None) -> None:
        """Evict entries down to *max_entries* (default: half current size)."""
        target = max_entries if max_entries is not None else len(self.entries) // 2
        while len(self.entries) > target:
            self._evict_oldest()

    # -- Private helpers -----------------------------------------------------

    def _normalize_path(self, path: str) -> str:
        """Normalize a file path for use as cache key component."""
        if not path:
            return ""
        expanded = os.path.expanduser(os.path.expandvars(path))
        return os.path.normpath(expanded)

    def _get_generation(self, key: str) -> int:
        """Get the generation counter embedded in a cache key."""
        # Keys end with :gN
        parts = key.rsplit(":g", 1)
        if len(parts) == 2:
            try:
                return int(parts[1])
            except ValueError:
                pass
        return 0

    def _get_path_generation(self, normalized_path: str) -> int:
        """Get the current generation for a file path."""
        return self.file_generations.get(normalized_path, 0)

    def _is_entry_affected_by_mutation(
        self, entry: CacheEntry, command: str,
    ) -> bool:
        """Check if a cache entry might be stale after a bash mutation."""
        # File-related capabilities are always affected
        if entry.capability_name in (
            "file_read", "file_list", "file_search",
            "code_analyze", "code_find_def", "code_find_refs",
            "project_map",
        ):
            # Check if the entry's path appears in the command
            path = entry.params.get("file_path") or entry.params.get("path", "")
            if path:
                normalized = self._normalize_path(path)
                basename = os.path.basename(normalized)
                dirname = os.path.dirname(normalized)
                # If the command mentions the file or its directory
                if basename and basename in command:
                    return True
                if dirname and dirname in command:
                    return True
            # For broad mutations (make, git, npm install etc.) invalidate all
            broad_patterns = [
                r"\bgit\s+(checkout|reset|merge|rebase|stash\s+pop)",
                r"\b(npm|yarn|pnpm)\s+install",
                r"\bmake\b",
                r"\bcargo\s+build",
            ]
            for pat_str in broad_patterns:
                if re.search(pat_str, command, re.I):
                    return True
        return False

    def _invalidate_all_file_entries(self, normalized_path: str) -> None:
        """Remove all cache entries that reference the given path."""
        keys_to_remove: list[str] = []
        for key, entry in self.entries.items():
            path = entry.params.get("file_path") or entry.params.get("path", "")
            if path and self._normalize_path(path) == normalized_path:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self._remove_entry(key)

    def _remove_entry(self, key: str) -> None:
        """Remove a single entry from cache and access order."""
        self.entries.pop(key, None)
        self._remove_from_access_order(key)

    def _format_cache_hit_output(self, entry: CacheEntry) -> str:
        """Format the output for a cache hit response."""
        parts = [
            f"[CACHE HIT — step {entry.cached_at_step}, "
            f"hits: {entry.hit_count}, "
            f"~{entry.estimated_tokens} tokens saved]",
        ]
        if entry.summary:
            parts.append(f"Summary: {entry.summary}")
        if entry.preview:
            parts.append(entry.preview)
        return "\n".join(parts)

    def _extract_preview(self, output: str) -> str:
        """Extract a head+tail preview from output text."""
        lines = output.splitlines()
        total = len(lines)
        if total <= PREVIEW_HEAD_LINES + PREVIEW_TAIL_LINES:
            return output

        head = lines[:PREVIEW_HEAD_LINES]
        tail = lines[-PREVIEW_TAIL_LINES:]
        omitted = total - PREVIEW_HEAD_LINES - PREVIEW_TAIL_LINES
        return "\n".join([
            *head,
            f"... ({omitted} lines omitted) ...",
            *tail,
        ])

    def _generate_summary(
        self, cap_name: str, params: dict[str, Any], output: str,
    ) -> str:
        """Generate a short summary for a cached entry."""
        line_count = output.count("\n") + 1
        char_count = len(output)

        match cap_name:
            case "file_read":
                path = params.get("file_path") or params.get("path", "")
                return f"Read {path} ({line_count} lines, {char_count} chars)"
            case "file_list":
                path = params.get("path", ".")
                return f"Listed {path} ({line_count} entries)"
            case "file_search":
                pattern = params.get("pattern", "")
                return f"Search '{pattern}' ({line_count} results)"
            case "code_analyze":
                path = params.get("file_path", "")
                return f"Analyzed {path}"
            case "code_find_def":
                symbol = params.get("symbol", "")
                return f"Found definition of '{symbol}'"
            case "code_find_refs":
                symbol = params.get("symbol", "")
                return f"Found references to '{symbol}'"
            case "project_map":
                return f"Project map ({line_count} lines)"
            case "web_search":
                query = params.get("query", "")
                return f"Web search: '{query}'"
            case "web_fetch":
                url = params.get("url", "")
                return f"Fetched {url} ({char_count} chars)"
            case _:
                return f"{cap_name} ({char_count} chars)"

    def _extract_meta(
        self, cap_name: str, params: dict[str, Any], output: str,
    ) -> dict[str, Any]:
        """Extract metadata for a cached entry."""
        meta: dict[str, Any] = {}
        line_count = output.count("\n") + 1
        meta["line_count"] = line_count

        if cap_name == "file_read":
            meta["is_full_read"] = (
                params.get("offset", 0) == 0
                and params.get("limit", 0) == 0
            )
            meta["is_large"] = line_count >= SMART_EXPAND_THRESHOLD_LINES

        if cap_name == "web_fetch":
            meta["url"] = params.get("url", "")

        return meta

    def _touch_access_order(self, key: str) -> None:
        """Move *key* to the end of the access order list (most recent)."""
        self._remove_from_access_order(key)
        self.access_order.append(key)

    def _remove_from_access_order(self, key: str) -> None:
        """Remove *key* from the access order list."""
        with contextlib.suppress(ValueError):
            self.access_order.remove(key)

    def _evict_oldest(self) -> None:
        """Evict the least recently accessed entry."""
        if not self.access_order:
            return
        oldest_key = self.access_order.pop(0)
        self.entries.pop(oldest_key, None)


# Public helper

def format_knowledge_inventory(inv: KnowledgeInventory) -> str | None:
    """Format a knowledge inventory as a human-readable string.

    Returns ``None`` if the inventory is empty.
    """
    parts: list[str] = []

    if inv.files_read:
        parts.append(f"Files read ({len(inv.files_read)}):")
        for f in inv.files_read[:20]:
            parts.append(f"  - {f}")
        if len(inv.files_read) > 20:
            parts.append(f"  ... and {len(inv.files_read) - 20} more")

    if inv.searches_performed:
        parts.append(f"Searches ({len(inv.searches_performed)}):")
        for s in inv.searches_performed[:10]:
            parts.append(f"  - {s}")

    if inv.code_analyses:
        parts.append(f"Code analyses ({len(inv.code_analyses)}):")
        for a in inv.code_analyses[:10]:
            parts.append(f"  - {a}")

    if inv.tokens_saved > 0:
        parts.append(f"Tokens saved by cache: ~{inv.tokens_saved:,}")

    if inv.hit_count > 0:
        parts.append(f"Cache hits: {inv.hit_count}")

    if not parts:
        return None

    return "\n".join(parts)
