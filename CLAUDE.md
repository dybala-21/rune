# RUNE Development Guide

## Vision

RUNE is a local-first, general-purpose AI assistant — like Jarvis, running on your machine. The core differentiator is **self-improving**: RUNE learns from its own executions, remembers what worked, and gets better over time.

## Development Principles

### Architecture

- **Extensibility first** — every subsystem must be replaceable or extendable without touching core
- **Flexibility** — support any LLM provider, any channel, any tool, any language
- **Performance** — sub-millisecond for hot paths (memory lookup, classification, caching)
- **Accuracy** — verify results with Evidence Gate and Quality Gate, never assume success
- **Speed** — minimize token usage, cache aggressively, stop searching when sufficient

### Anti-Patterns (NEVER use)

- **String pattern matching for natural language** — no regex for intent classification, no keyword lists per language. Use LLM classification instead. Regex is only acceptable for parsing structured formats (code, URLs, file paths).
- **Hardcoded language-specific rules** — no Korean-only patterns, no English-only patterns. All features must work across all languages without per-language maintenance.
- **God modules** — no file over 800 lines. Split by responsibility.
- **Silent error swallowing** — log at minimum debug level. Never bare `except: pass`.

### Self-Improving Loop

Every task execution feeds back into the system:
1. Episode saved with utility score (+1 success, -1 failure)
2. Facts auto-extracted to learned.md
3. Lessons recorded for future reference
4. Tool call patterns logged for behavior prediction
5. Proactive engine learns from rejections (reflexion)
6. Hit counts track which memories are actually useful

### Memory

Source of truth is markdown (MEMORY.md, learned.md, daily/*.md). SQLite and FAISS are derived caches — rebuildable from markdown at any time. User can edit, delete, or version-control their memory.

### Safety

Fail-closed: if Guardian crashes, deny the operation. Defense-in-depth: multiple independent checks (Guardian + Evidence Gate + Quality Gate). Never bypass safety for convenience.
