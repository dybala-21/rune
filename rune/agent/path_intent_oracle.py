"""Path intent oracle - LLM-based intentional path allowlist resolution.

Ported from src/agent/path-intent-oracle.ts (150 lines) - uses LLM
structured output to distinguish user-intended workspace paths from
paths that merely appear in logs/history/reference context.

resolveIntentionalPathAllowlist(): LLM oracle with caching + conservative fallback.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from rune.agent.workspace_guard import extract_explicit_path_allowlist
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Constants

PATH_INTENT_ORACLE_TTL_S = 2 * 60  # 2 minutes
PATH_INTENT_ORACLE_MAX = 200
DEFAULT_MIN_CONFIDENCE = 0.62
MAX_CANDIDATE_PATHS = 8

# Cache

@dataclass(slots=True)
class _CacheEntry:
    value: list[str]
    ts: float


_oracle_cache: dict[str, _CacheEntry] = {}


def _cache_key(goal: str, workspace_root: str, candidates: list[str]) -> str:
    return f"{workspace_root}::{goal.strip()}::{'|'.join(candidates)}"


def _get_cached(key: str) -> list[str] | None:
    entry = _oracle_cache.get(key)
    if entry is None:
        return None
    if time.time() - entry.ts > PATH_INTENT_ORACLE_TTL_S:
        del _oracle_cache[key]
        return None
    return list(entry.value)


def _set_cached(key: str, value: list[str]) -> None:
    if len(_oracle_cache) >= PATH_INTENT_ORACLE_MAX:
        # Evict oldest
        oldest = next(iter(_oracle_cache))
        del _oracle_cache[oldest]
    _oracle_cache[key] = _CacheEntry(value=list(value), ts=time.time())


# Path normalization

def _normalize_path_token(raw_path: str, workspace_root: str) -> str:
    """Normalize a single path token to absolute."""
    trimmed = raw_path.strip()
    if not trimmed:
        return ""
    if trimmed.startswith("$") or trimmed.startswith("${"):
        return ""

    home = os.environ.get("HOME", "")
    if trimmed == "~" and home:
        expanded = home
    elif home and trimmed.startswith("~/"):
        expanded = os.path.join(home, trimmed[2:])
    else:
        expanded = trimmed

    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(workspace_root, expanded))


def _normalize_many(paths: list[str], workspace_root: str) -> list[str]:
    """Normalize and deduplicate a list of paths."""
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in paths:
        token = _normalize_path_token(raw, workspace_root)
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


# Prompt builder

def _build_prompt(goal: str, candidates: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    return (
        f"User request:\n{goal}\n\n"
        f"Candidate paths (resolved):\n{numbered}\n\n"
        "Select only paths explicitly requested as current execution workspace.\n"
        "Reject paths that appear only in logs/history/examples/reference context."
    )


# Public API

@dataclass(slots=True)
class ResolvePathIntentOptions:
    """Options for path intent resolution."""

    model: Any = None  # Optional LLM model instance
    min_confidence: float = DEFAULT_MIN_CONFIDENCE


async def resolve_intentional_path_allowlist(
    goal: str,
    workspace_root: str,
    options: ResolvePathIntentOptions | None = None,
) -> list[str]:
    """Resolve user-intended workspace paths via LLM oracle.

    Strategy:
    1. Extract explicit path candidates from goal text.
    2. If LLM model available, ask oracle with structured output + confidence.
    3. Conservative fallback: empty allowlist (prefer false-negative over false-positive).
    """
    opts = options or ResolvePathIntentOptions()
    fallback: list[str] = []

    explicit_candidates = _normalize_many(
        extract_explicit_path_allowlist(goal, workspace_root)[:MAX_CANDIDATE_PATHS],
        workspace_root,
    )

    if not explicit_candidates:
        return fallback
    if opts.model is None:
        return fallback

    key = _cache_key(goal, workspace_root, explicit_candidates)
    cached = _get_cached(key)
    if cached is not None:
        return cached

    try:
        # Call LLM model for structured path intent extraction
        # This integrates with PydanticAI or any model implementing generate_object
        result = await _call_path_intent_oracle(
            goal, explicit_candidates, opts.model, opts.min_confidence
        )
        _set_cached(key, result)
        return result
    except Exception as exc:
        log.debug(
            "path_intent_oracle_failed",
            error=str(exc)[:200],
        )
        _set_cached(key, fallback)
        return fallback


async def _call_path_intent_oracle(
    goal: str,
    candidates: list[str],
    model: Any,
    min_confidence: float,
) -> list[str]:
    """Call the LLM oracle for path intent resolution via LiteLLM.

    Uses JSON response_format for structured output.
    Falls back to empty list on low confidence.
    """
    import json

    import litellm

    from rune.agent.litellm_adapter import _resolve_litellm_model

    fallback: list[str] = []

    model_str = model if isinstance(model, str) else getattr(model, "model", str(model))
    resolved = _resolve_litellm_model(model_str)

    try:
        resp = await litellm.acompletion(
            model=resolved,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract execution workspace intent from user text.\n"
                        "Return only candidate paths explicitly intended as active workspace for this turn.\n"
                        "Never invent paths.\n"
                        "When ambiguous, return empty requested_paths and low confidence.\n"
                        "Respond with JSON: {\"requested_paths\": [...], \"confidence\": 0.0-1.0}"
                    ),
                },
                {"role": "user", "content": _build_prompt(goal, candidates)},
            ],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content or "{}"
        obj = json.loads(raw)

        confidence = float(obj.get("confidence", 0.0))
        if confidence < min_confidence:
            return fallback

        requested = obj.get("requested_paths", [])
        if not isinstance(requested, list):
            return fallback

        allowed = set(candidates)
        selected = [
            p for p in _normalize_many(requested, os.path.dirname(candidates[0]))
            if p in allowed
        ]
        return selected if selected else fallback

    except Exception as exc:
        log.debug("path_intent_oracle_llm_failed", error=str(exc)[:200])
        return fallback
