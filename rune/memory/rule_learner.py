"""Rule Learner — learns rules from repeated failure patterns.

Trigger: same tool + similar error signature occurs 2+ times within 7 days.
Action: asks LLM to generate a one-line prevention rule, stored in learned.md.
Rules are domain-scoped and injected only into matching task types.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

from rune.memory.markdown_store import save_learned_fact
from rune.memory.state import (
    load_fact_meta,
    load_suppressed,
    save_fact_meta,
    update_fact_meta,
)
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Constants
_LOOKBACK_DAYS = 7
_MIN_OCCURRENCES = 2
_INITIAL_CONFIDENCE = 0.60
_DECAY_FACTOR = 0.9
_GC_THRESHOLD = 0.30
_SOFT_CAP = 30
_RULE_CATEGORY_PREFIX = "rule:"


def _error_signature(tool_name: str, error_message: str) -> str:
    """Create a stable signature from tool name + error message.

    Normalizes the error by stripping file paths and line numbers
    to match semantically similar errors.
    """
    # Normalize: strip paths, filenames, numbers for stable matching
    import re
    normalized = re.sub(r"/[\w/.\-]+", "<path>", error_message[:200])
    normalized = re.sub(r"[\w\-]+\.\w{1,4}", "<file>", normalized)  # filename.ext
    normalized = re.sub(r"\b\d+\b", "<n>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    raw = f"{tool_name}:{normalized}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def find_repeated_failures(
    store: Any,
    lookback_days: int = _LOOKBACK_DAYS,
    min_occurrences: int = _MIN_OCCURRENCES,
) -> list[dict[str, Any]]:
    """Find failure patterns that occurred 2+ times within the lookback window.

    Returns list of {signature, tool_name, error_sample, count, calls}.
    """
    cutoff = (datetime.now(UTC) - timedelta(days=lookback_days)).isoformat()

    try:
        rows = store.conn.execute(
            """SELECT tool_name, error_message, params, created_at
               FROM tool_call_log
               WHERE result_success = 0
                 AND error_message != ''
                 AND created_at > ?
               ORDER BY created_at DESC""",
            (cutoff,),
        ).fetchall()
    except Exception:
        return []

    # Group by error signature
    sig_groups: dict[str, dict[str, Any]] = {}
    for tool_name, error_msg, params_json, created_at in rows:
        sig = _error_signature(tool_name, error_msg)
        if sig not in sig_groups:
            sig_groups[sig] = {
                "signature": sig,
                "tool_name": tool_name,
                "error_sample": error_msg[:300],
                "count": 0,
                "calls": [],
            }
        sig_groups[sig]["count"] += 1
        sig_groups[sig]["calls"].append({
            "tool_name": tool_name,
            "error": error_msg[:200],
            "params": params_json,
            "created_at": created_at,
        })

    # Filter to patterns with min_occurrences
    return [g for g in sig_groups.values() if g["count"] >= min_occurrences]


async def generate_rule_from_failure(
    failure_pattern: dict[str, Any],
    domain: str,
) -> str | None:
    """Ask LLM to generate a one-line prevention rule from a failure pattern.

    Returns the rule text, or None if LLM fails.
    """
    try:
        import litellm

        tool_name = failure_pattern["tool_name"]
        error_sample = failure_pattern["error_sample"]
        count = failure_pattern["count"]

        prompt = (
            f"An AI coding agent failed {count} times with the same pattern:\n"
            f"Tool: {tool_name}\n"
            f"Error: {error_sample}\n\n"
            f"Write ONE short rule (under 15 words) that would prevent this failure. "
            f"Format: key_name: rule description\n"
            f"Example: verify_before_edit: re-read file before file_edit to avoid stale content\n"
            f"Rule:"
        )

        response = await litellm.acompletion(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
        )

        rule_text = response.choices[0].message.content.strip()
        # Clean up: remove leading "- " or bullet points
        rule_text = rule_text.lstrip("- •").strip()
        if not rule_text or len(rule_text) < 5:
            return None
        return rule_text

    except Exception as exc:
        log.debug("rule_generation_failed", error=str(exc)[:200])
        return None


async def learn_from_failures(store: Any, domain: str = "code_modify") -> list[str]:
    """Main entry point: find repeated failures and generate rules.

    Called after a task completes with utility=-1.
    Returns list of newly created rule keys.
    """
    # Check soft cap
    meta = load_fact_meta()
    existing_rules = sum(
        1 for k in meta if k.startswith(_RULE_CATEGORY_PREFIX)
    )
    if existing_rules >= _SOFT_CAP:
        log.debug("rule_learner_cap_reached", count=existing_rules)
        return []

    # Check suppressed rules
    suppressed = load_suppressed()

    # Find repeated failure patterns
    patterns = find_repeated_failures(store)
    if not patterns:
        return []

    new_rules: list[str] = []
    for pattern in patterns:
        sig = pattern["signature"]
        rule_key = f"{_RULE_CATEGORY_PREFIX}{domain}:{sig}"

        # Skip if already exists or suppressed
        if rule_key in meta:
            continue
        if rule_key in suppressed:
            continue

        # Generate rule via LLM
        rule_text = await generate_rule_from_failure(pattern, domain)
        if rule_text is None:
            continue

        # Parse key:value from rule_text
        if ":" in rule_text:
            key_part, value_part = rule_text.split(":", 1)
            key_part = key_part.strip().replace(" ", "_")[:40]
            value_part = value_part.strip()
        else:
            key_part = sig
            value_part = rule_text

        # Save to learned.md
        category = f"rule:{domain}"
        save_learned_fact(
            category=category,
            key=key_part,
            value=value_part,
            confidence=_INITIAL_CONFIDENCE,
        )

        # Save metadata
        update_fact_meta(rule_key, {
            "confidence": _INITIAL_CONFIDENCE,
            "hit_count": 0,
            "source": "rule_learner",
            "created_at": datetime.now(UTC).isoformat(),
            "failure_signature": sig,
            "failure_count": pattern["count"],
        })

        new_rules.append(key_part)
        log.info(
            "rule_learned",
            domain=domain,
            key=key_part,
            value=value_part[:80],
            failure_count=pattern["count"],
        )

    return new_rules


def get_rules_for_domain(domain: str) -> list[dict[str, Any]]:
    """Get active rules for a specific domain.

    Returns list of {key, value, confidence} for rules matching the domain.
    Max 10 rules.
    """
    from rune.memory.markdown_store import parse_learned_md

    try:
        facts = parse_learned_md()
    except Exception:
        return []

    category = f"rule:{domain}"
    meta = load_fact_meta()

    rules: list[dict[str, Any]] = []
    for fact in facts:
        if fact.get("category") != category:
            continue
        key = fact.get("key", "")
        confidence = fact.get("confidence", 0.5)
        if confidence < _GC_THRESHOLD:
            continue
        rules.append({
            "key": key,
            "value": fact.get("value", ""),
            "confidence": confidence,
            "hit_count": meta.get(f"{category}:{key}", {}).get("hit_count", 0),
        })

    # Sort by confidence desc, limit to 10
    rules.sort(key=lambda r: r["confidence"], reverse=True)
    return rules[:10]


def decay_unused_rules() -> int:
    """Decay confidence of rules with hit_count=0 and older than 30 days.

    Returns number of rules decayed.
    """
    meta = load_fact_meta()
    now = datetime.now(UTC)
    decayed = 0

    for key, entry in list(meta.items()):
        if not key.startswith(_RULE_CATEGORY_PREFIX):
            continue
        if entry.get("source") != "rule_learner":
            continue
        if entry.get("hit_count", 0) > 0:
            continue

        created = entry.get("created_at", "")
        if not created:
            continue

        try:
            created_dt = datetime.fromisoformat(created)
            age_days = (now - created_dt).days
        except (ValueError, TypeError):
            continue

        if age_days >= 30:
            old_conf = entry.get("confidence", _INITIAL_CONFIDENCE)
            new_conf = old_conf * _DECAY_FACTOR
            entry["confidence"] = new_conf
            decayed += 1

            if new_conf < _GC_THRESHOLD:
                log.info("rule_gc", key=key, confidence=new_conf)

    if decayed:
        save_fact_meta(meta)

    return decayed
