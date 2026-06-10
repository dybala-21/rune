"""Rule Learner — learns rules from repeated failure patterns.

Trigger: same tool + similar error signature occurs 2+ times within 7 days.
Action: asks LLM to generate a one-line prevention rule, stored in learned.md.
Rules are domain-scoped and injected only into matching task types.

Rules start at low confidence and must prove themselves through successful
task outcomes before being injected into prompts (trial-based validation).
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
_INITIAL_CONFIDENCE = 0.40
_INJECTION_THRESHOLD = 0.60
_DECAY_FACTOR = 0.9
_GC_THRESHOLD = 0.30
_SOFT_CAP = 30
_RULE_CATEGORY_PREFIX = "rule:"
_CONFIDENCE_UP = 0.03
_CONFIDENCE_DOWN = 0.05
# Token budget for rule generation. A reasoning fast-tier model spends hidden
# reasoning tokens against this cap, so a small value (e.g. 50) can be used up
# before any rule text is emitted, returning an empty completion.
_RULE_GEN_MAX_TOKENS = 600
# Crisp single-failure learning: a rule from one failure, used only when the
# signal is crisp. Starts at the injection threshold so the rule is usable
# immediately, but is still demoted by outcome feedback if it proves wrong.
_CRISP_INITIAL_CONFIDENCE = _INJECTION_THRESHOLD
# Environment-dependent failures are not crisp: a rule learned from them does
# not generalize, since the next run may have a different environment.
_NON_CRISP_ERROR_MARKERS = (
    "timed out",
    "timeout",
    "connection",
    "network",
    "econnrefused",
    "etimedout",
    "permission denied",
    "rate limit",
    "503",
    "502",
    "temporarily unavailable",
)
# Loop-end reasons that are not crisp: resource, exploration, or control
# outcomes that do not identify a reproducible mistake. A rule learned from
# "ran out of steps" would not generalize.
_NON_CRISP_LOOP_REASONS = frozenset({
    "stalled",
    "no_progress",
    "token_budget_exhausted",
    "max_iterations",
    "cancelled",
    "advisor_abort",
    "no_pydantic_ai",
})


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
    tool_name: str,
    error_sample: str,
    domain: str,
    *,
    occurrences: int = 1,
) -> str | None:
    """Ask the LLM for a one-line prevention rule from a failure.

    Shared by both learning paths (occurrences>=2 repeated, ==1 crisp). Steers
    toward a general rule and lets the model decline with NONE. Returns the rule
    text, or None on decline/empty/error.
    """
    try:
        from rune.llm.client import get_llm_client
        from rune.types import ModelTier

        seen = (
            "once with a clear, reproducible error"
            if occurrences <= 1
            else f"{occurrences} times with the same pattern"
        )
        prompt = (
            f"An AI coding agent failed {seen}.\n"
            f"Tool: {tool_name}\n"
            f"Error: {error_sample}\n\n"
            "Write ONE short, GENERAL rule (under 15 words) that prevents this "
            "class of failure on similar future tasks, not a fix specific to "
            "this one file or value. If no general rule applies, reply NONE.\n"
            "Format: key_name: rule description\n"
            "Example: close_brackets: ensure every opened bracket is closed before saving\n"
            "Rule:"
        )

        client = get_llm_client()
        response = await client.completion(
            messages=[{"role": "user", "content": prompt}],
            tier=ModelTier.FAST,
            max_tokens=_RULE_GEN_MAX_TOKENS,
        )

        # Extract text from response
        text = ""
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")
        else:
            try:
                text = response.choices[0].message.content
            except (AttributeError, IndexError):
                pass

        rule_text = (text or "").strip().lstrip("- •").strip()
        if not rule_text or len(rule_text) < 5 or rule_text.upper().startswith("NONE"):
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
        rule_text = await generate_rule_from_failure(
            pattern["tool_name"], pattern["error_sample"], domain,
            occurrences=pattern["count"],
        )
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
            "eval_count": 0,
            "source": "rule_learner",
            "created_at": datetime.now(UTC).isoformat(),
            "failure_signature": sig,
            "failure_count": pattern["count"],
            "human_key": key_part,
            "category": f"rule:{domain}",
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


def is_crisp_failure(error_message: str) -> bool:
    """Return True when a failure signal is crisp enough to learn from a single
    occurrence.

    Crisp means deterministic and reproducible (the same input fails the same
    way). Environment-dependent failures (network, timeout, permission, rate
    limit, transient HTTP) are not crisp: a rule learned from them does not
    generalize, since the next run may have a different environment.
    """
    if not error_message or not error_message.strip():
        return False
    lowered = error_message.lower()
    return not any(marker in lowered for marker in _NON_CRISP_ERROR_MARKERS)


def is_crisp_loop_reason(reason: str) -> bool:
    """Return True when an agent-loop end *reason* is a crisp, actionable
    failure (a reproducible mistake) rather than a resource/exploration/control
    outcome.

    A success reason ("completed"/"verified") is never a learnable failure.
    Resource and control reasons (stalled, out of steps/budget, cancelled) are
    excluded. Anything else defers to ``is_crisp_failure`` so that an
    ``error: <env failure>`` reason (network, timeout) is still rejected.
    """
    r = (reason or "").strip().lower()
    if not r or r in ("completed", "verified"):
        return False
    head = r.split(":", 1)[0].strip()  # "error: connection refused" -> "error"
    if head in _NON_CRISP_LOOP_REASONS:
        return False
    return is_crisp_failure(reason)


async def learn_from_crisp_failure(
    tool_name: str,
    error_message: str,
    domain: str = "code_modify",
) -> str | None:
    """Learn a prevention rule from a single crisp failure.

    Unlike ``learn_from_failures`` (which needs the pattern to repeat), this
    fires on one occurrence, but only for crisp signals. The new rule starts at
    ``_CRISP_INITIAL_CONFIDENCE`` (the injection threshold) so it is usable
    immediately, and it stays subject to outcome-feedback demotion, the soft
    cap, the suppressed list, and the domain/relevance gate, so a wrongly
    learned rule drops out of injection on its next negative outcome.

    Returns the new rule key, or None when the failure is not crisp, a rule
    already exists, the cap is hit, or generation fails.
    """
    if not is_crisp_failure(error_message):
        return None

    meta = load_fact_meta()
    existing_rules = sum(1 for k in meta if k.startswith(_RULE_CATEGORY_PREFIX))
    if existing_rules >= _SOFT_CAP:
        log.debug("crisp_rule_cap_reached", count=existing_rules)
        return None

    sig = _error_signature(tool_name, error_message)
    rule_key = f"{_RULE_CATEGORY_PREFIX}{domain}:{sig}"
    if rule_key in meta or rule_key in load_suppressed():
        return None

    rule_text = await generate_rule_from_failure(
        tool_name, error_message[:300], domain, occurrences=1
    )
    if rule_text is None:
        return None

    if ":" in rule_text:
        key_part, value_part = rule_text.split(":", 1)
        key_part = key_part.strip().replace(" ", "_")[:40]
        value_part = value_part.strip()
    else:
        key_part = sig
        value_part = rule_text

    save_learned_fact(
        category=f"rule:{domain}",
        key=key_part,
        value=value_part,
        confidence=_CRISP_INITIAL_CONFIDENCE,
    )
    update_fact_meta(rule_key, {
        "confidence": _CRISP_INITIAL_CONFIDENCE,
        "hit_count": 0,
        "eval_count": 0,
        "source": "crisp_failure",
        "created_at": datetime.now(UTC).isoformat(),
        "failure_signature": sig,
        "failure_count": 1,
        "human_key": key_part,
        "category": f"rule:{domain}",
    })
    log.info("crisp_rule_learned", domain=domain, key=key_part, value=value_part[:80])
    return key_part


def _find_meta_key(
    meta: dict[str, Any], category: str, human_key: str,
) -> str | None:
    """Find a meta entry by category + human_key.

    Tries direct key first, then scans for the ``human_key`` field that
    was added alongside the hash-based key.
    """
    direct = f"{category}:{human_key}"
    if direct in meta:
        return direct
    for k, v in meta.items():
        if (
            v.get("human_key") == human_key
            and v.get("category") == category
        ):
            return k
    return None


def _resolved_rule_candidates() -> list[dict[str, Any]]:
    """All ``rule:*`` facts with resolved confidence >= threshold.

    Each candidate carries its source ``domain`` (the part after ``rule:``) so
    callers can match exactly or across domains. Confidence prefers the
    outcome-updated meta value over the static one in learned.md.
    """
    from rune.memory.markdown_store import parse_learned_md

    try:
        facts = parse_learned_md()
    except Exception:
        return []

    meta = load_fact_meta()
    out: list[dict[str, Any]] = []
    for fact in facts:
        category = fact.get("category", "")
        if not category.startswith("rule:"):
            continue
        fdomain = category.split(":", 1)[1]
        key = fact.get("key", "")

        meta_key = _find_meta_key(meta, category, key)
        if meta_key and meta[meta_key].get("eval_count", 0) > 0:
            confidence = meta[meta_key].get("confidence", 0.5)
        else:
            confidence = fact.get("confidence", 0.5)
        if confidence < _INJECTION_THRESHOLD:
            continue

        meta_entry = meta.get(meta_key, {}) if meta_key else {}
        out.append({
            "key": key,
            "value": fact.get("value", ""),
            "confidence": confidence,
            "domain": fdomain,
            "hit_count": meta_entry.get("hit_count", 0),
        })
    return out


def get_rules_for_domain(domain: str) -> list[dict[str, Any]]:
    """Get active rules for a specific domain (EXACT match).

    Returns list of {key, value, confidence} for rules matching the domain.
    Max 10 rules.  Only rules with confidence >= ``_INJECTION_THRESHOLD``
    are returned — new rules must prove themselves first.
    """
    rules = [r for r in _resolved_rule_candidates() if r["domain"] == domain]
    rules.sort(key=lambda r: r["confidence"], reverse=True)
    return rules[:10]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0 if either is zero)."""
    import math

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# Semantic-similarity threshold for cross-domain rule retrieval. Tunable: too
# low injects unrelated rules, too high misses relevant ones.
_RULE_SIM_THRESHOLD_ENV = "RUNE_RULE_SIM_THRESHOLD"
_DEFAULT_RULE_SIM_THRESHOLD = 0.55


async def get_relevant_rules(
    goal: str,
    domain: str | None = None,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Rules to inject for ``goal``: exact-domain (precise) + semantically similar.

    Exact ``goal_type`` matching alone is brittle — the LLM classifier assigns
    different (all valid) enums to the same task across runs (e.g. code_modify vs
    the 'full' fallback), so a rule learned under one enum silently fails to
    inject when the same task later classifies to another. To be robust we ALSO
    retrieve rules whose text is semantically close to the goal, regardless of
    the domain they were stored under. Falls back to exact-domain only when the
    embedding provider is unavailable.
    """
    from rune.utils.env import env_float

    candidates = _resolved_rule_candidates()
    if not candidates:
        return []

    chosen: dict[str, dict[str, Any]] = {}

    # 1. Exact-domain matches — high precision, always included.
    if domain:
        for r in candidates:
            if r["domain"] == domain:
                chosen[r["key"]] = r

    # 2. Semantic matches across ALL domains — robust to classifier drift.
    if goal:
        try:
            from rune.memory.manager import get_memory_manager

            threshold = env_float(_RULE_SIM_THRESHOLD_ENV, _DEFAULT_RULE_SIM_THRESHOLD)
            texts = [goal] + [f"{r['key']}: {r['value']}" for r in candidates]
            vecs = await get_memory_manager().embed_batch(texts)
            gvec, rvecs = vecs[0], vecs[1:]
            for r, rv in zip(candidates, rvecs, strict=False):
                if _cosine(gvec, rv) >= threshold:
                    chosen.setdefault(r["key"], r)
        except Exception as exc:  # embeddings unavailable -> exact-domain only
            log.debug("rule_semantic_retrieval_skipped", error=str(exc)[:120])

    out = list(chosen.values())
    out.sort(key=lambda r: r["confidence"], reverse=True)
    return out[:limit]


def update_rules_from_outcome(
    domain: str,
    task_success: bool,
    goal: str = "",
    error_message: str = "",
) -> int:
    """Update confidence of domain rules based on task outcome.

    Only rules whose keywords appear in the task goal or error are updated,
    avoiding noise from unrelated rules.  No LLM call — pure keyword match.

    Returns number of rules updated.
    """
    meta = load_fact_meta()
    category = f"rule:{domain}"
    context = f"{goal} {error_message}".lower()
    updated = 0

    for key, entry in meta.items():
        if not key.startswith(_RULE_CATEGORY_PREFIX):
            continue
        # Demote rules from both learners on negative outcomes. This is what
        # lets a wrongly-learned one-shot rule fall back out of injection.
        if entry.get("source") not in ("rule_learner", "crisp_failure"):
            continue

        # Relevance check: rule keywords vs task context.
        human_key = entry.get("human_key", "")
        rule_words = {w for w in human_key.lower().split("_") if len(w) > 3}
        has_overlap = bool(rule_words) and any(w in context for w in rule_words)
        in_domain = entry.get("category") == category

        # Injection is cross-domain (semantic similarity, get_relevant_rules), so
        # demotion must also be able to reach a rule stored under another domain —
        # else a wrong rule injected cross-domain causes failures forever without
        # ever being penalized. Update a rule when it is the task's own domain, or
        # when it is positively keyword-relevant to this task. A cross-domain rule
        # with no positive overlap is never touched (avoids over-demotion).
        if not in_domain and not has_overlap:
            continue
        if in_domain and rule_words and context and not has_overlap:
            continue

        conf = entry.get("confidence", _INITIAL_CONFIDENCE)
        if task_success:
            conf = min(1.0, conf + _CONFIDENCE_UP)
        else:
            conf = max(0.0, conf - _CONFIDENCE_DOWN)

        entry["confidence"] = conf
        entry["eval_count"] = entry.get("eval_count", 0) + 1
        updated += 1

    if updated:
        save_fact_meta(meta)
        log.debug(
            "rules_updated_from_outcome",
            domain=domain,
            success=task_success,
            count=updated,
        )

    return updated


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
        if entry.get("source") not in ("rule_learner", "crisp_failure"):
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
