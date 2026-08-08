"""Rule outcomes that wait for the user's verdict.

Code-shaped work settles its outcome inside the run: an executed check
either passed or it did not. Non-code work has no such oracle — its
successes count the moment the run declares them, which is the exact
surface where the office bench caught confident wrong results being
promoted into rules. The one signal that does arrive comes later: the
user's next message. A correction means the "success" was nothing of
the kind.

So a non-code success with no executed evidence is recorded here as
pending instead of updating rule confidence. The next run's save pass
resolves it against the message that started that run: a correction
counts it as the failure it truly was; anything else — or expiry —
as the success it would have been counted as anyway. Only corrections
change a verdict, credit is delayed by at most one run or the TTL,
and resolution runs after the answer is out, so the user never waits
on it.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

_ENV_FLAG = "RUNE_DEFERRED_RULE_TRUTH"
_FILE = "pending-rule-outcomes.json"
_MAX_PENDING = 10
_TTL_HOURS = 24

_CLASSIFY_SYSTEM = (
    "You judge whether a user's newest message says that an earlier task "
    "result was wrong, broken, or not what they asked for. Corrections "
    "include pointing out mistakes, asking for the same thing again "
    "because the result failed, or complaining about the outcome — in any "
    "language. A new unrelated request, small talk, or a follow-up that "
    "builds on the result is NOT a correction.\n"
    "Respond with ONLY JSON: {\"corrected\": [<zero-based indices of the "
    "earlier tasks the message corrects>]}"
)


def deferred_truth_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "1") != "0"


def _load() -> list[dict[str, Any]]:
    from rune.memory.state import _read_json, _state_dir

    data = _read_json(_state_dir() / _FILE)
    return data if isinstance(data, list) else []


def _save(pendings: list[dict[str, Any]]) -> None:
    from rune.memory.state import _state_dir, _write_json

    _write_json(_state_dir() / _FILE, pendings)


def record_pending(
    domain: str,
    goal: str,
    error_message: str = "",
    relevant_keys: set[str] | None = None,
) -> None:
    """Hold a weak-evidence success until the user's next message."""
    pendings = _load()
    pendings.append({
        "domain": domain,
        "goal": goal[:500],
        "error_message": error_message[:300],
        "relevant_keys": sorted(relevant_keys) if relevant_keys else None,
        "created": datetime.now(UTC).isoformat(),
    })
    # Oldest entries fall off; they would have resolved as successes and
    # a backlog this deep means resolution has not been running anyway.
    del pendings[:-_MAX_PENDING]
    _save(pendings)
    log.debug("rule_outcome_deferred", domain=domain, pending=len(pendings))


def _expired(entry: dict[str, Any]) -> bool:
    try:
        created = datetime.fromisoformat(str(entry.get("created", "")))
    except ValueError:
        return True
    return datetime.now(UTC) - created > timedelta(hours=_TTL_HOURS)


async def _classify_corrections(
    pendings: list[dict[str, Any]], message: str
) -> list[int] | None:
    """Indices of pendings the message corrects, or None when the call
    fails (the caller leaves those pending for next time)."""
    from rune.llm.client import get_llm_client
    from rune.utils.fast_serde import json_decode

    tasks = "\n".join(
        f"{i}. {p.get('goal', '')[:200]}" for i, p in enumerate(pendings)
    )
    try:
        response = await get_llm_client().completion(
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Earlier tasks:\n{tasks}\n\n"
                        f"Newest message: {message[:500]}"
                    ),
                },
            ],
            tier="fast",  # type: ignore[arg-type]
            max_tokens=512,
            timeout=15.0,
        )
        text = ""
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "") or ""
        else:
            text = response.choices[0].message.content or ""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        raw = json_decode(text).get("corrected", [])
        return [int(i) for i in raw if 0 <= int(i) < len(pendings)]
    except Exception as exc:
        log.debug("pending_resolution_classify_failed", error=str(exc)[:120])
        return None


async def resolve_pendings(new_message: str) -> int:
    """Settle pending outcomes against the message that started this run.

    Returns how many were resolved. Corrections count as failures,
    everything else as the success it was provisionally. A failed
    classification leaves entries pending; expiry settles them as
    successes so an offline aux model cannot starve rule learning.
    """
    if not deferred_truth_enabled():
        return 0
    pendings = _load()
    if not pendings:
        return 0

    live = [p for p in pendings if not _expired(p)]
    expired = [p for p in pendings if _expired(p)]

    corrected: list[int] | None = []
    if live and new_message.strip():
        corrected = await _classify_corrections(live, new_message)

    resolved: list[tuple[dict[str, Any], bool]] = [(p, True) for p in expired]
    remaining: list[dict[str, Any]] = []
    if corrected is None:
        remaining = live  # classifier unavailable; try again next run
    else:
        for i, entry in enumerate(live):
            resolved.append((entry, i not in corrected))

    from rune.memory.rule_learner import update_rules_from_outcome

    for entry, success in resolved:
        keys = entry.get("relevant_keys")
        update_rules_from_outcome(
            str(entry.get("domain", "")),
            success,
            goal=str(entry.get("goal", "")),
            error_message=str(entry.get("error_message", "")),
            relevant_keys=set(keys) if keys else None,
        )
        if not success:
            log.info(
                "pending_outcome_corrected",
                goal=str(entry.get("goal", ""))[:80],
            )

    _save(remaining)
    return len(resolved)
