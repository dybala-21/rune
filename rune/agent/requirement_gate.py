"""Requirement-adherence gate: verify the produced output satisfies the user's
explicit requirements before finalizing, for general (non-test) tasks.

Enabled via ``RUNE_REQUIREMENT_GATE`` (off by default). The requirement checklist
is extracted once at task start and held in Python state, so it survives context
compaction. Skips (never blocks) when no checklist is extracted or the check is
inconclusive; runs after the existing gates in the finalize branch.

Independent-judge routing: a model is blind to its own errors (Self-Correction
Bench, arXiv 2507.02778: 64.5% blind-spot rate that vanishes when the same error
is surfaced externally) and weak models self-judge with poor calibration
(arXiv 2508.06225). So when the active provider's checker is weak (local ollama)
or its check call fails, the gate routes the same check to a configured
escalation provider (``escalationProvider`` / ``escalationModel``) as an
independent judge, instead of silently passing. The escalation judge must be a
strong/cloud model; a weak local judge is not a reliable independent verifier.
"""

from __future__ import annotations

import json

from rune.utils.env import env_flag
from rune.utils.logger import get_logger

log = get_logger(__name__)

_REQUIREMENT_GATE_ENV = "RUNE_REQUIREMENT_GATE"
_MAX_ARTIFACT_CHARS = 12_000
_MAX_CHECKLIST_ITEMS = 12

_EXTRACT_SYSTEM = (
    "You extract the explicit, checkable requirements from a user's task request "
    "so a reviewer can later confirm the produced output satisfied each one.\n"
    "Rules:\n"
    "- List only requirements the user actually stated (deliverables, constraints, "
    "formats, counts, fields, ordering, flags). Do NOT invent requirements.\n"
    "- Each item is one atomic, objectively checkable statement, short.\n"
    "- Ignore vague preferences that cannot be checked.\n"
    "- If the request has no checkable requirements, return an empty array.\n"
    'Output ONLY a JSON array of strings, e.g. ["...", "..."]. No prose, no fences.'
)

_CHECK_SYSTEM = (
    "You verify whether a produced OUTPUT satisfies a checklist of REQUIREMENTS.\n"
    "For each requirement decide if the output clearly satisfies it.\n"
    "Be conservative: only mark a requirement UNMET when the output CLEARLY fails "
    "it. If you are unsure, or the output plausibly satisfies it, treat it as MET "
    "(do not block on doubt).\n"
    'Output ONLY a JSON object: {"unmet": ["<the requirement text>", ...]}. '
    "An empty list means everything is satisfied. No prose, no fences."
)


def requirement_gate_enabled() -> bool:
    return env_flag(_REQUIREMENT_GATE_ENV)


def checker_capable() -> bool:
    """Whether the resolved checker model is strong enough to run the gate.

    A cloud provider, or an ollama '-cloud' model, counts as capable; a
    locally-installed ollama model does not (a weak checker false-blocks correct
    output). Returns False when the checker cannot be resolved.
    """
    try:
        from rune.config import get_config
        from rune.llm.client import get_llm_client
        from rune.types import ModelTier

        cfg = get_config().llm
        provider = (getattr(cfg, "active_provider", None)
                    or getattr(cfg, "default_provider", "") or "").lower()
        model = str(get_llm_client().resolve_model(ModelTier.BEST)).lower()
    except Exception as exc:
        log.warning("requirement_gate_checker_resolve_failed", error=str(exc)[:100])
        return False
    if provider == "ollama":
        return "-cloud" in model  # ollama cloud = strong; local install = weak
    return bool(provider)  # any cloud provider is strong enough


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _content_of(response: object) -> str:
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "") or ""
        return ""
    try:
        return response.choices[0].message.content or ""  # type: ignore[attr-defined]
    except (AttributeError, IndexError):
        return ""


def escalation_judge() -> tuple[object, str | None] | None:
    """Resolve the configured independent judge ``(provider, model)`` from
    ``escalationProvider`` / ``escalationModel``, or ``None`` when not set or the
    provider name is invalid. The escalation model is used as the verifier when
    the active checker is weak or its call fails."""
    try:
        from rune.config import get_config
        from rune.types import Provider

        cfg = get_config().llm
        name = (getattr(cfg, "escalation_provider", None) or "").strip().lower()
        if not name:
            return None
        provider = Provider(name)
    except Exception as exc:
        log.warning("requirement_gate_escalation_resolve_failed", error=str(exc)[:100])
        return None
    model = (getattr(get_config().llm, "escalation_model", None) or "").strip() or None
    return provider, model


async def _completion(
    system: str,
    user: str,
    max_tokens: int,
    judge: tuple[object, str | None] | None = None,
) -> str | None:
    """Run one best-tier completion. With ``judge`` set, route to that
    ``(provider, model)`` (the independent escalation judge) instead of the active
    provider. Returns the text, or None on failure."""
    try:
        from rune.llm.client import get_llm_client
        from rune.types import ModelTier

        provider = judge[0] if judge else None
        model = judge[1] if judge else None
        response = await get_llm_client().completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tier=ModelTier.BEST,
            provider=provider,  # type: ignore[arg-type]
            model=model,
            max_tokens=max_tokens,
            timeout=30.0,
        )
    except Exception as exc:  # never crash finalize on a checker failure
        log.warning("requirement_gate_llm_failed", error=str(exc)[:120])
        return None
    return _content_of(response)


async def extract_requirements(
    request: str, judge: tuple[object, str | None] | None = None
) -> list[str] | None:
    """Turn the user request into a checklist of checkable requirements.

    Returns the list (possibly empty), or ``None`` when extraction could not be
    done (call/parse failure). ``None`` and ``[]`` both mean "do not block".
    """
    text = await _completion(_EXTRACT_SYSTEM, f"Task request:\n{request}", 500, judge)
    if text is None:
        return None
    try:
        parsed = json.loads(_strip_fences(text))
    except (ValueError, TypeError):
        log.info("requirement_gate_extract_unparseable")
        return None
    if not isinstance(parsed, list):
        return None
    items = [str(x).strip() for x in parsed if str(x).strip()][:_MAX_CHECKLIST_ITEMS]
    return items


async def check_adherence(
    checklist: list[str], artifact: str,
    judge: tuple[object, str | None] | None = None,
) -> tuple[str, str | None]:
    """Judge the artifact against the checklist.

    Returns ``("pass"|"fail"|"skip", message)``:
    - ``"pass"`` - no requirement is clearly unmet.
    - ``"fail"`` - at least one requirement is clearly unmet; ``message`` lists them.
    - ``"skip"`` - inconclusive (call/parse failure); never blocks.
    """
    if not checklist:
        return "skip", None
    user = (
        "REQUIREMENTS:\n"
        + "\n".join(f"- {c}" for c in checklist)
        + "\n\nPRODUCED OUTPUT:\n"
        + (artifact or "(empty)")[:_MAX_ARTIFACT_CHARS]
    )
    text = await _completion(_CHECK_SYSTEM, user, 500, judge)
    if text is None:
        return "skip", None
    try:
        parsed = json.loads(_strip_fences(text))
    except (ValueError, TypeError):
        log.info("requirement_gate_check_unparseable")
        return "skip", None
    unmet_raw = parsed.get("unmet") if isinstance(parsed, dict) else None
    if not isinstance(unmet_raw, list):
        return "skip", None
    unmet = [str(x).strip() for x in unmet_raw if str(x).strip()]
    if not unmet:
        log.info("requirement_gate_pass", checklist=len(checklist))
        return "pass", None
    log.info("requirement_gate_fail", unmet=len(unmet), checklist=len(checklist))
    return "fail", build_block_message(unmet)


def build_block_message(unmet: list[str]) -> str:
    return (
        "[Requirement Gate] Your output does not yet satisfy every requirement the "
        "user stated. Do not finalize. Address each of these before finishing:\n"
        + "\n".join(f"- {u}" for u in unmet)
    )


class RequirementGate:
    """Holds the requirement checklist for one task run.

    The checklist is extracted once (on first ``verdict``) from the original
    request and cached in Python state, so it is immune to context compaction.
    """

    def __init__(self, request: str) -> None:
        self._request = request
        self._checklist: list[str] | None = None
        self._extracted = False

    async def _ensure_checklist(
        self, judge: tuple[object, str | None] | None
    ) -> None:
        """Extract the checklist once with ``judge``. Caches only on success, so a
        failed extraction on the active provider can be retried by the escalation
        judge on the next call."""
        if self._extracted:
            return
        self._checklist = await extract_requirements(self._request, judge)
        if self._checklist is None:
            return  # extraction failed; leave unextracted so a fallback can retry
        self._extracted = True
        if self._checklist:
            log.info("requirement_gate_checklist", n=len(self._checklist))

    async def _verdict_with(
        self, artifact: str, judge: tuple[object, str | None] | None
    ) -> tuple[str, str | None]:
        """Run extract + check with one judge. ``"skip"`` here means the call was
        inconclusive (the caller may try another judge)."""
        await self._ensure_checklist(judge)
        if self._checklist is None:
            return "skip", None  # extraction call failed
        if not self._checklist:
            return "pass", None  # no checkable requirements -> nothing to block on
        return await check_adherence(self._checklist, artifact, judge)

    async def verdict(self, artifact: str) -> tuple[str, str | None]:
        """``("pass"|"fail"|"skip", message)``.

        Tries the active provider when its checker is strong enough, then routes
        to the configured escalation judge if the active check is unavailable or
        its call fails. ``"skip"`` only when no capable judge can be reached -
        never a false block. A checker failure no longer silently passes the
        output: it falls through to the independent judge."""
        if checker_capable():
            state, msg = await self._verdict_with(artifact, None)
            if state != "skip":
                return state, msg
            log.info("requirement_gate_active_check_failed")
        else:
            log.info("requirement_gate_weak_active_checker")

        judge = escalation_judge()
        if judge is None:
            log.info("requirement_gate_no_independent_judge")
            return "skip", None
        log.info("requirement_gate_escalate_judge", provider=str(judge[0]))
        return await self._verdict_with(artifact, judge)
