"""Goal crystallization for the ``/goal`` command.

Turns a freeform request into an immutable, testable :class:`GoalSpec` that
gives the outer loop a concrete exit condition. Uses the LLM only, is
language-agnostic, and uses no regex or per-language rules. When the request
is too vague to derive verifiable criteria it returns ``ambiguous=True`` with
clarification questions; the command layer decides whether to ask them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from rune.agent.goal_loop import GoalSpec
from rune.utils.logger import get_logger

log = get_logger(__name__)


class LLMLike(Protocol):
    """Subset of :class:`rune.llm.client.LLMClient` used here."""

    async def completion(
        self, *, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Any: ...


@dataclass(slots=True)
class CrystallizeResult:
    spec: GoalSpec
    ambiguous: bool = False
    clarifications: list[str] = field(default_factory=list)
    notice: str = ""


_SYSTEM_PROMPT = """\
You convert a user's request into an immutable, testable specification for an \
autonomous coding loop. Respond in the user's language for free-text fields.

Produce ONLY a JSON object with these keys:
- "goal": one concise sentence restating the objective.
- "acceptance_criteria": list of objectively verifiable, observable criteria \
(a command passing, a file/endpoint existing, a measurable property). NOT vague \
intentions. These define when the loop is allowed to stop.
- "constraints": list of hard constraints that must hold (tech, scope, \
must-not-do). May be empty.
- "validation_commands": list of concrete, non-interactive shell commands that \
deterministically verify the acceptance criteria (tests, lint, build, type \
check). Empty list if none can be determined; never invent commands.
- "ambiguous": true if the request is too vague/underspecified to derive \
testable acceptance criteria.
- "clarifications": when ambiguous, 1-3 specific questions whose answers would \
make the spec testable. Otherwise [].
- "notice": one short sentence summarizing the derived spec for the user.

If you cannot derive at least one objectively verifiable acceptance criterion, \
set ambiguous=true and keep acceptance_criteria conservative.
"""


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [s.strip() for s in value if isinstance(s, str) and s.strip()]


def _extract_text(response: Any) -> str:
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "") or ""
    try:
        return response.choices[0].message.content or ""  # type: ignore[union-attr]
    except (AttributeError, IndexError):
        return ""


def _fallback(request: str, why: str) -> CrystallizeResult:
    """Conservative result so a vague goal is not accepted as verified."""
    return CrystallizeResult(
        spec=GoalSpec(goal=request.strip() or "(empty request)"),
        ambiguous=True,
        clarifications=[
            "The request is too vague to derive testable acceptance criteria. "
            "What concretely must be true for this to be done (a command that "
            "passes, a file/behavior that exists)?"
        ],
        notice=f"crystallization fell back to raw request ({why})",
    )


async def crystallize_goal(
    request: str,
    *,
    llm: LLMLike | None = None,
    tier: str = "fast",
) -> CrystallizeResult:
    """Crystallize *request* into a :class:`GoalSpec`.

    Reuses the shared LLM client (``rune.llm.client``) by default; an injected
    *llm* is used as-is (keeps this unit-testable without a provider).
    """
    request = (request or "").strip()
    if not request:
        return _fallback("", "empty request")

    try:
        if llm is None:
            from rune.llm.client import get_llm_client

            llm = get_llm_client()

        from rune.utils.fast_serde import json_decode

        response = await llm.completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": request},
            ],
            tier=tier,  # type: ignore[call-arg]
            # Some fast-tier models spend the token budget on hidden reasoning
            # and return empty visible text at small caps. The spec JSON is
            # sizeable, so a 1024 cap can yield "" and a false ambiguous.
            max_tokens=4096,
            timeout=30.0,
        )

        text = _extract_text(response).strip()
        if not text:
            raise ValueError("empty LLM response")
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        data = json_decode(text)
        if not isinstance(data, dict):
            raise ValueError("response is not a JSON object")

        goal = str(data.get("goal") or request).strip() or request
        acceptance = _str_list(data.get("acceptance_criteria"))
        constraints = _str_list(data.get("constraints"))
        commands = _str_list(data.get("validation_commands"))
        clarifications = _str_list(data.get("clarifications"))
        notice = str(data.get("notice") or "").strip()
        ambiguous = bool(data.get("ambiguous", False))

        # With no acceptance criteria the loop would accept on the inner gate
        # alone, so mark it ambiguous and let the command layer clarify.
        if not acceptance:
            ambiguous = True
            if not clarifications:
                clarifications = [
                    "No verifiable acceptance criteria could be derived. "
                    "What must be objectively true for this to be done?"
                ]

        base_notice = notice or f"Derived {len(acceptance)} acceptance criteria."
        if acceptance and not commands:
            base_notice = (
                "[no machine validation; verification relies on adversarial "
                f"review] {base_notice}"
            )
        return CrystallizeResult(
            spec=GoalSpec(
                goal=goal,
                acceptance_criteria=acceptance,
                constraints=constraints,
                validation_commands=commands,
            ),
            ambiguous=ambiguous,
            clarifications=clarifications,
            notice=base_notice,
        )

    except Exception as exc:
        log.debug("crystallize_fallback", error=str(exc)[:200])
        return _fallback(request, type(exc).__name__)
