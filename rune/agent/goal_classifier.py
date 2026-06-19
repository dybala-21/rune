"""Goal classifier for RUNE.

LLM-based classification. All user input goes directly to the LLM
classifier which natively handles all languages. No regex pre-filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

GoalType = Literal[
    "chat",          # Small talk, general conversation
    "web",           # Web search, browsing
    "research",      # Code analysis, reading (no write)
    "code_modify",   # File edits, code generation
    "execution",     # Running commands, testing
    "browser",       # Browser automation
    "full",          # Complex multi-step tasks
]

# All valid goal types (single source of truth).
VALID_GOAL_TYPES: set[str] = {
    "chat", "web", "research", "code_modify",
    "execution", "browser", "full",
}


_KNOWN_INTENT_CATEGORIES: frozenset[str] = frozenset({"email", "document"})


@dataclass(slots=True)
class ClassificationResult:
    goal_type: GoalType
    confidence: float
    tier: int  # 2 = LLM
    reason: str = ""
    # Extended classification fields (populated by prompt builder)
    is_continuation: bool = False
    is_domain_change: bool = False  # True when goal domain differs from previous turn
    is_complex_coding: bool = False
    is_multi_task: bool = False
    requires_code: bool = False
    requires_execution: bool = False
    complexity: str = "simple"  # simple / moderate / complex
    output_expectation: str = "text"  # text / file / either
    # Orthogonal intent flags (zero or more apply, language-agnostic).
    # Drives conditional inclusion of PROMPT_EMAIL_WORKFLOW / PROMPT_DOCUMENT
    # in agent/prompts.py:build_system_prompt.
    intent_categories: frozenset[str] = field(default_factory=frozenset)


# LLM classifier

_CLASSIFICATION_CATEGORIES = """\
- chat: Greetings, small talk, general questions about the assistant
- web: Web search, browsing, checking URLs, looking up information online
- research: Code/project analysis, review, assessment, finding improvements, understanding architecture, evaluating quality (read-only, no modifications)
- code_modify: Creating, editing, fixing, refactoring code or files. ANY request to create or save a file.
- execution: Running commands, tests, installing packages, building, deploying
- browser: Browser automation - clicking, filling forms, taking screenshots
- full: Complex multi-step tasks that span multiple categories"""

_CLASSIFICATION_RULES = """\
- Report/chart generation → full (NOT execution)
- Running commands, tests, installing packages → execution
- Analyzing, reviewing, assessing code (read-only) → research
- Fixing, editing, creating code or files → code_modify
- If the request combines multiple categories → full
- If evaluating without making changes → research"""

_INTENT_FLAGS = """\
Intent flags (default: empty list. Only set a flag when the goal explicitly mentions it. Detect across all languages.):
- email: ONLY when the goal is about email itself — sending mail, reading inbox, replying, drafting an email message. Examples: "check my inbox", "send a mail to X", "メールを書いて", "回复邮件". NOT for: writing a report, generating a file.
- document: ONLY when the goal is about producing a standalone document — report, proposal, business plan, formal write-up. Examples: "write a project report", "기획서 작성", "報告書を書いて". NOT for: sending an email, code generation.

If the goal combines both (e.g. "email the report"), set both. If neither clearly applies, return []."""

_INTENT_JSON_FIELD = '"intent_categories": [<zero or more of: "email", "document">]'

_REQUIRES_EXECUTION_FLAG = """\
requires_execution: true ONLY when verifying this output's correctness requires \
RUNNING code, tests, or commands (e.g. fix a bug and make tests pass, build or \
run a program, execute a script and check its result). false for prose, \
analysis, research, reports, plans, or documents whose correctness is judged by \
reading them. When in doubt, choose false."""

_REQUIRES_EXECUTION_JSON_FIELD = '"requires_execution": true/false'

_TIER2_SYSTEM_PROMPT = f"""\
You are a goal classifier. Given a user's request, classify it into exactly one category and detect any applicable intent flags.

Categories:
{_CLASSIFICATION_CATEGORIES}

{_INTENT_FLAGS}

Rules:
{_CLASSIFICATION_RULES}

{_REQUIRES_EXECUTION_FLAG}

Respond with ONLY a JSON object: {{"goal_type": "<category>", "confidence": <0.0-1.0>, "reason": "<brief reason>", {_REQUIRES_EXECUTION_JSON_FIELD}, {_INTENT_JSON_FIELD}}}
"""

_TIER2_SYSTEM_PROMPT_WITH_PREVIOUS = f"""\
You are a goal classifier. Given a user's request and the previous request context, classify the current request, detect any applicable intent flags, and determine if it is related to the previous one.

Categories:
{_CLASSIFICATION_CATEGORIES}

{_INTENT_FLAGS}

Rules:
{_CLASSIFICATION_RULES}

{_REQUIRES_EXECUTION_FLAG}

is_related_to_previous rules:
- true: the current request references, continues, or builds on the previous one
- false: the current request is about a completely different topic or task
- When in doubt, choose false

Respond with ONLY a JSON object: {{"goal_type": "<category>", "confidence": <0.0-1.0>, "reason": "<brief reason>", {_REQUIRES_EXECUTION_JSON_FIELD}, "is_related_to_previous": true/false, {_INTENT_JSON_FIELD}}}
"""


async def classify_tier2(
    goal: str,
    *,
    previous_goal: str = "",
    previous_goal_type: str = "",
) -> ClassificationResult:
    """LLM-based classification for all input.

    Uses LiteLLM (via the fast tier model) to classify goals.
    Natively handles all languages (Korean, English, Japanese, etc.)

    When *previous_goal* is provided, also determines whether the
    current goal is related to the previous one (domain change detection).
    """
    has_previous = bool(previous_goal and previous_goal_type)

    try:
        from rune.llm.client import get_llm_client
        from rune.utils.fast_serde import json_decode

        client = get_llm_client()

        if has_previous:
            system_prompt = _TIER2_SYSTEM_PROMPT_WITH_PREVIOUS
            user_content = (
                f"Previous request ({previous_goal_type}): {previous_goal[:200]}\n\n"
                f"Current request: {goal}"
            )
        else:
            system_prompt = _TIER2_SYSTEM_PROMPT
            user_content = goal

        tier = "fast"

        # max_tokens budget covers both visible JSON output (~80 tokens) and
        # any hidden reasoning the FAST-tier model uses. Reasoning models
        # (gpt-5-mini, etc.) can otherwise burn the budget on reasoning
        # alone and return an empty visible response, breaking parsing.
        response = await client.completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            tier=tier,  # type: ignore[arg-type]
            max_tokens=1024,
            timeout=15.0,
        )

        # Extract text from LiteLLM response
        text = ""
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")
        else:
            try:
                text = response.choices[0].message.content  # type: ignore[union-attr]
            except (AttributeError, IndexError):
                pass

        if not text:
            raise ValueError("Empty LLM response")

        # Parse JSON from the response (handle markdown fences)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        data = json_decode(text)

        goal_type = data.get("goal_type", "full")
        if goal_type not in VALID_GOAL_TYPES:
            goal_type = "full"

        confidence = float(data.get("confidence", 0.7))
        reason = str(data.get("reason", "LLM classification"))

        # Domain change detection
        is_domain_change = False
        if has_previous:
            is_related = data.get("is_related_to_previous", True)
            if not is_related:
                is_domain_change = True
            elif goal_type != previous_goal_type:
                # Different goal_type but LLM says related — trust LLM
                is_domain_change = False

        # Derive is_complex_coding from goal_type so multi-step coding
        # tasks get the larger tool-round and advisor budgets.
        is_complex_coding = goal_type in ("code_modify", "full")

        # Whether the output's correctness is established by running code/tests.
        # Defaults False on a missing/unparseable value (safe direction: the
        # requirement gate then still runs rather than being silently skipped).
        requires_execution = bool(data.get("requires_execution", False))

        # Validate intent_categories: keep only known values, ignore typos / hallucinated tags.
        raw_intents = data.get("intent_categories") or []
        if not isinstance(raw_intents, list):
            raw_intents = []
        intent_categories = frozenset(
            str(c).strip().lower()
            for c in raw_intents
            if isinstance(c, str) and str(c).strip().lower() in _KNOWN_INTENT_CATEGORIES
        )

        return ClassificationResult(
            goal_type=goal_type,  # type: ignore[arg-type]
            confidence=min(max(confidence, 0.0), 1.0),
            tier=2,
            reason=reason,
            is_domain_change=is_domain_change,
            is_complex_coding=is_complex_coding,
            requires_execution=requires_execution,
            intent_categories=intent_categories,
        )

    except Exception as exc:
        from rune.utils.logger import get_logger
        log = get_logger(__name__)
        log.debug("classification_fallback", error=str(exc)[:200])

        # Protective default: when classification fails we don't know the
        # intent, so include all known prompt sections rather than risk
        # missing email/document guidance.
        return ClassificationResult(
            goal_type="full",
            confidence=0.5,
            tier=2,
            reason=f"Fallback ({type(exc).__name__})",
            intent_categories=_KNOWN_INTENT_CATEGORIES,
        )


# Public API

async def classify_goal(
    goal: str,
    *,
    previous_goal: str = "",
    previous_goal_type: str = "",
) -> ClassificationResult:
    """Classify a user goal using LLM.

    When *previous_goal* and *previous_goal_type* are provided,
    also detects domain changes to prevent context bleed between
    unrelated turns in multi-turn conversations.
    """
    return await classify_tier2(
        goal,
        previous_goal=previous_goal,
        previous_goal_type=previous_goal_type,
    )
