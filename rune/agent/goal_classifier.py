"""Goal classifier for RUNE.

LLM-based classification. All user input goes directly to the LLM
classifier which natively handles all languages. No regex pre-filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(slots=True)
class ClassificationResult:
    goal_type: GoalType
    confidence: float
    tier: int  # 2 = LLM
    reason: str = ""
    # Extended classification fields (populated by prompt builder)
    is_continuation: bool = False
    is_complex_coding: bool = False
    is_multi_task: bool = False
    requires_code: bool = False
    requires_execution: bool = False
    complexity: str = "simple"  # simple / moderate / complex
    output_expectation: str = "text"  # text / file / either


# LLM classifier

_TIER2_SYSTEM_PROMPT = """\
You are a goal classifier. Given a user's request, classify it into exactly one category.

Categories:
- chat: Greetings, small talk, general questions about the assistant
- web: Web search, browsing, checking URLs, looking up information online
- research: Code/project analysis, review, assessment, finding improvements, understanding architecture, evaluating quality (read-only, no modifications)
- code_modify: Creating, editing, fixing, refactoring code or files. ANY request to create or save a file.
- execution: Running commands, tests, installing packages, building, deploying
- browser: Browser automation - clicking, filling forms, taking screenshots
- full: Complex multi-step tasks that span multiple categories

Rules:
- "build a report" or "make a chart" → full (NOT execution)
- "run tests" or "npm install" → execution
- "analyze the code" or "find improvements" or "review this project" → research
- "fix the bug in main.py" → code_modify
- ANY request mentioning file creation, code writing, or saving to a file → code_modify
- If the request combines multiple categories, choose "full"
- If the request is about evaluating, reviewing, or assessing (without making changes), choose "research"

Respond with ONLY a JSON object: {"goal_type": "<category>", "confidence": <0.0-1.0>, "reason": "<brief reason>"}
"""


async def classify_tier2(goal: str) -> ClassificationResult:
    """LLM-based classification for all input.

    Uses LiteLLM (via the fast tier model) to classify goals.
    Natively handles all languages (Korean, English, Japanese, etc.)
    """
    try:
        from rune.llm.client import get_llm_client
        from rune.utils.fast_serde import json_decode

        client = get_llm_client()
        response = await client.completion(
            messages=[
                {"role": "system", "content": _TIER2_SYSTEM_PROMPT},
                {"role": "user", "content": goal},
            ],
            tier="fast",  # type: ignore[arg-type]
            max_tokens=256,
            timeout=10.0,
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

        return ClassificationResult(
            goal_type=goal_type,  # type: ignore[arg-type]
            confidence=min(max(confidence, 0.0), 1.0),
            tier=2,
            reason=reason,
        )

    except Exception as exc:
        from rune.utils.logger import get_logger
        log = get_logger(__name__)
        log.debug("classification_fallback", error=str(exc)[:200])

        return ClassificationResult(
            goal_type="full",
            confidence=0.5,
            tier=2,
            reason=f"Fallback ({type(exc).__name__})",
        )


# Public API

async def classify_goal(goal: str) -> ClassificationResult:
    """Classify a user goal using LLM."""
    return await classify_tier2(goal)
