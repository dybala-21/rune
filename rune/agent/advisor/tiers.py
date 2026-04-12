"""Capability tier map for advisor pairing validation.

Provider-agnostic: adding a new model = one entry. Matching is by
(provider, model_prefix) so "claude-opus-4-6" matches the "claude-opus"
entry regardless of version suffix.

Rule: tier(advisor) >= tier(executor) + MIN_TIER_GAP. If violated, the
episode disables the advisor with a WARN log rather than silently
running with a weaker advisor.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_TIER_GAP = 10
UNKNOWN_TIER = 50

ADVISOR_TIER: dict[tuple[str, str], int] = {
    # Anthropic
    ("anthropic", "claude-opus"):      100,
    ("anthropic", "claude-sonnet"):     70,
    ("anthropic", "claude-haiku"):      40,
    # OpenAI
    ("openai", "o3"):                  100,
    ("openai", "o1"):                   95,
    ("openai", "gpt-5.4"):              92,
    ("openai", "gpt-5"):                90,
    ("openai", "gpt-4o"):               75,
    ("openai", "gpt-5-mini"):           50,
    ("openai", "gpt-4o-mini"):          45,
    # Google
    ("gemini", "gemini-2.5-pro"):       85,
    ("gemini", "gemini-2.5-flash"):     45,
    ("gemini", "gemini-2.0-flash-lite"): 30,
    ("gemini", "gemini-2.0-flash"):     45,
    ("gemini", "gemini-1.5-pro"):       70,
    ("gemini", "gemini-1.5-flash"):     40,
    # DeepSeek
    ("deepseek", "deepseek-reasoner"):  90,
    ("deepseek", "deepseek-chat"):      55,
    # xAI
    ("xai", "grok-3"):                  80,
    ("xai", "grok-2"):                  70,
    # Ollama / local — tier approximates real-world judgement capability
    ("ollama", "qwen3-coder:480b"):     95,
    ("ollama", "deepseek-v3.1:671b"):   95,
    ("ollama", "gpt-oss:120b"):         80,
    ("ollama", "qwen2.5-qwq"):          75,
    ("ollama", "qwen2.5:72b"):          70,
    ("ollama", "deepseek-r1:14b"):      65,
    ("ollama", "llama3.3:70b"):         65,
    ("ollama", "qwen2.5:32b"):          60,
    ("ollama", "gemma4:26b"):           55,
    ("ollama", "olmo-3:7b-think"):      50,
    ("ollama", "qwen2.5:14b"):          45,
    ("ollama", "llama3.1:8b"):          30,
    ("ollama", "qwen2.5:7b"):           28,
    ("ollama", "gemma4:e2b"):           20,
}


@dataclass(frozen=True, slots=True)
class PairingResult:
    ok: bool
    executor_tier: int
    advisor_tier: int
    reason: str = ""


def resolve_tier(provider: str, model: str) -> int:
    """Return the capability tier for (provider, model).

    Matches by the longest-prefix (provider, prefix) entry. Falls back
    to ``UNKNOWN_TIER`` so unknown models are treated neutrally.
    """
    provider = (provider or "").lower()
    model_lc = (model or "").lower()
    best_tier = UNKNOWN_TIER
    best_len = -1
    for (p, prefix), tier in ADVISOR_TIER.items():
        if p != provider:
            continue
        if model_lc.startswith(prefix.lower()) and len(prefix) > best_len:
            best_tier = tier
            best_len = len(prefix)
    return best_tier


def check_pairing(
    executor_provider: str,
    executor_model: str,
    advisor_provider: str,
    advisor_model: str,
    min_gap: int = MIN_TIER_GAP,
) -> PairingResult:
    """Validate that the advisor is strictly stronger than the executor."""
    exec_tier = resolve_tier(executor_provider, executor_model)
    adv_tier = resolve_tier(advisor_provider, advisor_model)
    if adv_tier < exec_tier + min_gap:
        return PairingResult(
            ok=False,
            executor_tier=exec_tier,
            advisor_tier=adv_tier,
            reason=(
                f"advisor tier {adv_tier} < executor tier {exec_tier} + "
                f"min_gap {min_gap}"
            ),
        )
    return PairingResult(ok=True, executor_tier=exec_tier, advisor_tier=adv_tier)


# Anthropic native advisor_20260301 tool pairings (Phase A).
# Per https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool
# these are the only officially supported executor→advisor combinations.
# Any other pair returns a 400 from Anthropic, so we gate client-side.
_NATIVE_ELIGIBLE_EXECUTORS = (
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
)
_NATIVE_ELIGIBLE_ADVISOR = "claude-opus-4-6"


def is_claude_native_eligible(
    executor_provider: str,
    executor_model: str,
    advisor_provider: str,
    advisor_model: str,
) -> bool:
    """Return True when the pair matches Anthropic's official advisor
    tool compatibility matrix.

    Eligibility is a hard prerequisite for sending the
    ``advisor_20260301`` tool schema + ``anthropic-beta`` header; any
    other combination falls back to RUNE's policy-driven path.
    """
    if executor_provider != "anthropic" or advisor_provider != "anthropic":
        return False
    if not any(
        executor_model.startswith(e) for e in _NATIVE_ELIGIBLE_EXECUTORS
    ):
        return False
    if not advisor_model.startswith(_NATIVE_ELIGIBLE_ADVISOR):
        return False
    return True


def extract_provider_and_model(model_str: str) -> tuple[str, str]:
    """Parse 'provider/model' or 'provider:model' or bare model names.

    Mirrors the logic already used by ``provider_capabilities.py`` so
    existing model strings work unchanged.
    """
    if not model_str:
        return "", ""
    if "/" in model_str:
        provider, _, model = model_str.partition("/")
        return provider.lower(), model
    if ":" in model_str:
        provider, _, model = model_str.partition(":")
        return provider.lower(), model
    lower = model_str.lower()
    if lower.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai", model_str
    if lower.startswith("gemini"):
        return "gemini", model_str
    if lower.startswith("claude"):
        return "anthropic", model_str
    if lower.startswith("grok"):
        return "xai", model_str
    if lower.startswith("deepseek"):
        return "deepseek", model_str
    return "", model_str
