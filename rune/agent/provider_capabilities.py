"""Provider-specific prompt supplements for multi-provider compatibility.

Adds minimal guidance to help non-Claude models use tools correctly.
Claude's base prompt is already optimized — no supplement needed.
"""

from __future__ import annotations

# Provider-specific prompt supplements (appended to system prompt)
PROVIDER_SUPPLEMENT: dict[str, str] = {
    "openai": (
        "\n## Tool Usage\n"
        "When asked to modify code, call file_edit immediately.\n"
        "Example: user says 'fix bug in main.py'\n"
        "  1. file_read(path='main.py')\n"
        "  2. file_edit(path='main.py', search='...', replace='...')\n"
        "Never describe what you will do — call the tool.\n"
    ),
    "gemini": (
        "\n## Tool Usage\n"
        "Call one tool at a time. Wait for the result before the next call.\n"
        "After reading a file, proceed to edit it — do not just describe changes.\n"
    ),
}


def get_prompt_supplement(model: str) -> str:
    """Get provider-specific prompt supplement from a model string.

    Accepts 'anthropic/claude-...' or 'gpt-4o' format.
    Returns empty string for unknown providers or providers that don't need supplements.
    """
    provider = _extract_provider(model)
    return PROVIDER_SUPPLEMENT.get(provider, "")


def _extract_provider(model: str) -> str:
    """Extract provider name from model string."""
    if "/" in model:
        return model.split("/", 1)[0]
    # OpenAI models have no prefix in LiteLLM
    lower = model.lower()
    if lower.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if lower.startswith(("gemini",)):
        return "gemini"
    if lower.startswith(("claude",)):
        return "anthropic"
    if lower.startswith(("grok",)):
        return "xai"
    return ""
