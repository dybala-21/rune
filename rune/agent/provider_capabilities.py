"""Small tool-use reminders for providers that need them."""

from __future__ import annotations

# Provider-specific prompt supplements (appended to system prompt)
PROVIDER_SUPPLEMENT: dict[str, str] = {
    "openai": (
        "\n## Tool Usage\n"
        "For requested changes, inspect the relevant context, then use the tools to implement and verify. "
        "A read or review request does not authorize edits.\n"
    ),
    "gemini": (
        "\n## Tool Usage\n"
        "Batch independent reads when useful. Wait for prerequisites before dependent calls, "
        "and keep changes to the same resource sequential. "
        "Implement requested changes with tools; preserve files for read or review requests.\n"
    ),
}


def get_prompt_supplement(model: str) -> str:
    """Return a provider supplement for a model name, or an empty string when unnecessary."""
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
