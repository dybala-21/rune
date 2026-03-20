"""Model capabilities registry for RUNE.

Ported from src/llm/model-capabilities.ts - tracks what each model supports.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ModelCapabilities:
    supports_tools: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True
    supports_json_mode: bool = True
    max_context: int = 128_000
    max_output: int = 16_384
    supports_prompt_caching: bool = False


# Known model capabilities
_CAPABILITIES: dict[str, ModelCapabilities] = {
    # OpenAI - latest
    "gpt-5.4": ModelCapabilities(
        supports_vision=True, max_context=200_000, max_output=32_768,
    ),
    "gpt-5.4-pro": ModelCapabilities(
        supports_vision=True, max_context=200_000, max_output=32_768,
    ),
    "gpt-5-mini": ModelCapabilities(
        supports_vision=True, max_context=200_000, max_output=16_384,
    ),
    "gpt-5-nano": ModelCapabilities(
        supports_vision=True, max_context=128_000, max_output=8_192,
    ),
    "gpt-5.3-codex": ModelCapabilities(
        supports_vision=True, max_context=200_000, max_output=32_768,
    ),
    "gpt-5.2": ModelCapabilities(
        supports_vision=True, max_context=200_000, max_output=32_768,
    ),
    # OpenAI - older
    "gpt-4.1": ModelCapabilities(
        supports_vision=True, max_context=128_000, max_output=32_768,
    ),
    "gpt-4.1-mini": ModelCapabilities(
        supports_vision=True, max_context=128_000, max_output=16_384,
    ),
    "gpt-4o": ModelCapabilities(
        supports_vision=True, max_context=128_000, max_output=16_384,
    ),
    "gpt-4o-mini": ModelCapabilities(
        supports_vision=True, max_context=128_000, max_output=16_384,
    ),
    "o4-mini": ModelCapabilities(
        supports_vision=True, max_context=200_000, max_output=16_384,
    ),
    "o3-mini": ModelCapabilities(
        supports_vision=False, max_context=200_000, max_output=16_384,
    ),
    # Anthropic
    "claude-opus-4-6": ModelCapabilities(
        supports_vision=True, max_context=200_000, max_output=32_768,
        supports_prompt_caching=True,
    ),
    "claude-opus-4-5-20251101": ModelCapabilities(
        supports_vision=True, max_context=200_000, max_output=32_768,
        supports_prompt_caching=True,
    ),
    "claude-sonnet-4-5-20250929": ModelCapabilities(
        supports_vision=True, max_context=200_000, max_output=16_384,
        supports_prompt_caching=True,
    ),
    "claude-sonnet-4-20250514": ModelCapabilities(
        supports_vision=True, max_context=200_000, max_output=16_384,
        supports_prompt_caching=True,
    ),
    "claude-haiku-4-5-20251001": ModelCapabilities(
        supports_vision=True, max_context=200_000, max_output=8_192,
        supports_prompt_caching=True,
    ),
    # Gemini
    "gemini-2.5-flash": ModelCapabilities(
        supports_vision=True, max_context=1_000_000, max_output=16_384,
    ),
    # Ollama (defaults)
    "llama3.2": ModelCapabilities(
        supports_tools=True, supports_vision=False,
        max_context=128_000, max_output=4_096,
    ),
    "codellama": ModelCapabilities(
        supports_tools=False, supports_vision=False,
        max_context=16_384, max_output=4_096,
    ),
}


def get_capabilities(model: str) -> ModelCapabilities:
    """Get capabilities for a model. Returns defaults for unknown models."""
    return _CAPABILITIES.get(model, ModelCapabilities())
