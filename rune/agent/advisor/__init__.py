"""Advisor subsystem: lightweight executor consults a stronger model at
escalation points. Provider-agnostic — uses litellm directly so every
provider supported by the executor (OpenAI, Anthropic, Gemini, DeepSeek,
xAI, Ollama, ...) is usable as an advisor with a one-line tier map entry.
"""

from rune.agent.advisor.protocol import (
    AdvisorDecision,
    AdvisorRequest,
    AdvisorTrigger,
)
from rune.agent.advisor.service import AdvisorService

__all__ = [
    "AdvisorDecision",
    "AdvisorRequest",
    "AdvisorService",
    "AdvisorTrigger",
]
