"""A 400 cannot be fixed by sending the identical request again.

The oil-price run died this way: the provider refused a parameter, the error
fell through to "unknown", and "unknown" retries. Three identical requests
failed, the circuit breaker counted three matching signatures and aborted the
run — so the one recovery that could have worked, switching to another
profile, was never reached.

Transient failures must keep retrying: 5xx and dropped connections are exactly
what the retry branch is for.
"""

from __future__ import annotations

import pytest

from rune.agent.failover import LLMProfile, classify_error, determine_strategy

PROFILE = LLMProfile(name="a", provider="openai", model="m1")
OTHER = LLMProfile(name="b", provider="anthropic", model="m2")


@pytest.mark.parametrize(
    "message,expected",
    [
        ("litellm.BadRequestError: OpenAIException - Function tools with "
         "reasoning_effort are not supported for gpt-5.6-sol", "bad_request"),
        ("400 BadRequest: unsupported parameter 'speed'", "invalid_request"),
        ("404 - The model 'foo' does not exist", "bad_request"),
        ("422 Unprocessable Entity: invalid parameter", "invalid_request"),
    ],
)
def test_deterministic_client_errors_are_classified(message, expected):
    assert classify_error(message) == expected, message
    strategy = determine_strategy(expected, PROFILE, 3, [PROFILE, OTHER])
    assert strategy.action not in ("retry", "compact")


@pytest.mark.parametrize(
    "message",
    [
        "500 Internal Server Error",
        "503 Service Unavailable",
        "connection reset by peer",
        "upstream connect error",
    ],
)
def test_transient_failures_stay_retryable(message):
    """These are why the retry branch exists; they must not be swept up."""
    reason = classify_error(message)
    strategy = determine_strategy(reason, PROFILE, 3, [PROFILE, OTHER])

    assert strategy.action == "retry", message


def test_a_deterministic_error_never_retries():
    reason = classify_error("400 BadRequest: reasoning_effort is not supported")

    for retries_left in (3, 2, 1, 0):
        strategy = determine_strategy(reason, PROFILE, retries_left, [PROFILE, OTHER])
        assert strategy.action != "retry", (
            f"retried a deterministic 400 with {retries_left} retries left"
        )


def test_a_deterministic_error_moves_to_another_profile():
    """Another provider may well accept what this one refused."""
    reason = classify_error("400 BadRequest: reasoning_effort is not supported")

    strategy = determine_strategy(reason, PROFILE, 3, [PROFILE, OTHER])

    assert strategy.action == "switch_profile"


def test_existing_categories_are_unchanged():
    """The new branch must not shadow what was already classified."""
    assert classify_error("401 unauthorized") == "auth"
    assert classify_error("429 rate limit exceeded") == "rate_limit"
    assert classify_error("402 billing issue") == "billing"
    assert classify_error("request timed out") == "timeout"
    assert classify_error("maximum context length exceeded") == "context_overflow"
