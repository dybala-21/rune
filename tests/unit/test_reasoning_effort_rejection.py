"""A model can accept reasoning and accept tools, yet reject both together.

gpt-5.6-sol does exactly that on /v1/chat/completions, and litellm's capability
DB has no way to say so: it reports the model reasons, we attach the parameter,
and every agent step — which always carries tools — dies on a 400. The failover
then retried the identical request three times and tripped the circuit breaker,
so the run ended with no answer at all.

The temperature path already solved this shape (detect, drop, retry, remember);
these tests pin the same loop for reasoning_effort.
"""

from __future__ import annotations

import pytest

from rune.agent import model_traits as mt

REAL_ERROR = (
    "litellm.BadRequestError: OpenAIException - Function tools with "
    "reasoning_effort are not supported for gpt-5.6-sol in "
    "/v1/chat/completions. To use function tools, use /v1/responses "
    "or set reasoning_effort to none."
)


@pytest.fixture(autouse=True)
def _clear_learned_state(monkeypatch):
    from rune.llm.reasoning import ReasoningControl

    monkeypatch.setattr("rune.llm.reasoning.reasoning_control", lambda model: ReasoningControl(("high",)))
    mt._REASONING_EFFORT_REJECTED.clear()
    mt._TEMPERATURE_REJECTED.clear()
    yield
    mt._REASONING_EFFORT_REJECTED.clear()


def test_the_real_error_is_recognised():
    assert mt.is_reasoning_effort_error(Exception(REAL_ERROR)) is True


@pytest.mark.parametrize(
    "message",
    [
        "OpenAIException - temperature is not supported for this model",
        "context_length_exceeded: maximum context is 8192 tokens",
        "429 rate limit exceeded",
        "",
    ],
)
def test_unrelated_errors_are_not_claimed(message):
    assert mt.is_reasoning_effort_error(Exception(message)) is False


def test_a_rejection_is_remembered():
    model = "gpt-5.6-sol"
    assert mt.reasoning_effort_rejected(model) is False

    mt.note_reasoning_effort_rejected(model)

    assert mt.reasoning_effort_rejected(model) is True


def test_learning_is_not_masked_by_the_capability_cache():
    """supports_reasoning_effort is lru_cached, so the check must sit outside it.

    Putting the learned rejection inside that function would be hidden by a
    True cached before the rejection ever happened.
    """
    model = "gpt-5.6-sol"
    before = mt.supports_reasoning_effort(model)

    mt.note_reasoning_effort_rejected(model)

    assert mt.supports_reasoning_effort(model) == before, (
        "the cached capability answer is expected to be unchanged"
    )
    assert mt.reasoning_effort_rejected(model) is True, (
        "the learned rejection must be readable without the cache"
    )


def test_one_model_rejecting_does_not_affect_another():
    mt.note_reasoning_effort_rejected("gpt-5.6-sol")

    assert mt.reasoning_effort_rejected("o3-mini") is False


# ---------------------------------------------------------------------------
# The retry loop itself: a rejected parameter must be dropped and the call
# retried, and a request carrying two bad parameters must not be abandoned
# after dropping only the first.
# ---------------------------------------------------------------------------

class _BadRequestError(Exception):
    pass


class _FakeLiteLLM:
    """Rejects the named kwargs, one complaint per call, like a real provider."""

    BadRequestError = _BadRequestError

    def __init__(self, rejects: set[str]):
        self._rejects = rejects
        self.calls: list[dict] = []

    async def acompletion(self, **kwargs):
        self.calls.append(dict(kwargs))
        for param in ("temperature", "reasoning_effort"):
            if param in self._rejects and param in kwargs:
                # Deliberately not the "/v1/chat/completions ... use
                # /v1/responses" wording: that refusal means the endpoint is
                # wrong, not the parameter, and dropping anything cannot fix
                # it. These tests are about the drop-and-retry path.
                raise _BadRequestError(
                    f"OpenAIException - Unsupported parameter: '{param}' is "
                    f"not supported with this model."
                )
        return "stream"

    async def aresponses(self, **kwargs):  # pragma: no cover - not this path
        raise AssertionError("a dropped parameter must not change endpoint")


async def _drive(fake, kwargs, model="fake-model"):
    """Drive the real adapter helper, not a copy of it."""
    from rune.llm.request_params import compatible_completion

    return await compatible_completion(fake.acompletion, fake.BadRequestError, kwargs)


@pytest.mark.asyncio
async def test_a_rejected_parameter_is_dropped_and_the_call_succeeds():
    fake = _FakeLiteLLM(rejects={"reasoning_effort"})

    result = await _drive(fake, {"model": "fake-model", "reasoning_effort": "high"})

    assert result == "stream"
    assert len(fake.calls) == 2, "expected exactly one retry"
    assert "reasoning_effort" not in fake.calls[1]
    assert mt.reasoning_effort_rejected("fake-model") is True


@pytest.mark.asyncio
async def test_two_rejected_parameters_are_both_dropped():
    """The single-parameter version gave up here, raising on the second 400."""
    fake = _FakeLiteLLM(rejects={"temperature", "reasoning_effort"})

    result = await _drive(
        fake,
        {"model": "fake-model", "temperature": 0.7, "reasoning_effort": "high"},
    )

    assert result == "stream"
    assert len(fake.calls) == 3
    assert "temperature" not in fake.calls[2]
    assert "reasoning_effort" not in fake.calls[2]


@pytest.mark.asyncio
async def test_an_unrelated_bad_request_is_not_swallowed():
    class _Unrelated(_FakeLiteLLM):
        async def acompletion(self, **kwargs):
            self.calls.append(dict(kwargs))
            raise _BadRequestError("OpenAIException - model not found")

        async def aresponses(self, **kwargs):  # pragma: no cover
            raise AssertionError("an unknown 400 must not change endpoint")

    fake = _Unrelated(rejects=set())

    with pytest.raises(_BadRequestError):
        await _drive(fake, {"model": "fake-model", "temperature": 0.7})

    assert len(fake.calls) == 1, "an unknown 400 must not be retried"
