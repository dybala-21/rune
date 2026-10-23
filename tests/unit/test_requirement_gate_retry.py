"""A read-only check survives one transient provider failure."""

from unittest.mock import AsyncMock

import pytest

from rune.agent.requirement_gate import _completion

ANSWER = {"choices": [{"message": {"content": "ok"}}]}


@pytest.mark.parametrize("errors,expected,calls", [
    ([TimeoutError("slow")], "ok", 2),                        # transient: retried once
    ([ValueError("bad request")], None, 1),                    # not transient: no retry
    ([TimeoutError("slow"), TimeoutError("still slow")], None, 2),  # bounded to one retry
])
async def test_checker_retries_only_transient_failures_once(monkeypatch, errors, expected, calls):
    client = AsyncMock()
    client.completion.side_effect = [*errors, ANSWER]
    monkeypatch.setattr("rune.llm.client.get_llm_client", lambda: client)
    assert await _completion("system", "user", 64) == expected
    assert client.completion.await_count == calls
