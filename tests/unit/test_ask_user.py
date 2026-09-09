"""Ask-user contracts, transport payloads, and concurrent run isolation."""

import asyncio

import pytest

from rune.api.questions import PendingQuestion, question_payload
from rune.capabilities.ask_user import (
    AskUserOption,
    AskUserParams,
    ask_user,
    get_ask_user_count,
    set_ask_user_callback,
    user_response,
)
from rune.utils.fast_serde import json_decode, json_encode


def question(text="어떤 정보를 원하시나요?"):
    return AskUserParams(question=text, reason="범위 확인", options=[
        AskUserOption(label="개념 설명", description="ESTABLISHED 의미"),
        AskUserOption(label="시스템 확인", description="현재 연결 확인"),
    ])


def test_payload_is_serializable_without_leaking_internal_reason():
    payload = json_decode(json_encode(question_payload(question(), "q1", "run1", "call1")))
    assert payload == {"id": "q1", "runId": "run1", "callId": "call1",
                       "question": "어떤 정보를 원하시나요?",
                       "options": [{"label": "개념 설명", "description": "ESTABLISHED 의미"},
                                   {"label": "시스템 확인", "description": "현재 연결 확인"}]}


@pytest.mark.parametrize("answer,index,expected,free_text", [
    ("untrusted label", 1, "시스템 확인", False),
    ("내 서버의 연결을 확인해 줘", None, "내 서버의 연결을 확인해 줘", True),
    ("", -1, "", True),
])
async def test_selection_and_free_text(answer, index, expected, free_text):
    pending = PendingQuestion(question())
    assert pending.resolve(answer, index)
    response = await pending.future
    assert response.answer == expected
    assert response.raw_input == answer
    assert response.free_text is free_text
    assert not pending.resolve("duplicate", None)


@pytest.mark.parametrize("index", [-2, 2, 99, True, "1"])
async def test_bad_selection_keeps_question_pending(index):
    pending = PendingQuestion(question())
    with pytest.raises(ValueError):
        pending.resolve("bad", index)
    assert not pending.future.done()
    assert pending.resolve("개념 설명", 0)


async def test_concurrent_runs_keep_their_own_callback_and_limits():
    ready = [asyncio.Event(), asyncio.Event()]

    async def run(index):
        async def callback(params):
            await ready[1 - index].wait()
            return user_response(params, str(index))

        set_ask_user_callback(callback)
        ready[index].set()
        results = await asyncio.gather(*(ask_user(question()) for _ in range(3)))
        assert [result.success for result in results] == [True, True, False]
        assert [result.output for result in results[:2]] == [f'User responded: "{index}"'] * 2
        assert get_ask_user_count() == 2

    await asyncio.wait_for(asyncio.gather(run(0), run(1)), timeout=2)


async def test_noninteractive_run_does_not_inherit_a_callback():
    async def callback(params):
        pytest.fail("previous run's callback leaked")

    set_ask_user_callback(callback)
    set_ask_user_callback(None)
    result = await ask_user(question())
    assert not result.success
    assert "non-interactive" in result.error


async def test_cancelled_question_releases_the_session_lock():
    entered = asyncio.Event()

    async def callback(params):
        if params.question == "cancel":
            entered.set()
            await asyncio.Future()
        return user_response(params, "after")

    set_ask_user_callback(callback)
    task = asyncio.create_task(ask_user(question("cancel")))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    result = await asyncio.wait_for(ask_user(question("next")), timeout=1)
    assert result.output == 'User responded: "after"'
