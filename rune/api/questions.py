"""Question payloads and response validation at the web transport boundary."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from rune.capabilities.ask_user import AskUserParams, UserResponse, user_response


def question_callback(run_id: str, pending: dict[str, PendingQuestion], emit, *, timeout: float = 300):
    from rune.agent.timing import timed

    lock = asyncio.Lock()

    @timed("question")
    async def ask(params: AskUserParams) -> UserResponse:
        from rune.agent.loop import current_tool_call_id

        async with lock:
            question_id = f"question:{run_id}:{uuid4().hex}"
            question = PendingQuestion(params)
            pending[question_id] = question
            try:
                await emit("question", {
                    **question_payload(params, question_id, run_id, current_tool_call_id()),
                    "expiresAt": time.time() * 1000 + timeout * 1000,
                })
                return await asyncio.wait_for(question.future, timeout=timeout)
            except TimeoutError:
                raise TimeoutError("Question expired without a user response") from None
            finally:
                pending.pop(question_id, None)
                if not question.future.done():
                    question.future.cancel()
                await emit("question_closed", {"id": question_id, "runId": run_id})

    return ask


def question_payload(params: AskUserParams, question_id: str, run_id: str, call_id: str = "") -> dict[str, Any]:
    return {
        "id": question_id, "question": params.question,
        "options": [option.model_dump(mode="json", exclude_none=True) for option in params.options or []],
        "runId": run_id, "callId": call_id,
    }


@dataclass
class PendingQuestion:
    params: AskUserParams
    future: asyncio.Future[UserResponse] = field(default_factory=lambda: asyncio.get_running_loop().create_future())

    def resolve(self, answer: str, selected_index: int | None) -> bool:
        if self.future.done():
            return False
        response = user_response(self.params, answer, selected_index)
        self.future.set_result(response)
        return True


class InteractionResponses:
    """Remember accepted response IDs so a lost HTTP response can be retried."""

    def __init__(self) -> None:
        self._accepted: OrderedDict[str, tuple[str, dict[str, Any]]] = OrderedDict()

    def replay(self, interaction_id: str, response_id: str, payload: dict[str, Any]) -> bool:
        if not response_id or interaction_id not in self._accepted:
            return False
        if self._accepted[interaction_id] != (response_id, payload):
            raise ValueError("This interaction already has a different response")
        return True

    def remember(self, interaction_id: str, response_id: str, payload: dict[str, Any]) -> None:
        if response_id:
            self._accepted[interaction_id] = response_id, payload
            while len(self._accepted) > 1024:
                self._accepted.popitem(last=False)
