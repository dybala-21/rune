"""Question payloads and response validation at the web transport boundary."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from rune.capabilities.ask_user import AskUserParams, UserResponse, user_response


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
