"""The hosted gateway refusing a model call because the month's budget is used up.

Behind rune-gateway, an owner's calls are held to their plan's monthly budget, and every owner on
a host to the host's (402 with `error.code` `budget_exhausted` or `host_budget_exhausted`). Neither
passes by retrying or switching models: it lasts until the month ends (UTC). So the run stops with
`REASON` and the owner is told when it starts again, in place of an error; a long task's step waits
until then instead of ending the task (rune.agent.background, rune.proactive.long_task).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone

# Korea has kept +09:00 all year since 1988: no tz database needed.
_SEOUL = timezone(timedelta(hours=9))
REASON = "gateway_budget_exhausted"
# A background run's status when its model calls were refused: a long task waits for `resume_at`.
STATUS = "model_budget"
CODES = ("budget_exhausted", "host_budget_exhausted")
# In an error's text, its JSON body as is or encoded once more (\"code\").
_CODE = re.compile(r'\\?"code\\?"\s*:\s*\\?"((?:host_)?budget_exhausted)\\?"')


@dataclass(frozen=True)
class BudgetRefusal:
    code: str
    resumes: datetime  # the 1st of next month, 00:00 UTC

    @property
    def resume_minute(self) -> int:
        return int(self.resumes.timestamp()) // 60

    def stopped_midway(self, locale: str = "en") -> str:
        """For a long task's step refused after it had done things: the owner checks them first."""
        when = self.message(locale)
        if locale.split("-", 1)[0] == "ko":
            return (f"{when} 이번 단계는 하던 중에 멈췄어요. 앱에서 한 일을 확인하고 확인 표시를 한 뒤, "
                    "그때 다시 켜면 거기서부터 이어서 해요.")
        return (f"{when} This step stopped part way through. Check what it did in the app and mark it "
                "checked, then turn the task on again after that to go on from there.")

    def waiting(self, locale: str = "en") -> str:
        """What a long task's owner is told when its step was refused: it waits, and goes on then."""
        if locale.split("-", 1)[0] == "ko":
            return f"{self.message(locale)} 이 작업은 그때까지 기다렸다가 멈춘 곳부터 이어서 해요."
        return f"{self.message(locale)} This task waits until then and goes on from where it was."

    def message(self, locale: str = "en") -> str:
        if locale.split("-", 1)[0] == "ko":
            local = self.resumes.astimezone(_SEOUL)
            when = f"{local.month}월 {local.day}일 {local.hour:02d}:{local.minute:02d}"
            if self.code == "host_budget_exhausted":
                return f"이 서버의 이번 달 전체 모델 사용 한도에 도달해서 지금은 답할 수 없어요. {when}(한국 시간)부터 다시 쓸 수 있어요."
            return f"이번 달 모델 사용 예산을 다 써서 지금은 답할 수 없어요. {when}(한국 시간)부터 다시 쓸 수 있어요."
        when = f"{self.resumes:%B} 1 (UTC)"
        if self.code == "host_budget_exhausted":
            return f"This server has reached its shared model budget for the month, so I can't answer right now. It starts again on {when}."
        return f"You've used this month's budget for your Rune's models, so I can't answer right now. It starts again on {when}."



def next_month(now: datetime | None = None) -> datetime:
    now = (now or datetime.now(UTC)).astimezone(UTC)
    return datetime(now.year + now.month // 12, now.month % 12 + 1, 1, tzinfo=UTC)


def _chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and all(current is not seen for seen in chain):
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _code(error: BaseException) -> str | None:
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        response = getattr(error, "response", None)
        try:
            body = response.json() if response is not None else None
        except Exception:
            body = None
    if isinstance(body, dict):
        inner = body.get("error", body)
        code = inner.get("code") if isinstance(inner, dict) else None
        if code in CODES:
            return code
    found = _CODE.search(str(error))
    return found.group(1) if found else None


def refusal(exc: BaseException, now: datetime | None = None) -> BudgetRefusal | None:
    """The gateway's budget refusal behind `exc`, or None (no gateway, or some other failure)."""
    from rune.llm.gateway import gateway_active

    if not gateway_active():
        return None
    # Only the gateway's own code: it passes a provider's answers on as they are, and a provider's
    # 402 (its account out of credit) is the operator's to fix, not the owner's month.
    code = next((c for c in map(_code, _chain(exc)) if c), None)
    return BudgetRefusal(code, next_month(now)) if code else None
