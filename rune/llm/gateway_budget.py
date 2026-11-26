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
from datetime import UTC, datetime, tzinfo

REASON = "gateway_budget_exhausted"
# A background run's status when its model calls were refused: a long task waits for `resume_at`.
STATUS = "model_budget"
CODES = ("budget_exhausted", "host_budget_exhausted")
# In an error's text, its JSON body as is or encoded once more (\\"code\\").
_CODE = re.compile(r'\\?"code\\?"\s*:\s*\\?"((?:host_)?budget_exhausted)\\?"')


def owner_zone() -> tzinfo:
    """The zone the owner's schedules are read in (the briefing's too); the machine's without one."""
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    from rune.daemon.external_schedule import schedule_timezone

    try:
        name = schedule_timezone()
        return ZoneInfo(name) if name else (datetime.now().astimezone().tzinfo or UTC)
    except (ZoneInfoNotFoundError, ValueError):
        return UTC


def zone_name(zone: tzinfo, moment: datetime, locale: str = "en") -> str:
    """A zone as the owner reads it: the catalog's name for it ("한국 시간"), else its abbreviation."""
    from rune import i18n

    names = dict(part.split("=", 1) for part in i18n.text("zone_names", locale).split("|") if "=" in part)
    return names.get(getattr(zone, "key", ""), "") or moment.tzname() or "UTC"


@dataclass(frozen=True)
class BudgetRefusal:
    code: str
    resumes: datetime  # the 1st of next month, 00:00 UTC

    @property
    def resume_minute(self) -> int:
        return int(self.resumes.timestamp()) // 60

    def when(self, locale: str = "en", zone: tzinfo | None = None) -> str:
        """When it starts again, in the owner's zone and words (rune/locales)."""
        from rune import i18n

        local = self.resumes.astimezone(zone or owner_zone())
        months = i18n.text("month_names", locale).split("|")
        return i18n.text("when", locale, month=local.month, month_name=months[local.month - 1]
                         if len(months) == 12 else local.month, day=local.day,
                         time=f"{local:%H:%M}", zone=zone_name(zone or owner_zone(), local, locale))

    def message(self, locale: str = "en", zone: tzinfo | None = None) -> str:
        from rune import i18n

        key = "budget.host" if self.code == "host_budget_exhausted" else "budget.owner"
        return i18n.text(key, locale, when=self.when(locale, zone))

    def waiting(self, locale: str = "en", zone: tzinfo | None = None) -> str:
        """What a long task's owner is told when its step was refused: it waits, and goes on then."""
        from rune import i18n

        return f"{self.message(locale, zone)} {i18n.text('budget.task_waits', locale)}"

    def stopped_midway(self, locale: str = "en", zone: tzinfo | None = None) -> str:
        """For a long task's step refused after it had changed something: the owner checks it first."""
        from rune import i18n

        return f"{self.message(locale, zone)} {i18n.text('budget.task_stopped', locale)}"


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
