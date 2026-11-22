"""The watch as a scheduled job: what a wake reads, what it tells, and that it always ends with a result."""

import asyncio
import json
from datetime import datetime, time, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from rune.briefing import labels as briefing_labels
from rune.capabilities import cron
from rune.memory.store import MemoryStore
from rune.proactive.routine import RoutinePolicy, should_notify
from rune.watch import job, schedule
from rune.watch.extract import Reading
from rune.watch.item import Item, Price
from rune.watch.judge import renew
from rune.watch.observe import Observation
from rune.watch.store import Store

SEOUL = ZoneInfo("Asia/Seoul")
DAY_ONE = datetime(2026, 11, 20, 9, 0, tzinfo=SEOUL)
COAT, MUG = "https://shop.example/p/coat", "https://shop.example/p/mug"


def won(low: int, high: int | None = None, currency: str = "KRW") -> Price:
    return Price(Decimal(low), Decimal(low if high is None else high), currency)


def at(day: int, hour: int = 9, minute: int = 0) -> datetime:
    return DAY_ONE.replace(hour=hour, minute=minute) + timedelta(days=day - 1)


class Pages:
    """The watched pages: what each answers when it is read, the time it is, and every read made."""

    def __init__(self) -> None:
        self.answers: dict[str, object] = {}
        self.reads: list[str] = []
        self.now = at(0)   # set up the day before: the reading a watch starts from is that day's read

    async def observe(self, reader, url, *, registering, now) -> Observation:
        self.reads.append(url)
        answer = self.answers[url]
        if callable(answer):
            answer = await answer()
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, str):
            return Observation(failure=answer, url=url, http_status=403 if answer == "blocked" else 0)
        assert isinstance(answer, Price)
        return Observation(reading=Reading(answer, 1, "jsonld"), url=url, http_status=200)

    def watch(self, url: str = COAT, price: Price | None = None, *, label: str = "", tell_when: str = "drop",
              below: int | None = None) -> Item:
        store = Store()
        try:
            return store.add(url=url, fetch_url=url, host="shop.example", label=label, tell_when=tell_when,
                             below=None if below is None else Decimal(below), price=price or won(100), now=self.now)
        finally:
            store.close()

    def kept(self, item: Item) -> Item:
        store = Store()
        try:
            found = store.get(item.id)
            assert found is not None
            return found
        finally:
            store.close()

    async def wake(self, now: datetime, **answers: Price | str) -> dict:
        """One wake at ``now``; ``coat=`` and ``mug=`` say what those pages answer from here on."""
        self.now = now
        self.answers.update({{"coat": COAT, "mug": MUG}[name]: answer for name, answer in answers.items()})
        return await job.run_watch()


@pytest.fixture
def pages(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Seoul")
    pages = Pages()
    monkeypatch.setattr(job, "observe", pages.observe)
    monkeypatch.setattr(job, "clock", lambda: pages.now)
    # A wake calls no model: one that did would fail the test here.
    monkeypatch.setattr("rune.llm.client.get_llm_client", lambda: pytest.fail("the watch called a model"))
    return pages


def quiet(result: dict) -> bool:
    return result == {"status": "completed", "success": True, "verified": True, "output": "Nothing to tell.",
                      "nothing_to_tell": True}


async def test_a_wake_with_nothing_due_asks_no_site_anything(pages, monkeypatch):
    with monkeypatch.context() as nothing_to_read_with:
        nothing_to_read_with.setattr(job, "Reader", lambda: pytest.fail("a wake with nothing due opened a client"))
        assert quiet(await pages.wake(at(1)))
        pages.watch()
        assert quiet(await pages.wake(at(2, 8, 59), coat=won(90))), "before the owner's time"
    assert pages.reads == []
    assert quiet(await pages.wake(at(2), coat=won(100))) and pages.reads == [COAT]
    assert quiet(await pages.wake(at(2, 10))) and quiet(await pages.wake(at(2, 23)))
    assert pages.reads == [COAT], "one read a day while nothing moved"


async def test_a_drop_is_told_after_its_second_reading_with_the_price_the_owner_knew(pages):
    item = pages.watch(label="울 코트")
    assert quiet(await pages.wake(at(2), coat=won(90))), "seen once: not told yet"
    assert pages.kept(item).told == won(100) and pages.kept(item).candidate == won(90)
    told = await pages.wake(at(2, 10))
    assert told["status"] == "completed" and told["success"] and told["verified"] and told["title"] == "Price watch"
    assert told["output"].splitlines() == [
        "A price you asked me to watch went down.", "",
        "울 코트 (shop.example)", "100 KRW → 90 KRW", "Last told 2026-11-19 · read 2026-11-21 10:00", "",
        "Prices are as the page writes them for a visitor who is not signed in."]
    assert told["buttons"] == [{"text": "Open the page", "url": COAT}]
    assert pages.kept(item).told == won(90) and pages.kept(item).candidate is None
    assert quiet(await pages.wake(at(2, 11))) and pages.reads == [COAT, COAT]


async def test_a_value_seen_once_and_gone_an_hour_later_is_never_told(pages):
    item = pages.watch()
    assert quiet(await pages.wake(at(2), coat=won(90)))
    assert quiet(await pages.wake(at(2, 10), coat=won(100)))
    assert pages.kept(item).told == won(100) and pages.kept(item).candidate is None


async def test_a_second_reading_that_fails_is_not_a_failed_day_and_the_next_days_read_settles_it(pages):
    item = pages.watch()
    assert quiet(await pages.wake(at(2), coat=won(90)))
    assert quiet(await pages.wake(at(2, 10), coat="server"))
    assert quiet(await pages.wake(at(2, 11))), "one second reading for a price, not one every hour"
    waiting = pages.kept(item)
    assert (waiting.candidate, waiting.failed_days, waiting.told) == (won(90), 0, won(100))
    assert "100 KRW → 90 KRW" in (await pages.wake(at(3), coat=won(90)))["output"]
    assert pages.reads == [COAT] * 3


async def test_a_page_that_stops_answering_is_said_once_on_its_second_day_without_the_old_price(pages):
    item = pages.watch()
    assert quiet(await pages.wake(at(2), coat="blocked"))
    said = await pages.wake(at(3))
    assert "Could not be read for 2 days in a row. The site refuses automated reading." in said["output"]
    assert "100" not in said["output"] and said["success"] and "nothing_to_tell" not in said
    assert quiet(await pages.wake(at(4))) and quiet(await pages.wake(at(5)))
    assert pages.reads == [COAT] * 4, "one read on a failed day, no more"
    assert quiet(await pages.wake(at(6), coat=won(100))), "back again: it goes on without a word"
    assert pages.kept(item).failed_days == 0


async def test_a_watch_ends_when_its_time_is_up_and_says_the_last_price_with_its_day(pages):
    item = pages.watch()
    assert quiet(await pages.wake(at(89), coat=won(100)))
    ended = await pages.wake(at(90))
    assert "watched for 90 days" in ended["output"] and "Last read: 100 KRW on 2027-02-17" in ended["output"]
    assert pages.kept(item).status == "ended"
    assert quiet(await pages.wake(at(91))) and len(pages.reads) == 2, "an ended watch is not read"


async def test_one_page_that_breaks_the_reading_does_not_stop_the_others(pages):
    coat, mug = pages.watch(COAT), pages.watch(MUG)
    assert quiet(await pages.wake(at(2), coat=RuntimeError("PAGE-WORDS in the error"), mug=won(90)))
    assert pages.reads == [COAT, MUG]
    assert pages.kept(coat).failed_days == 1 and pages.kept(mug).candidate == won(90)
    store = Store()
    assert [seen["failure"] for seen in store.observations(coat.id)] == ["error"]
    store.close()


async def test_one_page_whose_reading_cannot_be_kept_does_not_stop_the_others(pages, monkeypatch):
    coat, mug = pages.watch(COAT), pages.watch(MUG)
    settle = job.settle

    def breaks_on_the_coat(item, *rest):
        if item.id == coat.id:
            raise ValueError("PAGE-WORDS")
        return settle(item, *rest)

    monkeypatch.setattr(job, "settle", breaks_on_the_coat)
    assert quiet(await pages.wake(at(2), coat=won(90), mug=won(90)))
    assert pages.kept(coat).read_day == at(0).date() and pages.kept(mug).candidate == won(90)
    monkeypatch.setattr(job, "settle", settle)
    assert quiet(await pages.wake(at(2, 9, 10))) and pages.reads == [COAT, MUG, COAT], "read again at the next wake"


async def test_a_run_that_cannot_be_made_says_so_once_a_day_and_never_raises(pages, monkeypatch):
    pages.watch()
    monkeypatch.setattr(job, "Store", lambda: (_ for _ in ()).throw(OSError("PAGE-WORDS /private/path")))
    first = await pages.wake(at(2))
    assert first == {"status": "failed", "success": False, "verified": False, "output": "", "title": "Price watch",
                     "error": "The pages you asked me to watch could not be checked on 2026-11-21."}
    again = await pages.wake(at(2, 10))
    policy = RoutinePolicy.model_validate({"notify": "changes"})
    assert should_notify(policy, first, None) and not should_notify(policy, again, first)
    assert should_notify(policy, await pages.wake(at(3)), again), "the next day says it again"


async def test_pages_not_reached_before_the_deadline_wait_for_the_next_wake(pages, monkeypatch):
    coat, mug = pages.watch(COAT), pages.watch(MUG)
    monkeypatch.setattr(job, "DEADLINE", 0.05)

    async def slow() -> Price:
        await asyncio.sleep(1)
        return won(100)

    assert quiet(await pages.wake(at(2), coat=won(100), mug=slow))
    assert pages.kept(coat).read_day == at(2).date() and pages.kept(mug).read_day == at(0).date()
    assert quiet(await pages.wake(at(2, 10), mug=won(100)))
    assert pages.reads == [COAT, MUG, MUG] and pages.kept(mug).read_day == at(2).date()


async def test_what_the_owner_changed_while_its_page_was_being_read_stands(pages):
    item = pages.watch()
    assert quiet(await pages.wake(at(2), coat=won(90)))

    async def owner_changes_it() -> Price:
        store = Store()
        mine = store.get(item.id)
        renew(mine, at(2, 10), won(95), tell_when="change", below=None, label="mine")
        assert store.save(mine)
        store.close()
        return won(90)

    assert quiet(await pages.wake(at(2, 10), coat=owner_changes_it)), "the second reading is not told over their change"
    kept = pages.kept(item)
    assert (kept.told, kept.tell_when, kept.label, kept.candidate) == (won(95), "change", "mine", None)


async def test_a_closed_web_is_said_to_be_that_and_never_ends_a_watch(pages, monkeypatch):
    item = pages.watch()
    monkeypatch.setattr("rune.connectors.web_access.closed", AsyncMock(return_value={"closed": True}))
    assert quiet(await pages.wake(at(2), coat="unreachable"))
    said = await pages.wake(at(3))
    assert "This machine's web access is closed." in said["output"]
    for day in range(4, 40):
        await pages.wake(at(day))
    kept = pages.kept(item)
    assert (kept.status, kept.counted_days, kept.failed_days) == ("watching", 0, 38)
    # A site that answered, even with a refusal, was reached: that is about the page, not this machine.
    await pages.wake(at(40), coat="blocked")
    store = Store()
    assert store.observations(item.id)[0]["failure"] == "blocked" and pages.kept(item).counted_days == 1
    store.close()
    pages.answers[COAT] = "unreachable"
    monkeypatch.setattr("rune.connectors.web_access.closed", AsyncMock(return_value=None))
    for day in range(41, 54):
        result = await pages.wake(at(day))
    assert pages.kept(item).status == "ended" and "could not be read for 14 days" in result["output"]


async def test_the_message_is_worded_in_the_owners_language_from_labels_kept_beforehand(pages, monkeypatch):
    from rune.watch import labels

    korean = {**labels.ENGLISH, "title": "가격 지켜보기", "went_down": "지켜보던 가격이 내려갔어요."}
    model = SimpleNamespace(completion=AsyncMock(return_value={"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps(korean, ensure_ascii=False)}}]}))
    monkeypatch.setattr("rune.llm.client.get_llm_client", lambda: model)
    briefing_labels.set_language("ko")
    assert (await labels.load("ko"))["title"] == "가격 지켜보기"      # as watch_set does, once
    monkeypatch.setattr("rune.llm.client.get_llm_client", lambda: pytest.fail("the watch called a model"))
    pages.watch()
    await pages.wake(at(2), coat=won(90))
    told = await pages.wake(at(2, 10))
    assert told["title"] == "가격 지켜보기" and told["output"].startswith("지켜보던 가격이 내려갔어요.")
    assert "100 KRW → 90 KRW" in told["output"]


# -- Through the scheduler: the whole sequence, with no model anywhere ----------


@pytest.fixture
def scheduled(pages, tmp_path, monkeypatch):
    clock = [1_000_000.0]
    monkeypatch.setattr("rune.proactive.routine.time.time", lambda: clock[0])
    memory = MemoryStore(tmp_path / "memory.db")
    monkeypatch.setattr(cron, "_get_store", lambda: memory)
    sent = AsyncMock()
    monkeypatch.setattr(cron, "_send_to_channel", sent)

    async def wake(now: datetime, **answers: Price | str) -> str | None:
        """The scheduler fires the job at ``now``; what it sent to the owner, or None."""
        clock[0] += 60
        pages.now = now
        pages.answers.update({{"coat": COAT}[name]: answer for name, answer in answers.items()})
        before = sent.await_count
        row = memory.get_cron_job(schedule.standing(time(9, 0)).job_id)
        assert await cron.execute_cron_job(cron._row_to_cronjob(row)) is None
        return sent.await_args.args[2] if sent.await_count > before else None

    yield SimpleNamespace(wake=wake, sent=sent, pages=pages)
    memory.close()


async def test_eight_days_of_wakes_tell_three_things_and_read_at_most_twice_a_day(scheduled):
    pages, wake = scheduled.pages, scheduled.wake
    item = pages.watch()
    assert schedule.keep(time(9, 0)).state == "ok"
    assert await wake(at(1), coat=won(100)) is None
    assert await wake(at(2), coat=won(120)) is None                  # a rise is not a drop: nothing waits
    assert await wake(at(2, 10)) is None
    assert await wake(at(3), coat=won(100)) is None                  # back where the owner knows it
    assert await wake(at(4), coat=won(90)) is None                   # seen once
    assert "100 KRW → 90 KRW" in await wake(at(4, 10))
    assert await wake(at(5), coat="server") is None
    failing = await wake(at(6))
    assert "Could not be read for 2 days in a row." in failing and "90" not in failing
    assert await wake(at(7), coat=won(80)) is None
    assert "90 KRW → 85 KRW" in await wake(at(7, 10), coat=won(85)), "the higher of the two readings is told"
    assert await wake(at(8, 13), coat=won(90)) is None               # off at nine: read when it is back on
    assert scheduled.sent.await_count == 3
    assert [call.args[:2] for call in scheduled.sent.await_args_list] == [("telegram", "Price watch")] * 3
    by_day: dict[object, int] = {}
    store = Store()
    for seen in store.observations(item.id):
        day = datetime.fromisoformat(seen["observed_at"]).date()
        by_day[day] = by_day.get(day, 0) + 1
    store.close()
    assert len(pages.reads) == 10 and max(by_day.values()) == 2 and pages.kept(item).told == won(85)
