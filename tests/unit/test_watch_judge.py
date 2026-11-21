"""When a watched price is told: compared with what the owner knows, and only after a second reading."""

from datetime import UTC, datetime, time, timedelta
from decimal import Decimal

import pytest

from rune.watch.item import Ended, Failing, Item, Price, Told
from rune.watch.judge import due, meets, renew, settle

START = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)


def won(low: int, high: int | None = None, currency: str = "KRW") -> Price:
    return Price(Decimal(low), Decimal(low if high is None else high), currency)


def at(day: int, hour: int = 9, minute: int = 0) -> datetime:
    return START.replace(hour=hour, minute=minute) + timedelta(days=day)


def watch(tell_when: str = "drop", told: Price | None = None, below: int | None = None) -> Item:
    told = told or won(100)
    return Item(id=1, url="https://shop.example/p/1", fetch_url="https://shop.example/p/1", host="shop.example",
                label="", tell_when=tell_when, below=None if below is None else Decimal(below), told=told,
                told_at=START, renewed_at=START, last_read=told, last_read_at=START)


class Page:
    """Wakes an item at the times a test names and answers its reads; keeps what was told and asked."""

    def __init__(self, item: Item, check_time: time = time(9, 0)) -> None:
        self.item, self.check_time = item, check_time
        self.requests: list[datetime] = []

    def wake(self, steps: list[tuple[datetime, Price | str]]) -> list[tuple[datetime, list[object]]]:
        told = []
        for now, answer in steps:
            read = due(self.item, now, self.check_time)
            if read is None:
                continue
            self.requests.append(now)
            if events := settle(self.item, now, read, answer):
                told.append((now, list(events)))
        return told

    def most_requests_in_a_day(self) -> int:
        days: dict[object, int] = {}
        for now in self.requests:
            days[now.date()] = days.get(now.date(), 0) + 1
        return max(days.values(), default=0)


def hourly(day: int, answer: Price | str) -> list[tuple[datetime, Price | str]]:
    return [(at(day, hour), answer) for hour in range(24)]


def test_a_drop_is_told_once_after_a_second_reading_and_a_failure_on_its_second_day():
    page = Page(watch())
    told = page.wake([
        (at(1), won(100)), (at(2), won(120)), (at(2, 10), won(120)), (at(3), won(100)), (at(3, 10), won(100)),
        (at(4), won(90)), (at(4, 10), won(90)), (at(5), "server"), (at(6), "server"),
        (at(7), won(80)), (at(7, 10), won(85)), (at(8, 13), won(90)),
    ])
    assert told == [
        (at(4, 10), [Told(before=won(100), now=won(90))]),
        (at(6), [Failing(kind="server", days=2)]),
        # 80 and then 85 are both under the 90 the owner knows: told once, at the higher reading.
        (at(7, 10), [Told(before=won(90), now=won(85))]),
    ]
    assert page.item.told == won(85) and page.item.candidate is None
    assert page.most_requests_in_a_day() == 2


def test_a_price_that_rose_and_came_back_is_not_a_drop():
    page = Page(watch())
    assert page.wake([(at(1), won(120)), (at(1, 10), won(120)), (at(2), won(100)), (at(2, 10), won(100)),
                      (at(3), won(150)), (at(4), won(140)), (at(4, 10), won(140))]) == []
    # A rise is never what later readings are compared with, and gets no second read.
    assert page.item.told == won(100) and len(page.requests) == 4


def test_a_value_seen_once_and_gone_an_hour_later_is_not_told():
    page = Page(watch())
    assert page.wake([(at(1), won(60)), (at(1, 10), won(100)), (at(2), won(100))]) == []
    assert page.item.candidate is None and page.item.told == won(100)


def test_a_site_that_starts_refusing_gets_one_confirming_read_and_then_one_read_a_day():
    page = Page(watch())
    steps = [(at(1), won(90))] + [(at(1, hour), "blocked") for hour in range(10, 24)]
    told = page.wake(steps + hourly(2, "blocked") + hourly(3, "blocked"))
    assert told == [(at(3), [Failing(kind="blocked", days=2)])]
    assert len(page.requests) == 4 and page.most_requests_in_a_day() == 2
    assert page.item.candidate == won(90)
    # When it reads again with the price still down, that day's read is the second reading.
    assert page.wake([(at(4), won(90))]) == [(at(4), [Told(before=won(100), now=won(90))])]


def test_a_deeper_drop_on_the_second_reading_is_told_at_the_higher_one_first():
    page = Page(watch())
    assert page.wake([(at(1), won(90)), (at(1, 10), won(80)), (at(2), won(80)), (at(2, 10), won(80))]) == [
        (at(1, 10), [Told(before=won(100), now=won(90))]),
        (at(2, 10), [Told(before=won(90), now=won(80))]),
    ]


def test_a_price_that_moves_every_hour_is_told_once_two_readings_are_under_the_target():
    page = Page(watch(told=won(120), below=100))
    assert page.wake([(at(1), won(110)), (at(2), won(95)), (at(2, 10), won(93))]) == [
        (at(2, 10), [Told(before=won(120), now=won(95))])]
    assert page.item.failed_days == 0


def test_a_watch_registered_under_its_target_tells_only_a_lower_price():
    page = Page(watch(told=won(95), below=100))
    assert page.wake([(at(1), won(95)), (at(2), won(97)), (at(3), won(94)), (at(3, 10), won(94))]) == [
        (at(3, 10), [Told(before=won(95), now=won(94))])]


def test_any_change_is_told_as_it_stands_and_a_range_is_its_two_ends():
    page = Page(watch("change", told=won(100, 200)))
    assert page.wake([(at(1), won(100, 200)), (at(2), won(100, 180)), (at(2, 10), won(100, 180)),
                      (at(3), won(120, 180)), (at(3, 10), won(110, 180)), (at(4), won(110, 180))]) == [
        (at(2, 10), [Told(before=won(100, 200), now=won(100, 180))]),
        (at(3, 10), [Told(before=won(100, 180), now=won(110, 180))]),
    ]


def test_a_late_evening_check_time_confirms_after_midnight_with_no_extra_reads():
    page = Page(watch(), check_time=time(23, 30))
    told = page.wake([(at(1, 23, 30), won(90)), (at(2, 0, 30), won(90)), (at(2, 1, 30), won(90)),
                      (at(2, 23, 30), won(90))])
    assert told == [(at(2, 0, 30), [Told(before=won(100), now=won(90))])] and len(page.requests) == 3


def test_a_waiting_price_is_settled_by_the_first_wake_after_the_machine_was_off():
    page = Page(watch())
    assert page.wake([(at(1), won(90)), (at(3, 8), won(90)), (at(3, 9), won(90))]) == [
        (at(3, 8), [Told(before=won(100), now=won(90))])]
    page = Page(watch())
    assert page.wake([(at(1), won(90)), (at(3, 14), won(100))]) == [] and page.item.candidate is None


def test_days_the_machine_was_off_neither_break_nor_lengthen_a_failure_streak():
    page = Page(watch())
    assert page.wake([(at(1), "timeout"), (at(4), "timeout"), (at(5), "timeout"), (at(6), won(100)),
                      (at(7), "timeout")]) == [(at(4), [Failing(kind="timeout", days=2)])]
    assert page.item.failed_days == 1 and page.item.told_failure is False


def test_fourteen_failed_days_end_a_watch_but_days_the_web_was_closed_never_do():
    page = Page(watch())
    assert page.wake([(at(day), "blocked") for day in range(1, 16)]) == [
        (at(2), [Failing(kind="blocked", days=2)]), (at(14), [Ended(why="unreadable", last=won(100))])]
    assert page.item.status == "ended" and due(page.item, at(20), time(9, 0)) is None
    page = Page(watch())
    assert page.wake([(at(day), "web_closed") for day in range(1, 40)]) == [
        (at(2), [Failing(kind="web_closed", days=2)])]
    assert page.item.status == "watching"


def test_a_page_gone_two_days_running_ends_the_watch():
    page = Page(watch())
    assert page.wake([(at(1), won(100)), (at(2), "gone"), (at(3), "gone"), (at(4), won(100))]) == [
        (at(3), [Ended(why="gone", last=won(100))])]


def test_ninety_days_end_a_watch_after_that_days_read_and_never_before_a_waiting_drop_is_told():
    page = Page(watch())
    assert page.wake([(at(89), won(100)), (at(90), won(100)), (at(91), won(100))]) == [
        (at(90), [Ended(why="expired", last=won(100))])]
    page = Page(watch())
    assert page.wake([(at(90), won(90)), (at(90, 10), won(90)), (at(91), won(90))]) == [
        (at(90, 10), [Told(before=won(100), now=won(90)), Ended(why="expired", last=won(90))])]
    page = Page(watch())
    assert page.wake([(at(90), "timeout"), (at(91), "timeout")]) == [(at(90), [Ended(why="expired", last=won(100))])]


def test_hourly_wakes_read_a_quiet_page_once_a_day_and_a_late_start_still_reads_that_day():
    page = Page(watch())
    assert page.wake(hourly(1, won(100)) + hourly(2, won(100))) == [] and len(page.requests) == 2
    page = Page(watch())
    page.wake([(at(1, 14), won(100)), (at(1, 15), won(100))])  # the laptop was opened after one
    assert page.requests == [at(1, 14)]


def test_a_reading_in_another_currency_is_a_failed_day_not_a_price_move():
    page = Page(watch())
    assert page.wake([(at(1), won(70, currency="USD")), (at(2), won(70, currency="USD"))]) == [
        (at(2), [Failing(kind="incomparable", days=2)])]
    assert page.item.told == won(100) and page.item.candidate is None


@pytest.mark.parametrize(("tell_when", "below", "price", "expected"), [
    ("change", None, won(100), False), ("change", None, won(101), True), ("change", None, won(100, 130), True),
    ("drop", None, won(99), True), ("drop", None, won(100), False), ("drop", None, won(130), False),
    ("drop", 90, won(95), False), ("drop", 90, won(90), True), ("drop", 90, won(80), True),
])
def test_what_counts_as_the_owners_condition(tell_when, below, price, expected):
    assert meets(watch(tell_when, below=below), price) is expected


def test_setting_an_item_again_starts_from_the_price_just_shown():
    page = Page(watch())
    page.wake([(at(1), won(120)), (at(2), "blocked"), (at(3), "blocked")])
    item = page.item
    renew(item, at(4), won(120), tell_when="drop", below=Decimal(100), label="코트")
    assert (item.told, item.below, item.label, item.failed_days, item.renewed_at) == (
        won(120), Decimal(100), "코트", 0, at(4))
    # 95 is under the new target and under the 120 the owner was just shown.
    assert page.wake([(at(5), won(95)), (at(5, 10), won(95))]) == [(at(5, 10), [Told(before=won(120), now=won(95))])]
