"""What the watch keeps: items come back as they were saved, and two writers don't undo each other."""

from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal

import pytest

from rune.watch.item import Price
from rune.watch.judge import settle
from rune.watch.store import KEEP_OBSERVATIONS, MAX_WATCHING, Store, TooMany

NOW = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
PRICE = Price(Decimal("369000"), Decimal("369000"), "KRW")


@pytest.fixture
def store(tmp_path):
    kept = Store(tmp_path / "data" / "watch.db")
    yield kept
    kept.close()


def add(store: Store, n: int = 1, **more):
    return store.add(**{"url": f"https://shop.example/p/{n}", "fetch_url": f"https://www.shop.example/p/{n}",
                        "host": "shop.example", "label": "코트", "tell_when": "drop", "below": None,
                        "price": PRICE, "now": NOW, **more})


def test_an_item_comes_back_as_it_was_saved_whatever_state_it_is_in(store):
    item = add(store, below=Decimal("300000.50"))
    assert store.get(item.id) == item and store.by_url(item.url) == item
    assert (item.told, item.last_read, item.renewed_at, item.status) == (PRICE, PRICE, NOW, "watching")
    lower = Price(Decimal("290000"), Decimal("295000"), "KRW")   # under the target
    settle(item, NOW + timedelta(days=1), "day", lower)   # a waiting price
    assert store.save(item)
    back = store.get(item.id)
    assert back == item and back.candidate == lower and back.read_day == date(2026, 11, 21)
    settle(item, NOW + timedelta(days=1, hours=1), "confirm", lower)   # told
    settle(item, NOW + timedelta(days=2), "day", "gone")
    settle(item, NOW + timedelta(days=3), "day", "gone")               # ended
    assert store.save(item)
    back = store.get(item.id)
    assert back == item and (back.status, back.ended_why, back.told) == ("ended", "gone", lower)
    assert store.watching() == [] and [kept.id for kept in store.everything()] == [item.id]


def test_amounts_are_kept_as_the_same_price_however_they_were_written(store):
    item = add(store, price=Price(Decimal("56.00"), Decimal("56.5"), "USD"))
    back = store.get(item.id)
    assert back.told == Price(Decimal("56"), Decimal("56.5"), "USD")
    row = store._db.execute("SELECT told_low, told_high FROM items").fetchone()
    assert tuple(row) == ("56", "56.5")


def test_a_save_from_a_copy_read_before_another_save_is_refused(tmp_path):
    path = tmp_path / "watch.db"
    job, chat = Store(path), Store(path)   # the scheduled run and the chat's tool
    item = add(job)
    reading_the_page = job.get(item.id)
    changed = chat.get(item.id)
    changed.tell_when, changed.label = "change", "새 이름"
    assert chat.save(changed)
    reading_the_page.failed_days = 3
    assert not job.save(reading_the_page)          # the owner's change is not written over
    kept = job.get(item.id)
    assert (kept.tell_when, kept.label, kept.failed_days) == ("change", "새 이름", 0)
    assert job.save(kept) and chat.get(item.id).version == 2
    job.close()
    chat.close()


def test_no_more_than_the_limit_are_watched_and_ended_ones_do_not_count(store):
    items = [add(store, n) for n in range(MAX_WATCHING)]
    with pytest.raises(TooMany):
        add(store, 99)
    assert store.by_url("https://shop.example/p/99") is None
    items[0].status, items[0].ended_why, items[0].ended_at = "ended", "expired", NOW
    assert store.save(items[0])
    assert add(store, 99).id and len(store.watching()) == MAX_WATCHING


def test_the_same_address_is_one_item(store):
    add(store)
    with pytest.raises(Exception, match="UNIQUE"):
        add(store)
    assert len(store.everything()) == 1


def test_pages_are_listed_site_by_site(store):
    add(store, 1, host="b.example")
    add(store, 2, host="a.example")
    add(store, 3, host="b.example")
    assert [item.host for item in store.watching()] == ["a.example", "b.example", "b.example"]


def test_only_the_latest_reads_of_an_item_are_kept(store):
    one, other = add(store, 1), add(store, 2)
    for n in range(KEEP_OBSERVATIONS + 5):
        store.observed(one.id, NOW + timedelta(days=n), http_status=200, price=PRICE, offers=1, source="jsonld")
    store.observed(other.id, NOW, http_status=403, failure="blocked")
    kept = store.observations(one.id)
    assert len(kept) == KEEP_OBSERVATIONS and kept[0]["observed_at"] == (NOW + timedelta(days=64)).isoformat()
    assert store.observations(other.id) == [{"observed_at": NOW.isoformat(), "http_status": 403, "low": None,
                                             "high": None, "offers": 0, "source": "", "failure": "blocked"}]


def test_the_check_time_is_nine_until_the_owner_sets_one(store):
    assert store.check_time() == time(9, 0)
    store.set_check_time(time(23, 30, 59))
    store.set_check_time(time(7, 5))
    assert store.check_time() == time(7, 5)


def test_items_that_ended_long_ago_are_dropped_with_their_reads(store):
    old, recent, live = add(store, 1), add(store, 2), add(store, 3)
    for item, ended in ((old, NOW - timedelta(days=100)), (recent, NOW - timedelta(days=5))):
        item.status, item.ended_why, item.ended_at = "ended", "expired", ended
        assert store.save(item)
        store.observed(item.id, NOW, failure="gone")
    assert store.forget_ended(NOW - timedelta(days=90)) == 1
    assert [item.id for item in store.everything()] == [live.id, recent.id]
    assert store.observations(old.id) == [] and len(store.observations(recent.id)) == 1


def test_the_file_is_the_owners_alone(tmp_path):
    path = tmp_path / "watch.db"
    Store(path).close()
    assert path.stat().st_mode & 0o777 == 0o600


def test_a_runs_readings_are_written_as_one_change_leaving_out_what_the_owner_changed(tmp_path):
    path = tmp_path / "watch.db"
    job, chat = Store(path), Store(path)
    first, second, third = (job.get(add(job, n).id) for n in range(3))
    changed = chat.get(second.id)
    changed.label = "theirs"
    assert chat.save(changed)
    for item in (first, second, third):
        item.failed_days = 2
    written = job.save_all([first, second, third])
    assert [item.id for item in written] == [first.id, third.id]
    assert [chat.get(item.id).failed_days for item in (first, second, third)] == [2, 0, 2]
    assert chat.get(second.id).label == "theirs" and (first.version, second.version, third.version) == (1, 0, 1)
    assert job.save(first), "a written copy is the current one"

    def broken(item):
        raise OSError("disk")

    first.failed_days = third.failed_days = 5
    job._write, version = broken, first.version
    with pytest.raises(OSError):
        job.save_all([first, third])
    assert first.version == version and chat.get(first.id).failed_days == 2, "all or nothing"
    job.close()
    chat.close()
