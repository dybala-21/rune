"""The watch's message: figures code read, the owner's own name for the page, and fixed wording."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from rune.briefing import labels as briefing_labels
from rune.watch import labels
from rune.watch.compose import MAX_BUTTONS, News, amount, compose, money
from rune.watch.item import Ended, Failing, Item, Price, Told

SEOUL = ZoneInfo("Asia/Seoul")
NOW = datetime(2026, 11, 27, 10, 0, tzinfo=SEOUL)
TOLD_AT = datetime(2026, 11, 19, 16, 30, tzinfo=UTC)   # 01:30 on the 20th where the owner is


def won(low: int, high: int | None = None) -> Price:
    return Price(Decimal(low), Decimal(low if high is None else high), "KRW")


def item(ident: int = 3, label: str = "에어팟 프로 3", host: str = "www.apple.com", **more: object) -> Item:
    fields = {"id": ident, "url": f"https://{host}/p/{ident}", "fetch_url": f"https://{host}/p/{ident}", "host": host,
              "label": label, "tell_when": "drop", "below": None, "told": won(349000), "told_at": NOW,
              "renewed_at": TOLD_AT, **more}
    return Item(**fields)  # type: ignore[arg-type]


def told(before: int, now: int, **more: object) -> News:
    return News(item(**more), Told(won(before), won(now)), TOLD_AT)


@pytest.mark.parametrize(("value", "shown"), [
    ("369000", "369,000"), ("439990.0", "439,990"), ("4", "4"), ("19.9", "19.90"), ("19.90", "19.90"),
    ("19.995", "19.995"), ("0.5", "0.50"), ("1234567.5", "1,234,567.50"), ("1E+3", "1,000"),
])
def test_an_amount_is_written_whole_or_with_its_decimals(value, shown):
    assert amount(Decimal(value)) == shown


def test_a_price_is_one_amount_or_its_lowest_and_highest():
    assert money(won(369000)) == "369,000 KRW"
    assert money(won(149000, 189000)) == "149,000–189,000 KRW"


def test_a_drop_is_told_with_both_prices_both_dates_and_a_way_to_the_page():
    text, buttons = compose([told(369000, 349000)], NOW)
    assert text.splitlines() == [
        "A price you asked me to watch went down.", "",
        "에어팟 프로 3 (www.apple.com)", "369,000 KRW → 349,000 KRW",
        "Last told 2026-11-20 · read 2026-11-27 10:00", "",
        "Prices are as the page writes them for a visitor who is not signed in."]
    assert [(button.text, button.url) for button in buttons] == [("Open the page", "https://www.apple.com/p/3")]


def test_a_rise_is_a_change_and_a_page_without_a_name_goes_by_its_site():
    text, _ = compose([told(349000, 369000, label="")], NOW)
    assert text.splitlines()[:4] == ["A price you asked me to watch changed.", "", "www.apple.com",
                                     "349,000 KRW → 369,000 KRW"]


def test_a_page_that_could_not_be_read_is_said_to_be_that_with_no_price_beside_it():
    for kind, why in (("blocked", "refuses automated reading"), ("robots", "rules for automated programs"),
                      ("web_closed", "web access is closed"), ("no_price", "no longer writes a price"),
                      ("moved", "leads to another site"), ("incomparable", "another currency"),
                      ("timeout", "did not answer"), ("something new", "did not answer")):
        text, buttons = compose([News(item(last_read=won(349000), last_read_at=TOLD_AT), Failing(kind, 2), TOLD_AT)], NOW)
        assert "Could not be read for 2 days in a row." in text and why in text, kind
        # Not the price from before, on a day nothing was read; only the day count is a number.
        assert "349" not in text and "KRW" not in text and "not signed in" not in text
        assert len(buttons) == 1


def test_places_that_stopped_agreeing_ask_the_owner_for_the_price_and_name_no_amount():
    text, buttons = compose([News(item(last_read=won(349000), last_read_at=TOLD_AT), Failing("split", 1), TOLD_AT)], NOW)
    assert text.splitlines() == [
        "About the pages you asked me to watch:", "", "에어팟 프로 3 (www.apple.com)",
        "The page now gives more than one amount where its price was read, so I can't tell which is the price. "
        "Tell me the price it shows now and I will keep watching."]
    assert "349" not in text and "days" not in text and len(buttons) == 1


def test_a_watch_that_ended_says_why_and_an_old_price_carries_its_day():
    gone, _ = compose([News(item(), Ended("gone", won(349000)), TOLD_AT)], NOW)
    assert "The page is gone" in gone and "349" not in gone
    unread, _ = compose([News(item(counted_days=14), Ended("unreadable", won(349000)), TOLD_AT)], NOW)
    assert "could not be read for 14 days" in unread and "349" not in unread
    kept = item(last_read=won(349000), last_read_at=TOLD_AT + timedelta(days=4))
    expired, _ = compose([News(kept, Ended("expired", won(349000)), TOLD_AT)], NOW)
    assert "watched for 90 days" in expired and "Last read: 349,000 KRW on 2026-11-24" in expired
    never, _ = compose([News(item(), Ended("expired", None), TOLD_AT)], NOW)
    assert "watched for 90 days" in never and "Last read" not in never


def test_several_pages_share_one_message_with_a_button_each_up_to_a_few():
    news = [told(100, 90, ident=n, label=f"coat {n}", host="shop.example") for n in range(1, 8)]
    news.append(News(item(ident=1, label="coat 1", host="shop.example"), Failing("blocked", 2), TOLD_AT))
    text, buttons = compose(news, NOW)
    assert text.startswith("About the pages you asked me to watch:") and text.count("100 KRW → 90 KRW") == 7
    assert len(buttons) == MAX_BUTTONS and buttons[0].text == "coat 1 (shop.example)"
    assert len({button.url for button in buttons}) == MAX_BUTTONS


def test_the_wording_can_stand_in_another_language_and_keeps_its_figures():
    korean = {**dict.fromkeys(labels.ENGLISH, "한글"), "went_down": "지켜보던 가격이 내려갔어요.",
              "told_and_read": "지난번에 알려 드린 날 {told} · 이번 확인 {read}", "open": "상품 페이지 열기",
              "failing": "{days}일째 읽지 못했어요.", "ended_unreadable": "{days}일 동안 읽지 못해 그만 지켜봐요.",
              "ended_expired": "{days}일 동안 지켜봤어요.", "last_read": "마지막으로 읽은 값: {amount} ({date})",
              "failed": "{date}에는 확인하지 못했어요."}
    t = briefing_labels.checked(korean, labels.ENGLISH)
    text, buttons = compose([told(369000, 349000)], NOW, t)
    assert "지켜보던 가격이 내려갔어요." in text and "369,000 KRW → 349,000 KRW" in text
    assert "지난번에 알려 드린 날 2026-11-20 · 이번 확인 2026-11-27 10:00" in text and buttons[0].text == "상품 페이지 열기"


@pytest.mark.parametrize(("key", "bad"), [
    ("failing", "2일째 읽지 못했어요."),                 # a number of its own in place of the count
    ("told_and_read", "알려 드린 날 {told}"),             # a figure dropped
    ("why_blocked", "자세한 내용은 evil.example 에서"),   # a link
    ("open", ""),
])
def test_a_translation_that_adds_or_drops_a_fact_is_not_used(key, bad):
    with pytest.raises(ValueError):
        briefing_labels.checked({**dict.fromkeys(labels.ENGLISH, "한글"),
                                 **{k: v for k, v in labels.ENGLISH.items() if "{" in v}, key: bad}, labels.ENGLISH)


def test_every_english_label_would_pass_its_own_check_and_every_failure_has_wording():
    # A label the check would turn down in translation (a link-like word, a digit) could never be
    # shown in another language.
    assert briefing_labels.checked(dict(labels.ENGLISH), labels.ENGLISH) == labels.ENGLISH
    assert set(labels.WHY.values()) <= set(labels.ENGLISH) and labels.why("anything") in labels.ENGLISH
