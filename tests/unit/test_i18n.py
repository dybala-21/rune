"""The core's fixed wording in each language: every catalog has the English keys, and names only the
values the code passes for each."""

import json
from importlib import resources
from string import Formatter

import pytest

from rune import i18n

# What the code passes for each key.
VALUES = {
    "budget.owner": {"when"}, "budget.host": {"when"}, "budget.task_waits": set(), "budget.task_stopped": set(),
    "when": {"month", "month_name", "day", "time", "zone"}, "month_names": set(), "zone_names": set(),
}


def catalogs():
    return sorted(p.name for p in resources.files("rune.locales").iterdir() if p.name.endswith(".json"))


def test_english_has_exactly_the_keys_the_code_uses():
    assert set(i18n.catalog("en")) == set(VALUES)


@pytest.mark.parametrize("name", catalogs())
def test_every_catalog_has_every_key_and_names_only_what_the_code_passes(name):
    data = json.loads(resources.files("rune.locales").joinpath(name).read_text(encoding="utf-8"))
    assert set(data) == set(VALUES), name
    for key, text in data.items():
        named = {field for _, field, _, _ in Formatter().parse(text) if field}
        assert named <= VALUES[key], (name, key, named)
    assert len(data["month_names"].split("|")) == 12


def test_a_missing_language_or_a_broken_label_reads_english(monkeypatch):
    assert i18n.text("budget.owner", "xx", when="soon") == i18n.text("budget.owner", "en", when="soon")
    assert i18n.text("budget.owner", "ko-KR", when="곧").startswith("이번 달")
    monkeypatch.setitem(i18n.catalog("ko"), "budget.owner", "{nope} 끝")
    assert i18n.text("budget.owner", "ko", when="soon").startswith("You've used")


async def test_the_host_sets_the_owner_s_language(monkeypatch, tmp_path):
    from fastapi import HTTPException

    from rune.api.handlers.config import ConfigPatchRequest, get_config_endpoint, patch_config
    from rune.config import get_config, loader

    path = tmp_path / "config.yaml"
    path.write_text("general:\n  language: en\n")
    monkeypatch.setattr("rune.config.writer.config_file_path", lambda: path)
    monkeypatch.setattr(get_config().general, "locale", "en")
    assert (await patch_config(ConfigPatchRequest(locale="ko-KR"))).updated
    assert get_config().general.locale == "ko-kr"
    assert (await get_config_endpoint()).locale == "ko-kr"
    monkeypatch.setattr(loader, "_find_config_file", lambda: path)
    assert loader.load_config(force=True).general.locale == "ko-kr", "kept, the legacy key gone"
    assert "language" not in path.read_text()
    assert not (await patch_config(ConfigPatchRequest(locale="ko-KR"))).updated, "the same again changes nothing"
    with pytest.raises(HTTPException):
        await patch_config(ConfigPatchRequest(locale="not a tag!"))


def test_the_owner_s_zone_is_their_schedules_and_named_their_way(monkeypatch):
    from datetime import UTC, datetime
    from zoneinfo import ZoneInfo

    from rune.llm import gateway_budget

    refusal = gateway_budget.BudgetRefusal("budget_exhausted", datetime(2026, 12, 1, tzinfo=UTC))
    assert "12월 1일 09:00(한국 시간)" in refusal.message("ko", ZoneInfo("Asia/Seoul"))
    for name, expected in (("Asia/Seoul", "Asia/Seoul"), ("Not/AZone", "UTC")):
        monkeypatch.setattr("rune.daemon.external_schedule.schedule_timezone", lambda name=name: name)
        assert getattr(gateway_budget.owner_zone(), "key", "UTC") == expected


def test_a_catalog_that_isn_t_json_reads_english(monkeypatch):
    class Broken:
        def joinpath(self, name):
            return self

        def read_text(self, encoding="utf-8"):
            return "{not json"
    i18n.catalog.cache_clear()
    monkeypatch.setattr(i18n.resources, "files", lambda _: Broken())
    assert i18n.catalog("ko") == {}
    monkeypatch.undo()
    i18n.catalog.cache_clear()
