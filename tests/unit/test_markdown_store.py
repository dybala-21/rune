"""Tests for rune.memory.markdown_store."""

from __future__ import annotations

from pathlib import Path

import pytest

from rune.memory.markdown_store import (
    append_daily_entry,
    append_to_memory_md,
    ensure_memory_structure,
    learned_md_has_key,
    memory_md_has_key,
    parse_daily_log,
    parse_learned_md,
    parse_memory_md,
    parse_rules_md,
    parse_user_profile,
    prune_learned_md,
    remove_learned_fact,
    save_learned_fact,
    update_user_profile_section,
)


@pytest.fixture
def mem_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("rune.memory.markdown_store.memory_dir", lambda: tmp_path)
    return tmp_path


class TestMemoryMd:
    def test_parse_empty(self, mem_dir: Path) -> None:
        assert parse_memory_md(mem_dir / "MEMORY.md") == {}

    def test_parse_roundtrip(self, mem_dir: Path) -> None:
        md = mem_dir / "MEMORY.md"
        md.write_text("# Preferences\n\n- editor: vim\n- shell: zsh\n\n# Notes\n\n- keep it simple\n")
        sections = parse_memory_md(md)
        assert sections["Preferences"] == ["editor: vim", "shell: zsh"]
        assert sections["Notes"] == ["keep it simple"]

    def test_has_key(self, mem_dir: Path) -> None:
        md = mem_dir / "MEMORY.md"
        md.write_text("# Preferences\n\n- editor: vim\n")
        assert memory_md_has_key("editor", md) is True
        assert memory_md_has_key("shell", md) is False

    def test_append_existing_section(self, mem_dir: Path) -> None:
        md = mem_dir / "MEMORY.md"
        md.write_text("# Preferences\n\n- editor: vim\n\n# Notes\n\n- note1\n")
        append_to_memory_md("Preferences", "shell: zsh", md)
        sections = parse_memory_md(md)
        assert "shell: zsh" in sections["Preferences"]
        assert "note1" in sections["Notes"]

    def test_append_new_section(self, mem_dir: Path) -> None:
        md = mem_dir / "MEMORY.md"
        md.write_text("# Preferences\n\n- editor: vim\n")
        append_to_memory_md("Environment", "os: linux", md)
        sections = parse_memory_md(md)
        assert sections["Environment"] == ["os: linux"]


class TestLearnedMd:
    def test_parse_empty(self, mem_dir: Path) -> None:
        assert parse_learned_md(mem_dir / "learned.md") == []

    def test_save_and_parse(self, mem_dir: Path) -> None:
        save_learned_fact("preference", "editor", "vim", 0.85, mem_dir / "learned.md")
        facts = parse_learned_md(mem_dir / "learned.md")
        assert len(facts) == 1
        assert facts[0]["key"] == "editor"
        assert facts[0]["value"] == "vim"
        assert facts[0]["confidence"] == 0.85

    def test_update_existing(self, mem_dir: Path) -> None:
        path = mem_dir / "learned.md"
        save_learned_fact("preference", "editor", "vim", 0.8, path)
        save_learned_fact("preference", "editor", "neovim", 0.9, path)
        facts = parse_learned_md(path)
        assert len(facts) == 1
        assert facts[0]["value"] == "neovim"
        assert facts[0]["confidence"] == 0.9

    def test_has_key(self, mem_dir: Path) -> None:
        path = mem_dir / "learned.md"
        save_learned_fact("preference", "editor", "vim", 0.8, path)
        assert learned_md_has_key("editor", path) == "vim"
        assert learned_md_has_key("shell", path) is None

    def test_remove(self, mem_dir: Path) -> None:
        path = mem_dir / "learned.md"
        save_learned_fact("preference", "editor", "vim", 0.8, path)
        save_learned_fact("preference", "shell", "zsh", 0.7, path)
        assert remove_learned_fact("editor", path) is True
        facts = parse_learned_md(path)
        assert len(facts) == 1
        assert facts[0]["key"] == "shell"

    def test_remove_nonexistent(self, mem_dir: Path) -> None:
        path = mem_dir / "learned.md"
        save_learned_fact("preference", "editor", "vim", 0.8, path)
        assert remove_learned_fact("nonexistent", path) is False

    def test_lenient_parser_missing_confidence(self, mem_dir: Path) -> None:
        path = mem_dir / "learned.md"
        path.write_text("# Facts\n\n- [preference] editor: vim\n")
        facts = parse_learned_md(path)
        assert len(facts) == 1
        assert facts[0]["confidence"] == 0.5

    def test_lenient_parser_broken_format(self, mem_dir: Path) -> None:
        path = mem_dir / "learned.md"
        path.write_text("# Facts\n\n- just some random text\n")
        facts = parse_learned_md(path)
        assert len(facts) == 1
        assert facts[0]["category"] == "general"
        assert facts[0]["confidence"] == 0.3

    def test_colon_category_roundtrip(self, mem_dir: Path) -> None:
        """Categories with colons (e.g. rule:code_modify) must survive
        save → parse roundtrip.  Regression test for _LEARNED_RE fix."""
        path = mem_dir / "learned.md"
        save_learned_fact("rule:code_modify", "verify_edit", "re-read before edit", 0.60, path)
        save_learned_fact("rule:research", "check_source", "verify sources", 0.55, path)

        facts = parse_learned_md(path)
        assert len(facts) == 2

        by_cat = {f["category"]: f for f in facts}
        assert "rule:code_modify" in by_cat
        assert by_cat["rule:code_modify"]["key"] == "verify_edit"
        assert by_cat["rule:code_modify"]["confidence"] == 0.60

        assert "rule:research" in by_cat
        assert by_cat["rule:research"]["key"] == "check_source"

    def test_colon_category_update(self, mem_dir: Path) -> None:
        """Updating a fact with colon category should replace, not duplicate."""
        path = mem_dir / "learned.md"
        save_learned_fact("rule:code_modify", "verify_edit", "v1", 0.40, path)
        save_learned_fact("rule:code_modify", "verify_edit", "v2", 0.70, path)

        facts = parse_learned_md(path)
        rule_facts = [f for f in facts if f["category"] == "rule:code_modify"]
        assert len(rule_facts) == 1
        assert rule_facts[0]["value"] == "v2"
        assert rule_facts[0]["confidence"] == 0.70

    def test_colon_category_remove(self, mem_dir: Path) -> None:
        """Removing a fact with colon category should work."""
        path = mem_dir / "learned.md"
        save_learned_fact("rule:code_modify", "verify_edit", "re-read", 0.60, path)
        assert remove_learned_fact("verify_edit", path) is True
        assert parse_learned_md(path) == []

    def test_prune(self, mem_dir: Path) -> None:
        path = mem_dir / "learned.md"
        for i in range(10):
            save_learned_fact("test", f"key_{i}", f"val_{i}", i * 0.1, path)
        removed = prune_learned_md(cap=5, path=path)
        assert len(removed) == 5
        remaining = parse_learned_md(path)
        assert len(remaining) == 5
        # Lowest confidence ones should be removed
        remaining_keys = {f["key"] for f in remaining}
        assert "key_0" not in remaining_keys


class TestDailyLog:
    def test_append_and_parse(self, mem_dir: Path) -> None:
        (mem_dir / "daily").mkdir()
        path = append_daily_entry(
            title="Fix auth bug",
            actions=["read auth.py", "edited token check"],
            lessons=["check timezone first"],
            date="2026-03-16",
            time_str="14:23",
        )
        entries = parse_daily_log(path)
        assert len(entries) == 1
        assert entries[0]["title"] == "Fix auth bug"
        assert entries[0]["time"] == "14:23"
        assert "read auth.py" in entries[0]["actions"]
        assert "check timezone first" in entries[0]["lessons"]

    def test_multiple_entries(self, mem_dir: Path) -> None:
        (mem_dir / "daily").mkdir()
        append_daily_entry(title="Task 1", actions=["a1"], date="2026-03-16", time_str="10:00")
        append_daily_entry(title="Task 2", actions=["a2"], date="2026-03-16", time_str="11:00")
        path = mem_dir / "daily" / "2026-03-16.md"
        entries = parse_daily_log(path)
        assert len(entries) == 2
        assert entries[0]["title"] == "Task 1"
        assert entries[1]["title"] == "Task 2"


class TestRulesMd:
    def test_parse(self, mem_dir: Path) -> None:
        path = mem_dir / "rules.md"
        path.write_text(
            "# Safety Rules\n\n"
            "## rm_rf_protection\n"
            "- type: blocklist\n"
            "- pattern: rm -rf /\n"
            "- reason: Prevents catastrophic deletion\n\n"
            "## env_protection\n"
            "- type: blocklist\n"
            "- pattern: cat .env\n"
            "- reason: Prevents credential exposure\n"
        )
        rules = parse_rules_md(path)
        assert len(rules) == 2
        assert rules[0]["name"] == "rm_rf_protection"
        assert rules[0]["pattern"] == "rm -rf /"


class TestUserProfile:
    def test_parse_roundtrip(self, mem_dir: Path) -> None:
        path = mem_dir / "user-profile.md"
        path.write_text("# Communication\n\n- language: auto\n- verbosity: concise\n\n# Goals\n\n")
        sections = parse_user_profile(path)
        assert sections["Communication"] == ["language: auto", "verbosity: concise"]

    def test_update_section_preserves_others(self, mem_dir: Path) -> None:
        path = mem_dir / "user-profile.md"
        path.write_text(
            "# Communication\n\n- language: auto\n\n# Stats\n\n- old stat\n\n# Goals\n\n- my goal\n"
        )
        update_user_profile_section("Stats", ["python: 50 tasks, 90% success"], path)
        sections = parse_user_profile(path)
        assert sections["Communication"] == ["language: auto"]
        assert sections["Stats"] == ["python: 50 tasks, 90% success"]
        assert sections["Goals"] == ["my goal"]

    def test_custom_section_preserved(self, mem_dir: Path) -> None:
        path = mem_dir / "user-profile.md"
        path.write_text("# Communication\n\n- language: auto\n\n# MyCustom\n\n- custom data\n")
        update_user_profile_section("Stats", ["new stat"], path)
        sections = parse_user_profile(path)
        assert "MyCustom" in sections
        assert sections["MyCustom"] == ["custom data"]


class TestFirstRun:
    def test_ensure_memory_structure(self, mem_dir: Path) -> None:
        ensure_memory_structure()
        assert (mem_dir / "MEMORY.md").exists()
        assert (mem_dir / "learned.md").exists()
        assert (mem_dir / "daily").is_dir()
        assert (mem_dir / "user-profile.md").exists()
        assert (mem_dir / ".state" / "fact-meta.json").exists()
        assert (mem_dir / ".state" / "suppressed.json").exists()
        assert (mem_dir / ".state" / "conflicts.json").exists()
        assert (mem_dir / ".state" / "index-state.json").exists()

    def test_idempotent(self, mem_dir: Path) -> None:
        ensure_memory_structure()
        (mem_dir / "MEMORY.md").write_text("# Custom\n\n- my data\n")
        ensure_memory_structure()
        content = (mem_dir / "MEMORY.md").read_text()
        assert "my data" in content


class TestConcurrentWrite:
    def test_no_corruption(self, mem_dir: Path) -> None:
        import threading

        path = mem_dir / "learned.md"
        errors: list[str] = []

        def writer(idx: int) -> None:
            try:
                for j in range(5):
                    save_learned_fact("test", f"key_{idx}_{j}", f"val_{idx}_{j}", 0.5, path)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        facts = parse_learned_md(path)
        assert len(facts) == 50
