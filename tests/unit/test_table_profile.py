"""The table-profile footer: whole-file facts a windowed read can't see.

The failure this exists for is specific: a CSV with duplicated rows gets
summed in the model's head and the duplicates are counted twice, silently,
every run. The footer states what code found — both totals when they
differ — and leaves the judgement to the reader.
"""

from __future__ import annotations

import pytest

from rune.capabilities.file import FileReadParams, file_read
from rune.capabilities.table_profile import profile_table

CSV_WITH_DUPS = (
    "region,amount\n"
    "north,100\n"
    "south,250.50\n"
    "north,100\n"
    "east,49.50\n"
)

CSV_CLEAN = (
    "name,qty,note\n"
    "bolt,3,\n"
    "nut,5,loose\n"
)


class TestProfileTable:
    def test_duplicates_produce_both_sums(self):
        out = profile_table(CSV_WITH_DUPS, "sales.csv")
        assert "4 (plus header)" in out
        assert "1 exact duplicate row(s)" in out
        assert "line 4" in out  # the repeated north,100 row
        assert "amount sum=500" in out
        assert "400 without the duplicate rows" in out

    def test_clean_file_says_so(self):
        out = profile_table(CSV_CLEAN, "parts.csv")
        assert "no exact duplicate rows" in out
        assert "qty sum=8" in out
        # only one total when there is nothing to disagree about
        assert "without the duplicate rows" not in out

    def test_blank_cells_are_counted(self):
        out = profile_table(CSV_CLEAN, "parts.csv")
        assert "blank cells: note: 1" in out

    def test_text_columns_are_not_summed(self):
        out = profile_table(CSV_CLEAN, "parts.csv")
        assert "name sum" not in out
        assert "note sum" not in out

    def test_nan_and_infinity_are_not_numeric(self):
        # Decimal parses both; the footer must never state sum=NaN as a fact.
        out = profile_table("id,x\na,NaN\nb,Infinity\nc,3\n", "odd.csv")
        assert "x sum" not in out
        assert "NaN" not in out

    def test_non_tabular_extensions_get_nothing(self):
        assert profile_table(CSV_WITH_DUPS, "sales.txt") == ""
        assert profile_table("def f():\n    pass\n", "code.py") == ""

    def test_single_column_is_not_a_table(self):
        assert profile_table("id\n1\n2\n", "ids.csv") == ""

    def test_header_only_gets_nothing(self):
        assert profile_table("a,b,c\n", "empty.csv") == ""

    def test_tsv_uses_tab_delimiter(self):
        out = profile_table("a\tn\nx\t1\nx\t1\n", "data.tsv")
        assert "1 exact duplicate row(s)" in out
        assert "n sum=2" in out

    def test_flag_off_disables(self, monkeypatch):
        monkeypatch.setenv("RUNE_TABLE_PROFILE", "0")
        assert profile_table(CSV_WITH_DUPS, "sales.csv") == ""


@pytest.fixture(autouse=True)
def _allow_guardian(monkeypatch):
    """Guardian refuses paths under /var, where pytest puts tmp_path."""
    from rune.safety import guardian as g

    class _OK:
        allowed = True
        reason = ""

    monkeypatch.setattr(
        g.Guardian, "validate_file_read_path", lambda self, p: _OK()
    )


class TestFileReadCarriesTheProfile:
    @pytest.mark.asyncio
    async def test_csv_read_ends_with_the_footer(self, tmp_path):
        f = tmp_path / "sales.csv"
        f.write_text(CSV_WITH_DUPS)
        res = await file_read(FileReadParams(path=str(f)))
        assert res.success
        assert "[table profile" in res.output
        assert "400 without the duplicate rows" in res.output

    @pytest.mark.asyncio
    async def test_windowed_read_still_profiles_the_whole_file(self, tmp_path):
        f = tmp_path / "sales.csv"
        f.write_text(CSV_WITH_DUPS)
        res = await file_read(FileReadParams(path=str(f), limit=2))
        assert res.success
        assert "east" not in res.output.split("[table profile")[0]
        assert "data rows: 4" in res.output

    @pytest.mark.asyncio
    async def test_plain_text_read_is_untouched(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("just words\n")
        res = await file_read(FileReadParams(path=str(f)))
        assert res.success
        assert "[table profile" not in res.output
