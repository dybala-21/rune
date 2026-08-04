"""A run should know what is in the directory it is standing in.

The repository map only covers code — code-category goals, and only files
tree-sitter can parse. A folder of spreadsheets and notes satisfies neither,
so those runs used to begin with nothing but their own path. Measured cost:
asked for a revenue summary for a management meeting, the run read the CSV
the request named, summed the column called `revenue`, and never opened the
data dictionary beside it saying the reporting figure is `net_revenue`.
"""

from __future__ import annotations

from rune.agent import workspace_listing as wl


class TestBuildListing:
    def test_it_names_the_files_that_are_there(self, tmp_path):
        (tmp_path / "sales_june.csv").write_text("a,b\n")
        (tmp_path / "data_dictionary.md").write_text("# terms\n")
        out = wl.build_listing(tmp_path)
        assert "sales_june.csv" in out
        assert "data_dictionary.md" in out

    def test_directories_are_marked_and_descended(self, tmp_path):
        (tmp_path / "reports").mkdir()
        (tmp_path / "reports" / "q2.md").write_text("x")
        out = wl.build_listing(tmp_path)
        assert "reports/" in out
        assert "q2.md" in out

    def test_noise_directories_are_left_out(self, tmp_path):
        (tmp_path / "keep.csv").write_text("x")
        for junk in ("node_modules", "__pycache__", ".git", ".venv"):
            d = tmp_path / junk
            d.mkdir()
            (d / "f.txt").write_text("x")
        out = wl.build_listing(tmp_path)
        assert "keep.csv" in out
        for junk in ("node_modules", "__pycache__", ".git", ".venv"):
            assert junk not in out

    def test_a_long_listing_is_capped_and_says_so(self, tmp_path):
        for i in range(300):
            (tmp_path / f"f{i:03d}.txt").write_text("x")
        out = wl.build_listing(tmp_path, max_entries=10)
        assert out.count("\n") <= 11
        assert "truncated" in out

    def test_shallow_entries_survive_the_cap(self, tmp_path):
        # The file a request means is far likelier to be near the top than
        # buried, so depth loses to breadth when something has to go.
        deep = tmp_path / "sub"
        deep.mkdir()
        for i in range(20):
            (deep / f"deep{i}.txt").write_text("x")
        (tmp_path / "top.csv").write_text("x")
        out = wl.build_listing(tmp_path, max_entries=5)
        assert "top.csv" in out

    def test_an_empty_directory_says_nothing(self, tmp_path):
        assert wl.build_listing(tmp_path) == ""
        assert wl.listing_section(tmp_path) == ""

    def test_a_path_that_is_not_a_directory_is_harmless(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("x")
        assert wl.build_listing(f) == ""

    def test_it_can_be_switched_off(self, tmp_path, monkeypatch):
        (tmp_path / "a.csv").write_text("x")
        monkeypatch.setenv("RUNE_WORKSPACE_LISTING", "0")
        assert wl.build_listing(tmp_path) == ""


class TestSection:
    def test_the_section_warns_that_an_unasked_file_may_still_matter(self, tmp_path):
        (tmp_path / "sales.csv").write_text("x")
        (tmp_path / "data_dictionary.md").write_text("x")
        section = wl.listing_section(tmp_path)
        assert "data_dictionary.md" in section
        assert "define what the request means" in section


class TestPromptWiring:
    def test_the_listing_appears_when_there_is_no_repo_map(self, tmp_path):
        from rune.agent.prompts import build_system_prompt

        (tmp_path / "sales_june.csv").write_text("x")
        (tmp_path / "data_dictionary.md").write_text("x")
        prompt = build_system_prompt(
            "summarise june revenue", environment={"cwd": str(tmp_path)})
        assert "data_dictionary.md" in prompt

    def test_a_repo_map_keeps_the_slot(self, tmp_path):
        # Code tasks already get the better description; two would be noise.
        from rune.agent.prompts import build_system_prompt

        (tmp_path / "sales_june.csv").write_text("x")
        prompt = build_system_prompt(
            "fix the bug", environment={"cwd": str(tmp_path)},
            repo_map="app.py: main()")
        assert "Repository Map" in prompt
        assert "sales_june.csv" not in prompt
