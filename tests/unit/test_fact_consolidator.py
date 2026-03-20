"""Tests for rune.memory.fact_consolidator — fact extraction and consolidation."""


from rune.memory.fact_consolidator import (
    extract_facts_heuristic,
)

# ---------------------------------------------------------------------------
# Tests: Tier 1 heuristic extraction
# ---------------------------------------------------------------------------

class TestExtractFactsHeuristic:
    def test_extracts_package_manager_pnpm(self):
        facts = extract_facts_heuristic(
            "using pnpm for package management",
            "installed packages with pnpm",
        )
        assert any(f.key == "package_manager" and f.value == "pnpm" for f in facts)

    def test_extracts_package_manager_yarn(self):
        facts = extract_facts_heuristic("chose yarn", "")
        assert any(f.key == "package_manager" and f.value == "yarn" for f in facts)

    def test_extracts_package_manager_npm(self):
        facts = extract_facts_heuristic("prefer npm", "")
        assert any(f.key == "package_manager" and f.value == "npm" for f in facts)

    def test_extracts_test_runner(self):
        facts = extract_facts_heuristic("using vitest", "")
        assert any(f.key == "test_runner" and f.value == "vitest" for f in facts)

    def test_extracts_pytest_runner(self):
        facts = extract_facts_heuristic("using pytest for tests", "")
        assert any(f.key == "test_runner" and f.value == "pytest" for f in facts)

    def test_extracts_framework(self):
        facts = extract_facts_heuristic("built with react", "")
        assert any(f.key == "framework" for f in facts)

    def test_extracts_database(self):
        facts = extract_facts_heuristic("using postgresql", "")
        assert any(f.key == "database" and f.value == "postgresql" for f in facts)

    def test_extracts_bundler(self):
        facts = extract_facts_heuristic("using vite", "")
        assert any(f.key == "bundler" and f.value == "vite" for f in facts)

    def test_extracts_deploy_target(self):
        facts = extract_facts_heuristic("deploy to vercel", "")
        assert any(f.key == "deploy_target" and f.value == "vercel" for f in facts)

    def test_deduplicates_same_key(self):
        facts = extract_facts_heuristic(
            "using pnpm for packages. chose pnpm for monorepo",
            "",
        )
        pm_facts = [f for f in facts if f.key == "package_manager"]
        assert len(pm_facts) == 1

    def test_returns_empty_for_no_match(self):
        facts = extract_facts_heuristic(
            "hello world",
            "some random answer",
        )
        assert len(facts) == 0

    def test_extracts_from_answer_text(self):
        facts = extract_facts_heuristic(
            "setup project",
            "using pnpm as the package manager",
        )
        assert any(f.key == "package_manager" for f in facts)

    def test_category_is_preference_for_package_manager(self):
        facts = extract_facts_heuristic("using pnpm", "")
        pm = next(f for f in facts if f.key == "package_manager")
        assert pm.category == "preference"

    def test_category_is_environment_for_database(self):
        facts = extract_facts_heuristic("connect to postgresql", "")
        db = next(f for f in facts if f.key == "database")
        assert db.category == "environment"

    def test_extracts_port(self):
        facts = extract_facts_heuristic("port: 8080", "")
        assert any(f.key == "default_port" and f.value == "8080" for f in facts)
