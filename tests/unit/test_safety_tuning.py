"""Tests for rune.memory.tuning — memory tuning configuration."""



from rune.memory.tuning import (
    MEMORY_TUNING_DEFAULTS,
    PRESETS,
    get_tuning_config,
    parse_positive_int_env,
    parse_probability_env,
)

# ---------------------------------------------------------------------------
# Tests: Defaults and presets
# ---------------------------------------------------------------------------

class TestTuningDefaults:
    def test_defaults_have_all_expected_keys(self):
        expected_keys = {
            "semantic_limit",
            "semantic_min_score",
            "uncertain_semantic_limit",
            "uncertain_semantic_min_score",
            "max_episodes",
            "context_max_chars",
        }
        assert set(MEMORY_TUNING_DEFAULTS.keys()) == expected_keys

    def test_balanced_preset_matches_defaults(self):
        balanced = PRESETS["balanced"]
        for key, value in MEMORY_TUNING_DEFAULTS.items():
            assert balanced[key] == value

    def test_minimal_preset_has_lower_limits(self):
        minimal = PRESETS["minimal"]
        assert minimal["semantic_limit"] < MEMORY_TUNING_DEFAULTS["semantic_limit"]
        assert minimal["max_episodes"] < MEMORY_TUNING_DEFAULTS["max_episodes"]

    def test_aggressive_preset_has_higher_limits(self):
        aggressive = PRESETS["aggressive"]
        assert aggressive["semantic_limit"] > MEMORY_TUNING_DEFAULTS["semantic_limit"]
        assert aggressive["max_episodes"] > MEMORY_TUNING_DEFAULTS["max_episodes"]

    def test_research_preset_has_highest_limits(self):
        research = PRESETS["research"]
        aggressive = PRESETS["aggressive"]
        assert research["semantic_limit"] > aggressive["semantic_limit"]
        assert research["max_episodes"] > aggressive["max_episodes"]


# ---------------------------------------------------------------------------
# Tests: get_tuning_config
# ---------------------------------------------------------------------------

class TestGetTuningConfig:
    def test_returns_defaults_without_preset(self):
        config = get_tuning_config()
        for key, value in MEMORY_TUNING_DEFAULTS.items():
            assert config[key] == value

    def test_preset_overrides_defaults(self):
        config = get_tuning_config(preset="minimal")
        assert config["semantic_limit"] == PRESETS["minimal"]["semantic_limit"]

    def test_env_overrides_preset(self, monkeypatch):
        monkeypatch.setenv("RUNE_MEMORY_SEMANTIC_LIMIT", "99")
        config = get_tuning_config(preset="minimal")
        assert config["semantic_limit"] == 99

    def test_env_float_override(self, monkeypatch):
        monkeypatch.setenv("RUNE_MEMORY_SEMANTIC_MIN_SCORE", "0.75")
        config = get_tuning_config()
        assert config["semantic_min_score"] == 0.75

    def test_unknown_preset_falls_back_to_defaults(self):
        config = get_tuning_config(preset="nonexistent")  # type: ignore
        for key, value in MEMORY_TUNING_DEFAULTS.items():
            assert config[key] == value


# ---------------------------------------------------------------------------
# Tests: parse_probability_env
# ---------------------------------------------------------------------------

class TestParseProbabilityEnv:
    def test_returns_fallback_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("TEST_PROB", raising=False)
        assert parse_probability_env("TEST_PROB", 0.5) == 0.5

    def test_parses_valid_probability(self, monkeypatch):
        monkeypatch.setenv("TEST_PROB", "0.8")
        assert parse_probability_env("TEST_PROB", 0.5) == 0.8

    def test_rejects_out_of_range(self, monkeypatch):
        monkeypatch.setenv("TEST_PROB", "1.5")
        assert parse_probability_env("TEST_PROB", 0.5) == 0.5

    def test_rejects_negative(self, monkeypatch):
        monkeypatch.setenv("TEST_PROB", "-0.1")
        assert parse_probability_env("TEST_PROB", 0.5) == 0.5

    def test_rejects_non_numeric(self, monkeypatch):
        monkeypatch.setenv("TEST_PROB", "abc")
        assert parse_probability_env("TEST_PROB", 0.5) == 0.5


# ---------------------------------------------------------------------------
# Tests: parse_positive_int_env
# ---------------------------------------------------------------------------

class TestParsePositiveIntEnv:
    def test_returns_fallback_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("TEST_INT", raising=False)
        assert parse_positive_int_env("TEST_INT", 10) == 10

    def test_parses_valid_int(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "42")
        assert parse_positive_int_env("TEST_INT", 10) == 42

    def test_rejects_zero(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "0")
        assert parse_positive_int_env("TEST_INT", 10) == 10

    def test_rejects_negative(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "-5")
        assert parse_positive_int_env("TEST_INT", 10) == 10

    def test_rejects_non_numeric(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "xyz")
        assert parse_positive_int_env("TEST_INT", 10) == 10
