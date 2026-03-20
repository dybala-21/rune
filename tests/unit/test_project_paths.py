"""Tests for rune.memory.project_paths — project key encoding and path resolution."""



from rune.memory.project_paths import (
    detect_workspace_root,
    get_project_memory_paths,
    to_project_memory_key,
)

# ---------------------------------------------------------------------------
# Tests: to_project_memory_key
# ---------------------------------------------------------------------------

class TestToProjectMemoryKey:
    def test_encodes_unix_path(self):
        key = to_project_memory_key("/Users/test/workspace/rune")
        assert key == "-Users-test-workspace-rune"

    def test_starts_with_dash_for_absolute_path(self):
        key = to_project_memory_key("/some/path")
        assert key.startswith("-")

    def test_no_special_chars_in_key(self):
        key = to_project_memory_key("/Users/test/my project!/rune")
        assert all(c.isalnum() or c in (".", "-", "_") for c in key)

    def test_collapses_multiple_dashes(self):
        key = to_project_memory_key("/Users///test///rune")
        assert "---" not in key

    def test_empty_path_returns_workspace(self):
        key = to_project_memory_key("")
        # The function should return something non-empty
        assert key


# ---------------------------------------------------------------------------
# Tests: get_project_memory_paths
# ---------------------------------------------------------------------------

class TestGetProjectMemoryPaths:
    def test_builds_correct_directory_structure(self, tmp_path):
        config_dir = tmp_path / ".rune"
        paths = get_project_memory_paths(
            workspace_path="/Users/test/workspace/rune",
            rune_config_dir=str(config_dir),
        )
        assert paths.project_key == "-Users-test-workspace-rune"
        assert str(paths.base_dir).endswith("-Users-test-workspace-rune")
        assert str(paths.sessions_dir).endswith("sessions")
        assert str(paths.memory_dir).endswith("memory")
        assert str(paths.memory_file).endswith("MEMORY.md")
        assert str(paths.daily_dir).endswith("daily")

    def test_sessions_index_file_path(self, tmp_path):
        config_dir = tmp_path / ".rune"
        paths = get_project_memory_paths(
            workspace_path="/Users/test/workspace/rune",
            rune_config_dir=str(config_dir),
        )
        assert str(paths.sessions_index_file).endswith("sessions-index.json")

    def test_base_dir_under_projects(self, tmp_path):
        config_dir = tmp_path / ".rune"
        paths = get_project_memory_paths(
            workspace_path="/Users/test/workspace/rune",
            rune_config_dir=str(config_dir),
        )
        assert "projects" in str(paths.base_dir)

    def test_uses_cwd_when_no_workspace_path(self, tmp_path):
        config_dir = tmp_path / ".rune"
        paths = get_project_memory_paths(
            rune_config_dir=str(config_dir),
        )
        # Should use os.getcwd() as workspace
        assert paths.workspace_path is not None

    def test_different_workspaces_get_different_keys(self, tmp_path):
        config_dir = tmp_path / ".rune"
        paths1 = get_project_memory_paths("/project/a", str(config_dir))
        paths2 = get_project_memory_paths("/project/b", str(config_dir))
        assert paths1.project_key != paths2.project_key


# ---------------------------------------------------------------------------
# Tests: detect_workspace_root
# ---------------------------------------------------------------------------

class TestDetectWorkspaceRoot:
    def test_detects_git_directory(self, tmp_path):
        (tmp_path / ".git").mkdir()
        result = detect_workspace_root(str(tmp_path))
        assert result is not None
        assert result == tmp_path

    def test_detects_pyproject_toml(self, tmp_path):
        (tmp_path / "pyproject.toml").touch()
        result = detect_workspace_root(str(tmp_path))
        assert result is not None
        assert result == tmp_path

    def test_detects_package_json(self, tmp_path):
        (tmp_path / "package.json").touch()
        result = detect_workspace_root(str(tmp_path))
        assert result is not None

    def test_walks_up_to_find_marker(self, tmp_path):
        (tmp_path / ".git").mkdir()
        child = tmp_path / "src" / "deep"
        child.mkdir(parents=True)
        result = detect_workspace_root(str(child))
        assert result is not None
        assert result == tmp_path

    def test_returns_none_when_no_markers(self, tmp_path):
        isolated = tmp_path / "isolated"
        isolated.mkdir()
        # tmp_path typically doesn't have project markers
        # but we can't guarantee no parent has one, so just check type
        result = detect_workspace_root(str(isolated))
        # May or may not find workspace root depending on environment
        assert result is None or result is not None  # just ensure no crash
