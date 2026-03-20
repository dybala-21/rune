"""Tests for rune.services.connector — ported from connector.test.ts."""


from rune.services.connector import is_mcp_write_operation


class TestIsMCPWriteOperation:
    """Tests for is_mcp_write_operation()."""

    def test_detects_create_operations_as_write(self):
        assert is_mcp_write_operation("mcp.google-calendar.create-event") is True
        assert is_mcp_write_operation("mcp.notion.create-page") is True

    def test_detects_update_operations_as_write(self):
        assert is_mcp_write_operation("mcp.google-calendar.update-event") is True

    def test_detects_delete_operations_as_write(self):
        assert is_mcp_write_operation("mcp.google-calendar.delete-event") is True

    def test_detects_send_operations_as_write(self):
        assert is_mcp_write_operation("mcp.slack.send-message") is True

    def test_does_not_detect_list_as_write(self):
        assert is_mcp_write_operation("mcp.google-calendar.list-events") is False

    def test_does_not_detect_get_as_write(self):
        assert is_mcp_write_operation("mcp.google-calendar.get-freebusy") is False

    def test_does_not_detect_search_as_write(self):
        assert is_mcp_write_operation("mcp.google-calendar.search-events") is False

    def test_returns_false_for_non_mcp_capabilities(self):
        assert is_mcp_write_operation("file.read") is False
        assert is_mcp_write_operation("bash") is False
        assert is_mcp_write_operation("think") is False

    def test_returns_false_for_malformed_mcp_names(self):
        assert is_mcp_write_operation("mcp.") is False
        assert is_mcp_write_operation("mcp.only-server") is False

    def test_handles_nested_tool_names(self):
        assert is_mcp_write_operation("mcp.github.issues.create") is True
        assert is_mcp_write_operation("mcp.github.repos.list") is False
