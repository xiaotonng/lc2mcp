"""Tests for naming utilities."""

import pytest

from lc2mcp.naming import apply_prefix, resolve_conflict


class TestApplyPrefix:
    """Tests for apply_prefix function."""

    def test_with_prefix(self):
        """Test applying a prefix to a name."""
        result = apply_prefix("get_weather", "weather.")
        assert result == "weather.get_weather"

    def test_without_prefix(self):
        """Test that None prefix returns original name."""
        result = apply_prefix("get_weather", None)
        assert result == "get_weather"

    def test_empty_prefix(self):
        """Test that empty string prefix is applied."""
        result = apply_prefix("get_weather", "")
        assert result == "get_weather"

    def test_prefix_with_underscore(self):
        """Test prefix with underscore separator."""
        result = apply_prefix("analyze", "finance_")
        assert result == "finance_analyze"


class TestResolveConflict:
    """Tests for resolve_conflict function."""

    def test_no_conflict(self):
        """Test when there is no name conflict."""
        existing = {"tool_a", "tool_b"}
        result = resolve_conflict("tool_c", existing, "error")
        assert result == "tool_c"

    def test_error_strategy(self):
        """Test that 'error' strategy raises ValueError on conflict."""
        existing = {"tool_a", "tool_b"}
        with pytest.raises(ValueError) as exc_info:
            resolve_conflict("tool_a", existing, "error")
        assert "already exists" in str(exc_info.value)

    def test_overwrite_strategy(self):
        """Test that 'overwrite' strategy returns the same name."""
        existing = {"tool_a", "tool_b"}
        result = resolve_conflict("tool_a", existing, "overwrite")
        assert result == "tool_a"

    def test_suffix_strategy(self):
        """Test that 'suffix' strategy adds numeric suffix."""
        existing = {"tool_a", "tool_b"}
        result = resolve_conflict("tool_a", existing, "suffix")
        assert result == "tool_a_2"

    def test_suffix_strategy_multiple_conflicts(self):
        """Test suffix strategy with multiple existing suffixed names."""
        existing = {"tool_a", "tool_a_2", "tool_a_3"}
        result = resolve_conflict("tool_a", existing, "suffix")
        assert result == "tool_a_4"

    def test_suffix_finds_first_available(self):
        """Test that suffix finds the first available number."""
        existing = {"tool_a", "tool_a_2", "tool_a_4"}
        result = resolve_conflict("tool_a", existing, "suffix")
        # Should find tool_a_3 since it's available
        # Note: current implementation goes sequentially, so it will be tool_a_3
        assert result == "tool_a_3"
