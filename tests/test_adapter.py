"""Tests for the core adapter module."""


import pytest
from langchain_core.tools import StructuredTool, tool
from pydantic import BaseModel

from lc2mcp.adapter import _extract_tool_info, register_tools, to_mcp_tool


class SimpleArgs(BaseModel):
    """Simple arguments for testing."""

    query: str
    limit: int = 10


class TestExtractToolInfo:
    """Tests for _extract_tool_info function."""

    def test_extract_from_decorated_tool(self):
        """Test extraction from @tool decorated function."""

        @tool
        def search(query: str) -> str:
            """Search for something."""
            return f"Results for {query}"

        name, desc, schema = _extract_tool_info(search)

        assert name == "search"
        assert "Search" in desc or "search" in desc.lower()
        assert schema is not None

    def test_extract_from_structured_tool(self):
        """Test extraction from StructuredTool."""

        def my_func(query: str, limit: int = 10) -> str:
            return f"{query}: {limit}"

        tool_instance = StructuredTool.from_function(
            func=my_func,
            name="my_search",
            description="My search tool",
            args_schema=SimpleArgs,
        )

        name, desc, schema = _extract_tool_info(tool_instance)

        assert name == "my_search"
        assert desc == "My search tool"
        assert schema is not None
        assert "query" in schema.get("properties", {})


class TestToMcpTool:
    """Tests for to_mcp_tool function."""

    def test_basic_conversion(self):
        """Test basic tool conversion."""

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        wrapped = to_mcp_tool(add)

        assert wrapped.__name__ == "add"
        assert "Add" in wrapped.__doc__ or "add" in wrapped.__doc__.lower()
        assert hasattr(wrapped, "_mcp_tool_schema")

    def test_name_override(self):
        """Test overriding the tool name."""

        @tool
        def my_tool(x: str) -> str:
            """A tool."""
            return x

        wrapped = to_mcp_tool(my_tool, name="custom_name")

        assert wrapped._mcp_tool_name == "custom_name"

    def test_description_override(self):
        """Test overriding the tool description."""

        @tool
        def my_tool(x: str) -> str:
            """Original description."""
            return x

        wrapped = to_mcp_tool(my_tool, description="Custom description")

        assert wrapped._mcp_tool_description == "Custom description"

    def test_schema_override(self):
        """Test overriding the tool schema."""

        @tool
        def my_tool(x: str) -> str:
            """A tool."""
            return x

        custom_schema = {
            "type": "object",
            "properties": {"custom_field": {"type": "string"}},
        }

        wrapped = to_mcp_tool(my_tool, args_schema=custom_schema)

        assert wrapped._mcp_tool_schema == custom_schema

    @pytest.mark.asyncio
    async def test_wrapper_is_async(self):
        """Test that the wrapper is an async function."""

        @tool
        def sync_tool(x: str) -> str:
            """A sync tool."""
            return f"result: {x}"

        wrapped = to_mcp_tool(sync_tool)

        # The wrapper should be callable and async
        result = await wrapped(x="test")
        assert result == "result: test"


class TestRegisterTools:
    """Tests for register_tools function."""

    def test_register_single_tool(self):
        """Test registering a single tool."""
        from fastmcp import FastMCP

        mcp = FastMCP("test-server")

        @tool
        def my_tool(x: str) -> str:
            """A tool."""
            return x

        registered = register_tools(mcp, [my_tool])

        assert len(registered) == 1
        assert "my_tool" in registered

    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        from fastmcp import FastMCP

        mcp = FastMCP("test-server")

        @tool
        def tool_a(x: str) -> str:
            """Tool A."""
            return x

        @tool
        def tool_b(y: int) -> int:
            """Tool B."""
            return y * 2

        registered = register_tools(mcp, [tool_a, tool_b])

        assert len(registered) == 2
        assert "tool_a" in registered
        assert "tool_b" in registered

    def test_register_with_prefix(self):
        """Test registering tools with a name prefix."""
        from fastmcp import FastMCP

        mcp = FastMCP("test-server")

        @tool
        def analyze(data: str) -> str:
            """Analyze data."""
            return data

        registered = register_tools(mcp, [analyze], name_prefix="finance.")

        assert len(registered) == 1
        assert "finance.analyze" in registered

    def test_conflict_error_strategy(self):
        """Test that error strategy raises on conflict."""
        from fastmcp import FastMCP

        mcp = FastMCP("test-server")

        @tool
        def duplicate(x: str) -> str:
            """First tool."""
            return x

        @tool
        def duplicate(y: str) -> str:  # noqa: F811
            """Second tool with same name."""
            return y

        # Register first tool
        register_tools(mcp, [duplicate])

        # Attempting to register again should raise
        # Note: We need to create a new tool with the same name
        @tool
        def duplicate(z: str) -> str:  # noqa: F811
            """Third tool."""
            return z

        with pytest.raises(ValueError) as exc_info:
            register_tools(mcp, [duplicate], on_name_conflict="error")

        assert "already exists" in str(exc_info.value)

    def test_conflict_suffix_strategy(self):
        """Test that suffix strategy adds numeric suffix."""
        from fastmcp import FastMCP

        mcp = FastMCP("test-server")

        @tool
        def same_name(x: str) -> str:
            """First tool."""
            return x

        # Register first
        registered1 = register_tools(mcp, [same_name])

        @tool
        def same_name(y: str) -> str:  # noqa: F811
            """Second tool."""
            return y

        # Register with suffix strategy
        registered2 = register_tools(mcp, [same_name], on_name_conflict="suffix")

        assert "same_name" in registered1
        assert "same_name_2" in registered2
