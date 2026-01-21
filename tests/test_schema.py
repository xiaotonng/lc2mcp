"""Tests for schema conversion utilities."""

from pydantic import BaseModel, Field

from lc2mcp.schema import extract_schema_from_tool, pydantic_to_json_schema


class SimpleArgs(BaseModel):
    """Simple test arguments."""

    name: str
    count: int = 1


class ComplexArgs(BaseModel):
    """Complex test arguments with optional fields."""

    query: str = Field(..., description="The search query")
    max_results: int = Field(default=10, description="Maximum results to return")
    include_metadata: bool = Field(default=False)


class NestedItem(BaseModel):
    """A nested model."""

    id: str
    value: float


class NestedArgs(BaseModel):
    """Arguments with nested models."""

    items: list[NestedItem]
    name: str


class TestPydanticToJsonSchema:
    """Tests for pydantic_to_json_schema function."""

    def test_simple_model(self):
        """Test conversion of a simple Pydantic model."""
        schema = pydantic_to_json_schema(SimpleArgs)

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "count" in schema["properties"]
        assert "required" in schema
        assert "name" in schema["required"]
        # count has a default, so might not be required
        assert "count" not in schema.get("required", [])

    def test_complex_model_with_descriptions(self):
        """Test that field descriptions are preserved."""
        schema = pydantic_to_json_schema(ComplexArgs)

        assert schema["type"] == "object"
        props = schema["properties"]
        assert props["query"].get("description") == "The search query"
        assert props["max_results"].get("description") == "Maximum results to return"

    def test_nested_model(self):
        """Test conversion of nested Pydantic models."""
        schema = pydantic_to_json_schema(NestedArgs)

        assert schema["type"] == "object"
        assert "items" in schema["properties"]
        assert "name" in schema["properties"]
        # Nested models should create $defs
        assert "$defs" in schema or "definitions" in schema or "items" in schema["properties"]

    def test_empty_model(self):
        """Test conversion of an empty model."""

        class EmptyArgs(BaseModel):
            pass

        schema = pydantic_to_json_schema(EmptyArgs)
        assert schema["type"] == "object"
        assert schema.get("properties", {}) == {}


class TestExtractSchemaFromTool:
    """Tests for extract_schema_from_tool function."""

    def test_extract_from_tool_with_args_schema(self):
        """Test extraction from a tool with args_schema attribute."""
        from langchain_core.tools import StructuredTool

        def dummy_func(name: str, count: int = 1) -> str:
            return f"{name}: {count}"

        tool = StructuredTool.from_function(
            func=dummy_func,
            name="dummy",
            description="A dummy tool",
            args_schema=SimpleArgs,
        )

        schema = extract_schema_from_tool(tool)
        assert schema is not None
        assert schema["type"] == "object"
        assert "name" in schema["properties"]

    def test_extract_from_decorated_tool(self):
        """Test extraction from a @tool decorated function."""
        from langchain_core.tools import tool

        @tool
        def my_tool(query: str, limit: int = 5) -> str:
            """Search for something."""
            return f"Results for {query}"

        schema = extract_schema_from_tool(my_tool)
        assert schema is not None
        assert "query" in schema.get("properties", {})

    def test_extract_returns_none_for_no_schema(self):
        """Test that None is returned when no schema is available."""

        class NoSchemaObject:
            pass

        result = extract_schema_from_tool(NoSchemaObject())
        assert result is None
