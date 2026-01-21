"""Tests for context injection utilities."""

from dataclasses import dataclass
from typing import Any

from lc2mcp.context import (
    build_call_kwargs,
    detect_injectable_params,
    filter_schema_properties,
    get_non_injectable_params,
)


@dataclass
class MockContext:
    """Mock MCP context for testing."""

    request_id: str = "test-123"
    session: dict = None

    def __post_init__(self):
        if self.session is None:
            self.session = {}


@dataclass
class MockRuntime:
    """Mock ToolRuntime for testing."""

    context: Any


class TestDetectInjectableParams:
    """Tests for detect_injectable_params function."""

    def test_no_injectable_params(self):
        """Test function with no injectable params."""

        def func(a: str, b: int) -> str:
            return f"{a}{b}"

        result = detect_injectable_params(func)
        assert result == {}

    def test_mcp_ctx_param(self):
        """Test detection of mcp_ctx parameter."""

        def func(query: str, mcp_ctx: MockContext) -> str:
            return query

        result = detect_injectable_params(func)
        assert "mcp_ctx" in result

    def test_runtime_param(self):
        """Test detection of runtime parameter."""

        def func(query: str, runtime: MockRuntime) -> str:
            return query

        result = detect_injectable_params(func)
        assert "runtime" in result

    def test_both_injectable_params(self):
        """Test detection of both mcp_ctx and runtime."""

        def func(query: str, mcp_ctx: MockContext, runtime: MockRuntime) -> str:
            return query

        result = detect_injectable_params(func)
        assert "mcp_ctx" in result
        assert "runtime" in result


class TestGetNonInjectableParams:
    """Tests for get_non_injectable_params function."""

    def test_filters_mcp_ctx(self):
        """Test that mcp_ctx is filtered out."""

        def func(a: str, b: int, mcp_ctx: MockContext) -> str:
            return f"{a}{b}"

        result = get_non_injectable_params(func)
        assert "a" in result
        assert "b" in result
        assert "mcp_ctx" not in result

    def test_filters_runtime(self):
        """Test that runtime is filtered out."""

        def func(query: str, runtime: MockRuntime) -> str:
            return query

        result = get_non_injectable_params(func)
        assert "query" in result
        assert "runtime" not in result

    def test_all_params_kept_when_no_injectables(self):
        """Test that all params are kept when no injectables present."""

        def func(a: str, b: int, c: bool = True) -> str:
            return f"{a}{b}{c}"

        result = get_non_injectable_params(func)
        assert len(result) == 3
        assert "a" in result
        assert "b" in result
        assert "c" in result


class TestFilterSchemaProperties:
    """Tests for filter_schema_properties function."""

    def test_removes_injectable_from_properties(self):
        """Test removal of injectable params from properties."""
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "mcp_ctx": {"type": "object"},
                "limit": {"type": "integer"},
            },
            "required": ["query", "mcp_ctx"],
        }

        result = filter_schema_properties(schema, {"mcp_ctx"})

        assert "query" in result["properties"]
        assert "limit" in result["properties"]
        assert "mcp_ctx" not in result["properties"]
        assert "mcp_ctx" not in result.get("required", [])

    def test_removes_multiple_injectables(self):
        """Test removal of multiple injectable params."""
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "mcp_ctx": {"type": "object"},
                "runtime": {"type": "object"},
            },
            "required": ["query", "mcp_ctx", "runtime"],
        }

        result = filter_schema_properties(schema, {"mcp_ctx", "runtime"})

        assert "query" in result["properties"]
        assert "mcp_ctx" not in result["properties"]
        assert "runtime" not in result["properties"]
        assert result["required"] == ["query"]

    def test_empty_required_is_removed(self):
        """Test that empty required list is removed."""
        schema = {
            "type": "object",
            "properties": {"mcp_ctx": {"type": "object"}},
            "required": ["mcp_ctx"],
        }

        result = filter_schema_properties(schema, {"mcp_ctx"})

        assert "required" not in result

    def test_no_change_when_no_injectables(self):
        """Test that schema is unchanged when no injectables to remove."""
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

        result = filter_schema_properties(schema, set())

        assert result == schema


class TestBuildCallKwargs:
    """Tests for build_call_kwargs function."""

    def test_only_user_kwargs(self):
        """Test with only user kwargs, no injection."""
        user_kwargs = {"query": "test", "limit": 10}

        result = build_call_kwargs(
            user_kwargs=user_kwargs,
            injectable_params={},
            mcp_ctx=None,
            runtime_adapter=None,
        )

        assert result == user_kwargs

    def test_inject_mcp_ctx(self):
        """Test injection of mcp_ctx."""
        user_kwargs = {"query": "test"}
        ctx = MockContext()

        result = build_call_kwargs(
            user_kwargs=user_kwargs,
            injectable_params={"mcp_ctx": MockContext},
            mcp_ctx=ctx,
            runtime_adapter=None,
        )

        assert result["query"] == "test"
        assert result["mcp_ctx"] is ctx

    def test_inject_runtime(self):
        """Test injection of runtime via adapter."""
        user_kwargs = {"query": "test"}
        ctx = MockContext()

        def adapter(mcp_ctx):
            return MockRuntime(context={"user": "test_user"})

        result = build_call_kwargs(
            user_kwargs=user_kwargs,
            injectable_params={"runtime": MockRuntime},
            mcp_ctx=ctx,
            runtime_adapter=adapter,
        )

        assert result["query"] == "test"
        assert isinstance(result["runtime"], MockRuntime)
        assert result["runtime"].context == {"user": "test_user"}

    def test_inject_both(self):
        """Test injection of both mcp_ctx and runtime."""
        user_kwargs = {"query": "test"}
        ctx = MockContext()

        def adapter(mcp_ctx):
            return MockRuntime(context={"request_id": mcp_ctx.request_id})

        result = build_call_kwargs(
            user_kwargs=user_kwargs,
            injectable_params={"mcp_ctx": MockContext, "runtime": MockRuntime},
            mcp_ctx=ctx,
            runtime_adapter=adapter,
        )

        assert result["query"] == "test"
        assert result["mcp_ctx"] is ctx
        assert result["runtime"].context["request_id"] == "test-123"

    def test_preserves_user_kwargs(self):
        """Test that user kwargs are not modified."""
        user_kwargs = {"a": 1, "b": 2}
        original = user_kwargs.copy()

        build_call_kwargs(
            user_kwargs=user_kwargs,
            injectable_params={"mcp_ctx": MockContext},
            mcp_ctx=MockContext(),
            runtime_adapter=None,
        )

        assert user_kwargs == original
