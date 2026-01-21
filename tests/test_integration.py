"""Integration tests for lc2mcp."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastmcp import Context, FastMCP
from langchain_core.tools import StructuredTool, tool
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel, Field

from lc2mcp import register_tools, to_mcp_tool


class SearchArgs(BaseModel):
    """Arguments for search tool."""

    query: str = Field(..., description="The search query")
    max_results: int = Field(default=10, description="Maximum results")


@dataclass(frozen=True)
class UserContext:
    """User context for testing."""

    user_id: str
    tenant_id: str


@dataclass
class AppState:
    """App state for testing."""

    request_count: int = 0


class TestEndToEndBasic:
    """End-to-end tests for basic functionality."""

    def test_quickstart_example(self):
        """Test the quickstart example from README."""

        @tool
        def get_weather(city: str) -> str:
            """Get the current weather for a specific city."""
            return f"Sunny, 25°C in {city}"

        mcp = FastMCP("weather-server")
        registered = register_tools(mcp, [get_weather])

        assert "get_weather" in registered

    def test_structured_tool_registration(self):
        """Test registration of StructuredTool instances."""

        def search_func(query: str, max_results: int = 10) -> str:
            return f"Found {max_results} results for: {query}"

        search_tool = StructuredTool.from_function(
            func=search_func,
            name="web_search",
            description="Search the web",
            args_schema=SearchArgs,
        )

        mcp = FastMCP("search-server")
        registered = register_tools(mcp, [search_tool])

        assert "web_search" in registered

    def test_multiple_tools_different_types(self):
        """Test registering different types of tools together."""

        @tool
        def tool_a(x: str) -> str:
            """Tool A using decorator."""
            return x

        def tool_b_func(y: int) -> int:
            return y * 2

        tool_b = StructuredTool.from_function(
            func=tool_b_func,
            name="tool_b",
            description="Tool B as StructuredTool",
        )

        mcp = FastMCP("mixed-server")
        registered = register_tools(mcp, [tool_a, tool_b])

        assert len(registered) == 2
        assert "tool_a" in registered
        assert "tool_b" in registered


class TestEndToEndWithContext:
    """End-to-end tests for context injection."""

    def test_context_injection_setup(self):
        """Test that context injection can be configured."""

        @tool
        def whoami(runtime: ToolRuntime[UserContext]) -> str:
            """Return current user."""
            return f"User: {runtime.context.user_id}"

        def runtime_adapter(mcp_ctx: Context) -> ToolRuntime[UserContext]:
            return ToolRuntime(
                state={},
                context=UserContext(user_id="test", tenant_id="default"),
                config={},
                stream_writer=lambda x: None,
                tool_call_id=None,
                store=None,
            )

        mcp = FastMCP("auth-server")
        registered = register_tools(
            mcp,
            [whoami],
            runtime_adapter=runtime_adapter,
        )

        assert "whoami" in registered

    def test_mcp_ctx_injection_setup(self):
        """Test that mcp_ctx injection can be configured."""

        @tool
        def get_request_id(mcp_ctx: Context) -> str:
            """Get request ID from context."""
            return mcp_ctx.request_id

        mcp = FastMCP("ctx-server")
        registered = register_tools(
            mcp,
            [get_request_id],
            inject_mcp_ctx=True,
        )

        assert "get_request_id" in registered


class TestEndToEndNaming:
    """End-to-end tests for naming features."""

    def test_prefix_namespacing(self):
        """Test that prefixes properly namespace tools."""

        @tool
        def analyze(data: str) -> str:
            """Analyze data."""
            return f"Analyzed: {data}"

        mcp = FastMCP("multi-domain")

        # Register with finance prefix
        registered = register_tools(mcp, [analyze], name_prefix="finance.")

        assert "finance.analyze" in registered

    def test_multiple_prefixes(self):
        """Test multiple tool sets with different prefixes."""

        @tool
        def get_data(id: str) -> str:
            """Get data by ID."""
            return f"Data: {id}"

        mcp = FastMCP("multi-domain")

        # Register with different prefixes
        reg1 = register_tools(mcp, [get_data], name_prefix="users.")

        @tool
        def get_data(id: str) -> str:  # noqa: F811
            """Get data by ID (products)."""
            return f"Product: {id}"

        reg2 = register_tools(mcp, [get_data], name_prefix="products.")

        assert "users.get_data" in reg1
        assert "products.get_data" in reg2


class TestToMcpToolManual:
    """Tests for manual tool wrapping with to_mcp_tool."""

    def test_manual_wrap_and_register(self):
        """Test manually wrapping and registering a tool."""

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        wrapped = to_mcp_tool(
            add,
            name="math.add",
            description="Add two integers.",
        )

        assert wrapped._mcp_tool_name == "math.add"
        assert wrapped._mcp_tool_description == "Add two integers."

    def test_manual_wrap_with_schema_override(self):
        """Test manual wrapping with schema override."""

        @tool
        def process(data: str) -> str:
            """Process data."""
            return data

        custom_schema = {
            "type": "object",
            "properties": {
                "data": {"type": "string", "description": "Custom description"},
            },
            "required": ["data"],
        }

        wrapped = to_mcp_tool(process, args_schema=custom_schema)

        assert wrapped._mcp_tool_schema == custom_schema


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_tool_with_no_args(self):
        """Test tool with no arguments."""

        @tool
        def get_timestamp() -> str:
            """Get current timestamp."""
            return "2024-01-01T00:00:00Z"

        mcp = FastMCP("timestamp-server")
        registered = register_tools(mcp, [get_timestamp])

        assert "get_timestamp" in registered

    def test_tool_with_complex_return(self):
        """Test tool that returns complex data."""

        @tool
        def get_user(user_id: str) -> dict:
            """Get user info."""
            return {"id": user_id, "name": "Test User", "active": True}

        mcp = FastMCP("user-server")
        registered = register_tools(mcp, [get_user])

        assert "get_user" in registered

    def test_empty_tools_list(self):
        """Test registering an empty list of tools."""
        mcp = FastMCP("empty-server")
        registered = register_tools(mcp, [])

        assert registered == []

    def test_tool_with_optional_args(self):
        """Test tool with all optional arguments."""

        @tool
        def configure(
            option_a: str = "default_a",
            option_b: int = 10,
            option_c: bool = False,
        ) -> str:
            """Configure settings."""
            return f"{option_a}, {option_b}, {option_c}"

        mcp = FastMCP("config-server")
        registered = register_tools(mcp, [configure])

        assert "configure" in registered


# ============================================================================
# End-to-End Execution Tests (actually call the tools)
# ============================================================================


def _create_mock_context() -> MagicMock:
    """Create a mock FastMCP Context for testing."""
    mock_ctx = MagicMock(spec=Context)
    mock_ctx.request_id = "test-request-123"
    mock_ctx.session_id = "test-session-456"
    mock_ctx.get_state = MagicMock(return_value=None)
    return mock_ctx


class TestToolExecution:
    """Tests that actually execute the wrapped tools."""

    @pytest.mark.asyncio
    async def test_basic_tool_execution(self):
        """Test that basic tool can be executed via wrapper."""

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        wrapped = to_mcp_tool(add)
        result = await wrapped(a=2, b=3)

        assert result == 5

    @pytest.mark.asyncio
    async def test_string_tool_execution(self):
        """Test string-based tool execution."""

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        wrapped = to_mcp_tool(greet)
        result = await wrapped(name="World")

        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_tool_with_defaults_execution(self):
        """Test tool with default arguments."""

        @tool
        def search(query: str, limit: int = 10) -> str:
            """Search for something."""
            return f"Found {limit} results for: {query}"

        wrapped = to_mcp_tool(search)

        # With explicit limit
        result1 = await wrapped(query="test", limit=5)
        assert result1 == "Found 5 results for: test"

        # With default limit
        result2 = await wrapped(query="test")
        assert result2 == "Found 10 results for: test"


class TestRuntimeInjectionExecution:
    """Tests for ToolRuntime injection with actual execution."""

    @pytest.mark.asyncio
    async def test_runtime_single_generic_execution(self):
        """Test ToolRuntime[UserContext] single generic execution."""

        @tool
        def whoami(runtime: ToolRuntime[UserContext]) -> str:
            """Return current user."""
            return f"User: {runtime.context.user_id}, Tenant: {runtime.context.tenant_id}"

        def runtime_adapter(mcp_ctx: Context) -> ToolRuntime[UserContext]:
            return ToolRuntime(
                state={"key": "value"},
                context=UserContext(user_id="alice", tenant_id="acme"),
                config={},
                stream_writer=lambda x: None,
                tool_call_id=None,
                store=None,
            )

        wrapped = to_mcp_tool(whoami, runtime_adapter=runtime_adapter)
        mock_ctx = _create_mock_context()

        result = await wrapped(mcp_ctx=mock_ctx)

        assert result == "User: alice, Tenant: acme"

    @pytest.mark.asyncio
    async def test_runtime_dual_generic_execution(self):
        """Test ToolRuntime[UserContext, AppState] dual generic execution."""

        @tool
        def get_info(runtime: ToolRuntime[UserContext, AppState]) -> str:
            """Get user info with state."""
            return f"User: {runtime.context.user_id}, Count: {runtime.state.request_count}"

        def runtime_adapter(mcp_ctx: Context) -> ToolRuntime[UserContext, AppState]:
            return ToolRuntime(
                state=AppState(request_count=42),
                context=UserContext(user_id="bob", tenant_id="corp"),
                config={},
                stream_writer=lambda x: None,
                tool_call_id=None,
                store=None,
            )

        wrapped = to_mcp_tool(get_info, runtime_adapter=runtime_adapter)
        mock_ctx = _create_mock_context()

        result = await wrapped(mcp_ctx=mock_ctx)

        assert result == "User: bob, Count: 42"

    @pytest.mark.asyncio
    async def test_runtime_with_dict_state_execution(self):
        """Test ToolRuntime with dict state (default StateT)."""

        @tool
        def get_state_value(key: str, runtime: ToolRuntime[UserContext]) -> str:
            """Get value from state dict."""
            return f"State[{key}] = {runtime.state.get(key, 'not found')}"

        def runtime_adapter(mcp_ctx: Context) -> ToolRuntime[UserContext]:
            return ToolRuntime(
                state={"foo": "bar", "count": 123},
                context=UserContext(user_id="user1", tenant_id="tenant1"),
                config={},
                stream_writer=lambda x: None,
                tool_call_id=None,
                store=None,
            )

        wrapped = to_mcp_tool(get_state_value, runtime_adapter=runtime_adapter)
        mock_ctx = _create_mock_context()

        result = await wrapped(mcp_ctx=mock_ctx, key="foo")

        assert result == "State[foo] = bar"

    @pytest.mark.asyncio
    async def test_runtime_with_mixed_args(self):
        """Test tool with both regular args and runtime injection."""

        @tool
        def process_data(data: str, multiplier: int, runtime: ToolRuntime[UserContext]) -> str:
            """Process data with user context."""
            return f"User {runtime.context.user_id} processed '{data}' x{multiplier}"

        def runtime_adapter(mcp_ctx: Context) -> ToolRuntime[UserContext]:
            return ToolRuntime(
                state={},
                context=UserContext(user_id="processor", tenant_id="t1"),
                config={},
                stream_writer=lambda x: None,
                tool_call_id=None,
                store=None,
            )

        wrapped = to_mcp_tool(process_data, runtime_adapter=runtime_adapter)
        mock_ctx = _create_mock_context()

        result = await wrapped(mcp_ctx=mock_ctx, data="hello", multiplier=3)

        assert result == "User processor processed 'hello' x3"


class TestMcpCtxInjectionExecution:
    """Tests for direct mcp_ctx: Context injection with actual execution."""

    @pytest.mark.asyncio
    async def test_mcp_ctx_injection_execution(self):
        """Test direct mcp_ctx injection execution."""

        @tool
        def get_request_id(mcp_ctx: Context) -> str:
            """Get request ID."""
            return f"Request: {mcp_ctx.request_id}"

        wrapped = to_mcp_tool(get_request_id, inject_mcp_ctx=True)
        mock_ctx = _create_mock_context()

        result = await wrapped(mcp_ctx=mock_ctx)

        assert result == "Request: test-request-123"

    @pytest.mark.asyncio
    async def test_mcp_ctx_with_other_args(self):
        """Test mcp_ctx injection with other arguments."""

        @tool
        def log_action(action: str, mcp_ctx: Context) -> str:
            """Log an action with request ID."""
            return f"[{mcp_ctx.request_id}] Action: {action}"

        wrapped = to_mcp_tool(log_action, inject_mcp_ctx=True)
        mock_ctx = _create_mock_context()

        result = await wrapped(mcp_ctx=mock_ctx, action="user_login")

        assert result == "[test-request-123] Action: user_login"

    @pytest.mark.asyncio
    async def test_mcp_ctx_get_state(self):
        """Test accessing mcp_ctx.get_state."""

        @tool
        def get_user_from_ctx(mcp_ctx: Context) -> str:
            """Get user from context state."""
            user = mcp_ctx.get_state("user_id")
            return f"User: {user or 'anonymous'}"

        wrapped = to_mcp_tool(get_user_from_ctx, inject_mcp_ctx=True)

        # Test with no user in state
        mock_ctx = _create_mock_context()
        result1 = await wrapped(mcp_ctx=mock_ctx)
        assert result1 == "User: anonymous"

        # Test with user in state
        mock_ctx.get_state = MagicMock(return_value="alice")
        result2 = await wrapped(mcp_ctx=mock_ctx)
        assert result2 == "User: alice"


class TestBothInjectionExecution:
    """Tests for using both runtime_adapter and inject_mcp_ctx together."""

    @pytest.mark.asyncio
    async def test_both_injections_same_tool_set(self):
        """Test registering tools with both injection types."""

        @tool
        def tool_with_runtime(runtime: ToolRuntime[UserContext]) -> str:
            """Tool using runtime."""
            return f"Runtime user: {runtime.context.user_id}"

        @tool
        def tool_with_ctx(mcp_ctx: Context) -> str:
            """Tool using mcp_ctx."""
            return f"Request: {mcp_ctx.request_id}"

        @tool
        def tool_with_args_only(value: int) -> int:
            """Tool with only regular args."""
            return value * 2

        def runtime_adapter(mcp_ctx: Context) -> ToolRuntime[UserContext]:
            return ToolRuntime(
                state={},
                context=UserContext(user_id="mixed_user", tenant_id="t"),
                config={},
                stream_writer=lambda x: None,
                tool_call_id=None,
                store=None,
            )

        mcp = FastMCP("mixed-server")
        registered = register_tools(
            mcp,
            [tool_with_runtime, tool_with_ctx, tool_with_args_only],
            runtime_adapter=runtime_adapter,
            inject_mcp_ctx=True,
        )

        assert len(registered) == 3
        assert "tool_with_runtime" in registered
        assert "tool_with_ctx" in registered
        assert "tool_with_args_only" in registered

    @pytest.mark.asyncio
    async def test_runtime_adapter_receives_mcp_ctx(self):
        """Test that runtime_adapter receives the mcp_ctx properly."""
        received_ctx = None

        @tool
        def check_ctx(runtime: ToolRuntime[UserContext]) -> str:
            """Check context."""
            return f"User: {runtime.context.user_id}"

        def runtime_adapter(mcp_ctx: Context) -> ToolRuntime[UserContext]:
            nonlocal received_ctx
            received_ctx = mcp_ctx
            # Extract user from context state
            user_id = mcp_ctx.get_state("user_id") or "default_user"
            return ToolRuntime(
                state={},
                context=UserContext(user_id=user_id, tenant_id="t"),
                config={},
                stream_writer=lambda x: None,
                tool_call_id=None,
                store=None,
            )

        wrapped = to_mcp_tool(check_ctx, runtime_adapter=runtime_adapter)
        mock_ctx = _create_mock_context()
        mock_ctx.get_state = MagicMock(return_value="ctx_user")

        result = await wrapped(mcp_ctx=mock_ctx)

        assert received_ctx is mock_ctx
        assert result == "User: ctx_user"


class TestAsyncToolExecution:
    """Tests for async tool execution."""

    @pytest.mark.asyncio
    async def test_async_tool_execution(self):
        """Test async tool is properly awaited."""

        @tool
        async def async_fetch(url: str) -> str:
            """Fetch data asynchronously."""
            return f"Fetched: {url}"

        wrapped = to_mcp_tool(async_fetch)
        result = await wrapped(url="https://example.com")

        assert result == "Fetched: https://example.com"

    @pytest.mark.asyncio
    async def test_async_tool_with_runtime(self):
        """Test async tool with runtime injection."""

        @tool
        async def async_with_runtime(data: str, runtime: ToolRuntime[UserContext]) -> str:
            """Async tool with runtime."""
            return f"User {runtime.context.user_id} fetched: {data}"

        def runtime_adapter(mcp_ctx: Context) -> ToolRuntime[UserContext]:
            return ToolRuntime(
                state={},
                context=UserContext(user_id="async_user", tenant_id="t"),
                config={},
                stream_writer=lambda x: None,
                tool_call_id=None,
                store=None,
            )

        wrapped = to_mcp_tool(async_with_runtime, runtime_adapter=runtime_adapter)
        mock_ctx = _create_mock_context()

        result = await wrapped(mcp_ctx=mock_ctx, data="async_data")

        assert result == "User async_user fetched: async_data"


class TestStructuredToolExecution:
    """Tests for StructuredTool execution."""

    @pytest.mark.asyncio
    async def test_structured_tool_execution(self):
        """Test StructuredTool can be executed."""

        def calculate(a: int, b: int, operation: str = "add") -> int:
            if operation == "add":
                return a + b
            elif operation == "multiply":
                return a * b
            return 0

        structured = StructuredTool.from_function(
            func=calculate,
            name="calculator",
            description="Perform calculations",
        )

        wrapped = to_mcp_tool(structured)

        result1 = await wrapped(a=5, b=3, operation="add")
        assert result1 == 8

        result2 = await wrapped(a=5, b=3, operation="multiply")
        assert result2 == 15

    @pytest.mark.asyncio
    async def test_structured_tool_with_pydantic_schema(self):
        """Test StructuredTool with Pydantic schema execution."""

        class CalcArgs(BaseModel):
            x: int = Field(..., description="First number")
            y: int = Field(..., description="Second number")

        def calc_func(x: int, y: int) -> int:
            return x + y

        structured = StructuredTool.from_function(
            func=calc_func,
            name="pydantic_calc",
            description="Calculate with Pydantic args",
            args_schema=CalcArgs,
        )

        wrapped = to_mcp_tool(structured)
        result = await wrapped(x=10, y=20)

        assert result == 30


class TestContextMethodsExecution:
    """Tests for Context methods (debug, info, progress, etc.) being called correctly."""

    @pytest.mark.asyncio
    async def test_ctx_log_methods(self):
        """Test that ctx.debug/info/warning/error methods are called correctly."""
        log_calls = []

        @tool
        async def logging_tool(message: str, mcp_ctx: Context) -> str:
            """Tool that uses various logging methods."""
            await mcp_ctx.debug(f"Debug: {message}")
            await mcp_ctx.info(f"Info: {message}")
            await mcp_ctx.warning(f"Warning: {message}")
            await mcp_ctx.error(f"Error: {message}")
            return f"Logged: {message}"

        wrapped = to_mcp_tool(logging_tool, inject_mcp_ctx=True)

        # Create mock context with tracked log calls
        mock_ctx = MagicMock(spec=Context)

        async def mock_debug(msg, **kwargs):
            log_calls.append(("debug", msg))

        async def mock_info(msg, **kwargs):
            log_calls.append(("info", msg))

        async def mock_warning(msg, **kwargs):
            log_calls.append(("warning", msg))

        async def mock_error(msg, **kwargs):
            log_calls.append(("error", msg))

        mock_ctx.debug = mock_debug
        mock_ctx.info = mock_info
        mock_ctx.warning = mock_warning
        mock_ctx.error = mock_error

        result = await wrapped(mcp_ctx=mock_ctx, message="test")

        assert result == "Logged: test"
        assert ("debug", "Debug: test") in log_calls
        assert ("info", "Info: test") in log_calls
        assert ("warning", "Warning: test") in log_calls
        assert ("error", "Error: test") in log_calls

    @pytest.mark.asyncio
    async def test_ctx_report_progress(self):
        """Test that ctx.report_progress is called correctly."""
        progress_calls = []

        @tool
        async def progress_tool(steps: int, mcp_ctx: Context) -> str:
            """Tool that reports progress."""
            for i in range(steps):
                await mcp_ctx.report_progress(i + 1, steps, f"Step {i + 1}")
            return f"Completed {steps} steps"

        wrapped = to_mcp_tool(progress_tool, inject_mcp_ctx=True)

        mock_ctx = MagicMock(spec=Context)

        async def mock_report_progress(progress, total=None, message=None):
            progress_calls.append((progress, total, message))

        mock_ctx.report_progress = mock_report_progress

        result = await wrapped(mcp_ctx=mock_ctx, steps=3)

        assert result == "Completed 3 steps"
        assert len(progress_calls) == 3
        assert progress_calls[0] == (1, 3, "Step 1")
        assert progress_calls[1] == (2, 3, "Step 2")
        assert progress_calls[2] == (3, 3, "Step 3")

    @pytest.mark.asyncio
    async def test_ctx_state_management(self):
        """Test that ctx.set_state and get_state work correctly."""
        state_storage = {}

        @tool
        def stateful_tool(key: str, value: str, mcp_ctx: Context) -> str:
            """Tool that uses state management."""
            # Set state
            mcp_ctx.set_state(key, value)
            # Get state back
            retrieved = mcp_ctx.get_state(key)
            return f"Set {key}={value}, got {retrieved}"

        wrapped = to_mcp_tool(stateful_tool, inject_mcp_ctx=True)

        mock_ctx = MagicMock(spec=Context)

        def mock_set_state(k, v):
            state_storage[k] = v

        def mock_get_state(k):
            return state_storage.get(k)

        mock_ctx.set_state = mock_set_state
        mock_ctx.get_state = mock_get_state

        result = await wrapped(mcp_ctx=mock_ctx, key="user", value="alice")

        assert result == "Set user=alice, got alice"
        assert state_storage["user"] == "alice"

    @pytest.mark.asyncio
    async def test_ctx_properties_access(self):
        """Test that ctx properties like request_id, session_id are accessible."""

        @tool
        def id_tool(mcp_ctx: Context) -> dict:
            """Tool that accesses context properties."""
            return {
                "request_id": mcp_ctx.request_id,
                "session_id": mcp_ctx.session_id,
            }

        wrapped = to_mcp_tool(id_tool, inject_mcp_ctx=True)

        mock_ctx = MagicMock(spec=Context)
        mock_ctx.request_id = "req-abc-123"
        mock_ctx.session_id = "sess-xyz-789"

        result = await wrapped(mcp_ctx=mock_ctx)

        assert result["request_id"] == "req-abc-123"
        assert result["session_id"] == "sess-xyz-789"

    @pytest.mark.asyncio
    async def test_ctx_combined_operations(self):
        """Test a realistic tool that combines multiple context operations."""
        operations_log = []

        @tool
        async def complex_tool(data: str, mcp_ctx: Context) -> str:
            """Tool that uses multiple context features."""
            # Log start
            await mcp_ctx.info(f"Starting processing: {data}")

            # Report progress
            await mcp_ctx.report_progress(0, 100, "Starting")

            # Store intermediate state
            mcp_ctx.set_state("processing", data)

            # Report more progress
            await mcp_ctx.report_progress(50, 100, "Halfway")

            # Get stored state
            stored = mcp_ctx.get_state("processing")

            # Report completion
            await mcp_ctx.report_progress(100, 100, "Complete")

            # Log completion
            await mcp_ctx.info(f"Completed processing: {stored}")

            return f"Processed: {stored}"

        wrapped = to_mcp_tool(complex_tool, inject_mcp_ctx=True)

        state_storage = {}

        mock_ctx = MagicMock(spec=Context)

        async def mock_info(msg, **kwargs):
            operations_log.append(("info", msg))

        async def mock_report_progress(progress, total=None, message=None):
            operations_log.append(("progress", progress, total, message))

        def mock_set_state(k, v):
            state_storage[k] = v

        def mock_get_state(k):
            return state_storage.get(k)

        mock_ctx.info = mock_info
        mock_ctx.report_progress = mock_report_progress
        mock_ctx.set_state = mock_set_state
        mock_ctx.get_state = mock_get_state

        result = await wrapped(mcp_ctx=mock_ctx, data="test-data")

        assert result == "Processed: test-data"

        # Verify operations sequence
        assert operations_log[0] == ("info", "Starting processing: test-data")
        assert operations_log[1] == ("progress", 0, 100, "Starting")
        assert operations_log[2] == ("progress", 50, 100, "Halfway")
        assert operations_log[3] == ("progress", 100, 100, "Complete")
        assert operations_log[4] == ("info", "Completed processing: test-data")

    @pytest.mark.asyncio
    async def test_ctx_in_runtime_adapter(self):
        """Test that context is properly passed through runtime adapter."""
        adapter_received_ctx = None

        @tool
        def user_tool(runtime: ToolRuntime[UserContext]) -> str:
            """Tool using runtime with context from adapter."""
            return f"User: {runtime.context.user_id}"

        def runtime_adapter(mcp_ctx: Context) -> ToolRuntime[UserContext]:
            nonlocal adapter_received_ctx
            adapter_received_ctx = mcp_ctx

            # Use context to get user info
            user_id = mcp_ctx.get_state("current_user") or "unknown"

            return ToolRuntime(
                state={},
                context=UserContext(user_id=user_id, tenant_id="default"),
                config={},
                stream_writer=lambda x: None,
                tool_call_id=None,
                store=None,
            )

        wrapped = to_mcp_tool(user_tool, runtime_adapter=runtime_adapter)

        mock_ctx = MagicMock(spec=Context)
        mock_ctx.get_state = MagicMock(return_value="alice")

        result = await wrapped(mcp_ctx=mock_ctx)

        assert result == "User: alice"
        assert adapter_received_ctx is mock_ctx
        mock_ctx.get_state.assert_called_with("current_user")

    @pytest.mark.asyncio
    async def test_ctx_async_resource_operations(self):
        """Test async context operations like list_resources."""

        @tool
        async def resource_tool(mcp_ctx: Context) -> int:
            """Tool that lists resources."""
            resources = await mcp_ctx.list_resources()
            return len(resources)

        wrapped = to_mcp_tool(resource_tool, inject_mcp_ctx=True)

        mock_ctx = MagicMock(spec=Context)

        async def mock_list_resources():
            return [{"uri": "file:///a.txt"}, {"uri": "file:///b.txt"}]

        mock_ctx.list_resources = mock_list_resources

        result = await wrapped(mcp_ctx=mock_ctx)

        assert result == 2
