"""
Advanced example: Context injection with ToolRuntime.

This demonstrates how to inject MCP context into LangChain tools using:
1. ToolRuntime[ContextT] - single generic (StateT defaults to dict)
2. ToolRuntime[ContextT, StateT] - dual generics with custom state
3. Direct mcp_ctx: Context injection for raw MCP context access
"""

from dataclasses import dataclass

from fastmcp import Context, FastMCP
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from lc2mcp import register_tools

# ============================================================================
# 1. Define your App Context and State
# ============================================================================


@dataclass(frozen=True)
class UserContext:
    """User context injected into tools."""

    user_id: str
    tenant_id: str


@dataclass
class AppState:
    """Custom app state (optional, for dual-generic scenario)."""

    request_count: int = 0


# ============================================================================
# 2. Define tools with different injection patterns
# ============================================================================


# Pattern A: ToolRuntime[UserContext] - single generic, StateT defaults to dict
@tool
def whoami(runtime: ToolRuntime[UserContext]) -> str:
    """Return the current user (single generic pattern)."""
    return f"Hello, user {runtime.context.user_id} from tenant {runtime.context.tenant_id}"


@tool
def get_state_value(key: str, runtime: ToolRuntime[UserContext]) -> str:
    """Get a value from runtime state (dict by default)."""
    # runtime.state is dict type when using single generic
    value = runtime.state.get(key, "not found")
    return f"State[{key}] = {value}"


# Pattern B: ToolRuntime[UserContext, AppState] - dual generics with custom state
@tool
def whoami_with_state(runtime: ToolRuntime[UserContext, AppState]) -> str:
    """Return user info with request count (dual generic pattern)."""
    return (
        f"Hello, user {runtime.context.user_id} "
        f"(request #{runtime.state.request_count})"
    )


# Pattern C: Direct mcp_ctx injection for raw MCP context access
@tool
def request_id(mcp_ctx: Context) -> str:
    """Return the current request ID from MCP context."""
    return mcp_ctx.request_id


@tool
def get_session_id(mcp_ctx: Context) -> str:
    """Return the current session ID from MCP context."""
    return mcp_ctx.session_id


# Pattern D: Using Context methods (debug, info, progress, etc.)
@tool
async def process_with_logging(data: str, mcp_ctx: Context) -> str:
    """Process data with logging and progress reporting.

    This demonstrates using Context methods to send logs and progress
    updates back to the MCP client.
    """
    # Log start
    await mcp_ctx.info(f"Starting to process: {data}")

    # Report progress (will be sent to MCP client if progress token is set)
    await mcp_ctx.report_progress(0, 100, "Starting")

    # Store state for this request
    mcp_ctx.set_state("current_data", data)

    # Simulate processing steps with progress
    await mcp_ctx.debug("Step 1: Validation")
    await mcp_ctx.report_progress(33, 100, "Validating")

    await mcp_ctx.debug("Step 2: Processing")
    await mcp_ctx.report_progress(66, 100, "Processing")

    # Get stored state
    stored = mcp_ctx.get_state("current_data")

    await mcp_ctx.debug("Step 3: Finalizing")
    await mcp_ctx.report_progress(100, 100, "Complete")

    # Log completion
    await mcp_ctx.info(f"Completed processing: {stored}")

    return f"Processed: {stored}"


# ============================================================================
# 3. Runtime adapters for different scenarios
# ============================================================================


def runtime_adapter_simple(mcp_ctx: Context) -> ToolRuntime[UserContext]:
    """Adapter for single generic: ToolRuntime[UserContext].

    StateT defaults to dict, useful when you don't need custom state.
    """
    user_id = mcp_ctx.get_state("user_id") or "demo_user"
    tenant_id = mcp_ctx.get_state("tenant_id") or "demo_tenant"

    return ToolRuntime(
        state={"initialized": True},  # dict is default StateT
        context=UserContext(user_id=user_id, tenant_id=tenant_id),
        config={},
        stream_writer=lambda x: None,
        tool_call_id=None,
        store=None,
    )


def runtime_adapter_with_state(mcp_ctx: Context) -> ToolRuntime[UserContext, AppState]:
    """Adapter for dual generics: ToolRuntime[UserContext, AppState].

    Use this when you need custom typed state.
    """
    user_id = mcp_ctx.get_state("user_id") or "demo_user"
    tenant_id = mcp_ctx.get_state("tenant_id") or "demo_tenant"
    count = mcp_ctx.get_state("request_count") or 0

    return ToolRuntime(
        state=AppState(request_count=count),
        context=UserContext(user_id=user_id, tenant_id=tenant_id),
        config={},
        stream_writer=lambda x: None,
        tool_call_id=None,
        store=None,
    )


# ============================================================================
# 4. FastMCP server setup
# ============================================================================

mcp = FastMCP("auth-server")

# Register all tools with runtime adapter
# Note: All tools share the same adapter, which works for both patterns
register_tools(
    mcp,
    [
        # Single generic pattern
        whoami,
        get_state_value,
        # Dual generic pattern
        whoami_with_state,
        # Direct mcp_ctx pattern
        request_id,
        get_session_id,
        # Context methods pattern (logging, progress)
        process_with_logging,
    ],
    runtime_adapter=runtime_adapter_simple,  # or runtime_adapter_with_state
    inject_mcp_ctx=True,
)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)
