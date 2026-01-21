"""
Example: Using name prefixes and conflict handling.

This demonstrates how to namespace tools and handle name collisions
when registering multiple tool sets.
"""

from fastmcp import FastMCP
from langchain_core.tools import tool

from lc2mcp import register_tools


# Define some finance tools
@tool
def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a symbol."""
    return f"Stock {symbol}: $150.00"


@tool
def analyze(data: str) -> str:
    """Analyze financial data."""
    return f"Financial analysis of: {data}"


# Define some ops tools (note: also has an 'analyze' tool)
@tool
def check_status(service: str) -> str:
    """Check the status of a service."""
    return f"Service {service}: healthy"


@tool
def analyze(logs: str) -> str:  # noqa: F811 - intentional redefinition
    """Analyze operational logs."""
    return f"Ops analysis of: {logs}"


# Create server
mcp = FastMCP("multi-domain")

# Register finance tools with prefix
# If a name collision occurs: raise an error (default)
register_tools(
    mcp,
    [get_stock_price, analyze],
    name_prefix="finance.",
    on_name_conflict="error",
)

# Register ops tools with prefix and auto-suffix on collision
register_tools(
    mcp,
    [check_status, analyze],
    name_prefix="ops.",
    on_name_conflict="suffix",
)

if __name__ == "__main__":
    print("Registered tools:")
    # In practice, tools would be: finance.get_stock_price, finance.analyze,
    # ops.check_status, ops.analyze (no conflict due to different prefixes)
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8003)
