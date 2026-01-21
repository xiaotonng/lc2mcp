"""
Basic example: Turn a simple LangChain tool into an MCP server.

This demonstrates the quickstart use case from the README.
"""

from fastmcp import FastMCP
from langchain_core.tools import tool

from lc2mcp import register_tools


# 1. Define a LangChain tool (or import one)
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a specific city."""
    # In a real app, you'd call an API here
    return f"Sunny, 25°C in {city}"


@tool
def get_temperature(city: str, unit: str = "celsius") -> str:
    """Get the temperature for a city in the specified unit."""
    temp = 25 if unit == "celsius" else 77
    return f"The temperature in {city} is {temp}°{unit[0].upper()}"


# 2. Initialize FastMCP server
mcp = FastMCP("weather-server")

# 3. Register the tools
# This automatically converts the schema and wires up the handler
register_tools(mcp, [get_weather, get_temperature])

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
