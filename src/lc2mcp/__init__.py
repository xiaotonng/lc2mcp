"""
lc2mcp - Convert LangChain tools to FastMCP tools.

This module provides utilities to convert LangChain tools into FastMCP tools,
allowing seamless integration with MCP clients like Claude and Cursor.
"""

from lc2mcp.adapter import register_tools, to_mcp_tool
from lc2mcp.types import (
    BaseTool,
    Context,
    FastMCP,
    JsonSchemaDict,
    LangChainTool,
    RuntimeAdapter,
    StructuredTool,
    ToolRuntime,
)

__version__ = "0.1.2"
__all__ = [
    # Core functions
    "register_tools",
    "to_mcp_tool",
    # Type aliases (for user convenience)
    "BaseTool",
    "Context",
    "FastMCP",
    "JsonSchemaDict",
    "LangChainTool",
    "RuntimeAdapter",
    "StructuredTool",
    "ToolRuntime",
]
