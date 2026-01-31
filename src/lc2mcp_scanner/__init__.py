"""
lc2mcp-scanner - Scan directories for LangChain and FastMCP tools/resources.

This module provides utilities to discover tools and resources from Python
directories, supporting both LangChain and FastMCP decorators.
"""

from lc2mcp_scanner.scanner import (
    get_tool_info,
    is_fastmcp_resource,
    is_fastmcp_tool,
    is_langchain_tool,
    scan_fastmcp_tools,
    scan_resources,
    scan_tools,
)

__version__ = "0.1.0"
__all__ = [
    # Scan functions
    "scan_tools",
    "scan_fastmcp_tools",
    "scan_resources",
    # Detection functions
    "is_langchain_tool",
    "is_fastmcp_tool",
    "is_fastmcp_resource",
    # Utility functions
    "get_tool_info",
]
