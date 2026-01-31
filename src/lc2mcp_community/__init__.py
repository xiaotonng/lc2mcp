"""
lc2mcp-community - Community tools and resources for lc2mcp.

This package provides ready-to-use LangChain tools and FastMCP resources
that can be scanned and registered with lc2mcp-server or any FastMCP application.
"""

from lc2mcp_community.context import (
    ChatContext,
    UserInfo,
    extract_user_id_from_mcp_context,
    get_context,
)

__version__ = "0.1.0"
__all__ = [
    "UserInfo",
    "ChatContext",
    "get_context",
    "extract_user_id_from_mcp_context",
]
