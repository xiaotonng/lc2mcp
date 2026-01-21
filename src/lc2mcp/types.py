"""
Type definitions for lc2mcp.

This module provides structured type aliases and imports for use throughout the library.
"""

from typing import Any, Callable, TypeAlias, TypeVar

from fastmcp import Context, FastMCP
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel

# Type variable for generic context
ContextT = TypeVar("ContextT")
StateT = TypeVar("StateT")

# JSON Schema type (matches pydantic.json_schema.JsonSchemaValue)
JsonSchemaDict: TypeAlias = dict[str, Any]

# LangChain tool types
LangChainTool: TypeAlias = BaseTool | Callable[..., Any]

# Runtime adapter function type
# Context is FastMCP's context class (fastmcp.server.context.Context)
# ToolRuntime supports both single and dual generic parameters:
#   - ToolRuntime[ContextT] -> ContextT with default StateT=dict
#   - ToolRuntime[ContextT, StateT] -> explicit ContextT and StateT
RuntimeAdapter: TypeAlias = Callable[[Context], ToolRuntime[Any, Any]]

# Re-export commonly used types
__all__ = [
    # FastMCP types
    "FastMCP",
    "Context",
    # LangChain types
    "BaseTool",
    "StructuredTool",
    "ToolRuntime",
    # Pydantic types
    "BaseModel",
    # Type aliases
    "JsonSchemaDict",
    "LangChainTool",
    "RuntimeAdapter",
    # Type variables
    "ContextT",
    "StateT",
]
