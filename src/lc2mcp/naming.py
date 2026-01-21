"""
Naming utilities for tool name prefixing and conflict resolution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from fastmcp import FastMCP

OnNameConflict = Literal["error", "overwrite", "suffix"]


def apply_prefix(name: str, prefix: str | None) -> str:
    """
    Apply a prefix to a tool name.

    Args:
        name: The original tool name.
        prefix: The prefix to apply (e.g., "finance."). Can be None.

    Returns:
        The prefixed name, or the original name if prefix is None.
    """
    if prefix is None:
        return name
    return f"{prefix}{name}"


def resolve_conflict(
    name: str,
    existing_names: set[str],
    strategy: OnNameConflict,
) -> str:
    """
    Resolve a name conflict based on the specified strategy.

    Args:
        name: The proposed tool name.
        existing_names: Set of already registered tool names.
        strategy: How to handle conflicts - "error", "overwrite", or "suffix".

    Returns:
        The resolved name to use.

    Raises:
        ValueError: If strategy is "error" and name conflicts.
    """
    if name not in existing_names:
        return name

    if strategy == "error":
        raise ValueError(
            f"Tool name '{name}' already exists. "
            "Use on_name_conflict='overwrite' or 'suffix' to handle duplicates."
        )

    if strategy == "overwrite":
        return name

    # strategy == "suffix"
    counter = 2
    while True:
        new_name = f"{name}_{counter}"
        if new_name not in existing_names:
            return new_name
        counter += 1


def get_registered_tool_names(mcp: FastMCP) -> set[str]:
    """
    Get the set of already registered tool names from a FastMCP server.

    Args:
        mcp: A FastMCP server instance.

    Returns:
        Set of registered tool names.
    """
    # FastMCP stores tools in _tool_manager or similar
    # We need to access the internal tool registry
    if hasattr(mcp, "_tool_manager"):
        tool_manager = mcp._tool_manager
        if hasattr(tool_manager, "_tools"):
            return set(tool_manager._tools.keys())

    # Fallback: try to access tools directly
    if hasattr(mcp, "_tools"):
        return set(mcp._tools.keys())

    return set()
