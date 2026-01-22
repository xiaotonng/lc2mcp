"""
Context injection utilities for MCP context and ToolRuntime.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, get_type_hints

from lc2mcp.types import Context, JsonSchemaDict, RuntimeAdapter

if TYPE_CHECKING:
    pass

# Parameter names that indicate injectable context
INJECTABLE_PARAMS: frozenset[str] = frozenset({"mcp_ctx", "runtime"})


def detect_injectable_params(func: Callable[..., Any]) -> dict[str, type]:
    """
    Detect which parameters in a function signature should be injected.

    Args:
        func: The function to analyze.

    Returns:
        A dict mapping parameter names to their type hints for injectable params.
        Keys can be "mcp_ctx" or "runtime".
    """
    try:
        hints = get_type_hints(func)
    except Exception:
        # get_type_hints can fail with forward references
        hints = getattr(func, "__annotations__", {})

    injectables: dict[str, type] = {}
    for param_name in INJECTABLE_PARAMS:
        if param_name in hints:
            injectables[param_name] = hints[param_name]

    return injectables


def get_non_injectable_params(func: Callable[..., Any]) -> dict[str, inspect.Parameter]:
    """
    Get parameters that should be exposed in the MCP schema (non-injectable).

    Args:
        func: The function to analyze.

    Returns:
        A dict of parameter name -> Parameter for non-injectable params.
    """
    sig = inspect.signature(func)
    return {name: param for name, param in sig.parameters.items() if name not in INJECTABLE_PARAMS}


def filter_schema_properties(
    schema: JsonSchemaDict,
    injectable_params: set[str],
) -> JsonSchemaDict:
    """
    Remove injectable parameters from a JSON schema.

    Args:
        schema: The original JSON schema.
        injectable_params: Set of parameter names to remove.

    Returns:
        A new schema with injectable params removed from properties and required.
    """
    if not injectable_params:
        return schema

    result = schema.copy()

    # Filter properties
    if "properties" in result:
        result["properties"] = {
            k: v for k, v in result["properties"].items() if k not in injectable_params
        }

    # Filter required
    if "required" in result:
        result["required"] = [r for r in result["required"] if r not in injectable_params]
        if not result["required"]:
            del result["required"]

    return result


def build_call_kwargs(
    user_kwargs: dict[str, Any],
    injectable_params: dict[str, type],
    mcp_ctx: Context | None,
    runtime_adapter: RuntimeAdapter | None,
) -> dict[str, Any]:
    """
    Build the complete kwargs for calling a LangChain tool.

    Args:
        user_kwargs: Arguments provided by the MCP client.
        injectable_params: Dict of injectable param names to their types.
        mcp_ctx: The FastMCP Context object (if available).
        runtime_adapter: Function to convert Context to ToolRuntime (if provided).

    Returns:
        Complete kwargs including both user args and injected context.
    """
    kwargs = user_kwargs.copy()

    # Inject mcp_ctx if the tool expects it
    if "mcp_ctx" in injectable_params and mcp_ctx is not None:
        kwargs["mcp_ctx"] = mcp_ctx

    # Inject runtime if the tool expects it and adapter is provided
    if "runtime" in injectable_params and runtime_adapter is not None and mcp_ctx is not None:
        kwargs["runtime"] = runtime_adapter(mcp_ctx)

    return kwargs
