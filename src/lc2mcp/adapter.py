"""
Core adapter logic for converting LangChain tools to FastMCP tools.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any, Callable, Sequence, cast

from pydantic import Field, create_model

from lc2mcp.context import (
    build_call_kwargs,
    detect_injectable_params,
    filter_schema_properties,
)
from lc2mcp.naming import (
    OnNameConflict,
    apply_prefix,
    get_registered_tool_names,
    resolve_conflict,
)
from lc2mcp.schema import extract_schema_from_tool
from lc2mcp.types import (
    BaseModel,
    BaseTool,
    Context,
    FastMCP,
    JsonSchemaDict,
    LangChainTool,
    RuntimeAdapter,
)

if TYPE_CHECKING:
    pass


def _is_default_structuredtool_method(method: Callable[..., Any]) -> bool:
    """Check if a method is the default StructuredTool._arun/_run.

    StructuredTool's default _arun requires a 'config' argument which we don't
    provide, so we need to detect and avoid calling it directly.
    """
    # Check if method belongs to StructuredTool class (not overridden)
    from langchain_core.tools import StructuredTool

    try:
        # Get the class that defines this method
        if hasattr(method, "__self__"):
            obj = method.__self__
            method_name = method.__name__
            # Check if the method is inherited from StructuredTool
            for cls in type(obj).__mro__:
                if method_name in cls.__dict__:
                    return cls is StructuredTool
    except Exception:
        pass
    return False


def _extract_tool_info(tool: LangChainTool) -> tuple[str, str, JsonSchemaDict | None]:
    """
    Extract name, description, and schema from a LangChain tool.

    Args:
        tool: A LangChain BaseTool or callable decorated with @tool.

    Returns:
        Tuple of (name, description, args_schema or None).
    """
    # Handle BaseTool instances
    if isinstance(tool, BaseTool):
        name = tool.name
        description = tool.description or ""
        schema = extract_schema_from_tool(tool)
        return name, description, schema

    # Handle @tool decorated functions (they become BaseTool instances)
    # But if passed as callable, extract from function metadata
    if callable(tool):
        raw_name = getattr(tool, "name", None) or getattr(tool, "__name__", "unknown_tool")
        func_name = cast(str, raw_name)
        func_desc = cast(str, getattr(tool, "description", None) or tool.__doc__ or "")
        schema = extract_schema_from_tool(tool)
        return func_name, func_desc, schema

    raise TypeError(f"Expected BaseTool or callable, got {type(tool)}")


def _create_wrapper(
    tool: LangChainTool,
    injectable_params: dict[str, type],
    inject_mcp_ctx: bool,
    runtime_adapter: RuntimeAdapter | None,
) -> Callable[..., Any]:
    """
    Create an async wrapper function for a LangChain tool.

    The wrapper handles:
    - Converting sync tools to async
    - Injecting mcp_ctx and runtime if configured
    - Calling the underlying tool with the correct arguments
    """

    async def wrapper(mcp_ctx: Context | None = None, **kwargs: Any) -> Any:
        # Build complete kwargs with injections
        call_kwargs = build_call_kwargs(
            user_kwargs=kwargs,
            injectable_params=injectable_params if (inject_mcp_ctx or runtime_adapter) else {},
            mcp_ctx=mcp_ctx if (inject_mcp_ctx or runtime_adapter) else None,
            runtime_adapter=runtime_adapter,
        )

        # Call the tool
        if isinstance(tool, BaseTool):
            # For BaseTool, we prefer calling the underlying function directly
            # to bypass LangChain's internal Pydantic validation (which doesn't
            # know about our injected parameters like runtime/mcp_ctx).
            #
            # For StructuredTool (created via @tool decorator), .func contains
            # the original function; for custom BaseTool subclasses, use _run/_arun.
            func = getattr(tool, "func", None)
            coroutine = getattr(tool, "coroutine", None)

            if coroutine is not None:
                # StructuredTool with async coroutine (async @tool decorator)
                result = await coroutine(**call_kwargs)
            elif func is not None:
                # StructuredTool: call the underlying function directly
                if inspect.iscoroutinefunction(func):
                    result = await func(**call_kwargs)
                else:
                    result = await asyncio.to_thread(func, **call_kwargs)
            else:
                # Custom BaseTool subclass: check for _arun/_run methods
                # Note: _arun in StructuredTool requires config, so we avoid it
                _arun = getattr(tool, "_arun", None)
                _run = getattr(tool, "_run", None)

                # Check if _arun is overridden (not default StructuredTool._arun)
                if _arun is not None and not _is_default_structuredtool_method(_arun):
                    result = await _arun(**call_kwargs)
                elif _run is not None:
                    result = await asyncio.to_thread(_run, **call_kwargs)
                else:
                    # Last resort: use invoke (may fail with injected params)
                    if hasattr(tool, "ainvoke"):
                        result = await tool.ainvoke(call_kwargs)
                    else:
                        result = await asyncio.to_thread(tool.invoke, call_kwargs)
        else:
            # Regular callable
            if inspect.iscoroutinefunction(tool):
                result = await tool(**call_kwargs)
            else:
                result = await asyncio.to_thread(tool, **call_kwargs)

        return result

    return wrapper


def to_mcp_tool(
    tool: LangChainTool,
    *,
    name: str | None = None,
    description: str | None = None,
    args_schema: JsonSchemaDict | None = None,
    inject_mcp_ctx: bool = False,
    runtime_adapter: RuntimeAdapter | None = None,
) -> Callable[..., Any]:
    """
    Convert a single LangChain tool into a FastMCP-compatible callable.

    This function wraps a LangChain tool and returns an async function that can
    be registered with FastMCP using `mcp.add_tool()`.

    Args:
        tool: A LangChain BaseTool or @tool decorated function.
        name: Override the tool name. Defaults to the tool's name.
        description: Override the tool description. Defaults to the tool's description.
        args_schema: Override the arguments schema. Defaults to the tool's schema.
        inject_mcp_ctx: If True, inject `mcp_ctx: Context` into tools that declare it.
        runtime_adapter: A function that converts FastMCP Context to ToolRuntime.

    Returns:
        An async callable suitable for FastMCP registration.
    """
    # Extract tool info
    tool_name, tool_desc, tool_schema = _extract_tool_info(tool)

    # Apply overrides
    final_name = name or tool_name
    final_desc = description or tool_desc
    final_schema: JsonSchemaDict = (
        args_schema
        or tool_schema
        or {
            "type": "object",
            "properties": {},
        }
    )

    # Detect injectable parameters
    # For StructuredTool, use .coroutine (async) or .func (sync) to get the original function
    # For other BaseTool, use ._run; for callables, use the tool directly
    if isinstance(tool, BaseTool):
        func = getattr(tool, "coroutine", None) or getattr(tool, "func", None) or tool._run
    else:
        func = tool
    needs_injection = inject_mcp_ctx or runtime_adapter is not None
    injectable_params = detect_injectable_params(func) if needs_injection else {}

    # Filter schema to remove injectable params
    if injectable_params:
        final_schema = filter_schema_properties(final_schema, set(injectable_params.keys()))

    # Create the wrapper
    wrapper = _create_wrapper(tool, injectable_params, inject_mcp_ctx, runtime_adapter)

    # Attach metadata for FastMCP
    wrapper.__name__ = final_name
    wrapper.__doc__ = final_desc
    setattr(wrapper, "_mcp_tool_schema", final_schema)
    setattr(wrapper, "_mcp_tool_name", final_name)
    setattr(wrapper, "_mcp_tool_description", final_desc)

    return wrapper


def register_tools(
    mcp: FastMCP,
    tools: Sequence[LangChainTool],
    *,
    name_prefix: str | None = None,
    on_name_conflict: OnNameConflict = "error",
    inject_mcp_ctx: bool = False,
    runtime_adapter: RuntimeAdapter | None = None,
) -> list[str]:
    """
    Register a list of LangChain tools on a FastMCP server.

    This is the primary function for bulk-registering LangChain tools as MCP tools.

    Args:
        mcp: A FastMCP server instance.
        tools: List of LangChain tools (BaseTool instances or @tool decorated functions).
        name_prefix: Optional prefix for all tool names (e.g., "finance." -> "finance.get_stock").
        on_name_conflict: How to handle name collisions:
            - "error": Raise ValueError (default)
            - "overwrite": Replace the existing tool
            - "suffix": Add numeric suffix (tool_2, tool_3, ...)
        inject_mcp_ctx: If True, inject `mcp_ctx: Context` into tools that declare it.
        runtime_adapter: A function that converts FastMCP Context to ToolRuntime.
            If provided, tools declaring `runtime: ToolRuntime[T]` will receive it.

    Returns:
        List of registered tool names.

    Raises:
        ValueError: If on_name_conflict="error" and a name collision occurs.
        TypeError: If a tool is not a valid LangChain tool.
    """
    registered_names: list[str] = []
    existing_names = get_registered_tool_names(mcp)

    for tool in tools:
        # Convert to MCP tool
        wrapped = to_mcp_tool(
            tool,
            inject_mcp_ctx=inject_mcp_ctx,
            runtime_adapter=runtime_adapter,
        )

        # Get the tool name and apply prefix
        tool_name: str = getattr(wrapped, "_mcp_tool_name")
        prefixed_name = apply_prefix(tool_name, name_prefix)

        # Resolve conflicts
        final_name = resolve_conflict(prefixed_name, existing_names, on_name_conflict)

        # Update wrapper with final name
        wrapped.__name__ = final_name
        setattr(wrapped, "_mcp_tool_name", final_name)

        # Get schema and description
        schema: JsonSchemaDict = getattr(
            wrapped, "_mcp_tool_schema", {"type": "object", "properties": {}}
        )
        description: str = getattr(wrapped, "_mcp_tool_description", "")

        # Register with FastMCP using the @mcp.tool pattern
        # FastMCP expects us to use the decorator or add_tool method
        needs_context = inject_mcp_ctx or runtime_adapter is not None
        _register_single_tool(mcp, wrapped, final_name, description, schema, needs_context)

        # Track registered names
        existing_names.add(final_name)
        registered_names.append(final_name)

    return registered_names


def _json_type_to_python(json_type: str | list[str], prop_schema: JsonSchemaDict) -> Any:
    """Convert JSON Schema type to Python type."""
    if isinstance(json_type, list):
        # Handle nullable types like ["string", "null"]
        non_null = [t for t in json_type if t != "null"]
        if non_null:
            json_type = non_null[0]
        else:
            return Any

    type_map: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return type_map.get(json_type, Any)


def _create_pydantic_model_from_schema(
    name: str,
    schema: JsonSchemaDict,
) -> type[BaseModel]:
    """
    Create a Pydantic model from a JSON schema.

    Args:
        name: Name for the model class.
        schema: JSON schema dict with properties and required fields.

    Returns:
        A dynamically created Pydantic model class.
    """
    properties: dict[str, JsonSchemaDict] = schema.get("properties", {})
    required_fields: set[str] = set(schema.get("required", []))

    field_definitions: dict[str, tuple[Any, Any]] = {}
    for field_name, field_schema in properties.items():
        field_type = _json_type_to_python(
            field_schema.get("type", "string"),
            field_schema,
        )
        field_desc: str = field_schema.get("description", "")
        default = field_schema.get("default", ...)

        if field_name in required_fields and default is ...:
            # Required field with no default
            field_definitions[field_name] = (
                field_type,
                Field(..., description=field_desc) if field_desc else ...,
            )
        else:
            # Optional field or has default
            if default is ...:
                default = None
                field_type = field_type | None
            field_definitions[field_name] = (
                field_type,
                Field(default=default, description=field_desc) if field_desc else default,
            )

    # Create a valid Python identifier for the model name
    model_name = "".join(c if c.isalnum() else "_" for c in name).strip("_") + "Args"

    return create_model(model_name, **field_definitions)  # type: ignore[call-overload, no-any-return]


def _register_single_tool(
    mcp: FastMCP,
    wrapper: Callable[..., Any],
    name: str,
    description: str,
    schema: JsonSchemaDict,
    needs_context: bool,
) -> None:
    """
    Register a single wrapped tool with FastMCP.

    This handles the actual registration with FastMCP's tool system.

    Args:
        mcp: FastMCP server instance.
        wrapper: The wrapped tool function.
        name: Tool name.
        description: Tool description.
        schema: JSON Schema for tool arguments.
        needs_context: Whether the tool needs MCP context injection.
    """
    # Create a Pydantic model from the schema for proper parameter handling
    ArgsModel = _create_pydantic_model_from_schema(name, schema)

    # Create the tool function that FastMCP expects
    # We need to create a function with explicit parameters, not **kwargs
    # Note: We use a factory function to avoid Python 3.14 annotation issues
    if needs_context:
        _create_and_register_tool_with_ctx(mcp, name, description, ArgsModel, wrapper)
    else:
        _create_and_register_tool_no_ctx(mcp, name, description, ArgsModel, wrapper)


def _create_and_register_tool_with_ctx(
    mcp: FastMCP,
    name: str,
    description: str,
    args_model: type[BaseModel],
    wrapper: Callable[..., Any],
) -> None:
    """Create and register a tool that needs context injection."""
    # Build annotations from model fields + ctx
    annotations: dict[str, Any] = {"ctx": Context}
    for field_name, field_info in args_model.model_fields.items():
        annotations[field_name] = field_info.annotation
    annotations["return"] = Any

    # Create wrapper that unpacks kwargs
    async def tool_func(ctx, **kwargs) -> Any:
        return await wrapper(mcp_ctx=ctx, **kwargs)

    tool_func.__annotations__ = annotations
    tool_func.__name__ = name
    tool_func.__doc__ = description

    # Set default values from model
    import inspect

    params = [inspect.Parameter("ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Context)]
    for field_name, field_info in args_model.model_fields.items():
        default = field_info.default if field_info.default is not None else inspect.Parameter.empty
        if field_info.is_required():
            default = inspect.Parameter.empty
        params.append(
            inspect.Parameter(
                field_name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=field_info.annotation,
            )
        )
    tool_func.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]

    mcp.tool(name=name, description=description)(tool_func)


def _create_and_register_tool_no_ctx(
    mcp: FastMCP,
    name: str,
    description: str,
    args_model: type[BaseModel],
    wrapper: Callable[..., Any],
) -> None:
    """Create and register a tool without context injection."""
    # Build annotations from model fields
    annotations: dict[str, Any] = {}
    for field_name, field_info in args_model.model_fields.items():
        annotations[field_name] = field_info.annotation
    annotations["return"] = Any

    async def tool_func(**kwargs) -> Any:
        return await wrapper(**kwargs)

    tool_func.__annotations__ = annotations
    tool_func.__name__ = name
    tool_func.__doc__ = description

    # Set default values from model
    import inspect

    params = []
    for field_name, field_info in args_model.model_fields.items():
        default = field_info.default if field_info.default is not None else inspect.Parameter.empty
        if field_info.is_required():
            default = inspect.Parameter.empty
        params.append(
            inspect.Parameter(
                field_name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=field_info.annotation,
            )
        )
    tool_func.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]

    mcp.tool(name=name, description=description)(tool_func)
