"""
Schema conversion utilities for transforming Pydantic models to JSON Schema.
"""

from typing import Any, get_origin

from langchain_core.tools import BaseTool
from pydantic import BaseModel, create_model

from lc2mcp.types import JsonSchemaDict, LangChainTool

# Import LangChain's injected tool arg marker
try:
    from langchain_core.tools.base import _DirectlyInjectedToolArg

    HAS_INJECTED_ARG = True
except ImportError:
    HAS_INJECTED_ARG = False
    _DirectlyInjectedToolArg = None  # type: ignore[misc, assignment]


def _is_injected_arg_type(annotation: Any) -> bool:
    """
    Check if a type annotation is an injected tool argument.

    This includes ToolRuntime, InjectedState, and other LangChain injected types.

    Args:
        annotation: A type annotation to check.

    Returns:
        True if the annotation is an injected arg type.
    """
    if not HAS_INJECTED_ARG or _DirectlyInjectedToolArg is None:
        return False

    # Check direct type
    if isinstance(annotation, type) and issubclass(annotation, _DirectlyInjectedToolArg):
        return True

    # Check generic origin (e.g., ToolRuntime[Context, State])
    origin = get_origin(annotation)
    if origin is not None:
        if isinstance(origin, type) and issubclass(origin, _DirectlyInjectedToolArg):
            return True

    return False


def _filter_injected_fields(model: type[BaseModel]) -> type[BaseModel]:
    """
    Create a new Pydantic model with injected arg fields removed.

    Args:
        model: Original Pydantic model class.

    Returns:
        A new model class without injected arg fields.
    """
    if not HAS_INJECTED_ARG:
        return model

    fields_to_keep = {}
    for field_name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        if not _is_injected_arg_type(annotation):
            fields_to_keep[field_name] = (annotation, field_info)

    if len(fields_to_keep) == len(model.model_fields):
        # No fields were filtered, return original
        return model

    # Create a new model with only non-injected fields
    return create_model(  # type: ignore[call-overload, no-any-return]
        f"{model.__name__}Filtered",
        **fields_to_keep,
    )


def pydantic_to_json_schema(
    model: type[BaseModel],
    filter_injected: bool = True,
) -> JsonSchemaDict:
    """
    Convert a Pydantic BaseModel to a JSON Schema dict suitable for MCP.

    Args:
        model: A Pydantic BaseModel class (not an instance).
        filter_injected: If True, filter out LangChain injected arg fields.

    Returns:
        A JSON Schema dictionary with 'type', 'properties', and 'required' fields.
    """
    # Filter out injected fields before generating schema
    if filter_injected:
        model = _filter_injected_fields(model)

    schema = model.model_json_schema()

    # Extract only the fields needed for MCP tool schema
    result: JsonSchemaDict = {
        "type": "object",
        "properties": schema.get("properties", {}),
    }

    # Only include 'required' if there are required fields
    required = schema.get("required", [])
    if required:
        result["required"] = required

    # Handle $defs (nested models) if present
    if "$defs" in schema:
        result["$defs"] = schema["$defs"]

    return result


def extract_schema_from_tool(tool: LangChainTool) -> JsonSchemaDict | None:
    """
    Extract JSON schema from a LangChain tool.

    Supports:
    - Tools with args_schema (Pydantic BaseModel)
    - Tools with get_input_schema() method
    - StructuredTool instances

    Args:
        tool: A LangChain tool (BaseTool, StructuredTool, or @tool decorated function).

    Returns:
        A JSON Schema dict, or None if no schema can be extracted.
    """
    # Try args_schema attribute (common for BaseTool subclasses)
    if hasattr(tool, "args_schema") and tool.args_schema is not None:
        args_schema = tool.args_schema
        if isinstance(args_schema, type) and issubclass(args_schema, BaseModel):
            return pydantic_to_json_schema(args_schema)

    # Try get_input_schema method (for BaseTool instances)
    if isinstance(tool, BaseTool) and hasattr(tool, "get_input_schema"):
        try:
            input_schema = tool.get_input_schema()
            if isinstance(input_schema, type) and issubclass(input_schema, BaseModel):
                return pydantic_to_json_schema(input_schema)
        except Exception:
            pass

    return None
