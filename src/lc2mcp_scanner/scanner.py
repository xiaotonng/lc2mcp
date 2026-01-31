"""
Tool and resource scanner for lc2mcp-scanner.

Provides utilities to scan directories for LangChain tools, FastMCP tools,
and FastMCP resources.
"""

import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def is_langchain_tool(obj: Any) -> bool:
    """
    Check if an object is a LangChain tool.

    Detects:
    - langchain.tools.StructuredTool instances
    - Functions decorated with @tool (have __tool_schema__ or similar markers)
    """
    # Check for StructuredTool or BaseTool
    try:
        from langchain_core.tools import BaseTool as LCBaseTool
        if isinstance(obj, LCBaseTool):
            return True
    except ImportError:
        pass

    # Check for @tool decorated functions (they become StructuredTool)
    if hasattr(obj, "name") and hasattr(obj, "description") and hasattr(obj, "invoke"):
        return True

    return False


def is_fastmcp_tool(obj: Any) -> bool:
    """
    Check if an object is a FastMCP tool function.

    Detects functions registered with @mcp.tool decorator.
    FastMCP tools have specific attributes set by the decorator.
    """
    if callable(obj) and hasattr(obj, "_mcp_tool"):
        return True
    
    # Also check for Tool class instances from fastmcp
    try:
        from fastmcp.tools import Tool as FastMCPTool
        if isinstance(obj, FastMCPTool):
            return True
    except ImportError:
        pass
    
    return False


def is_fastmcp_resource(obj: Any) -> bool:
    """
    Check if an object is a FastMCP resource function.

    Detects functions registered with @mcp.resource decorator.
    """
    if callable(obj) and hasattr(obj, "_mcp_resource"):
        return True
    return False


def _load_module_from_file(file_path: Path) -> Any | None:
    """
    Dynamically load a Python module from a file path.

    Args:
        file_path: Path to the .py file

    Returns:
        Loaded module or None if loading fails
    """
    try:
        module_name = file_path.stem
        # Create unique module name to avoid conflicts
        unique_name = f"_lc2mcp_scanned_{file_path.parent.name}_{module_name}"

        spec = importlib.util.spec_from_file_location(unique_name, file_path)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.warning(f"Failed to load module from {file_path}: {e}")
        return None


def _iter_python_files(directory: Path, recursive: bool = True) -> list[Path]:
    """
    Iterate over Python files in a directory.

    Args:
        directory: Directory to scan
        recursive: Whether to scan subdirectories

    Returns:
        List of .py file paths
    """
    if not directory.exists() or not directory.is_dir():
        return []

    pattern = "**/*.py" if recursive else "*.py"
    files = list(directory.glob(pattern))

    # Filter out __pycache__, __init__.py (unless it contains tools)
    return [
        f for f in files
        if "__pycache__" not in str(f) and f.name != "__init__.py"
    ]


def scan_tools(
    dirs: list[str | Path],
    recursive: bool = True,
    include_init: bool = False,
) -> list[Callable]:
    """
    Scan directories for LangChain tools.

    Looks for:
    - Functions decorated with @tool from langchain_core.tools
    - StructuredTool or BaseTool instances

    Args:
        dirs: List of directories to scan
        recursive: Whether to scan subdirectories
        include_init: Whether to scan __init__.py files

    Returns:
        List of discovered LangChain tools
    """
    tools: list[Callable] = []
    seen_names: set[str] = set()

    for dir_path in dirs:
        directory = Path(dir_path) if isinstance(dir_path, str) else dir_path
        if not directory.exists():
            logger.debug(f"Directory does not exist: {directory}")
            continue

        files = _iter_python_files(directory, recursive)
        if include_init:
            init_file = directory / "__init__.py"
            if init_file.exists():
                files.append(init_file)

        for file_path in files:
            module = _load_module_from_file(file_path)
            if module is None:
                continue

            # Scan module for tools
            for name, obj in inspect.getmembers(module):
                if name.startswith("_"):
                    continue

                if is_langchain_tool(obj):
                    tool_name = getattr(obj, "name", name)
                    if tool_name not in seen_names:
                        tools.append(obj)
                        seen_names.add(tool_name)
                        logger.info(f"Discovered tool: {tool_name} from {file_path}")

    return tools


def scan_fastmcp_tools(
    dirs: list[str | Path],
    recursive: bool = True,
) -> list[Callable]:
    """
    Scan directories for FastMCP tool functions.

    Looks for functions with @mcp.tool decorator markers.

    Args:
        dirs: List of directories to scan
        recursive: Whether to scan subdirectories

    Returns:
        List of discovered FastMCP tool functions
    """
    tools: list[Callable] = []
    seen_names: set[str] = set()

    for dir_path in dirs:
        directory = Path(dir_path) if isinstance(dir_path, str) else dir_path
        if not directory.exists():
            continue

        for file_path in _iter_python_files(directory, recursive):
            module = _load_module_from_file(file_path)
            if module is None:
                continue

            for name, obj in inspect.getmembers(module):
                if name.startswith("_"):
                    continue

                if is_fastmcp_tool(obj):
                    if name not in seen_names:
                        tools.append(obj)
                        seen_names.add(name)
                        logger.info(f"Discovered FastMCP tool: {name} from {file_path}")

    return tools


def scan_resources(
    dirs: list[str | Path],
    recursive: bool = True,
) -> list[Callable]:
    """
    Scan directories for FastMCP resource registrar functions.

    Looks for functions named `register_xxx(mcp: FastMCP)` that register resources.
    This is the standard pattern for FastMCP resource registration.

    Args:
        dirs: List of directories to scan
        recursive: Whether to scan subdirectories

    Returns:
        List of registrar functions that accept (mcp: FastMCP)
    """
    registrars: list[Callable] = []
    seen_names: set[str] = set()

    for dir_path in dirs:
        directory = Path(dir_path) if isinstance(dir_path, str) else dir_path
        if not directory.exists():
            logger.debug(f"Directory does not exist: {directory}")
            continue

        files = _iter_python_files(directory, recursive)

        for file_path in files:
            module = _load_module_from_file(file_path)
            if module is None:
                continue

            for name, obj in inspect.getmembers(module):
                # Look for register_xxx functions
                if name.startswith("register_") and callable(obj):
                    # Check function signature accepts mcp parameter
                    try:
                        sig = inspect.signature(obj)
                        params = list(sig.parameters.keys())
                        if params and params[0] == "mcp":
                            if name not in seen_names:
                                registrars.append(obj)
                                seen_names.add(name)
                                logger.info(f"Discovered resource registrar: {name} from {file_path}")
                    except (ValueError, TypeError):
                        continue

    return registrars


def get_tool_info(tool: Callable) -> dict[str, Any]:
    """
    Extract metadata from a tool for display purposes.

    Args:
        tool: A LangChain or FastMCP tool

    Returns:
        Dictionary with tool metadata (name, description, source, etc.)
    """
    info: dict[str, Any] = {
        "name": getattr(tool, "name", tool.__name__ if hasattr(tool, "__name__") else "unknown"),
        "description": "",
        "source_file": "",
        "source_code": "",
    }

    # Get description
    if hasattr(tool, "description"):
        info["description"] = tool.description
    elif hasattr(tool, "__doc__") and tool.__doc__:
        info["description"] = tool.__doc__.strip().split("\n")[0]

    # Get source file and code
    try:
        if hasattr(tool, "func"):
            # StructuredTool wraps the actual function
            func = tool.func
        else:
            func = tool

        source_file = inspect.getfile(func)
        info["source_file"] = source_file

        source_code = inspect.getsource(func)
        info["source_code"] = source_code
    except (TypeError, OSError):
        pass

    return info
