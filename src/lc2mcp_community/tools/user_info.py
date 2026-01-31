"""User information tool."""

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from lc2mcp_community.context import ChatContext, get_context


@tool(parse_docstring=True)
def whoami(
    runtime: ToolRuntime[ChatContext],
    format: str = "text",
) -> str:
    """Return the current authenticated user information.

    Args:
        format: Output format, either 'text' or 'json'
    """
    ctx = get_context(runtime)
    if not ctx or not ctx.user:
        return "User information not available."

    user = ctx.user

    if format == "json":
        import json

        return json.dumps(
            {
                "user_id": user.user_id,
                "username": user.username,
                "display_name": user.display_name,
                "email": user.email,
                "age": user.age,
                "gender": user.gender,
            },
            ensure_ascii=False,
        )

    return (
        f"User ID: {user.user_id}\n"
        f"Username: {user.username}\n"
        f"Display Name: {user.display_name}\n"
        f"Email: {user.email or 'N/A'}\n"
        f"Age: {user.age or 'N/A'}\n"
        f"Gender: {user.gender or 'N/A'}"
    )
