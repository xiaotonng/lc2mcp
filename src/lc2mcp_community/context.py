"""Common context types for lc2mcp-community tools."""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime


@dataclass(frozen=True)
class UserInfo:
    """User information passed to tools."""

    user_id: int
    username: str
    display_name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    avatar_url: Optional[str] = None
    email: Optional[str] = None


@dataclass(frozen=True)
class ChatContext:
    """Chat context passed to all tools via ToolRuntime."""

    user: Optional[UserInfo] = None
    query: str = ""
    session_id: Optional[int] = None
    file_paths: tuple[str, ...] = field(default_factory=tuple)


def get_context(runtime: "ToolRuntime[ChatContext]") -> Optional[ChatContext]:
    """
    Extract ChatContext from ToolRuntime.
    
    This is a common utility for all tools to get context from runtime.
    
    Args:
        runtime: The ToolRuntime instance passed to the tool
        
    Returns:
        ChatContext if available, None otherwise
    """
    if runtime.context:
        return runtime.context
    if runtime.config and "configurable" in runtime.config:
        return runtime.config["configurable"].get("context")
    return None


def extract_user_id_from_mcp_context(ctx) -> Optional[int]:
    """
    Extract user_id from MCP Context (OAuth token).
    
    This is a shared utility for extracting user_id from FastMCP Context
    objects, typically from OAuth Bearer token claims.
    
    Args:
        ctx: FastMCP Context object
        
    Returns:
        user_id as int if found, None otherwise
    """
    try:
        request_context = ctx.request_context
        if request_context and hasattr(request_context, "request"):
            request = request_context.request
            if request and hasattr(request, "user") and request.user:
                user = request.user
                if hasattr(user, "access_token") and user.access_token:
                    claims = user.access_token.claims
                    user_id = claims.get("user_id") or claims.get("sub")
                    if user_id:
                        return int(user_id)
    except Exception:
        pass
    return None
