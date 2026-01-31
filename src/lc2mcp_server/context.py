"""ChatContext and runtime_adapter for lc2mcp tools."""

import asyncio
import logging
from typing import Optional

from fastmcp import Context
from langgraph.prebuilt import ToolRuntime

# Re-export from community for compatibility
from lc2mcp_community.context import (
    ChatContext,
    UserInfo,
    extract_user_id_from_mcp_context,
)

from .database import async_session
from .models import User

logger = logging.getLogger(__name__)


async def get_user_info_by_id(user_id: int) -> Optional[UserInfo]:
    """Fetch user info from database by user ID."""
    async with async_session() as db:
        user = await db.get(User, user_id)
        if not user:
            return None
        return UserInfo(
            user_id=user.id,
            username=user.username,
            display_name=user.display_name,
            age=user.age,
            gender=user.gender,
            avatar_url=user.avatar_url,
            email=user.email,
        )


def _get_user_id(mcp_ctx: Context) -> Optional[int]:
    """
    Extract user_id from MCP Context.
    
    Sources (in order):
    1. OAuth Bearer token (via request.user.access_token.claims)
    2. State set by the application (via mcp_ctx.get_state)
    """
    # Try OAuth token first
    user_id = extract_user_id_from_mcp_context(mcp_ctx)
    if user_id:
        return user_id

    # Fallback to state (for internal calls)
    state_user_id = mcp_ctx.get_state("user_id")
    if state_user_id:
        return int(state_user_id)

    return None


def _fetch_user_info_sync(user_id: int) -> Optional[UserInfo]:
    """Synchronously fetch user info (handles event loop edge cases)."""
    import concurrent.futures

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, get_user_info_by_id(user_id))
                return future.result()
        else:
            return loop.run_until_complete(get_user_info_by_id(user_id))
    except Exception:
        return None


def runtime_adapter(mcp_ctx: Context) -> ToolRuntime[ChatContext]:
    """
    lc2mcp runtime_adapter: Convert MCP Context to ToolRuntime[ChatContext].

    This function is called by lc2mcp for each tool invocation.
    It extracts user information from OAuth token or state,
    then fetches the full user data from database.
    """
    # Extract user_id
    user_id = _get_user_id(mcp_ctx)

    # Extract other state
    query = mcp_ctx.get_state("query") or ""
    session_id = mcp_ctx.get_state("session_id")
    file_paths = mcp_ctx.get_state("file_paths") or []

    # Fetch user info
    user_info = _fetch_user_info_sync(user_id) if user_id else None

    # Fallback to anonymous user
    if not user_info:
        user_info = UserInfo(
            user_id=0,
            username="anonymous",
            display_name="Anonymous",
        )

    return ToolRuntime(
        context=ChatContext(
            user=user_info,
            query=query,
            session_id=int(session_id) if session_id else None,
            file_paths=tuple(file_paths),
        ),
        state={},
        config={},
        stream_writer=lambda x: None,
        tool_call_id=None,
        store=None,
    )
