"""LangChain chat service with GPT-5.2/Gemini-3-Pro and tools."""

import logging
import time
from typing import AsyncIterator, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from sqlalchemy import select

from lc2mcp_community.tools import ALL_TOOLS
from lc2mcp_scanner import scan_tools

from .config import config
from .context import ChatContext, UserInfo
from .database import async_session, get_db_session
from .models import Conversation, Session

logger = logging.getLogger(__name__)

# Cached tools list
_cached_tools: list | None = None


def get_llm(model: str, streaming: bool = False):
    """Get LLM instance based on model selection."""
    if model == "gemini-3-pro":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            streaming=streaming,
            temperature=0.3,
        )
    else:
        # Default to GPT-4o
        return ChatOpenAI(
            model="gpt-4o",
            streaming=streaming,
            temperature=0.3,
        )

def get_tools() -> list:
    """
    Get available tools for the chat agent.
    
    Uses lc2mcp_scanner.scan_tools() to discover tools from community and external directories.
    
    Returns:
        List of LangChain tools
    """
    global _cached_tools
    
    if _cached_tools is not None:
        return _cached_tools
    
    from pathlib import Path
    import lc2mcp_community.tools as community_tools_module
    
    community_tools_dir = Path(community_tools_module.__file__).parent
    dirs_to_scan = [community_tools_dir]
    
    for ext_dir in config.tools.external_dirs:
        ext_path = Path(ext_dir)
        if ext_path.exists():
            dirs_to_scan.append(ext_path)
    
    _cached_tools = scan_tools(dirs_to_scan, recursive=True, include_init=True)
    logger.info(f"Using {len(_cached_tools)} tools: {[t.name for t in _cached_tools]}")
    
    return _cached_tools if _cached_tools else ALL_TOOLS


SYSTEM_PROMPT = """You are a helpful AI assistant. Use tools proactively when appropriate.

Guidelines:
- Prefer action over clarification
- Be helpful, accurate, and concise
- Use available tools to provide real-time information when needed"""


async def get_session_conversations(session_id: int) -> list[Conversation]:
    """Get all conversations for a session."""
    async with async_session() as db:
        result = await db.execute(
            select(Conversation)
            .where(Conversation.session_id == session_id)
            .order_by(Conversation.created_at)
        )
        return list(result.scalars().all())


async def save_conversation(
    session_id: int,
    input: str,
    output: str,
    model: str,
    input_files: Optional[list[str]] = None,
    tool_calls: Optional[list[dict]] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    cost: Optional[float] = None,
    latency_ms: Optional[int] = None,
) -> Conversation:
    """Save a complete conversation turn to the database."""
    async with get_db_session() as db:
        conversation = Conversation(
            session_id=session_id,
            input=input,
            input_files=input_files,
            output=output,
            tool_calls=tool_calls,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
            latency_ms=latency_ms,
        )
        db.add(conversation)
        await db.flush()
        await db.refresh(conversation)

        # Update session title if it's the first conversation
        session = await db.get(Session, session_id)
        if session and session.title == "New Chat":
            # Use first 50 chars of input as title
            session.title = input[:50] + ("..." if len(input) > 50 else "")

        return conversation


async def create_session(
    user_id: int,
    model: str = "gpt-5.2",
) -> Session:
    """Create a new chat session."""
    async with get_db_session() as db:
        session = Session(
            user_id=user_id,
            model=model,
        )
        db.add(session)
        await db.flush()
        await db.refresh(session)
        return session


async def get_user_sessions(user_id: int) -> list[Session]:
    """Get all sessions for a user."""
    async with async_session() as db:
        result = await db.execute(
            select(Session)
            .where(Session.user_id == user_id)
            .order_by(Session.updated_at.desc())
        )
        return list(result.scalars().all())


async def get_session(session_id: int) -> Optional[Session]:
    """Get a session by ID."""
    async with async_session() as db:
        return await db.get(Session, session_id)


async def update_session_model(session_id: int, model: str) -> None:
    """Update model selection for a session."""
    async with get_db_session() as db:
        session = await db.get(Session, session_id)
        if session:
            session.model = model


async def delete_session(session_id: int) -> None:
    """Delete a chat session and all its messages."""
    async with get_db_session() as db:
        session = await db.get(Session, session_id)
        if session:
            await db.delete(session)


def build_messages_from_conversations(conversations: list[Conversation], user_message: str) -> list:
    """Build LangChain message list from conversation history."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for conv in conversations:
        messages.append(HumanMessage(content=conv.input))
        messages.append(AIMessage(content=conv.output))

    messages.append(HumanMessage(content=user_message))
    return messages


async def chat_stream(
    user_info: UserInfo,
    session_id: int,
    user_message: str,
    file_paths: Optional[list[str]] = None,
    model: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Stream chat response using LangChain with GPT-5.2 or Gemini-3-Pro.
    Note: Streaming mode doesn't support tool calling, use chat_with_agent for that.

    Args:
        user_info: Current user information
        session_id: Chat session ID
        user_message: User's message
        file_paths: Optional list of uploaded file paths
        model: Model to use (defaults to session's model)
    """
    start_time = time.time()
    
    # Get session to determine model if not provided
    if model is None:
        session = await get_session(session_id)
        model = session.model if session else "gpt-5.2"

    # Get conversation history
    conversations = await get_session_conversations(session_id)

    # Build message list
    messages = build_messages_from_conversations(conversations, user_message)

    # Get LLM based on model selection
    llm = get_llm(model, streaming=True)

    # Stream response (no tool calling in stream mode)
    full_response = ""
    try:
        async for chunk in llm.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                full_response += chunk.content
                yield chunk.content
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        full_response = error_msg
        yield error_msg

    latency_ms = int((time.time() - start_time) * 1000)

    # Save complete conversation
    await save_conversation(
        session_id=session_id,
        input=user_message,
        output=full_response,
        model=model,
        input_files=file_paths,
        latency_ms=latency_ms,
    )


async def chat_with_agent(
    user_info: UserInfo,
    session_id: int,
    user_message: str,
    file_paths: Optional[list[str]] = None,
    model: Optional[str] = None,
) -> AsyncIterator[dict]:
    """
    Streaming chat with full agent support (tool calling via MCP).
    
    Tools are fetched from the local MCP server and called via MCP protocol.
    Uses astream with stream_mode="messages" to get streaming content,
    tool calls, and token usage in real-time.
    
    Yields dicts with keys:
    - type: "content" | "tool_call" | "tool_result" | "usage" | "done" | "error"
    - data: The corresponding data
    """
    from langgraph.prebuilt import create_react_agent

    start_time = time.time()

    # Get session to determine model if not provided
    if model is None:
        session = await get_session(session_id)
        model = session.model if session else "gpt-5.2"

    # Get conversation history
    conversations = await get_session_conversations(session_id)

    # Get LLM based on model selection (enable streaming)
    llm = get_llm(model, streaming=True)

    # Build chat history from previous conversations
    chat_history = []
    for conv in conversations:
        chat_history.append(HumanMessage(content=conv.input))
        chat_history.append(AIMessage(content=conv.output))
    chat_history.append(HumanMessage(content=user_message))

    # Get tools for agent
    tools = get_tools()

    # Create agent with LangGraph create_react_agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )

    # Track state for conversation saving
    full_response = ""
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    thinking_tokens = 0
    cache_read_tokens = 0
    current_tool_calls = {}  # Track tool calls by ID
    
    try:
        # Use stream_mode="messages" for streaming content chunks
        async for event in agent.astream(
            {"messages": chat_history},
            stream_mode="messages",
        ):
            # event is a tuple (chunk, metadata) in messages mode
            if isinstance(event, tuple) and len(event) == 2:
                chunk, metadata = event
            else:
                chunk = event
            
            # Handle ToolMessage (tool results) - check this FIRST before content
            if isinstance(chunk, ToolMessage):
                tool_result = {
                    "id": chunk.tool_call_id,
                    "name": chunk.name,
                    "result": chunk.content,
                }
                yield {"type": "tool_result", "data": tool_result}
                
                # Update tool_calls_data with result
                if chunk.tool_call_id in current_tool_calls:
                    current_tool_calls[chunk.tool_call_id]["result"] = chunk.content
                continue
            
            # Handle AIMessageChunk (streaming content and tool calls)
            if hasattr(chunk, "content") and chunk.content:
                full_response += chunk.content
                yield {"type": "content", "data": chunk.content}
            
            # Handle tool call chunks
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                for tc in chunk.tool_calls:
                    if tc.get("name") and tc.get("id"):
                        tool_id = tc["id"]
                        if tool_id not in current_tool_calls:
                            current_tool_calls[tool_id] = {
                                "name": tc["name"],
                                "args": tc.get("args", {}),
                                "id": tool_id,
                            }
                            yield {"type": "tool_call", "data": current_tool_calls[tool_id]}
            
            # Handle usage metadata
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage = chunk.usage_metadata
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
                total_tokens += usage.get("total_tokens", 0)
                
                # Extract detailed token info (OpenAI specific)
                input_details = usage.get("input_token_details", {})
                output_details = usage.get("output_token_details", {})
                cache_read = input_details.get("cache_read", 0) if input_details else 0
                reasoning = output_details.get("reasoning", 0) if output_details else 0
                
                cache_read_tokens += cache_read
                thinking_tokens += reasoning
                
                yield {"type": "usage", "data": {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "thinking_tokens": reasoning,
                    "cache_read_tokens": cache_read,
                }}
        
        if not full_response:
            full_response = "No response generated."
            yield {"type": "content", "data": full_response}
            
    except Exception as e:
        logger.exception(f"Agent error: {e}")
        full_response = f"Error: {str(e)}"
        yield {"type": "error", "data": str(e)}

    latency_ms = int((time.time() - start_time) * 1000)
    
    # Convert current_tool_calls to list
    tool_calls_list = list(current_tool_calls.values()) if current_tool_calls else None

    # Save complete conversation with all metadata
    await save_conversation(
        session_id=session_id,
        input=user_message,
        output=full_response,
        model=model,
        input_files=file_paths,
        tool_calls=tool_calls_list,
        input_tokens=input_tokens if input_tokens > 0 else None,
        output_tokens=output_tokens if output_tokens > 0 else None,
        total_tokens=total_tokens if total_tokens > 0 else None,
        latency_ms=latency_ms,
    )

    # Signal completion
    yield {"type": "done", "data": {
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "thinking_tokens": thinking_tokens,
        "cache_read_tokens": cache_read_tokens,
    }}
