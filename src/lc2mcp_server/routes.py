"""API routes for the server."""

import os
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import select

from .auth import (
    authenticate_user,
    get_current_user,
    require_user,
)
from .chat import (
    chat_with_agent,
    create_session,
    delete_session,
    get_session,
    get_session_conversations,
    get_user_sessions,
    update_session_model,
)
from .config import config
from .database import get_db_session
from .models import MCPConnection, ModelConfig, User

# Templates setup
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Routers
router = APIRouter()
api_router = APIRouter(prefix="/api", tags=["api"])


# === Pydantic Models ===


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message: str
    file_paths: Optional[list[str]] = None


class SessionCreate(BaseModel):
    model: str = "gpt-5.2"


class SessionUpdate(BaseModel):
    model: Optional[str] = None


class ModelConfigCreate(BaseModel):
    name: str
    provider: str  # openai, google, openrouter, custom
    model_id: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    config: Optional[dict] = None


class ModelConfigUpdate(BaseModel):
    name: Optional[str] = None
    provider: Optional[str] = None
    model_id: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    is_enabled: Optional[bool] = None
    config: Optional[dict] = None


# Predefined MCP platforms with icons and info
# Using jsDelivr CDN for simple-icons SVGs (more reliable)
# Format: https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/{slug}.svg
MCP_PLATFORMS = [
    {
        "id": "chatgpt",
        "name": "ChatGPT",
        "icon": "https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/openai.svg",
        "lucide": "bot",
        "color": "#412991",
        "description": "OpenAI ChatGPT",
        "oauth_enabled": True,
    },
    {
        "id": "claude",
        "name": "Claude",
        "icon": "https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/anthropic.svg",
        "lucide": "sparkles",
        "color": "#191919",
        "description": "Anthropic Claude Desktop",
        "oauth_enabled": True,
    },
    {
        "id": "cursor",
        "name": "Cursor",
        "icon": "https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/cursor.svg",
        "lucide": "mouse-pointer-2",
        "color": "#000000",
        "description": "Cursor AI IDE",
        "oauth_enabled": False,
    },
    {
        "id": "windsurf",
        "name": "Windsurf",
        "icon": "https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/codeium.svg",
        "lucide": "wind",
        "color": "#09B6A2",
        "description": "Codeium Windsurf IDE",
        "oauth_enabled": False,
    },
    {
        "id": "manus",
        "name": "Manus",
        "icon": "https://manus.im/favicon.svg",
        "lucide": "hand",
        "color": "#5046E5",
        "description": "Manus AI Agent",
        "oauth_enabled": True,
    },
    {
        "id": "gemini",
        "name": "Gemini",
        "icon": "https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/googlegemini.svg",
        "lucide": "sparkle",
        "color": "#8E75B2",
        "description": "Google Gemini",
        "oauth_enabled": True,
    },
    {
        "id": "cline",
        "name": "Cline",
        "icon": "https://raw.githubusercontent.com/cline/cline/main/assets/icons/icon.png",
        "lucide": "terminal",
        "color": "#E91E63",
        "description": "Cline VSCode Extension",
        "oauth_enabled": False,
    },
    {
        "id": "continue",
        "name": "Continue",
        "icon": "https://raw.githubusercontent.com/continuedev/continue/main/docs/static/img/logo.png",
        "lucide": "play",
        "color": "#FBBD23",
        "description": "Continue Dev Extension",
        "oauth_enabled": False,
    },
    {
        "id": "copilot",
        "name": "GitHub Copilot",
        "icon": "https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/github.svg",
        "lucide": "github",
        "color": "#181717",
        "description": "GitHub Copilot Agent",
        "oauth_enabled": True,
    },
    {
        "id": "vscode",
        "name": "VS Code",
        "icon": "https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/visualstudiocode.svg",
        "lucide": "code-2",
        "color": "#007ACC",
        "description": "Visual Studio Code MCP",
        "oauth_enabled": False,
    },
]


# === Web Pages ===


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, user: Optional[User] = Depends(get_current_user)):
    """Home page - redirect to chat or login."""
    if user:
        return RedirectResponse(url="/chat", status_code=302)
    return RedirectResponse(url="/login", status_code=302)


@router.get("/login", response_class=HTMLResponse)
async def login_page(
    request: Request,
    user: Optional[User] = Depends(get_current_user),
    oauth: Optional[str] = None,
    client_id: Optional[str] = None,
    redirect_uri: Optional[str] = None,
    code_challenge: Optional[str] = None,
    state: Optional[str] = None,
    scope: Optional[str] = None,
):
    """Login page - also handles OAuth authorization."""
    is_oauth = oauth == "1" and client_id and redirect_uri

    # If user is logged in and this is OAuth, show consent view
    if user and is_oauth:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "user": user,
                "is_oauth": True,
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code_challenge": code_challenge,
                "state": state,
                "scope": scope,
            },
        )

    # If user is logged in (not OAuth), go to chat
    if user:
        return RedirectResponse(url="/chat", status_code=302)

    # Show login form (with OAuth params if present)
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "is_oauth": is_oauth,
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "state": state,
            "scope": scope,
        },
    )


@router.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    # OAuth params (optional)
    oauth: Optional[str] = Form(None),
    client_id: Optional[str] = Form(None),
    redirect_uri: Optional[str] = Form(None),
    code_challenge: Optional[str] = Form(None),
    state: Optional[str] = Form(None),
    scope: Optional[str] = Form(None),
):
    """Handle login form submission."""
    is_oauth = oauth == "1" and client_id and redirect_uri

    user = await authenticate_user(username, password)
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Invalid username or password",
                "is_oauth": is_oauth,
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code_challenge": code_challenge,
                "state": state,
                "scope": scope,
            },
            status_code=400,
        )

    # Set session
    request.session["user_id"] = user.id

    # If OAuth, redirect back to login page to show consent
    if is_oauth:
        from urllib.parse import urlencode

        params = {
            "oauth": "1",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
        }
        if state:
            params["state"] = state
        if scope:
            params["scope"] = scope
        return RedirectResponse(url=f"/login?{urlencode(params)}", status_code=302)

    return RedirectResponse(url="/chat", status_code=302)


@router.post("/authorize")
async def authorize_submit(
    request: Request,
    client_id: str = Form(...),
    redirect_uri: str = Form(...),
    code_challenge: str = Form(...),
    state: Optional[str] = Form(None),
    scope: Optional[str] = Form(None),
    action: str = Form(...),
    user: User = Depends(require_user),
):
    """Handle OAuth authorization consent."""
    import secrets
    import time
    from urllib.parse import urlencode

    from .database import get_db_session
    from .models import OAuthCode

    if action == "deny":
        params = {"error": "access_denied"}
        if state:
            params["state"] = state
        return RedirectResponse(url=f"{redirect_uri}?{urlencode(params)}", status_code=302)

    # Generate authorization code
    code = secrets.token_urlsafe(32)
    expires_at = time.time() + 600  # 10 minutes

    async with get_db_session() as db:
        oauth_code = OAuthCode(
            code=code,
            user_id=user.id,
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            scope=scope or "",
            expires_at=expires_at,
        )
        db.add(oauth_code)

    # Redirect back to client with code
    params = {"code": code}
    if state:
        params["state"] = state
    return RedirectResponse(url=f"{redirect_uri}?{urlencode(params)}", status_code=302)


@router.get("/logout")
async def logout(request: Request):
    """Handle logout."""
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, user: User = Depends(require_user)):
    """Redirect to dashboard."""
    return RedirectResponse(url="/dashboard", status_code=302)


@router.get("/dashboard", response_class=HTMLResponse)
@router.get("/dashboard/{tab}", response_class=HTMLResponse)
async def dashboard_page(request: Request, user: User = Depends(require_user), tab: str = "tools"):
    """Dashboard page with tools, connections, and debug center."""
    # Validate tab
    if tab not in ("tools", "connections", "debug"):
        tab = "tools"

    sessions = await get_user_sessions(user.id)
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": user,
            "sessions": sessions,
            "platforms": MCP_PLATFORMS,
            "initial_tab": tab,
        },
    )


# === Chat API ===


@api_router.get("/sessions")
async def list_sessions(user: User = Depends(require_user)):
    """List user's chat sessions."""
    sessions = await get_user_sessions(user.id)
    return [
        {
            "id": s.id,
            "title": s.title,
            "model": getattr(s, "model", "gpt-5.2"),
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat(),
        }
        for s in sessions
    ]


@api_router.post("/sessions")
async def create_new_session(
    data: SessionCreate,
    user: User = Depends(require_user),
):
    """Create a new chat session."""
    session = await create_session(user.id, data.model)
    return {
        "id": session.id,
        "title": session.title,
        "model": session.model,
    }


@api_router.get("/sessions/{session_id}/conversations")
async def get_conversations(session_id: int, user: User = Depends(require_user)):
    """Get all conversations for a session."""
    session = await get_session(session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found")

    conversations = await get_session_conversations(session_id)
    return [
        {
            "id": conv.id,
            "input": conv.input,
            "input_files": conv.input_files,
            "output": conv.output,
            "tool_calls": conv.tool_calls,
            "model": conv.model,
            "input_tokens": conv.input_tokens,
            "output_tokens": conv.output_tokens,
            "total_tokens": conv.total_tokens,
            "cost": conv.cost,
            "latency_ms": conv.latency_ms,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
        }
        for conv in conversations
    ]


@api_router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: int,
    data: ChatRequest,
    user: User = Depends(require_user),
):
    """Send a message and stream AI response with tool calls (SSE).

    Returns SSE events:
    - data: {"type": "content", "data": "text chunk"}
    - data: {"type": "tool_call", "data": {"name": "...", "args": {...}}}
    - data: {"type": "tool_result", "data": {"name": "...", "result": "..."}}
    - data: {"type": "usage", "data": {"input_tokens": N, "output_tokens": N}}
    - data: {"type": "done", "data": {"latency_ms": N, ...}}
    - data: [DONE]
    """
    import json

    session = await get_session(session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found")

    from .context import UserInfo

    user_info = UserInfo(
        user_id=user.id,
        username=user.username,
        display_name=user.display_name,
        age=user.age,
        gender=user.gender,
        avatar_url=user.avatar_url,
        email=user.email,
    )

    async def generate():
        async for event in chat_with_agent(
            user_info=user_info,
            session_id=session_id,
            user_message=data.message,
            file_paths=data.file_paths,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@api_router.delete("/sessions/{session_id}")
async def delete_session_api(
    session_id: int,
    user: User = Depends(require_user),
):
    """Delete a chat session."""
    session = await get_session(session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found")

    await delete_session(session_id)
    return {"success": True}


@api_router.put("/sessions/{session_id}")
async def update_session_settings(
    session_id: int,
    data: SessionUpdate,
    user: User = Depends(require_user),
):
    """Update session settings (model)."""
    session = await get_session(session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found")

    if data.model is not None:
        await update_session_model(session_id, data.model)

    return {"success": True}


@api_router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user: User = Depends(require_user),
):
    """Upload a file."""
    # Generate unique filename
    ext = os.path.splitext(file.filename or "")[1]
    filename = f"{uuid.uuid4()}{ext}"
    filepath = config.upload_dir / filename

    # Save file
    content = await file.read()
    if len(content) > config.max_upload_size:
        raise HTTPException(status_code=400, detail="File too large")

    with open(filepath, "wb") as f:
        f.write(content)

    return {
        "filename": filename,
        "path": str(filepath),
        "size": len(content),
        "content_type": file.content_type,
    }


# === MCP Management APIs ===


@api_router.get("/mcp/tools")
async def get_mcp_tools(user: User = Depends(require_user)):
    """Get list of registered MCP tools."""
    import inspect

    from lc2mcp_community.tools import ALL_TOOLS

    tools = []
    for tool in ALL_TOOLS:
        # Get source code
        try:
            source = inspect.getsource(tool.func)
        except Exception:
            source = "# Source code not available"

        tools.append(
            {
                "name": tool.name,
                "description": tool.description or "",
                "args": list(tool.args.keys()) if hasattr(tool, "args") else [],
                "source": source,
            }
        )
    return {"tools": tools}


@api_router.get("/mcp/resources")
async def get_mcp_resources(user: User = Depends(require_user)):
    """Get list of registered MCP resources."""
    # Import the mcp instance to get resources
    from .main import mcp

    resources = []
    # FastMCP stores resources internally
    if hasattr(mcp, "_resource_manager") and mcp._resource_manager:
        for uri, resource in mcp._resource_manager._resources.items():
            resources.append(
                {
                    "uri": uri,
                    "name": resource.name if hasattr(resource, "name") else str(uri),
                    "description": resource.description if hasattr(resource, "description") else "",
                    "mime_type": resource.mime_type
                    if hasattr(resource, "mime_type")
                    else "text/plain",
                }
            )
    return {"resources": resources}


@api_router.get("/mcp/resources/preview")
async def preview_mcp_resource(uri: str, user: User = Depends(require_user)):
    """Preview a MCP resource content."""
    # For user://profile, generate a preview with current user data
    if uri == "user://profile":
        html_content = f"""
<div style="font-family: system-ui; padding: 20px; max-width: 400px;">
    <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 20px;">
        <div style="width: 64px; height: 64px; border-radius: 50%;
            background: linear-gradient(135deg, #8b5cf6, #7c3aed);
            display: flex; align-items: center; justify-content: center;
            color: white; font-size: 24px; font-weight: bold;">
            {user.display_name[0] if user.display_name else "U"}
        </div>
        <div>
            <h2 style="margin: 0; font-size: 20px;">{user.display_name}</h2>
            <p style="margin: 4px 0 0; color: #888;">@{user.username}</p>
        </div>
    </div>
    <div style="background: rgba(128,128,128,0.1); border-radius: 12px; padding: 16px;">
        <div style="display: grid; gap: 12px;">
            <div><span style="color: #888;">用户ID：</span>{user.id}</div>
            <div><span style="color: #888;">邮箱：</span>{user.email or "未设置"}</div>
            <div><span style="color: #888;">年龄：</span>{user.age or "未设置"}</div>
            <div><span style="color: #888;">性别：</span>{user.gender or "未设置"}</div>
        </div>
    </div>
</div>
"""
        return {"content": html_content, "mime_type": "text/html"}

    # For other resources, try to execute
    from .main import mcp

    if hasattr(mcp, "_resource_manager") and mcp._resource_manager:
        if uri in mcp._resource_manager._resources:
            resource = mcp._resource_manager._resources[uri]
            try:
                if hasattr(resource, "fn"):
                    result = await resource.fn() if callable(resource.fn) else str(resource.fn)
                    return {
                        "content": result,
                        "mime_type": getattr(resource, "mime_type", "text/plain"),
                    }
                return {"content": "Resource preview not available", "mime_type": "text/plain"}
            except Exception as e:
                return {"content": f"Error: {str(e)}", "mime_type": "text/plain"}

    raise HTTPException(status_code=404, detail="Resource not found")


@api_router.get("/mcp/platforms")
async def get_mcp_platforms(user: User = Depends(require_user)):
    """Get MCP platforms and their connection status."""
    # Get connection records from database
    async with get_db_session() as db:
        result = await db.execute(select(MCPConnection))
        connections = result.scalars().all()

    connection_map = {c.platform_id: c for c in connections}

    platforms = []
    for platform in MCP_PLATFORMS:
        conn = connection_map.get(platform["id"])
        platforms.append(
            {
                **platform,
                "is_connected": conn.is_connected if conn else False,
                "last_call_at": conn.last_call_at.isoformat()
                if conn and conn.last_call_at
                else None,
                "call_count": conn.call_count if conn else 0,
            }
        )

    return {"platforms": platforms}


@api_router.get("/mcp/config")
async def get_mcp_config(user: User = Depends(require_user)):
    """Get MCP server configuration."""
    from .config import config as app_config

    return {
        "server_url": f"{app_config.base_url}/mcp",
        "oauth_enabled": True,
        "oauth_metadata_url": f"{app_config.base_url}/.well-known/oauth-authorization-server",
        "supported_models": [
            {"id": "gpt-5.2", "name": "GPT-5.2", "provider": "OpenAI"},
            {"id": "gemini-3-pro", "name": "Gemini 3 Pro", "provider": "Google"},
        ],
    }


# === Model Configuration APIs ===


# Preset models that are always available
PRESET_MODELS = [
    {
        "name": "GPT-5.2",
        "provider": "openai",
        "model_id": "gpt-5.2",
        "is_preset": True,
        "icon": "sparkles",
        "color": "emerald",
    },
    {
        "name": "Gemini 3 Pro",
        "provider": "google",
        "model_id": "gemini-3-pro",
        "is_preset": True,
        "icon": "diamond",
        "color": "blue",
    },
]


@api_router.get("/models")
async def list_models(user: User = Depends(require_user)):
    """List all available models (presets + user configured)."""
    async with get_db_session() as db:
        result = await db.execute(select(ModelConfig).where(ModelConfig.user_id == user.id))
        user_models = result.scalars().all()

    # Combine presets with user models
    models = []

    # Add presets
    for preset in PRESET_MODELS:
        # Check if user has API key configured for this preset
        user_config = next(
            (m for m in user_models if m.model_id == preset["model_id"] and m.is_preset), None
        )
        models.append(
            {
                "id": preset["model_id"],
                "name": preset["name"],
                "provider": preset["provider"],
                "model_id": preset["model_id"],
                "is_preset": True,
                "is_enabled": user_config.is_enabled if user_config else True,
                "has_api_key": bool(user_config and user_config.api_key),
                "icon": preset.get("icon", "cpu"),
                "color": preset.get("color", "gray"),
                "config_id": user_config.id if user_config else None,
            }
        )

    # Add custom models
    for model in user_models:
        if not model.is_preset:
            models.append(
                {
                    "id": model.id,
                    "name": model.name,
                    "provider": model.provider,
                    "model_id": model.model_id,
                    "base_url": model.base_url,
                    "is_preset": False,
                    "is_enabled": model.is_enabled,
                    "has_api_key": bool(model.api_key),
                    "icon": "settings-2",
                    "color": "purple",
                    "config_id": model.id,
                }
            )

    return {"models": models}


@api_router.post("/models")
async def create_model(
    data: ModelConfigCreate,
    user: User = Depends(require_user),
):
    """Create a new model configuration."""
    async with get_db_session() as db:
        model = ModelConfig(
            user_id=user.id,
            name=data.name,
            provider=data.provider,
            model_id=data.model_id,
            base_url=data.base_url,
            api_key=data.api_key,
            is_preset=False,
            is_enabled=True,
            config=data.config,
        )
        db.add(model)
        await db.flush()
        await db.refresh(model)
        return {
            "id": model.id,
            "name": model.name,
            "provider": model.provider,
            "model_id": model.model_id,
            "is_enabled": model.is_enabled,
        }


@api_router.put("/models/{model_id}")
async def update_model(
    model_id: int,
    data: ModelConfigUpdate,
    user: User = Depends(require_user),
):
    """Update a model configuration."""
    async with get_db_session() as db:
        result = await db.execute(
            select(ModelConfig).where(ModelConfig.id == model_id, ModelConfig.user_id == user.id)
        )
        model = result.scalar_one_or_none()

        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        if data.name is not None:
            model.name = data.name
        if data.provider is not None:
            model.provider = data.provider
        if data.model_id is not None:
            model.model_id = data.model_id
        if data.base_url is not None:
            model.base_url = data.base_url
        if data.api_key is not None:
            model.api_key = data.api_key
        if data.is_enabled is not None:
            model.is_enabled = data.is_enabled
        if data.config is not None:
            model.config = data.config

        return {"success": True}


@api_router.delete("/models/{model_id}")
async def delete_model(
    model_id: int,
    user: User = Depends(require_user),
):
    """Delete a model configuration."""
    async with get_db_session() as db:
        result = await db.execute(
            select(ModelConfig).where(ModelConfig.id == model_id, ModelConfig.user_id == user.id)
        )
        model = result.scalar_one_or_none()

        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        if model.is_preset:
            raise HTTPException(status_code=400, detail="Cannot delete preset models")

        await db.delete(model)
        return {"success": True}


@api_router.post("/models/preset/{model_id}/configure")
async def configure_preset_model(
    model_id: str,
    api_key: str = Form(...),
    user: User = Depends(require_user),
):
    """Configure API key for a preset model."""
    # Validate preset model exists
    preset = next((p for p in PRESET_MODELS if p["model_id"] == model_id), None)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset model not found")

    async with get_db_session() as db:
        # Check if config already exists
        result = await db.execute(
            select(ModelConfig).where(
                ModelConfig.user_id == user.id,
                ModelConfig.model_id == model_id,
                ModelConfig.is_preset.is_(True),
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            existing.api_key = api_key
        else:
            model = ModelConfig(
                user_id=user.id,
                name=preset["name"],
                provider=preset["provider"],
                model_id=model_id,
                api_key=api_key,
                is_preset=True,
                is_enabled=True,
            )
            db.add(model)

    return {"success": True}
