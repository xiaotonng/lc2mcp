"""Main entry point for lc2mcp-server."""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastmcp import FastMCP
from starlette.middleware.sessions import SessionMiddleware

from lc2mcp import register_tools
from lc2mcp_scanner import scan_resources, scan_tools

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("mcp-debug")

from lc2mcp_community.tools import ALL_TOOLS

from .config import config
from .context import runtime_adapter, get_user_info_by_id
from .database import init_db
from .oauth_provider import CustomOAuthProvider
from .routes import api_router, router


async def create_demo_user():
    """Create a demo user if not exists."""
    from .auth import create_user, get_user_by_username

    demo_cfg = config.demo_user
    demo_user = await get_user_by_username(demo_cfg.username)
    if not demo_user:
        await create_user(
            username=demo_cfg.username,
            password=demo_cfg.password,
            display_name=demo_cfg.display_name,
            age=25,
            gender="other",
            email=demo_cfg.email,
        )
        print(f"Demo user created: {demo_cfg.username} / {demo_cfg.password}")


async def mcp_logging_middleware(request: Request, call_next):
    """Simple middleware to log MCP requests."""
    # Only log requests to /mcp
    if request.url.path == "/mcp" and request.method == "POST":
        # Read body once (cached by Starlette)
        body = await request.body()
        body_str = body.decode("utf-8") if body else ""

        # Parse JSON-RPC to get method
        method = "unknown"
        try:
            if body_str:
                data = json.loads(body_str)
                method = data.get("method", "unknown")
        except json.JSONDecodeError:
            pass

        # Log the request
        logger.info("=" * 60)
        logger.info(f"[MCP REQUEST] Method: {method}")
        logger.info(f"[MCP REQUEST] Path: {request.url.path}")

        # Special highlight for resources/list
        if method == "resources/list":
            logger.info("*" * 60)
            logger.info("*** resources/list CALLED! ***")
            logger.info("*" * 60)

    response = await call_next(request)
    return response


def load_tools() -> list:
    """
    Load tools using lc2mcp_scanner.scan_tools().
    
    Scans community and external directories for LangChain tools.
    """
    import lc2mcp_community.tools as community_tools_module
    
    community_tools_dir = Path(community_tools_module.__file__).parent
    
    # Collect directories to scan
    dirs_to_scan = [community_tools_dir]
    logger.info(f"Added community tools directory: {community_tools_dir}")
    
    for ext_dir in config.tools.external_dirs:
        ext_path = Path(ext_dir)
        if ext_path.exists():
            dirs_to_scan.append(ext_path)
            logger.info(f"Added external tools directory: {ext_path}")
    
    # Use lc2mcp_scanner.scan_tools() to discover tools
    tools = scan_tools(dirs_to_scan, recursive=True, include_init=True)
    logger.info(f"Loaded {len(tools)} tools: {[t.name for t in tools]}")
    return tools if tools else ALL_TOOLS


def load_resources() -> list:
    """
    Load resource registrars using lc2mcp_scanner.scan_resources().
    
    Scans community and external directories for register_xxx(mcp) functions.
    """
    import lc2mcp_community.resources as community_resources_module
    
    community_resources_dir = Path(community_resources_module.__file__).parent
    
    # Collect directories to scan
    dirs_to_scan = [community_resources_dir]
    logger.info(f"Added community resources directory: {community_resources_dir}")
    
    for ext_dir in config.resources.external_dirs:
        ext_path = Path(ext_dir)
        if ext_path.exists():
            dirs_to_scan.append(ext_path)
            logger.info(f"Added external resources directory: {ext_path}")
    
    # Use lc2mcp_scanner.scan_resources() to discover registrars
    registrars = scan_resources(dirs_to_scan, recursive=True)
    logger.info(f"Loaded {len(registrars)} resource registrars")
    return registrars


def create_mcp() -> FastMCP:
    """Create and configure the MCP server with OAuth authentication."""
    # Set up user info fetcher for community resources
    from lc2mcp_community.resources.user_profile import set_user_info_fetcher
    set_user_info_fetcher(get_user_info_by_id)
    
    # Get base URL from config or environment
    # This is required for OAuth metadata endpoints
    base_url = config.base_url
    if not base_url:
        # Fallback to local URL if not configured
        base_url = f"http://{config.host}:{config.port}"
    
    # Create OAuth provider - provides complete OAuth 2.1 server
    # Includes: /.well-known/oauth-authorization-server, /authorize, /token
    oauth_provider = CustomOAuthProvider(base_url=base_url)

    # Create MCP server with OAuth authentication
    mcp = FastMCP("lc2mcp-server", auth=oauth_provider)

    # Load tools dynamically from configured directories
    tools = load_tools()
    
    # Register tools with runtime_adapter for context injection
    register_tools(
        mcp,
        tools,
        runtime_adapter=runtime_adapter,
    )

    # Load and register resources
    resource_registrars = load_resources()
    for registrar in resource_registrars:
        registrar(mcp)

    return mcp


# Create MCP instance first (needed for mounting)
mcp = create_mcp()
mcp_http_app = mcp.http_app()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - includes MCP lifespan."""
    # Initialize database
    await init_db()
    await create_demo_user()
    
    # Run MCP lifespan
    async with mcp_http_app.lifespan(app):
        yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="lc2mcp-server",
        description="A FastAPI server with OAuth, chat interface, and MCP tools",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add session middleware for web auth
    app.add_middleware(SessionMiddleware, secret_key=config.secret_key)

    # Add MCP logging middleware for debugging
    @app.middleware("http")
    async def log_mcp_requests(request: Request, call_next):
        return await mcp_logging_middleware(request, call_next)

    # Include routers FIRST (higher priority)
    # OAuth endpoints are provided by FastMCP's OAuthProvider
    app.include_router(router)
    app.include_router(api_router)

    # Mount static files
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Mount MCP HTTP endpoint LAST (fallback for /mcp path)
    # Access via: http://localhost:8000/mcp
    app.mount("", mcp_http_app)

    return app


# Create FastAPI app
app = create_app()


def run():
    """Run the FastAPI server."""
    uvicorn.run(
        "lc2mcp_server.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
    )


def run_mcp():
    """Run the MCP server (stdio mode)."""
    mcp.run()


if __name__ == "__main__":
    run()
