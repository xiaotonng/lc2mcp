"""User profile resource."""

from typing import Callable, Optional

from fastmcp import Context, FastMCP

from lc2mcp_community.context import UserInfo, extract_user_id_from_mcp_context

# Default user info fetcher (returns None, should be overridden)
_user_info_fetcher: Optional[Callable[[int], UserInfo | None]] = None


def set_user_info_fetcher(fetcher: Callable[[int], UserInfo | None]) -> None:
    """Set the function to fetch user info by user_id."""
    global _user_info_fetcher
    _user_info_fetcher = fetcher


def register_user_profile(mcp: FastMCP) -> None:
    """Register the user_profile resource."""

    @mcp.resource(
        "user://profile",
        name="user_profile",
        title="User Profile",
        description="Current user's profile page.",
        mime_type="text/html",
    )
    async def user_profile(ctx: Context) -> str:
        """
        Current user's profile page.
        Returns an HTML page showing user information.
        """
        user_id = extract_user_id_from_mcp_context(ctx)

        if not user_id:
            return _error_html("未登录或无法获取用户信息")

        # Fetch user info
        user_info = None
        if _user_info_fetcher:
            import asyncio
            import concurrent.futures

            try:
                fetcher = _user_info_fetcher
                result = fetcher(user_id)
                # Handle async fetcher
                if asyncio.iscoroutine(result):
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, result)
                            user_info = future.result()
                    else:
                        user_info = loop.run_until_complete(result)
                else:
                    user_info = result
            except Exception:
                user_info = None

        if not user_info:
            return _error_html(f"用户 ID {user_id} 不存在")

        return _profile_html(user_info)


def _error_html(message: str) -> str:
    """Generate error HTML page."""
    return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>User Profile</title>
    <style>
        body {{
            font-family: system-ui, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
        }}
        .error {{ color: #dc3545; }}
    </style>
</head>
<body>
    <h1>User Profile</h1>
    <p class="error">{message}</p>
</body>
</html>
"""


def _profile_html(user_info: UserInfo) -> str:
    """Generate profile HTML page."""
    return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{user_info.display_name} - Profile</title>
    <style>
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .profile-card {{
            background: white;
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        .avatar {{
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 32px;
            font-weight: bold;
            margin: 0 auto 20px;
        }}
        h1 {{
            text-align: center;
            margin: 0 0 8px;
            color: #333;
        }}
        .username {{
            text-align: center;
            color: #666;
            margin-bottom: 24px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }}
        .info-item {{
            background: #f8f9fa;
            padding: 12px 16px;
            border-radius: 8px;
        }}
        .info-label {{
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .info-value {{
            font-size: 16px;
            color: #333;
            margin-top: 4px;
        }}
        .badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="profile-card">
        <div class="avatar">{user_info.display_name[0].upper()}</div>
        <h1>{user_info.display_name}</h1>
        <p class="username">
            @{user_info.username} <span class="badge">ID: {user_info.user_id}</span>
        </p>

        <div class="info-grid">
            <div class="info-item">
                <div class="info-label">Email</div>
                <div class="info-value">{user_info.email or "Not set"}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Age</div>
                <div class="info-value">{user_info.age or "Not set"}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Gender</div>
                <div class="info-value">{user_info.gender or "Not set"}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Status</div>
                <div class="info-value">Active</div>
            </div>
        </div>
    </div>
</body>
</html>
"""
