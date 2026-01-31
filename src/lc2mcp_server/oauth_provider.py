"""Custom OAuth Provider using FastMCP's OAuthProvider."""

import hashlib
import secrets
import time
from typing import Optional

from fastmcp.server.auth import AccessToken, OAuthProvider
from mcp.server.auth.provider import AuthorizationCode, AuthorizationParams, RefreshToken
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyHttpUrl

from .database import get_db_session
from .models import OAuthCode, OAuthToken as OAuthTokenModel, User


class CustomOAuthProvider(OAuthProvider):
    """
    Custom OAuth Provider that implements FastMCP's OAuthProvider.
    
    This provides a complete OAuth 2.1 server with:
    - /.well-known/oauth-authorization-server metadata
    - /authorize endpoint
    - /token endpoint
    - Token verification
    
    All storage uses our SQLite database.
    """

    def __init__(self, base_url: str):
        """
        Initialize the OAuth provider.
        
        Args:
            base_url: The public base URL of the server
        """
        super().__init__(
            base_url=base_url,
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                valid_scopes=["openid", "profile", "email"],
                default_scopes=["openid", "profile"],
            ),
            revocation_options=RevocationOptions(enabled=True),
        )
        # In-memory client storage (for dynamic registration)
        self._clients: dict[str, OAuthClientInformationFull] = {}
        # Pre-register our known client
        self._register_default_client()

    def _register_default_client(self):
        """Register the default ChatGPT client."""
        from .config import config
        
        self._clients[config.oauth_client_id] = OAuthClientInformationFull(
            client_id=config.oauth_client_id,
            client_secret=config.oauth_client_secret,
            redirect_uris=[AnyHttpUrl("https://chatgpt.com/aip/g/oauth/callback")],
            client_name="ChatGPT",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="client_secret_post",
        )

    # === Client Management ===

    async def get_client(self, client_id: str) -> Optional[OAuthClientInformationFull]:
        """Retrieve client information by ID."""
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        """Store new client registration."""
        self._clients[client_info.client_id] = client_info

    # === Authorization Flow ===

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        """
        Handle authorization request.
        
        Redirects to login page with OAuth parameters.
        Login page will handle both login and authorization consent.
        """
        from urllib.parse import urlencode
        
        auth_params = {
            "oauth": "1",
            "client_id": client.client_id,
            "redirect_uri": str(params.redirect_uri),
            "code_challenge": params.code_challenge,
        }
        if params.state:
            auth_params["state"] = params.state
        if params.scopes:
            auth_params["scope"] = " ".join(params.scopes)
        
        # Redirect to login page (handles both login and consent)
        return f"/login?{urlencode(auth_params)}"

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> Optional[AuthorizationCode]:
        """Load authorization code from database."""
        async with get_db_session() as db:
            from sqlalchemy import select
            
            result = await db.execute(
                select(OAuthCode).where(
                    OAuthCode.code == authorization_code,
                    OAuthCode.client_id == client.client_id,
                )
            )
            code_obj = result.scalar_one_or_none()
            
            if not code_obj:
                return None
            
            # Check if expired
            if code_obj.expires_at < time.time():
                return None
            
            return AuthorizationCode(
                code=code_obj.code,
                scopes=code_obj.scope.split() if code_obj.scope else [],
                expires_at=code_obj.expires_at,
                client_id=code_obj.client_id,
                code_challenge=code_obj.code_challenge or "",
                redirect_uri=AnyHttpUrl(code_obj.redirect_uri),
                redirect_uri_provided_explicitly=True,
            )

    # === Token Management ===

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        """Exchange authorization code for tokens."""
        async with get_db_session() as db:
            from sqlalchemy import select, delete
            
            # Get the code to find user_id
            result = await db.execute(
                select(OAuthCode).where(OAuthCode.code == authorization_code.code)
            )
            code_obj = result.scalar_one_or_none()
            
            if not code_obj:
                raise ValueError("Invalid authorization code")
            
            user_id = code_obj.user_id
            
            # Delete the used code
            await db.execute(
                delete(OAuthCode).where(OAuthCode.code == authorization_code.code)
            )
            
            # Generate tokens
            access_token = secrets.token_urlsafe(32)
            refresh_token = secrets.token_urlsafe(32)
            expires_in = 3600 * 24  # 24 hours
            
            # Store access token
            token_obj = OAuthTokenModel(
                access_token=access_token,
                refresh_token=refresh_token,
                user_id=user_id,
                client_id=client.client_id,
                scope=authorization_code.scopes[0] if authorization_code.scopes else "",
                expires_at=time.time() + expires_in,
            )
            db.add(token_obj)
            
            return OAuthToken(
                access_token=access_token,
                token_type="bearer",
                expires_in=expires_in,
                refresh_token=refresh_token,
                scope=" ".join(authorization_code.scopes) if authorization_code.scopes else None,
            )

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> Optional[RefreshToken]:
        """Load refresh token from database."""
        async with get_db_session() as db:
            from sqlalchemy import select
            
            result = await db.execute(
                select(OAuthTokenModel).where(
                    OAuthTokenModel.refresh_token == refresh_token,
                    OAuthTokenModel.client_id == client.client_id,
                )
            )
            token_obj = result.scalar_one_or_none()
            
            if not token_obj:
                return None
            
            return RefreshToken(
                token=token_obj.refresh_token,
                client_id=token_obj.client_id,
                scopes=token_obj.scope.split() if token_obj.scope else [],
            )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        """Exchange refresh token for new tokens."""
        async with get_db_session() as db:
            from sqlalchemy import select
            
            # Find the existing token
            result = await db.execute(
                select(OAuthTokenModel).where(
                    OAuthTokenModel.refresh_token == refresh_token.token,
                    OAuthTokenModel.client_id == client.client_id,
                )
            )
            token_obj = result.scalar_one_or_none()
            
            if not token_obj:
                raise ValueError("Invalid refresh token")
            
            # Generate new tokens
            new_access_token = secrets.token_urlsafe(32)
            new_refresh_token = secrets.token_urlsafe(32)
            expires_in = 3600 * 24  # 24 hours
            
            # Update token in database
            token_obj.access_token = new_access_token
            token_obj.refresh_token = new_refresh_token
            token_obj.expires_at = time.time() + expires_in
            
            return OAuthToken(
                access_token=new_access_token,
                token_type="bearer",
                expires_in=expires_in,
                refresh_token=new_refresh_token,
                scope=" ".join(scopes) if scopes else None,
            )

    async def load_access_token(self, token: str) -> Optional[AccessToken]:
        """Load access token from database."""
        async with get_db_session() as db:
            from sqlalchemy import select
            
            result = await db.execute(
                select(OAuthTokenModel).where(OAuthTokenModel.access_token == token)
            )
            token_obj = result.scalar_one_or_none()
            
            if not token_obj:
                return None
            
            # Check if expired
            if token_obj.expires_at and token_obj.expires_at < time.time():
                return None
            
            return AccessToken(
                token=token_obj.access_token,
                client_id=token_obj.client_id,
                scopes=token_obj.scope.split() if token_obj.scope else [],
                expires_at=int(token_obj.expires_at) if token_obj.expires_at else None,
                claims={
                    "sub": str(token_obj.user_id),
                    "user_id": token_obj.user_id,
                },
            )

    async def revoke_token(self, token) -> None:
        """Revoke a token."""
        async with get_db_session() as db:
            from sqlalchemy import delete
            
            if isinstance(token, AccessToken):
                await db.execute(
                    delete(OAuthTokenModel).where(
                        OAuthTokenModel.access_token == token.token
                    )
                )
            elif isinstance(token, RefreshToken):
                await db.execute(
                    delete(OAuthTokenModel).where(
                        OAuthTokenModel.refresh_token == token.token
                    )
                )

    async def verify_token(self, token: str) -> Optional[AccessToken]:
        """Verify bearer token for incoming requests."""
        # First try as OAuth access token
        access_token = await self.load_access_token(token)
        if access_token:
            return access_token
        
        # Fallback: try as JWT token (for internal use)
        from .auth import decode_access_token
        
        user_id = decode_access_token(token)
        if user_id:
            return AccessToken(
                token=token,
                client_id="jwt-client",
                scopes=[],
                claims={"sub": str(user_id), "user_id": user_id},
            )
        
        return None
