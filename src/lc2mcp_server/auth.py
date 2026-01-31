"""Authentication implementation."""

import hashlib
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import select

from .config import config
from .database import get_db_session
from .models import OAuthToken, User

# JWT settings
ALGORITHM = "HS256"

# HTTP Bearer for API auth
security = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt (simple demo implementation)."""
    # For demo purposes - in production use bcrypt/argon2 with proper Python version
    salt = config.secret_key[:16]
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return hash_password(plain_password) == hashed_password


def create_access_token(user_id: int, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=config.access_token_expire_minutes)
    )
    to_encode = {"sub": str(user_id), "exp": expire}
    return jwt.encode(to_encode, config.secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[int]:
    """Decode and validate a JWT access token. Returns user_id or None."""
    try:
        payload = jwt.decode(token, config.secret_key, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            return None
        return int(user_id)
    except JWTError:
        return None


async def get_user_by_username(username: str) -> Optional[User]:
    """Get user by username."""
    async with get_db_session() as db:
        result = await db.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()


async def get_user_by_id(user_id: int) -> Optional[User]:
    """Get user by ID."""
    async with get_db_session() as db:
        return await db.get(User, user_id)


async def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate user with username and password."""
    user = await get_user_by_username(username)
    if not user or not verify_password(password, user.password_hash):
        return None
    return user


async def create_user(
    username: str,
    password: str,
    display_name: str,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    email: Optional[str] = None,
) -> User:
    """Create a new user."""
    async with get_db_session() as db:
        user = User(
            username=username,
            password_hash=hash_password(password),
            display_name=display_name,
            age=age,
            gender=gender,
            email=email,
        )
        db.add(user)
        await db.flush()
        await db.refresh(user)
        return user


# OAuth Token Validation (tokens are managed by FastMCP OAuthProvider)


async def validate_oauth_token(access_token: str) -> Optional[User]:
    """Validate OAuth access token and return user."""
    async with get_db_session() as db:
        result = await db.execute(
            select(OAuthToken).where(
                OAuthToken.access_token == access_token,
                OAuthToken.expires_at > datetime.utcnow(),
            )
        )
        token = result.scalar_one_or_none()

        if not token:
            return None

        return await db.get(User, token.user_id)


# FastAPI Dependencies


async def get_current_user_from_session(request: Request) -> Optional[User]:
    """Get current user from session cookie."""
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return await get_user_by_id(user_id)


async def get_current_user_from_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[User]:
    """Get current user from Bearer token."""
    if not credentials:
        return None

    token = credentials.credentials

    # Try JWT token first
    user_id = decode_access_token(token)
    if user_id:
        return await get_user_by_id(user_id)

    # Try OAuth token
    return await validate_oauth_token(token)


async def get_current_user(
    request: Request,
    token_user: Optional[User] = Depends(get_current_user_from_token),
) -> Optional[User]:
    """Get current user from session or token."""
    if token_user:
        return token_user
    return await get_current_user_from_session(request)


async def require_user(user: Optional[User] = Depends(get_current_user)) -> User:
    """Require authenticated user."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return user
