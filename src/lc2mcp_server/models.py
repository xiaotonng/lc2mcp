"""Database models."""

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class User(Base):
    """User model."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    gender: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)  # male/female/other
    avatar_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    sessions: Mapped[list["Session"]] = relationship(back_populates="user", cascade="all, delete")


class Session(Base):
    """Chat session model."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(200), default="New Chat")
    model: Mapped[str] = mapped_column(String(50), default="gpt-5.2")  # gpt-5.2 or gemini-3-pro
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="sessions")
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="session", cascade="all, delete"
    )


class Conversation(Base):
    """Single conversation turn (input + output + tool calls + cost)."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("sessions.id"), nullable=False)
    
    # Input
    input: Mapped[str] = mapped_column(Text, nullable=False)  # User message
    input_files: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # Uploaded file paths
    
    # Output
    output: Mapped[str] = mapped_column(Text, nullable=False)  # AI response
    
    # Tool calls: [{name, args, result, latency_ms}]
    tool_calls: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    
    # Model info
    model: Mapped[str] = mapped_column(String(50), nullable=False)  # Model used
    
    # Token usage
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Cost (in USD or credits)
    cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Performance
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Total response time
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    # Relationships
    session: Mapped["Session"] = relationship(back_populates="conversations")


class OAuthCode(Base):
    """OAuth authorization code model."""

    __tablename__ = "oauth_codes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    client_id: Mapped[str] = mapped_column(String(100), nullable=False)
    redirect_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    code_challenge: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    scope: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    expires_at: Mapped[float] = mapped_column(Float, nullable=False)  # Unix timestamp

    # Relationships
    user: Mapped["User"] = relationship()


class OAuthToken(Base):
    """OAuth access token model."""

    __tablename__ = "oauth_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    access_token: Mapped[str] = mapped_column(String(500), unique=True, nullable=False)
    refresh_token: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    client_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    scope: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    expires_at: Mapped[float] = mapped_column(Float, nullable=False)  # Unix timestamp

    # Relationships
    user: Mapped["User"] = relationship()


class MCPConnection(Base):
    """MCP platform connection record."""

    __tablename__ = "mcp_connections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform_id: Mapped[str] = mapped_column(String(50), nullable=False)  # chatgpt, cursor, claude, etc.
    client_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    last_call_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_method: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    call_count: Mapped[int] = mapped_column(Integer, default=0)
    is_connected: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class ModelConfig(Base):
    """User model configuration."""

    __tablename__ = "model_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)  # Display name
    provider: Mapped[str] = mapped_column(String(50), nullable=False)  # openai, google, openrouter, custom
    model_id: Mapped[str] = mapped_column(String(100), nullable=False)  # gpt-5.2, gemini-3-pro, etc.
    base_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # Custom API base URL
    api_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # Encrypted API key
    is_preset: Mapped[bool] = mapped_column(Boolean, default=False)  # Whether this is a preset model
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # Extra configuration
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship()
